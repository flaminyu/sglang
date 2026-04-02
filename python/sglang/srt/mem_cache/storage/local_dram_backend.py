# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project
"""
HiCache Local DRAM Storage Backend.

This backend stores KV cache in CPU pinned memory (RAM) instead of disk.
It's designed to enable fast CPU->GPU load_back for the Continuum TTL spill mechanism.

Key advantages over "file" backend:
- ~50-100 GB/s bandwidth vs disk I/O at 1-5 GB/s
- No filesystem overhead (no open/read/write syscalls)
- Enables successful write_backup() and load_back() operations

The storage is organized as a simple dict mapping key -> pinned CPU tensor.
Each entry stores the actual KV data for one prefix (key) as a pinned CPU tensor.
"""

import logging
import threading
from typing import Dict, List, Optional

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)

logger = logging.getLogger(__name__)


class HiCacheLocalDRAM(HiCacheStorage):
    """
    Local DRAM (CPU pinned memory) storage backend for HiCache.

    Uses a simple dict-based storage with thread-safe access.
    KV pairs are stored as pinned CPU tensors for fast GPU transfer.

    This is the recommended backend for single-node setups where
    load_back activation is critical for TTL spill performance.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        pinned: bool = True,
    ):
        self.storage_config = storage_config
        self.tp_rank = storage_config.tp_rank
        self.tp_size = storage_config.tp_size
        self.pp_rank = storage_config.pp_rank
        self.pp_size = storage_config.pp_size
        self.is_mla_model = storage_config.is_mla_model

        # Initialize storage dict
        self._storage: Dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

        # Optionally pin memory for faster GPU transfer
        self.pinned = pinned

        # Page size will be set when register_mem_pool_host is called
        self._page_size: int = 0

        logger.info(
            f"HiCacheLocalDRAM initialized: "
            f"pinned={pinned}, "
            f"tp_rank={self.tp_rank}, tp_size={self.tp_size}, "
            f"is_mla={self.is_mla_model}"
        )

    def register_mem_pool_host(self, mem_pool_host):
        """Store reference to host memory pool for reading data at load-back time."""
        self.mem_pool_host = mem_pool_host
        self._page_size = mem_pool_host.page_size

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional = None,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a value from DRAM storage.

        Copies the stored tensor to target_location if provided.
        """
        with self._lock:
            if key not in self._storage:
                return None

            stored_tensor = self._storage[key]

            if target_location is not None:
                target_location.copy_(stored_tensor)
                return target_location
            return stored_tensor

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional = None,
    ) -> bool:
        """
        Store a value in DRAM storage.

        Accepts either `value` (a tensor) or `target_location` (host indices).
        The stored tensor is always a CPU tensor (pinned if self.pinned=True).
        """
        with self._lock:
            if key in self._storage:
                return True

            try:
                if target_location is not None:
                    # Clone the data (target_location points to host memory).
                    # We need to clone so the data persists after host memory is reused.
                    if self.pinned:
                        data = target_location.clone().pin_memory()
                    else:
                        data = target_location.clone()
                    self._storage[key] = data
                elif value is not None:
                    if self.pinned and not value.is_pinned():
                        value = value.pin_memory()
                    self._storage[key] = value
                else:
                    logger.error(f"HiCacheLocalDRAM.set(): neither value nor target_location provided for key={key}")
                    return False

                return True

            except Exception as e:
                logger.error(f"HiCacheLocalDRAM.set() failed for key={key}: {e}")
                return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        with self._lock:
            return key in self._storage

    # Deprecated: batch_set/batch_get are superseded by batch_set_v1/batch_get_v1
    def batch_get(
        self,
        keys: List[str],
        target_locations=None,
        target_sizes=None,
    ) -> List[Optional[torch.Tensor]]:
        """Deprecated. Delegate to get()."""
        return [self.get(key, loc) for key, loc in zip(keys, target_locations or [None] * len(keys))]

    def batch_set(
        self,
        keys: List[str],
        values=None,
        target_locations=None,
        target_sizes=None,
    ) -> bool:
        """Deprecated. Delegate to set()."""
        for key, value in zip(keys, values or []):
            if not self.set(key, value):
                return False
        return True

    def clear(self) -> bool:
        """Clear all stored KV pairs."""
        with self._lock:
            self._storage.clear()
            logger.info("HiCacheLocalDRAM storage cleared")
        return True

    def _extract_page_from_host(self, start_idx: int) -> Optional[torch.Tensor]:
        """Extract a full page of data from host memory pool at start_idx.

        The start_idx is a token index into the host memory pool.
        The method extracts tokens [start_idx, start_idx + page_size).

        Returns a tensor with shape [page_size * total_features] flattened.
        The returned tensor preserves the full shape so _write_page_to_host
        can reconstruct it correctly.
        """
        page_size = self._page_size
        layout = getattr(self.mem_pool_host, 'layout', None)
        kv_buffer = self.mem_pool_host.kv_buffer

        try:
            if layout == "layer_first":
                # Shape: [2, layer_num, total, head_num, head_dim]
                # Tokens are at dimension 1 (after K+V and layer_num).
                # Extract tokens [start_idx, start_idx + page_size) at dimension 1.
                # Shape: [2, layer_num, page_size, head_num, head_dim]
                data = kv_buffer[:, :, start_idx:start_idx + page_size, :, :].clone()
            elif layout == "page_first":
                # Shape: [2, total, layer_num, head_num, head_dim]
                # Tokens are at dimension 1.
                data = kv_buffer[:, start_idx:start_idx + page_size, :, :, :].clone()
            elif layout in ("page_first_direct", "page_head"):
                # Shape: [2, page_num, layer_num, page_size, head_num, head_dim]
                # start_idx is a flat token index; page_idx = start_idx // page_size
                page_idx = start_idx // page_size
                data = kv_buffer[:, page_idx:page_idx + 1, :, :, :, :].clone()
            else:
                # Fallback: treat as contiguous, try to slice last dim
                data = kv_buffer[..., start_idx:start_idx + page_size].clone()

            # Pin if needed and flatten
            if self.pinned and not data.is_pinned():
                data = data.pin_memory()
            return data.flatten().contiguous()
        except Exception as e:
            logger.error(f"HiCacheLocalDRAM._extract_page_from_host failed at start_idx={start_idx}: {e}")
            return None

    def _write_page_to_host(self, dest_idx: int, page_tensor: torch.Tensor) -> bool:
        """Write a flat page_tensor back to host memory pool at dest_idx.

        The dest_idx is a token index. The page_tensor is flat.
        IMPORTANT: The shape to reshape to depends on the layout, NOT on a
        generic (2, ...) order. The layout-specific shapes are:
          layer_first:        (2, layer_num, page_size, head_num, head_dim)
          page_first:         (2, page_size, layer_num, head_num, head_dim)
          page_first_direct:  (2, 1, layer_num, page_size, head_num, head_dim)
          page_head:          (2, 1, head_num, page_size, layer_num, head_dim)
        """
        page_size = self._page_size
        layout = getattr(self.mem_pool_host, 'layout', None)
        kv_buffer = self.mem_pool_host.kv_buffer
        layer_num = getattr(self.mem_pool_host, 'layer_num', None)
        head_num = getattr(self.mem_pool_host, 'head_num', None)
        head_dim = getattr(self.mem_pool_host, 'head_dim', None)

        try:
            if layout == "layer_first":
                # Extract shape: (2, layer_num, page_size, head_num, head_dim)
                # Must reshape to match exactly before writing to kv_buffer
                # which has shape (2, layer_num, size, head_num, head_dim)
                reshaped = page_tensor.reshape(2, layer_num, page_size, head_num, head_dim)
                kv_buffer[:, :, dest_idx:dest_idx + page_size, :, :] = reshaped

            elif layout == "page_first":
                # Extract shape: (2, page_size, layer_num, head_num, head_dim)
                # kv_buffer shape: (2, size, layer_num, head_num, head_dim)
                reshaped = page_tensor.reshape(2, page_size, layer_num, head_num, head_dim)
                kv_buffer[:, dest_idx:dest_idx + page_size, :, :, :] = reshaped

            elif layout == "page_first_direct":
                # Extract shape: (2, 1, layer_num, page_size, head_num, head_dim)
                # kv_buffer shape: (2, page_num, layer_num, page_size, head_num, head_dim)
                page_idx = dest_idx // page_size
                reshaped = page_tensor.reshape(
                    2, 1, layer_num, page_size, head_num, head_dim
                )
                kv_buffer[:, page_idx:page_idx + 1, :, :, :, :] = reshaped

            elif layout == "page_head":
                # Extract shape: (2, 1, head_num, page_size, layer_num, head_dim)
                # kv_buffer shape: (2, page_num, head_num, page_size, layer_num, head_dim)
                page_idx = dest_idx // page_size
                reshaped = page_tensor.reshape(
                    2, 1, head_num, page_size, layer_num, head_dim
                )
                kv_buffer[:, page_idx:page_idx + 1, :, :, :, :] = reshaped

            else:
                # Fallback: treat as contiguous, infer from available dims
                kv_buffer[..., dest_idx:dest_idx + page_size] = page_tensor.reshape(
                    -1, page_size
                )
            return True
        except Exception as e:
            logger.error(f"HiCacheLocalDRAM._write_page_to_host failed at dest_idx={dest_idx}: {e}")
            return False

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Batch version of set() for storage backup.

        Called by HiCacheController._page_backup() in the backup thread.
        Each key corresponds to one page of KV data at host_indices.

        The host_indices is a flat tensor of ALL token indices to back up.
        Pages are extracted by reading consecutive groups of page_size tokens.

        Args:
            keys: List of hash strings (one per page)
            host_indices: Flat tensor of token indices in host memory pool.
                        Keys[i] corresponds to tokens host_indices[i*page_size : (i+1)*page_size].
            extra_info: Optional metadata (prefix_keys etc.)

        Returns:
            List of booleans indicating success for each key.
        """
        results = []
        page_size = self._page_size

        if page_size == 0:
            logger.error("HiCacheLocalDRAM.batch_set_v1: page_size not set (call register_mem_pool_host first)")
            return [False] * len(keys)

        host_indices_list = host_indices.tolist()

        with self._lock:
            for i, key in enumerate(keys):
                if key in self._storage:
                    results.append(True)
                    continue

                try:
                    # Extract the page: tokens [i*page_size, (i+1)*page_size)
                    page_start = host_indices_list[i * page_size]
                    data = self._extract_page_from_host(page_start)
                    if data is None:
                        results.append(False)
                        continue

                    self._storage[key] = data
                    results.append(True)

                except Exception as e:
                    logger.error(f"HiCacheLocalDRAM.batch_set_v1() failed for key={key}: {e}")
                    results.append(False)

        return results

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Batch version of get() for storage prefetch/load-back.

        Copies stored KV data to the host memory pool at host_indices.

        Args:
            keys: List of hash strings
            host_indices: Flat tensor of destination indices in host memory pool.
                         Keys[i] corresponds to tokens at host_indices[i*page_size : (i+1)*page_size].

        Returns:
            List of booleans indicating whether each key was found and written.
        """
        results = []
        page_size = self._page_size

        if page_size == 0:
            logger.error("HiCacheLocalDRAM.batch_get_v1: page_size not set")
            return [False] * len(keys)

        host_indices_list = host_indices.tolist()

        with self._lock:
            for i, key in enumerate(keys):
                if key not in self._storage:
                    results.append(False)
                    continue

                try:
                    stored_tensor = self._storage[key]
                    page_start = host_indices_list[i * page_size]
                    ok = self._write_page_to_host(page_start, stored_tensor)
                    results.append(ok)

                except Exception as e:
                    logger.error(f"HiCacheLocalDRAM.batch_get_v1() failed for key={key}: {e}")
                    results.append(False)

        return results

    def size(self) -> int:
        """Return approximate storage size in bytes."""
        with self._lock:
            total = 0
            for tensor in self._storage.values():
                total += tensor.numel() * tensor.element_size()
            return total

    def count(self) -> int:
        """Return number of stored KV pairs."""
        with self._lock:
            return len(self._storage)

    def __repr__(self) -> str:
        return (
            f"HiCacheLocalDRAM("
            f"count={self.count()}, "
            f"size_mb={self.size() / 1024**2:.1f}MB, "
            f"pinned={self.pinned})"
        )
