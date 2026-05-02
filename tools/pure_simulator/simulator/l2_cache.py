"""
L2CacheManager: Manages L2 (Host Memory) cache for spilled KV cache entries.

When L1 evicts entries, they are spilled to L2 instead of being discarded.
On L2 hit, the cache needs to be "loaded back" to L1, incurring a penalty.

This module integrates with StateManager to track L2 load/backup durations,
which are needed by the AIConfigurator adapter for accurate time prediction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time as time_module

from .state_manager import StateManager


def calc_transfer_time(
    num_tokens: int,
    bytes_per_token: float,
    bandwidth_GBps: float,
    overhead_us: float,
    efficiency: float = 0.85
) -> float:
    """
    Calculate memory transfer time using amortization model.

    The amortization model accounts for fixed overhead in memory transfers:
    - Small transfers are dominated by the fixed overhead per operation
    - Large transfers approach the raw bandwidth limit

    Formula: time = size / (bandwidth * efficiency) + overhead
    where:
    - size / bandwidth = transfer time at full bandwidth
    - overhead = fixed latency per transfer operation

    Args:
        num_tokens: Number of tokens to transfer
        bytes_per_token: Bytes per token (KV cache size)
        bandwidth_GBps: Raw bandwidth in GB/s
        overhead_us: Fixed overhead per transfer in microseconds
        efficiency: Bandwidth efficiency factor (0-1, default 0.85)

    Returns:
        Transfer time in seconds.
    """
    if num_tokens <= 0:
        return 0.0

    size_bytes = num_tokens * bytes_per_token
    bandwidth_bps = bandwidth_GBps * 1e9 * efficiency  # Convert GB/s to bytes/s with efficiency
    overhead_s = overhead_us * 1e-6  # Convert microseconds to seconds

    # Transfer time = size / bandwidth + fixed overhead
    # For small transfers: dominated by overhead
    # For large transfers: approaches size/bandwidth
    transfer_time = size_bytes / bandwidth_bps + overhead_s

    return transfer_time


@dataclass
class L2CacheEntry:
    """An entry in L2 cache."""
    node_id: int
    token_count: int
    token_ids: List[int]  # Store actual token sequence for prefix matching
    extra_key: str
    created_at: float = 0.0
    last_access: float = 0.0
    access_count: int = 0


@dataclass
class L2CacheStats:
    """L2 cache statistics."""
    l1_to_l2_spills: int = 0
    l2_to_l1_loads: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l2_evictions: int = 0
    total_tokens_spilled: int = 0
    total_tokens_loaded: int = 0


class L2CacheManager:
    """
    Manages L2 (Host Memory) cache.

    L2 serves as a fallback when entries are evicted from L1 (GPU HBM).
    When accessed from L2, entries are "loaded back" to L1 with a penalty.

    LMCache-style state machine:
    None --> pinned --> ready --> evictable
                  (TTL active) (in L2) (can be evicted)

    Write Policy:
    - write_back: TTL-protected nodes are NOT spilled to L2
    - write_through: Always spill to L2 (Continuum default)
    """

    def __init__(
        self,
        capacity_tokens: int = 100000,
        load_back_tokens: int = 1000,  # Default tokens to load back per access
        h2d_penalty_factor: float = 0.3,  # H2D transfer penalty (fraction of original cost)
        write_policy: str = "write_through",  # "write_back" or "write_through"
        # Bandwidth model parameters
        bytes_per_token: float = 512.0,
        bandwidth_GBps: float = 32.0,
        h2d_overhead_us: float = 6.67,
        d2h_overhead_us: float = 4.0,
        memory_bandwidth_GBps: float = 64.0,
    ):
        self.capacity_tokens = capacity_tokens
        self.load_back_tokens = load_back_tokens
        self.h2d_penalty_factor = h2d_penalty_factor
        self.write_policy = write_policy

        # Bandwidth model parameters
        self.bytes_per_token = bytes_per_token
        self.bandwidth_GBps = bandwidth_GBps
        self.h2d_overhead_us = h2d_overhead_us
        self.d2h_overhead_us = d2h_overhead_us
        self.memory_bandwidth_GBps = memory_bandwidth_GBps

        self.entries: Dict[str, L2CacheEntry] = {}  # key -> entry
        self.access_order: List[str] = []  # LRU order for L2
        self.current_tokens = 0

        self.stats = L2CacheStats()
    
    def should_spill(self, node_is_pinned: bool, ttl_expired: bool = False) -> bool:
        """
        Determine if a node should be spilled to L2 based on write policy.
        
        Args:
            node_is_pinned: Whether the node is currently TTL-protected
            ttl_expired: Whether the TTL has expired (default False)
        
        Returns:
            True if the node should be spilled to L2, False otherwise
        
        LMCache-style behavior:
        - write_back: TTL-protected nodes stay in L1
        - write_through: Always spill, including expired TTL nodes
        
        Key insight: When TTL expires, the data should still go to L2
        (via normal eviction) to allow future recovery. Only active
        pinned nodes are protected from L2 spill.
        """
        if self.write_policy == "write_back":
            # write_back: Active TTL-protected nodes stay in L1
            # But EXPIRED TTL nodes should be spilled to L2 (like normal eviction)
            if node_is_pinned and not ttl_expired:
                return False
        # write_through: always spill
        return True

    def _find_contiguous_segments(self, token_count: int) -> List[int]:
        """
        Detect contiguous memory segments.

        For L2 cache with LRU eviction, tokens are stored contiguously
        within each entry. This returns [token_count] for simplicity.

        In a more advanced model, this would track actual memory addresses.

        Args:
            token_count: Number of tokens to check

        Returns:
            List of segment lengths. Currently returns [token_count] for
            a single contiguous segment per entry.
        """
        # Current model: each entry is one contiguous segment
        return [token_count]

    def spill_to_l2(
        self,
        node_id: int,
        token_count: int,
        token_ids: List[int],
        extra_key: str,
        current_time: float = 0.0,
    ) -> bool:
        """
        Spill an evicted entry from L1 to L2.
        
        Args:
            node_id: L1 node ID
            token_count: Number of tokens in this entry
            token_ids: Actual token sequence for prefix matching
            extra_key: Namespace key
            current_time: Current simulation time
        
        Returns True if spill succeeded, False if L2 is full and couldn't evict.
        """
        # Generate L2 key
        l2_key = f"{extra_key}:node_{node_id}"
        
        # Check if already in L2
        if l2_key in self.entries:
            return True
        
        # Need to make space?
        while self.current_tokens + token_count > self.capacity_tokens:
            if not self._evict_lru():
                # L2 is completely full
                return False
        
        # Add to L2
        entry = L2CacheEntry(
            node_id=node_id,
            token_count=token_count,
            token_ids=token_ids,
            extra_key=extra_key,
            created_at=current_time,
            last_access=current_time,
            access_count=0,
        )
        self.entries[l2_key] = entry
        self.access_order.append(l2_key)
        self.current_tokens += token_count
        
        self.stats.l1_to_l2_spills += 1
        self.stats.total_tokens_spilled += token_count

        # Track L2 backup time in StateManager using bandwidth amortization model
        segments = self._find_contiguous_segments(token_count)
        total_backup_time = 0.0
        for seg_len in segments:
            seg_time = calc_transfer_time(
                num_tokens=seg_len,
                bytes_per_token=self.bytes_per_token,
                bandwidth_GBps=self.memory_bandwidth_GBps,
                overhead_us=self.d2h_overhead_us,  # D2H uses d2h_overhead_us
                efficiency=0.85
            )
            total_backup_time += seg_time
        StateManager.inc_hicache_l2_backup_dur(total_backup_time)

        return True
    
    def access_l2(
        self,
        extra_key: str,
        node_id: int,
        current_time: float = 0.0,
    ) -> Tuple[bool, int]:
        """
        Access an entry from L2, loading it back to L1.
        
        Returns: (found, tokens_loaded)
            - found: True if entry was in L2
            - tokens_loaded: Number of tokens to load back
        """
        l2_key = f"{extra_key}:node_{node_id}"
        
        if l2_key not in self.entries:
            self.stats.l2_misses += 1
            return False, 0
        
        # Update LRU
        if l2_key in self.access_order:
            self.access_order.remove(l2_key)
        self.access_order.append(l2_key)
        
        entry = self.entries[l2_key]
        entry.last_access = current_time
        entry.access_count += 1
        
        # Calculate tokens to load back
        tokens_to_load = min(self.load_back_tokens, entry.token_count)
        
        # Remove from L2 (will be loaded to L1)
        del self.entries[l2_key]
        self.current_tokens -= entry.token_count
        self.access_order.remove(l2_key)
        
        self.stats.l2_to_l1_loads += 1
        self.stats.l2_hits += 1
        self.stats.total_tokens_loaded += tokens_to_load
        
        return True, tokens_to_load
    
    def _match_tokens(self, req_tokens: List[int], cached_tokens: List[int]) -> int:
        """Calculate longest common prefix between two token sequences."""
        match_len = 0
        for i in range(min(len(req_tokens), len(cached_tokens))):
            if req_tokens[i] == cached_tokens[i]:
                match_len += 1
            else:
                break
        return match_len
    
    def match_l2(
        self,
        token_ids: List[int],
        extra_key: str
    ) -> Tuple[int, List[Tuple[int, int, List[int]]]]:
        """
        Match tokens against L2 cache entries using prefix matching.
        
        Args:
            token_ids: Request token sequence to match
            extra_key: Namespace key to filter entries
        
        Returns:
            Tuple of (matched_length, list of (node_id, token_count, token_ids))
            The list contains entries that have the maximum match length.
        """
        if not token_ids:
            return 0, []
        
        matched = 0
        results = []
        prefix = f"{extra_key}:node_"
        
        for l2_key, entry in self.entries.items():
            if not l2_key.startswith(prefix):
                continue
            
            # Calculate longest common prefix between request and this L2 entry
            match_len = self._match_tokens(token_ids, entry.token_ids)
            
            if match_len > matched:
                matched = match_len
                results = [(entry.node_id, entry.token_count, entry.token_ids)]
            elif match_len == matched and match_len > 0:
                results.append((entry.node_id, entry.token_count, entry.token_ids))
        
        return matched, results
    
    def find_by_extra_key(self, extra_key: str) -> List[Tuple[int, int, List[int]]]:
        """
        Find all entries for a given extra_key.
        Returns list of (node_id, token_count, token_ids) tuples.
        """
        result = []
        prefix = f"{extra_key}:node_"
        for l2_key, entry in self.entries.items():
            if l2_key.startswith(prefix):
                result.append((entry.node_id, entry.token_count, entry.token_ids))
        return result
    
    def load_all_for_program(
        self,
        extra_key: str,
        current_time: float = 0.0,
    ) -> Tuple[int, List[Tuple[int, int, List[int]]]]:
        """
        Load all cached data for a program from L2.

        Returns: (total_tokens, list of (node_id, tokens_loaded, token_ids))
        """
        entries_to_load = self.find_by_extra_key(extra_key)

        if not entries_to_load:
            self.stats.l2_misses += 1
            return 0, []

        loaded = []
        total_tokens = 0

        for node_id, token_count, token_ids in entries_to_load:
            l2_key = f"{extra_key}:node_{node_id}"
            if l2_key in self.entries:
                entry = self.entries[l2_key]
                tokens_to_load = min(self.load_back_tokens, entry.token_count)
                total_tokens += tokens_to_load
                loaded.append((node_id, tokens_to_load, entry.token_ids))

                # Remove from L2
                self.current_tokens -= entry.token_count
                del self.entries[l2_key]
                self.access_order.remove(l2_key)

                entry.last_access = current_time
                entry.access_count += 1

        self.stats.l2_to_l1_loads += len(loaded)
        self.stats.l2_hits += len(loaded)
        self.stats.total_tokens_loaded += total_tokens

        # Track L2 load time in StateManager using bandwidth amortization model
        if total_tokens > 0:
            segments = self._find_contiguous_segments(total_tokens)
            total_load_time = 0.0
            for seg_len in segments:
                seg_time = calc_transfer_time(
                    num_tokens=seg_len,
                    bytes_per_token=self.bytes_per_token,
                    bandwidth_GBps=self.bandwidth_GBps,
                    overhead_us=self.h2d_overhead_us,  # H2D uses h2d_overhead_us
                    efficiency=0.85
                )
                total_load_time += seg_time
            StateManager.inc_hicache_l2_load_dur(total_load_time)

        return total_tokens, loaded
    
    def lookup(self, extra_key: str, node_id: int) -> bool:
        """Check if an entry exists in L2 without loading it."""
        l2_key = f"{extra_key}:node_{node_id}"
        return l2_key in self.entries
    
    def remove_entry(self, extra_key: str, node_id: int) -> bool:
        """
        Remove a specific entry from L2 cache.
        Used when loading L2 data back to L1.
        
        Returns True if entry was found and removed, False otherwise.
        """
        l2_key = f"{extra_key}:node_{node_id}"
        
        if l2_key not in self.entries:
            return False
        
        entry = self.entries[l2_key]
        self.current_tokens -= entry.token_count
        del self.entries[l2_key]
        if l2_key in self.access_order:
            self.access_order.remove(l2_key)
        
        return True
    
    def _evict_lru(self) -> bool:
        """Evict the least recently used entry from L2."""
        if not self.access_order:
            return False
        
        l2_key = self.access_order.pop(0)
        if l2_key in self.entries:
            entry = self.entries[l2_key]
            self.current_tokens -= entry.token_count
            del self.entries[l2_key]
            self.stats.l2_evictions += 1
            return True
        
        return False
    
    def clear(self):
        """Clear all L2 entries."""
        self.entries.clear()
        self.access_order.clear()
        self.current_tokens = 0
    
    def get_stats(self) -> Dict:
        """Get L2 cache statistics."""
        return {
            "l1_to_l2_spills": self.stats.l1_to_l2_spills,
            "l2_to_l1_loads": self.stats.l2_to_l1_loads,
            "l2_hits": self.stats.l2_hits,
            "l2_misses": self.stats.l2_misses,
            "l2_evictions": self.stats.l2_evictions,
            "total_tokens_spilled": self.stats.total_tokens_spilled,
            "total_tokens_loaded": self.stats.total_tokens_loaded,
            "current_tokens": self.current_tokens,
            "capacity_tokens": self.capacity_tokens,
            "hit_rate": self.stats.l2_hits / (self.stats.l2_hits + self.stats.l2_misses) 
                        if (self.stats.l2_hits + self.stats.l2_misses) > 0 else 0.0,
        }
