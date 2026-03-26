from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)
_jit_rope_disabled = False


def _apply_rope_torch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int,
) -> None:
    pos = positions.to(torch.long)
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    half_dim = rope_dim // 2
    cos_sin = cos_sin_cache.index_select(0, pos)
    cos = cos_sin[:, :half_dim]
    sin = cos_sin[:, half_dim:rope_dim]

    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]
    q[..., :rope_dim] = apply_rotary_emb(q_rope, cos, sin, is_neox)
    k[..., :rope_dim] = apply_rotary_emb(k_rope, cos, sin, is_neox)


@cache_once
def _jit_fused_rope_module(is_neox: bool, rope_dim: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_neox, rope_dim, is_arch_support_pdl(), dtype)
    return load_jit(
        "fused_rope",
        *args,
        cuda_files=["elementwise/rope.cuh"],
        cuda_wrappers=[
            ("run_rope", f"FusedRopeKernel<{args}>::run"),
            ("run_rope_store", f"FusedRopeKernel<{args}>::run_fused"),
        ],
    )


@dataclass
class FusedSetKVBufferArg:
    """
    value : Optional[torch.Tensor]
        Value tensor, shape: ``(nnz, num_v_heads * head_size)``.
    k_buffer : Optional[torch.Tensor]
        Buffer for keys, shape: ``(nnz, num_k_heads * head_size)``.
    v_buffer : Optional[torch.Tensor]
        Buffer for values, shape: ``(nnz, num_v_heads * head_size)``.
    cache_loc : Optional[torch.Tensor]
        Cache location tensor, used for indexing kv cache.
    """

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    cache_loc: torch.Tensor


@register_custom_op(mutates_args=["q", "k"])
def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace rotary position embedding for query and key tensors.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, rope_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, rope_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim],
            where the first half along dim=-1 is cos and the second half is sin.
            Must be float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved style (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    global _jit_rope_disabled
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    if _jit_rope_disabled:
        _apply_rope_torch_fallback(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
        return

    try:
        module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
        module.run_rope(q, k, cos_sin_cache, positions)
    except Exception as exc:
        _jit_rope_disabled = True
        logger.warning(
            "Failed to load/run JIT fused rope kernel. Falling back to torch implementation. error=%s",
            exc,
        )
        _apply_rope_torch_fallback(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )


@register_custom_op(mutates_args=["q", "k_cache", "v_cache"])
def apply_rope_inplace_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
) -> None:
    """
    Fused inplace RoPE + KV cache store.

    Applies rotary position embedding to q inplace. For k, applies RoPE and
    stores the result in k_cache. The original v is also stored in v_cache.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        v: Value tensor of shape [num_tokens, num_kv_heads, head_dim].
        k_cache: Key cache of shape [cache_size, num_kv_heads * head_dim].
        v_cache: Value cache of shape [cache_size, num_kv_heads * head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        out_loc: Cache write locations of shape [num_tokens], same dtype as positions.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
    """
    global _jit_rope_disabled
    rope_dim = rope_dim or cos_sin_cache.size(-1)
    v = v.view_as(k)

    if _jit_rope_disabled:
        _apply_rope_torch_fallback(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
        k_cache.index_copy_(0, out_loc.to(torch.long), k.reshape(k.shape[0], -1))
        v_cache.index_copy_(0, out_loc.to(torch.long), v.reshape(v.shape[0], -1))
        return

    try:
        module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
        module.run_rope_store(q, k, v, k_cache, v_cache, cos_sin_cache, positions, out_loc)
    except Exception as exc:
        _jit_rope_disabled = True
        logger.warning(
            "Failed to load/run JIT fused rope+store kernel. Falling back to torch implementation. error=%s",
            exc,
        )
        _apply_rope_torch_fallback(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
        k_cache.index_copy_(0, out_loc.to(torch.long), k.reshape(k.shape[0], -1))
        v_cache.index_copy_(0, out_loc.to(torch.long), v.reshape(v.shape[0], -1))


# NOTE: this name is intentionally set as the old kernel in `sgl_kernel`
def apply_rope_with_cos_sin_cache_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    is_neox: bool,
    rope_dim: int = 0,
    fused_args: Optional[FusedSetKVBufferArg] = None,
) -> None:
    """
    Apply RoPE to q and k inplace, with optional fused kv cache store.

    If `fused_args` is provided, it will perform fused RoPE and KV cache store.
    Otherwise, it will only apply RoPE inplace.

    Args:
        q: Query tensor of shape [num_tokens, num_qo_heads, head_dim].
        k: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
        cos_sin_cache: Cosine/sine cache of shape [max_position, rope_dim], float32.
        positions: Position indices of shape [num_tokens], int32 or int64.
        is_neox: Whether to use GPT-NeoX style (True) or GPT-J interleaved (False).
        rope_dim: Rotary embedding dimension. Defaults to cos_sin_cache.size(-1).
        fused_args: Optional arguments for fused RoPE + KV cache store. If None,
            only RoPE will be applied inplace without touching kv cache.
    """
    if fused_args is not None:
        apply_rope_inplace_with_kvcache(
            q,
            k,
            fused_args.value,
            fused_args.k_buffer,
            fused_args.v_buffer,
            cos_sin_cache,
            positions,
            fused_args.cache_loc,
            is_neox=is_neox,
            rope_dim=rope_dim,
        )
    else:
        apply_rope_inplace(
            q, k, cos_sin_cache, positions, is_neox=is_neox, rope_dim=rope_dim
        )
