from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, Optional


_TRACE_LOCK = threading.Lock()
_NAMESPACE_TRACE: Dict[str, Dict[str, Any]] = {}


def _clean_text_excerpt(text: Optional[str], max_chars: int = 240) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _update_trace(extra_key: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    if not extra_key:
        return {}
    filtered = {k: v for k, v in payload.items() if v is not None}
    with _TRACE_LOCK:
        slot = _NAMESPACE_TRACE.setdefault(extra_key, {})
        slot.update(filtered)
        return copy.deepcopy(slot)


def register_namespace_request(
    extra_key: Optional[str],
    *,
    program_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    task_type: Optional[str] = None,
    prompt_text: Optional[str] = None,
    turn_index: Optional[int] = None,
    turn_count: Optional[int] = None,
    ttl_sec: Optional[float] = None,
    ttl_source: Optional[str] = None,
) -> Dict[str, Any]:
    return _update_trace(
        extra_key,
        {
            "program_id": program_id,
            "tool_name": tool_name,
            "task_type": task_type,
            "turn_index": turn_index,
            "turn_count": turn_count,
            "ttl_sec": round(float(ttl_sec), 6) if ttl_sec is not None else None,
            "ttl_source": ttl_source,
            "prompt_excerpt": _clean_text_excerpt(prompt_text),
            "last_request_ts": round(time.time(), 6),
        },
    )


def register_namespace_response(
    extra_key: Optional[str],
    *,
    output_text: Optional[str] = None,
    queueing_sec: Optional[float] = None,
    e2e_latency: Optional[float] = None,
) -> Dict[str, Any]:
    return _update_trace(
        extra_key,
        {
            "output_excerpt": _clean_text_excerpt(output_text),
            "last_queueing_sec": round(float(queueing_sec), 6)
            if queueing_sec is not None
            else None,
            "last_e2e_latency_sec": round(float(e2e_latency), 6)
            if e2e_latency is not None
            else None,
            "last_response_ts": round(time.time(), 6),
        },
    )


def record_namespace_load_back(
    extra_key: Optional[str],
    *,
    node_id: int,
    num_tokens: int,
    kv_cache_size_bytes: int,
    duration_sec: float,
) -> Dict[str, Any]:
    return _update_trace(
        extra_key,
        {
            "last_load_back_node_id": int(node_id),
            "last_load_back_tokens": int(num_tokens),
            "last_load_back_bytes": int(kv_cache_size_bytes),
            "last_load_back_duration_sec": round(float(duration_sec), 6),
            "last_load_back_ts": round(time.time(), 6),
        },
    )


def get_namespace_trace(extra_key: Optional[str]) -> Dict[str, Any]:
    if not extra_key:
        return {}
    with _TRACE_LOCK:
        slot = _NAMESPACE_TRACE.get(extra_key)
        return copy.deepcopy(slot) if slot else {}