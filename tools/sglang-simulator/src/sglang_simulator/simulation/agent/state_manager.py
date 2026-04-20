#!/usr/bin/env python3
"""
扩展的 StateManager - 多级缓存命中追踪

支持追踪 L0（GPU）、L1（Host）、L2（Disk）三层缓存的命中情况。
用于精确模拟 Baseline 和 Continuum 策略的差异。
"""
from __future__ import annotations

from typing import Dict, Optional, Any
import time


# ============================================================================
# 数据类定义
# ============================================================================

class CacheHitStats:
    """缓存命中统计"""
    def __init__(self):
        self.l0_hit_tokens: int = 0
        self.l1_hit_tokens: int = 0
        self.l2_hit_tokens: int = 0
        self.miss_tokens: int = 0
        self.l0_hit_count: int = 0
        self.l1_hit_count: int = 0
        self.l2_hit_count: int = 0
        self.miss_count: int = 0
    
    @property
    def total_hit_tokens(self) -> int:
        return self.l0_hit_tokens + self.l1_hit_tokens + self.l2_hit_tokens
    
    @property
    def total_hit_count(self) -> int:
        return self.l0_hit_count + self.l1_hit_count + self.l2_hit_count
    
    @property
    def l0_hit_rate(self) -> float:
        total = self.total_hit_tokens + self.miss_tokens
        return self.l0_hit_tokens / max(1, total)
    
    @property
    def l1_hit_rate(self) -> float:
        total = self.total_hit_tokens + self.miss_tokens
        return self.l1_hit_tokens / max(1, total)
    
    @property
    def l2_hit_rate(self) -> float:
        total = self.total_hit_tokens + self.miss_tokens
        return self.l2_hit_tokens / max(1, total)
    
    @property
    def overall_hit_rate(self) -> float:
        total = self.total_hit_tokens + self.miss_tokens
        return self.total_hit_tokens / max(1, total)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "l0_hit_tokens": self.l0_hit_tokens,
            "l1_hit_tokens": self.l1_hit_tokens,
            "l2_hit_tokens": self.l2_hit_tokens,
            "miss_tokens": self.miss_tokens,
            "l0_hit_count": self.l0_hit_count,
            "l1_hit_count": self.l1_hit_count,
            "l2_hit_count": self.l2_hit_count,
            "miss_count": self.miss_count,
            "total_hit_tokens": self.total_hit_tokens,
            "total_hit_count": self.total_hit_count,
            "l0_hit_rate": self.l0_hit_rate,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hit_rate": self.l2_hit_rate,
            "overall_hit_rate": self.overall_hit_rate,
        }


class LatencyStats:
    """延迟统计"""
    def __init__(self):
        self.inference_dur: float = 0.0
        self.l1_load_dur: float = 0.0
        self.l1_backup_dur: float = 0.0
        self.l2_load_dur: float = 0.0
        self.l2_backup_dur: float = 0.0
    
    @property
    def total_dur(self) -> float:
        return self.inference_dur + self.l1_load_dur + self.l1_backup_dur + self.l2_load_dur + self.l2_backup_dur
    
    @property
    def non_inference_dur(self) -> float:
        return self.l1_load_dur + self.l1_backup_dur + self.l2_load_dur + self.l2_backup_dur
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_dur": self.inference_dur,
            "l1_load_dur": self.l1_load_dur,
            "l1_backup_dur": self.l1_backup_dur,
            "l2_load_dur": self.l2_load_dur,
            "l2_backup_dur": self.l2_backup_dur,
            "total_dur": self.total_dur,
            "non_inference_dur": self.non_inference_dur,
        }


class RequestStats:
    """请求级统计"""
    def __init__(
        self,
        rid: str,
        program_id: Optional[str] = None,
        turn_index: int = 0,
        is_tool_call: bool = False,
        tool_name: Optional[str] = None,
        tool_duration: float = 0.0,
    ):
        self.rid = rid
        self.program_id = program_id
        self.turn_index = turn_index
        self.l0_hit_tokens: int = 0
        self.l1_hit_tokens: int = 0
        self.l2_hit_tokens: int = 0
        self.miss_tokens: int = 0
        self.queue_time: float = 0.0
        self.l1_load_time: float = 0.0
        self.l2_load_time: float = 0.0
        self.inference_time: float = 0.0
        self.is_tool_call = is_tool_call
        self.tool_name = tool_name
        self.tool_duration = tool_duration
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.l0_hit_tokens + self.l1_hit_tokens + self.l2_hit_tokens + self.miss_tokens
        hit = self.l0_hit_tokens + self.l1_hit_tokens + self.l2_hit_tokens
        return hit / max(1, total)
    
    @property
    def total_time(self) -> float:
        return self.queue_time + self.l1_load_time + self.l2_load_time + self.inference_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rid": self.rid,
            "program_id": self.program_id,
            "turn_index": self.turn_index,
            "l0_hit_tokens": self.l0_hit_tokens,
            "l1_hit_tokens": self.l1_hit_tokens,
            "l2_hit_tokens": self.l2_hit_tokens,
            "miss_tokens": self.miss_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "queue_time": self.queue_time,
            "l1_load_time": self.l1_load_time,
            "l2_load_time": self.l2_load_time,
            "inference_time": self.inference_time,
            "total_time": self.total_time,
            "is_tool_call": self.is_tool_call,
            "tool_name": self.tool_name,
            "tool_duration": self.tool_duration,
        }


# ============================================================================
# 扩展的 StateManager
# ============================================================================

class AgentStateManager:
    """
    扩展的 StateManager，支持多级缓存追踪
    """
    
    # 全局状态
    _iteration: int = 0
    _global_clock: float = 0.0
    _last_inference_dur: float = 0.0
    _current_inference_dur: float = 0.0
    
    # 延迟统计
    _hicache_l1_load_dur: float = 0.0
    _hicache_l1_backup_dur: float = 0.0
    _hicache_l2_load_dur: float = 0.0
    _hicache_l2_backup_dur: float = 0.0
    
    # 缓存命中统计
    _cache_hit_stats: Optional[CacheHitStats] = None
    
    # 请求统计
    _request_stats: Dict[str, RequestStats] = {}
    
    # TTL 节点追踪
    _ttl_nodes: Dict[str, float] = {}
    
    # 当前运行的请求
    _running_requests: Dict[str, Dict[str, Any]] = {}
    
    # 调度器类型
    _scheduler_type: str = "fcfs"
    
    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0.0
        cls._last_inference_dur = 0.0
        cls._current_inference_dur = 0.0
        cls._hicache_l1_load_dur = 0.0
        cls._hicache_l1_backup_dur = 0.0
        cls._hicache_l2_load_dur = 0.0
        cls._hicache_l2_backup_dur = 0.0
        cls._cache_hit_stats = CacheHitStats()
        cls._request_stats = {}
        cls._ttl_nodes = {}
        cls._running_requests = {}
    
    # ========================================================================
    # 迭代和时钟
    # ========================================================================
    
    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1
    
    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration
    
    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock
    
    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        cls._global_clock += dur
    
    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock
    
    @classmethod
    def set_scheduler_type(cls, scheduler_type: str) -> None:
        cls._scheduler_type = scheduler_type
    
    @classmethod
    def get_scheduler_type(cls) -> str:
        return cls._scheduler_type
    
    # ========================================================================
    # 推理延迟
    # ========================================================================
    
    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur
    
    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur
    
    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur
    
    # ========================================================================
    # L1 延迟
    # ========================================================================
    
    @classmethod
    def inc_hicache_l1_load_dur(cls, dur: float) -> None:
        cls._hicache_l1_load_dur += dur
    
    @classmethod
    def inc_hicache_l1_backup_dur(cls, dur: float) -> None:
        cls._hicache_l1_backup_dur += dur
    
    @classmethod
    def get_hicache_l1_load_dur(cls) -> float:
        return cls._hicache_l1_load_dur
    
    @classmethod
    def get_hicache_l1_backup_dur(cls) -> float:
        return cls._hicache_l1_backup_dur
    
    @classmethod
    def pop_hicache_l1_load_dur(cls) -> float:
        dur = cls._hicache_l1_load_dur
        cls._hicache_l1_load_dur = 0.0
        return dur
    
    @classmethod
    def pop_hicache_l1_backup_dur(cls) -> float:
        dur = cls._hicache_l1_backup_dur
        cls._hicache_l1_backup_dur = 0.0
        return dur
    
    # ========================================================================
    # L2 延迟
    # ========================================================================
    
    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur
    
    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur
    
    @classmethod
    def get_hicache_l2_load_dur(cls) -> float:
        return cls._hicache_l2_load_dur
    
    @classmethod
    def get_hicache_l2_backup_dur(cls) -> float:
        return cls._hicache_l2_backup_dur
    
    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0.0
        return dur
    
    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0.0
        return dur
    
    # ========================================================================
    # 缓存命中统计
    # ========================================================================
    
    @classmethod
    def inc_cache_hit(cls, tier: str, tokens: int) -> None:
        if cls._cache_hit_stats is None:
            cls._cache_hit_stats = CacheHitStats()
        stats = cls._cache_hit_stats
        if tier == "l0":
            stats.l0_hit_tokens += tokens
            stats.l0_hit_count += 1
        elif tier == "l1":
            stats.l1_hit_tokens += tokens
            stats.l1_hit_count += 1
        elif tier == "l2":
            stats.l2_hit_tokens += tokens
            stats.l2_hit_count += 1
        else:
            stats.miss_tokens += tokens
            stats.miss_count += 1
    
    @classmethod
    def get_cache_hit_stats(cls) -> CacheHitStats:
        if cls._cache_hit_stats is None:
            cls._cache_hit_stats = CacheHitStats()
        return cls._cache_hit_stats
    
    @classmethod
    def get_cache_hit_stats_dict(cls) -> Dict[str, Any]:
        return cls.get_cache_hit_stats().to_dict()
    
    # ========================================================================
    # 请求级统计
    # ========================================================================
    
    @classmethod
    def create_request_stats(
        cls,
        rid: str,
        program_id: Optional[str] = None,
        turn_index: int = 0,
        is_tool_call: bool = False,
        tool_name: Optional[str] = None,
        tool_duration: float = 0.0,
    ) -> RequestStats:
        stats = RequestStats(
            rid=rid,
            program_id=program_id,
            turn_index=turn_index,
            is_tool_call=is_tool_call,
            tool_name=tool_name,
            tool_duration=tool_duration,
        )
        cls._request_stats[rid] = stats
        return stats
    
    @classmethod
    def get_request_stats(cls, rid: str) -> Optional[RequestStats]:
        return cls._request_stats.get(rid)
    
    @classmethod
    def update_request_cache_hit(
        cls,
        rid: str,
        l0_tokens: int = 0,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
        miss_tokens: int = 0,
    ) -> None:
        stats = cls._request_stats.get(rid)
        if stats:
            stats.l0_hit_tokens += l0_tokens
            stats.l1_hit_tokens += l1_tokens
            stats.l2_hit_tokens += l2_tokens
            stats.miss_tokens += miss_tokens
    
    @classmethod
    def update_request_time(
        cls,
        rid: str,
        queue_time: float = 0.0,
        l1_load_time: float = 0.0,
        l2_load_time: float = 0.0,
        inference_time: float = 0.0,
    ) -> None:
        stats = cls._request_stats.get(rid)
        if stats:
            if queue_time > 0:
                stats.queue_time = queue_time
            if l1_load_time > 0:
                stats.l1_load_time = l1_load_time
            if l2_load_time > 0:
                stats.l2_load_time = l2_load_time
            if inference_time > 0:
                stats.inference_time = inference_time
    
    @classmethod
    def get_all_request_stats(cls) -> Dict[str, RequestStats]:
        return cls._request_stats.copy()
    
    # ========================================================================
    # TTL 节点管理
    # ========================================================================
    
    @classmethod
    def add_ttl_node(cls, key: str, expires_at: float) -> None:
        cls._ttl_nodes[key] = expires_at
    
    @classmethod
    def is_ttl_node_expired(cls, key: str, current_time: Optional[float] = None) -> bool:
        if key not in cls._ttl_nodes:
            return True
        if current_time is None:
            current_time = time.time()
        return current_time > cls._ttl_nodes[key]
    
    @classmethod
    def remove_ttl_node(cls, key: str) -> None:
        cls._ttl_nodes.pop(key, None)
    
    @classmethod
    def get_active_ttl_nodes(cls, current_time: Optional[float] = None) -> Dict[str, float]:
        if current_time is None:
            current_time = time.time()
        return {
            key: expires_at
            for key, expires_at in cls._ttl_nodes.items()
            if current_time <= expires_at
        }
    
    @classmethod
    def cleanup_expired_ttl_nodes(cls, current_time: Optional[float] = None) -> int:
        if current_time is None:
            current_time = time.time()
        expired_keys = [
            key for key, expires_at in cls._ttl_nodes.items()
            if current_time > expires_at
        ]
        for key in expired_keys:
            cls._ttl_nodes.pop(key, None)
        return len(expired_keys)
    
    # ========================================================================
    # 运行请求追踪
    # ========================================================================
    
    @classmethod
    def start_request(
        cls,
        rid: str,
        program_id: str,
        turn_index: int,
        start_time: Optional[float] = None,
    ) -> None:
        if start_time is None:
            start_time = cls._global_clock
        cls._running_requests[rid] = {
            "program_id": program_id,
            "turn_index": turn_index,
            "start_time": start_time,
        }
    
    @classmethod
    def finish_request(cls, rid: str, end_time: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if rid not in cls._running_requests:
            return None
        if end_time is None:
            end_time = cls._global_clock
        info = cls._running_requests.pop(rid)
        info["end_time"] = end_time
        info["duration"] = end_time - info["start_time"]
        return info
    
    @classmethod
    def get_running_requests(cls) -> Dict[str, Dict[str, Any]]:
        return cls._running_requests.copy()
    
    # ========================================================================
    # 延迟统计
    # ========================================================================
    
    @classmethod
    def get_latency_stats(cls) -> LatencyStats:
        stats = LatencyStats()
        stats.inference_dur = cls._current_inference_dur
        stats.l1_load_dur = cls._hicache_l1_load_dur
        stats.l1_backup_dur = cls._hicache_l1_backup_dur
        stats.l2_load_dur = cls._hicache_l2_load_dur
        stats.l2_backup_dur = cls._hicache_l2_backup_dur
        return stats
    
    @classmethod
    def get_latency_stats_dict(cls) -> Dict[str, Any]:
        return cls.get_latency_stats().to_dict()
    
    # ========================================================================
    # 综合统计
    # ========================================================================
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        return {
            "iteration": cls._iteration,
            "global_clock": cls._global_clock,
            "scheduler_type": cls._scheduler_type,
            "cache_hit": cls.get_cache_hit_stats_dict(),
            "latency": cls.get_latency_stats_dict(),
            "running_requests": len(cls._running_requests),
            "total_requests": len(cls._request_stats),
            "active_ttl_nodes": len(cls.get_active_ttl_nodes()),
        }
    
    @classmethod
    def print_summary(cls) -> None:
        summary = cls.get_summary()
        print("\n" + "=" * 60)
        print("AgentStateManager Summary")
        print("=" * 60)
        print(f"Global Clock: {summary['global_clock']:.4f}s")
        print(f"Scheduler Type: {summary['scheduler_type']}")
        print(f"Running Requests: {summary['running_requests']}")
        print(f"Total Requests: {summary['total_requests']}")
        
        cache = summary['cache_hit']
        print(f"\nCache Hit Stats:")
        print(f"  L0 (GPU) Hit: {cache['l0_hit_tokens']:,} tokens ({cache['l0_hit_rate']:.1%})")
        print(f"  L1 (Host) Hit: {cache['l1_hit_tokens']:,} tokens ({cache['l1_hit_rate']:.1%})")
        print(f"  L2 (Disk) Hit: {cache['l2_hit_tokens']:,} tokens ({cache['l2_hit_rate']:.1%})")
        print(f"  Miss: {cache['miss_tokens']:,} tokens")
        print(f"  Overall Hit Rate: {cache['overall_hit_rate']:.1%}")
        
        latency = summary['latency']
        print(f"\nLatency Stats:")
        print(f"  Inference: {latency['inference_dur']:.4f}s")
        print(f"  L1 Load (H2D): {latency['l1_load_dur']:.4f}s")
        print(f"  L1 Backup (D2H): {latency['l1_backup_dur']:.4f}s")
        print(f"  L2 Load (Disk): {latency['l2_load_dur']:.4f}s")
        print(f"  L2 Backup (Disk): {latency['l2_backup_dur']:.4f}s")
        print(f"  Total: {latency['total_dur']:.4f}s")
        print("=" * 60)
