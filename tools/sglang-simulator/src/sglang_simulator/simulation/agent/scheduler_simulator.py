#!/usr/bin/env python3
"""
Scheduler Hook 和调度策略模拟

支持 Baseline (FCFS) 和 DTTL (Dynamic TTL) 两种调度策略的模拟。
正确模拟 L0 (GPU) + L1 (Host) 两级缓存架构。

核心设计：
1. Baseline: FCFS + LRU，GPU 满时淘汰到 Host，GPU miss 时查 Host 并 H2D 传输
2. DTTL (Dynamic TTL): 同样架构 + 动态 TTL Pinning
   - 基于历史工具执行数据动态计算 TTL
   - 工具调用时锁定减少淘汰

NOTE: DTTL 是动态的，不是固定的静态值。TTL 通过以下公式计算：
    τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ

Usage:
    from sglang_simulator.simulation.agent.scheduler_simulator import SchedulerSimulator

    simulator = SchedulerSimulator(
        mode="dttl",
        max_tokens=...,
        ttl_simulator=ttl_sim,
        state_manager=state_mgr,
    )
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_simulator.simulation.agent.dataset import AgentProgram, TurnSpec
    from sglang_simulator.simulation.agent.state_manager import AgentStateManager
    from sglang_simulator.simulation.agent.continuum_ttl import ContinuumTTLSimulator


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class QueuedRequest:
    """排队的请求"""
    rid: str
    program_id: str
    turn_index: int
    arrival_time: float  # 到达时间
    input_tokens: int
    output_tokens: int
    is_tool_call: bool
    tool_name: Optional[str]
    tool_duration: float
    priority: int = 0  # 优先级
    
    @property
    def total_tokens(self) -> int:
        """总 tokens (输入 + 输出)"""
        return self.input_tokens + self.output_tokens


@dataclass
class PinnedRequest:
    """被 pin 的请求（DTTL 动态 TTL 策略）"""
    rid: str
    program_id: str
    turn_index: int
    pinned_at: float
    ttl_sec: float  # 动态计算的 TTL 值
    tokens: int
    
    @property
    def expires_at(self) -> float:
        return self.pinned_at + self.ttl_sec
    
    def is_expired(self, current_time: float) -> bool:
        return current_time > self.expires_at


@dataclass
class CacheHitResult:
    """缓存命中结果
    
    命中层级：
    - l0_tokens: GPU (HBM) 直接命中
    - l1_tokens: Host 命中，需要 H2D 传输
    - miss_tokens: 未命中，需要重新计算
    """
    l0_tokens: int = 0  # GPU 命中
    l1_tokens: int = 0  # Host 命中 (H2D)
    miss_tokens: int = 0  # 未命中
    
    @property
    def total_hit_tokens(self) -> int:
        return self.l0_tokens + self.l1_tokens


@dataclass 
class ScheduleResult:
    """调度结果"""
    rid: str
    program_id: str
    turn_index: int
    cache_hit: CacheHitResult
    queue_time: float
    h2d_time: float  # Host -> GPU 传输时间
    inference_time: float
    total_time: float  # 处理时间 (不含 queue_time)
    was_pinned: bool
    evicted_tokens: int  # 从 GPU 淘汰到 Host 的 tokens


# ============================================================================
# Baseline 调度器模拟
# ============================================================================

class BaselineScheduler:
    """
    Baseline 调度器: FCFS + LRU
    
    缓存架构:
    - L0 (GPU): 主要 KV Cache
    - L1 (Host): GPU 淘汰的 KV Cache 备份
    
    行为:
    1. 请求按到达顺序处理 (FCFS)
    2. GPU miss 时查 Host Cache，然后 H2D 传输
    3. GPU 满时 LRU 淘汰到 Host
    """
    
    def __init__(
        self,
        max_tokens: int,
        max_host_tokens: int,
        state_manager: "AgentStateManager",
        h2d_bandwidth_gb: float = 50.0,
        inference_speed: float = 0.0001,  # tokens/ms
    ):
        self.max_tokens = max_tokens  # GPU 缓存容量
        self.max_host_tokens = max_host_tokens  # Host 缓存容量
        self.state_manager = state_manager
        self.h2d_bandwidth_gb = h2d_bandwidth_gb
        self.inference_speed = inference_speed  # tokens per ms
        
        # 队列
        self.queue: deque = deque()
        
        # 两级缓存
        # L0 (GPU): Dict[(program_id, turn_index)] = (tokens, last_access_time)
        self.l0_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l0_tokens = 0
        
        # L1 (Host): Dict[(program_id, turn_index)] = (tokens, last_access_time)
        self.l1_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l1_tokens = 0
        
        # 统计
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "evicted_tokens": 0,
        }
    
    def reset(self):
        """重置调度器"""
        self.queue.clear()
        self.l0_cache.clear()
        self.l1_cache.clear()
        self.current_l0_tokens = 0
        self.current_l1_tokens = 0
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "evicted_tokens": 0,
        }
    
    def enqueue(self, request: QueuedRequest):
        """入队请求"""
        self.queue.append(request)
    
    def add_request(self, request: QueuedRequest):
        """添加请求到队列"""
        self.queue.append(request)
    
    def _find_cache_hit(self, request: QueuedRequest) -> CacheHitResult:
        """
        查找两级缓存命中
        
        1. 先查 L0 (GPU) 缓存
        2. L0 miss，查 L1 (Host) 缓存
        3. L1 hit，产生 H2D 传输延迟
        4. 都 miss，需要重新计算
        """
        # 在 L0 (GPU) 缓存中查找最佳匹配
        l0_match = self._find_best_match(
            request.program_id,
            request.turn_index,
            request.input_tokens,
            self.l0_cache
        )
        
        if l0_match > 0:
            self.stats["l0_hits"] += 1
            # 更新 LRU 时间
            self._touch_cache(request.program_id, request.turn_index - 1, self.l0_cache)
            return CacheHitResult(
                l0_tokens=l0_match,
                l1_tokens=0,
                miss_tokens=request.input_tokens - l0_match,
            )
        
        # L0 miss，查 L1 (Host) 缓存
        l1_match = self._find_best_match(
            request.program_id,
            request.turn_index,
            request.input_tokens,
            self.l1_cache
        )
        
        if l1_match > 0:
            self.stats["l1_hits"] += 1
            # 更新 LRU 时间
            self._touch_cache(request.program_id, request.turn_index - 1, self.l1_cache)
            return CacheHitResult(
                l0_tokens=0,
                l1_tokens=l1_match,
                miss_tokens=request.input_tokens - l1_match,
            )
        
        # 完全 miss
        self.stats["misses"] += 1
        return CacheHitResult(
            l0_tokens=0,
            l1_tokens=0,
            miss_tokens=request.input_tokens,
        )
    
    def _find_best_match(
        self,
        program_id: str,
        turn_index: int,
        input_tokens: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]]
    ) -> int:
        """
        在缓存中查找最佳匹配
        
        假设 100% 匹配率：完全复用已有缓存
        """
        best_match = 0
        
        for key, (cached_tokens, _) in cache.items():
            prog_id, cached_turn = key
            if prog_id != program_id:
                continue
            if cached_turn >= turn_index:
                continue
            
            # 100% 匹配率
            matched = min(cached_tokens, input_tokens)
            best_match = max(best_match, matched)
        
        return best_match
    
    def _touch_cache(
        self,
        program_id: str,
        turn_index: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]]
    ):
        """更新缓存条目的访问时间"""
        key = (program_id, turn_index)
        if key in cache:
            tokens, _ = cache[key]
            cache[key] = (tokens, time.time())
    
    def _evict_l0_for(self, tokens_needed: int) -> int:
        """
        为新请求腾出 GPU 空间
        
        LRU 淘汰到 Host Cache
        返回淘汰的 tokens 数
        """
        evicted = 0
        
        while self.current_l0_tokens + tokens_needed > self.max_tokens and self.l0_cache:
            # LRU: 找到最老的条目
            oldest_key = None
            oldest_time = float('inf')
            
            for key, (_, last_access) in self.l0_cache.items():
                if last_access < oldest_time:
                    oldest_time = last_access
                    oldest_key = key
            
            if oldest_key is None:
                break
            
            tokens, _ = self.l0_cache.pop(oldest_key)
            self.current_l0_tokens -= tokens
            evicted += tokens
            
            # 移到 Host Cache
            if self.current_l1_tokens + tokens <= self.max_host_tokens:
                self.l1_cache[oldest_key] = (tokens, oldest_time)
                self.current_l1_tokens += tokens
        
        return evicted
    
    def _calculate_h2d_time(self, tokens: int) -> float:
        """计算 H2D 传输时间"""
        if tokens <= 0:
            return 0.0
        # 128 bytes per token, bandwidth in GB/s
        bytes_per_token = 128
        total_bytes = tokens * bytes_per_token
        gb = total_bytes / (1024 ** 3)
        return gb / self.h2d_bandwidth_gb  # seconds
    
    def _estimate_inference_time(self, input_tokens: int, output_tokens: int) -> float:
        """估算推理时间 (ms)"""
        return (input_tokens + output_tokens) * self.inference_speed
    
    def process_next(self, current_time: float) -> Optional[ScheduleResult]:
        """处理下一个请求"""
        if not self.queue:
            return None
        
        request = self.queue.popleft()
        
        # 查找缓存命中
        hit_result = self._find_cache_hit(request)
        
        # 计算队列等待时间
        queue_time = max(0, current_time - request.arrival_time)
        
        # 计算 H2D 传输时间 (L1 hit)
        h2d_time = self._calculate_h2d_time(hit_result.l1_tokens)
        
        # 淘汰腾出空间
        tokens_needed = hit_result.miss_tokens + request.output_tokens
        evicted = self._evict_l0_for(tokens_needed)
        
        # 估算推理时间 (只计算 miss 的部分)
        inference_time = self._estimate_inference_time(
            hit_result.miss_tokens,
            request.output_tokens
        )
        
        # 添加到 GPU 缓存
        if request.output_tokens > 0:
            total_new_tokens = hit_result.miss_tokens + request.output_tokens
            
            if self.current_l0_tokens + total_new_tokens <= self.max_tokens:
                self.l0_cache[(request.program_id, request.turn_index)] = (
                    total_new_tokens,
                    current_time
                )
                self.current_l0_tokens += total_new_tokens
            else:
                # GPU 满了，添加到 Host Cache
                if self.current_l1_tokens + total_new_tokens <= self.max_host_tokens:
                    self.l1_cache[(request.program_id, request.turn_index)] = (
                        total_new_tokens,
                        current_time
                    )
                    self.current_l1_tokens += total_new_tokens
        
        self.stats["requests_processed"] += 1
        self.stats["evicted_tokens"] += evicted
        
        return ScheduleResult(
            rid=request.rid,
            program_id=request.program_id,
            turn_index=request.turn_index,
            cache_hit=hit_result,
            queue_time=queue_time,
            h2d_time=h2d_time,
            inference_time=inference_time,
            total_time=inference_time + h2d_time,
            was_pinned=False,
            evicted_tokens=evicted,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self.stats["l0_hits"] + self.stats["l1_hits"] + self.stats["misses"]
        hit_rate = (self.stats["l0_hits"] + self.stats["l1_hits"]) / max(1, total)
        return {
            **self.stats,
            "l0_cache_tokens": self.current_l0_tokens,
            "l1_cache_tokens": self.current_l1_tokens,
            "l0_cache_entries": len(self.l0_cache),
            "l1_cache_entries": len(self.l1_cache),
            "hit_rate": hit_rate,
        }


# ============================================================================
# Continuum 调度器模拟
# ============================================================================

class ContinuumScheduler:
    """
    Continuum 调度器: 程序亲和性 + TTL Pinning
    
    基于 Baseline 的两级缓存架构：
    - L0 (GPU): 主要 KV Cache
    - L1 (Host): GPU 淘汰的 KV Cache 备份
    
    额外特性：
    1. 程序亲和性调度：同程序请求优先处理
    2. TTL Pinning：工具调用时锁定，减少淘汰
    3. 动态 TTL：根据历史数据调整 TTL
    """
    
    def __init__(
        self,
        max_tokens: int,
        max_host_tokens: int,
        ttl_simulator: "ContinuumTTLSimulator",
        state_manager: "AgentStateManager",
        h2d_bandwidth_gb: float = 50.0,
        inference_speed: float = 0.0001,
    ):
        self.max_tokens = max_tokens
        self.max_host_tokens = max_host_tokens
        self.ttl_simulator = ttl_simulator
        self.state_manager = state_manager
        self.h2d_bandwidth_gb = h2d_bandwidth_gb
        self.inference_speed = inference_speed
        
        # 队列
        self.waiting_queue: deque = deque()
        # 使用 (program_id, turn_index) 作为key来追踪pin
        self.pinned_cache: Dict[Tuple[str, int], float] = {}  # (prog, turn) -> expires_at
        
        # 两级缓存
        self.l0_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l0_tokens = 0
        self.l1_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l1_tokens = 0
        
        # 程序追踪
        self.program_first_entry: Dict[str, float] = {}
        self.program_stats: Dict[str, Dict] = {}
        
        # 统计
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "pins": 0,
            "unpins": 0,
            "evicted_tokens": 0,
            "ttl_selections": 0,
        }
    
    def reset(self):
        """重置调度器"""
        self.waiting_queue.clear()
        self.pinned_cache.clear()
        self.l0_cache.clear()
        self.l1_cache.clear()
        self.current_l0_tokens = 0
        self.current_l1_tokens = 0
        self.program_first_entry.clear()
        self.program_stats.clear()
        self.ttl_simulator.reset()
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "pins": 0,
            "unpins": 0,
            "evicted_tokens": 0,
            "ttl_selections": 0,
        }
    
    def add_request(self, request: QueuedRequest):
        """添加请求到队列"""
        self.waiting_queue.append(request)
    
    def _unpin_expired(self, current_time: float):
        """清理过期的 pin"""
        expired_keys = []
        for key, expires_at in self.pinned_cache.items():
            if current_time > expires_at:
                expired_keys.append(key)
                self.stats["unpins"] += 1
        
        for key in expired_keys:
            del self.pinned_cache[key]
    
    def _sort_by_affinity(self, current_time: float) -> List[QueuedRequest]:
        """
        按程序亲和性排序
        
        优先级:
        1. 同程序的连续轮次 (已在 GPU 或 Host 有缓存)
        2. TTL 即将过期的请求
        3. 其他请求
        """
        requests = list(self.waiting_queue)
        
        def priority(req: QueuedRequest) -> Tuple[int, float]:
            prog_id = req.program_id
            
            # 检查是否在 GPU 缓存
            in_l0 = any(k[0] == prog_id and k[1] < req.turn_index 
                        for k in self.l0_cache.keys())
            # 检查是否在 Host 缓存
            in_l1 = any(k[0] == prog_id and k[1] < req.turn_index 
                        for k in self.l1_cache.keys())
            # 检查是否被 pin
            key = (prog_id, req.turn_index - 1)  # 检查前一轮是否被 pin
            is_pinned = key in self.pinned_cache and current_time <= self.pinned_cache[key]
            # 检查 pin 是否快过期
            pin_expiry = float('inf')
            if is_pinned:
                pin_expiry = self.pinned_cache[key] - current_time
            
            # 优先级: GPU命中(0) > Host命中(1) > 其他(2)
            if in_l0:
                base = 0
            elif in_l1:
                base = 1
            else:
                base = 2
            
            # 越快过期越优先
            return (base, pin_expiry)
        
        return sorted(requests, key=priority)
    
    def _select_ttl(self, request: QueuedRequest, queue_time: float) -> float:
        """选择动态 TTL"""
        if not request.is_tool_call:
            return 0.0
        
        ttl, strategy = self.ttl_simulator.select_dynamic_ttl(
            request.program_id,
            request.tool_name,
            queue_time
        )
        self.stats["ttl_selections"] += 1
        return ttl
    
    def _pin_request(self, request: QueuedRequest, ttl: float, current_time: float):
        """Pin 请求的缓存条目"""
        key = (request.program_id, request.turn_index)
        expires_at = current_time + ttl
        self.pinned_cache[key] = expires_at
        self.stats["pins"] += 1
        
        # 添加到 TTL 模拟器
        self.ttl_simulator.add_node(
            request.program_id,
            request.turn_index,
            ttl,
            request.total_tokens,
            current_time,
        )
    
    def _find_cache_hit(self, request: QueuedRequest) -> CacheHitResult:
        """查找两级缓存命中"""
        # L0 查找
        l0_match = self._find_best_match(
            request.program_id,
            request.turn_index,
            request.input_tokens,
            self.l0_cache
        )
        
        if l0_match > 0:
            self.stats["l0_hits"] += 1
            self._touch_cache(request.program_id, request.turn_index - 1, self.l0_cache)
            return CacheHitResult(
                l0_tokens=l0_match,
                l1_tokens=0,
                miss_tokens=request.input_tokens - l0_match,
            )
        
        # L1 查找
        l1_match = self._find_best_match(
            request.program_id,
            request.turn_index,
            request.input_tokens,
            self.l1_cache
        )
        
        if l1_match > 0:
            self.stats["l1_hits"] += 1
            self._touch_cache(request.program_id, request.turn_index - 1, self.l1_cache)
            return CacheHitResult(
                l0_tokens=0,
                l1_tokens=l1_match,
                miss_tokens=request.input_tokens - l1_match,
            )
        
        self.stats["misses"] += 1
        return CacheHitResult(
            l0_tokens=0,
            l1_tokens=0,
            miss_tokens=request.input_tokens,
        )
    
    def _find_best_match(
        self,
        program_id: str,
        turn_index: int,
        input_tokens: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]]
    ) -> int:
        """在缓存中查找最佳匹配（100% 匹配率）"""
        best_match = 0
        
        for key, (cached_tokens, _) in cache.items():
            prog_id, cached_turn = key
            if prog_id != program_id:
                continue
            if cached_turn >= turn_index:
                continue
            
            # 100% 匹配率
            matched = min(cached_tokens, input_tokens)
            best_match = max(best_match, matched)
        
        return best_match
    
    def _touch_cache(
        self,
        program_id: str,
        turn_index: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]]
    ):
        """更新缓存访问时间"""
        key = (program_id, turn_index)
        if key in cache:
            tokens, _ = cache[key]
            cache[key] = (tokens, time.time())
    
    def _evict_l0_for(self, tokens_needed: int, current_time: float) -> int:
        """
        为新请求腾出 GPU 空间
        
        与 Baseline 的区别：
        - pinned 的条目不会被淘汰
        - 淘汰的条目移到 Host Cache
        """
        evicted = 0
        
        while self.current_l0_tokens + tokens_needed > self.max_tokens and self.l0_cache:
            # LRU: 找最老的非 pinned 条目
            oldest_key = None
            oldest_time = float('inf')
            
            for key, (_, last_access) in self.l0_cache.items():
                # 检查是否被 pin (使用 (program_id, turn_index))
                if key in self.pinned_cache:
                    expires_at = self.pinned_cache[key]
                    if current_time <= expires_at:
                        continue  # 跳过未过期的 pinned 条目
                if last_access < oldest_time:
                    oldest_time = last_access
                    oldest_key = key
            
            if oldest_key is None:
                # 所有条目都被 pin，无法淘汰
                break
            
            tokens, _ = self.l0_cache.pop(oldest_key)
            self.current_l0_tokens -= tokens
            evicted += tokens
            
            # 移到 Host Cache
            if self.current_l1_tokens + tokens <= self.max_host_tokens:
                self.l1_cache[oldest_key] = (tokens, oldest_time)
                self.current_l1_tokens += tokens
        
        return evicted
    
    def _calculate_h2d_time(self, tokens: int) -> float:
        """计算 H2D 传输时间"""
        if tokens <= 0:
            return 0.0
        bytes_per_token = 128
        total_bytes = tokens * bytes_per_token
        gb = total_bytes / (1024 ** 3)
        return gb / self.h2d_bandwidth_gb
    
    def _estimate_inference_time(self, input_tokens: int, output_tokens: int) -> float:
        """估算推理时间"""
        return (input_tokens + output_tokens) * self.inference_speed
    
    def process_next(self, current_time: float) -> Optional[ScheduleResult]:
        """处理下一个请求"""
        if not self.waiting_queue:
            return None
        
        # 清理过期的 pin
        self._unpin_expired(current_time)
        
        # 按亲和性排序
        sorted_requests = self._sort_by_affinity(current_time)
        self.waiting_queue = deque(sorted_requests)
        
        request = self.waiting_queue.popleft()
        
        # 查找缓存命中
        hit_result = self._find_cache_hit(request)
        
        # 计算队列等待时间
        queue_time = max(0, current_time - request.arrival_time)
        
        # 计算 H2D 传输时间
        h2d_time = self._calculate_h2d_time(hit_result.l1_tokens)
        
        # 估算推理时间
        inference_time = self._estimate_inference_time(
            hit_result.miss_tokens,
            request.output_tokens
        )
        
        # 计算需要空间
        tokens_needed = hit_result.miss_tokens + request.output_tokens if request.output_tokens > 0 else 0
        
        # PIN 前一轮的缓存（关键修复！）
        # 前一轮的缓存是当前轮次需要的依赖，应该保护它不被驱逐
        was_pinned = False
        if request.turn_index > 0:
            prev_turn = request.turn_index - 1
            prev_key = (request.program_id, prev_turn)
            if prev_key in self.l0_cache:
                ttl = self._select_ttl(request, queue_time)
                if ttl > 0:
                    self.pinned_cache[prev_key] = current_time + ttl
                    was_pinned = True
                    self.stats["pins"] += 1
                    # Track pinned turn for statistics
                    if request.program_id not in self.program_stats:
                        self.program_stats[request.program_id] = {}
                    self.program_stats[request.program_id]["pinned_turn"] = prev_turn
        
        # 淘汰腾出空间 (Pin 已经在上面执行，会被保护)
        evicted = self._evict_l0_for(tokens_needed, current_time)
        
        # 记录工具执行 (for history)
        if request.is_tool_call and request.tool_duration > 0:
            prev_request = self.program_stats.get(request.program_id, {}).get("last_request")
            if prev_request:
                idle_gap = current_time - prev_request - request.tool_duration
                if idle_gap > 0:
                    self.ttl_simulator.record_idle_gap(request.program_id, idle_gap)
            
            self.ttl_simulator.record_tool_execution(
                request.program_id,
                request.tool_name or "unknown",
                request.tool_duration,
            )
        
        # Add to GPU cache (pin was already done before eviction)
        if request.output_tokens > 0:
            total_new_tokens = hit_result.miss_tokens + request.output_tokens
            # Only add if there's room (eviction already freed some space)
            if self.current_l0_tokens + total_new_tokens <= self.max_tokens:
                self.l0_cache[(request.program_id, request.turn_index)] = (
                    total_new_tokens,
                    current_time
                )
                self.current_l0_tokens += total_new_tokens
            # If no room, the request's output is not cached (too much pressure)
        
        # Update program stats
        if request.program_id not in self.program_stats:
            self.program_stats[request.program_id] = {}
        self.program_stats[request.program_id]["last_request"] = current_time + inference_time
        
        self.stats["requests_processed"] += 1
        self.stats["evicted_tokens"] += evicted
        if was_pinned:
            self.stats["pins"] += 1
        
        return ScheduleResult(
            rid=request.rid,
            program_id=request.program_id,
            turn_index=request.turn_index,
            cache_hit=hit_result,
            queue_time=queue_time,
            h2d_time=h2d_time,
            inference_time=inference_time,
            total_time=inference_time + h2d_time,
            was_pinned=was_pinned,
            evicted_tokens=evicted,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = self.stats["l0_hits"] + self.stats["l1_hits"] + self.stats["misses"]
        hit_rate = (self.stats["l0_hits"] + self.stats["l1_hits"]) / max(1, total)
        return {
            **self.stats,
            "l0_cache_tokens": self.current_l0_tokens,
            "l1_cache_tokens": self.current_l1_tokens,
            "l0_cache_entries": len(self.l0_cache),
            "l1_cache_entries": len(self.l1_cache),
            "pinned_count": len(self.pinned_cache),
            "hit_rate": hit_rate,
            "ttl_stats": self.ttl_simulator.get_stats(),
        }


# ============================================================================
# 统一的调度器模拟器
# ============================================================================

class SchedulerSimulator:
    """
    统一的调度器模拟器
    
    支持 Baseline (FCFS) 和 Continuum 两种调度策略。
    """
    
    def __init__(
        self,
        mode: str,
        max_tokens: int,
        max_host_tokens: int = 0,
        ttl_simulator: Optional["ContinuumTTLSimulator"] = None,
        state_manager: Optional["AgentStateManager"] = None,
        h2d_bandwidth_gb: float = 50.0,
        inference_speed: float = 0.0001,
    ):
        """
        Args:
            mode: 调度模式 ("baseline" 或 "continuum")
            max_tokens: GPU 缓存容量 (tokens)
            max_host_tokens: Host 缓存容量 (tokens)
            ttl_simulator: TTL 模拟器（仅 continuum 模式需要）
            state_manager: 状态管理器
            h2d_bandwidth_gb: H2D 带宽 (GB/s)
            inference_speed: 推理速度 (tokens per ms)
        """
        self.mode = mode
        self.max_tokens = max_tokens
        self.max_host_tokens = max_host_tokens if max_host_tokens > 0 else max_tokens * 10
        
        if state_manager is None:
            from sglang_simulator.simulation.agent.state_manager import AgentStateManager
            state_manager = AgentStateManager
        
        self.state_manager = state_manager
        state_manager.set_scheduler_type(mode)
        
        if mode == "baseline":
            self.scheduler = BaselineScheduler(
                max_tokens=max_tokens,
                max_host_tokens=self.max_host_tokens,
                state_manager=state_manager,
                h2d_bandwidth_gb=h2d_bandwidth_gb,
                inference_speed=inference_speed,
            )
        elif mode == "continuum":
            if ttl_simulator is None:
                from sglang_simulator.simulation.agent.continuum_ttl import ContinuumTTLSimulator
                ttl_simulator = ContinuumTTLSimulator()
            self.scheduler = ContinuumScheduler(
                max_tokens=max_tokens,
                max_host_tokens=self.max_host_tokens,
                ttl_simulator=ttl_simulator,
                state_manager=state_manager,
                h2d_bandwidth_gb=h2d_bandwidth_gb,
                inference_speed=inference_speed,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 结果收集
        self.results: List[ScheduleResult] = []
        self.current_time = 0.0
        self._request_counter = 0
    
    def reset(self):
        """重置模拟器"""
        self.scheduler.reset()
        self.state_manager.reset()
        self.results.clear()
        self.current_time = 0.0
        self._request_counter = 0
    
    def add_request(
        self,
        program_id: str,
        turn_index: int,
        input_tokens: int,
        output_tokens: int,
        arrival_time: float,
        is_tool_call: bool = False,
        tool_name: Optional[str] = None,
        tool_duration: float = 0.0,
    ):
        """添加请求"""
        self._request_counter += 1
        request = QueuedRequest(
            rid=f"req_{self._request_counter}",
            program_id=program_id,
            turn_index=turn_index,
            arrival_time=arrival_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_tool_call=is_tool_call,
            tool_name=tool_name,
            tool_duration=tool_duration,
        )
        self.scheduler.add_request(request)
    
    def step(self, dt: float = 0.1) -> List[ScheduleResult]:
        """
        执行一个时间步
        
        正确的模拟流程：
        1. 检查是否有请求可以处理
        2. 如果有，到达时间 <= 当前时间，则处理
        3. 如果没有，到达时间 > 当前时间，推进到最早到达时间
        """
        results = []
        
        # 获取调度器的队列属性
        queue = getattr(self.scheduler, 'waiting_queue', None) or getattr(self.scheduler, 'queue', None)
        
        if not queue:
            return results
        
        # 找到最早可处理的请求
        earliest_arrival = min(req.arrival_time for req in queue)
        
        # 如果最早到达时间在未来，推进时间
        if earliest_arrival > self.current_time:
            self.current_time = earliest_arrival
        
        # 处理已到达的请求 - 按到达顺序逐个处理
        while queue:
            # 获取队首请求
            next_req = queue[0]
            
            # 如果请求还没到达，等待
            if next_req.arrival_time > self.current_time:
                # 推进时间到下一个请求的到达时间
                self.current_time = next_req.arrival_time
            
            # 处理队首请求
            result = self.scheduler.process_next(self.current_time)
            if result is None:
                break
            
            results.append(result)
            self.results.append(result)
            
            # 更新时间到请求完成
            self.current_time += result.total_time
        
        return results
    
    def run_until_complete(self, max_steps: int = 100000) -> List[ScheduleResult]:
        """运行直到所有请求完成"""
        for _ in range(max_steps):
            queue = getattr(self.scheduler, 'waiting_queue', None) or getattr(self.scheduler, 'queue', None)
            if not queue or len(queue) == 0:
                break  # 队列为空，模拟完成
            
            results = self.step(dt=0.1)
            if not results:
                # 没有请求被处理（可能所有请求都在未来）
                # 推进时间到最早到达的请求
                earliest_arrival = min(req.arrival_time for req in queue)
                if earliest_arrival > self.current_time:
                    self.current_time = earliest_arrival
                else:
                    # 如果无法推进时间且没有请求被处理，退出
                    break
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """获取模拟摘要"""
        scheduler_stats = self.scheduler.get_stats()
        
        # 墙钟时间
        wall_time = self.current_time
        
        # 各部分时间
        total_inference_time = sum(r.inference_time for r in self.results)
        total_h2d_time = sum(r.h2d_time for r in self.results)
        total_process_time = total_inference_time + total_h2d_time
        total_queue_time = max(0, wall_time - total_process_time)
        
        total_evicted = sum(r.evicted_tokens for r in self.results)
        pinned_count = sum(1 for r in self.results if r.was_pinned)
        
        # 缓存统计
        total_l0 = sum(r.cache_hit.l0_tokens for r in self.results)
        total_l1 = sum(r.cache_hit.l1_tokens for r in self.results)
        total_miss = sum(r.cache_hit.miss_tokens for r in self.results)
        total_cache = total_l0 + total_l1
        total_tokens = total_cache + total_miss
        hit_rate = total_cache / max(1, total_tokens)
        
        return {
            "mode": self.mode,
            "total_requests": len(self.results),
            "wall_time": wall_time,
            "total_time": wall_time,
            "total_queue_time": total_queue_time,
            "total_h2d_time": total_h2d_time,
            "total_l1_time": total_h2d_time,  # L1 cache hit requires H2D transfer
            "total_l2_time": total_inference_time,  # L2/backend processing time
            "total_inference_time": total_inference_time,
            "total_evicted_tokens": total_evicted,
            "pinned_requests": pinned_count,
            "cache_stats": {
                "l0_hit_tokens": total_l0,
                "l1_hit_tokens": total_l1,
                "l2_hit_tokens": 0,  # Not used in current architecture
                "miss_tokens": total_miss,
                "total_cache_tokens": total_cache,
                "total_tokens": total_tokens,
                "hit_rate": hit_rate,
            },
            "scheduler_stats": scheduler_stats,
        }

    def get_full_state(self) -> Dict[str, Any]:
        """
        获取当前完整状态用于可视化。

        Returns:
            包含时间、缓存状态、队列状态的字典
        """
        # 获取底层调度器
        scheduler = self.scheduler

        # 获取队列
        queue = getattr(scheduler, 'waiting_queue', None) or getattr(scheduler, 'queue', deque())
        queue_list = []
        for req in queue:
            queue_list.append({
                "rid": req.rid,
                "program_id": req.program_id,
                "turn_index": req.turn_index,
                "arrival_time": req.arrival_time,
                "input_tokens": req.input_tokens,
                "output_tokens": req.output_tokens,
                "is_tool_call": req.is_tool_call,
                "tool_name": req.tool_name,
                "tool_duration": req.tool_duration,
            })

        # 格式化缓存条目
        def format_cache_entry(key, value):
            prog_id, turn = key
            tokens, last_access = value
            return {
                "program_id": prog_id,
                "turn_index": turn,
                "tokens": tokens,
                "last_access": last_access,
            }

        l0_entries = [format_cache_entry(k, v) for k, v in scheduler.l0_cache.items()]
        l1_entries = [format_cache_entry(k, v) for k, v in scheduler.l1_cache.items()]

        # 获取 pinned 请求（仅 Continuum 模式）
        pinned_entries = []
        if hasattr(scheduler, 'pinned_cache'):
            for (prog_id, turn), expires_at in scheduler.pinned_cache.items():
                pinned_entries.append({
                    "program_id": prog_id,
                    "turn_index": turn,
                    "expires_at": expires_at,
                })

        return {
            "time": self.current_time,
            "mode": self.mode,
            "cache": {
                "l0_total_tokens": scheduler.current_l0_tokens,
                "l0_max_tokens": scheduler.max_tokens,
                "l0_entries": l0_entries,
                "l1_total_tokens": scheduler.current_l1_tokens,
                "l1_max_tokens": scheduler.max_host_tokens,
                "l1_entries": l1_entries,
                "pinned_entries": pinned_entries,
            },
            "queue": queue_list,
            "queue_length": len(queue_list),
            "results_count": len(self.results),
        }
