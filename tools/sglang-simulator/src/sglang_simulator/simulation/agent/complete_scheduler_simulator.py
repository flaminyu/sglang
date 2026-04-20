#!/usr/bin/env python3
"""
KVCache TTL 完整模拟器 - 论文对齐版

修改点（对照 Continuum 论文 Algorithm 1）:
1. PIN 改为程序级别 (program_id)，不是轮次级别
2. PIN 在 OnRequestFinish 时设置，不是 OnRequestArrive
3. PIN 在 Schedule 时释放（请求被调度时）
4. 优先级：PIN 程序 > FCFS
5. 完整 Utility Model: τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CacheHitResult:
    l0_tokens: int = 0
    l1_tokens: int = 0
    miss_tokens: int = 0


@dataclass
class QueuedRequest:
    rid: str
    program_id: str
    turn_index: int
    arrival_time: float
    input_tokens: int
    output_tokens: int
    is_tool_call: bool
    tool_name: Optional[str]
    tool_duration: float
    
    cached_tokens: int = 0
    miss_tokens: int = 0


@dataclass
class ScheduleResult:
    rid: str
    program_id: str
    turn_index: int
    cache_hit: CacheHitResult
    queue_time: float
    h2d_time: float
    inference_time: float
    total_time: float
    was_pinned: bool
    evicted_tokens: int


# ============================================================================
# Continuum TTL 模拟器 - 论文对齐版
# ============================================================================

@dataclass
class ProgramStats:
    tool_durations: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1024)))
    idle_gaps: deque = field(default_factory=lambda: deque(maxlen=1024))
    turn_count: int = 0


class ContinuumTTLSimulator:
    """
    Continuum TTL 模拟器 - 论文对齐版
    
    完整 Utility Model:
    τ* = argmax_τ P(τ, f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ
    
    Where:
    - P(τ, f) = empirical CDF of tool f execution time ≤ τ
    - T = average queueing delay per unit memory
    - η = memoryfulness factor (-Corr(k, N-k))
    - Prefill-Reload = profiled prefill or CPU reload time
    - MemUsage/M = relative memory usage
    """
    
    def __init__(
        self,
        default_ttl: float = 3.0,
        min_ttl: float = 0.1,
        max_ttl: float = 10.0,
        history_threshold: int = 1,
        # 论文参数
        avg_queue_delay: float = 1.0,  # T
        memoryfulness: float = 0.8,       # η
        prefill_time: float = 0.1,        # Prefill-Reload
        reload_time: float = 0.05,         # CPU reload
        enable_adaptive_ttl: bool = True,
    ):
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.history_threshold = history_threshold
        
        # 论文参数
        self.avg_queue_delay = avg_queue_delay  # T
        self.memoryfulness = memoryfulness       # η
        self.prefill_time = prefill_time        # Prefill-Reload
        self.reload_time = reload_time          # CPU reload
        
        self.enable_adaptive_ttl = enable_adaptive_ttl
        
        self.program_stats: Dict[str, ProgramStats] = {}
        self.global_idle_gaps: deque = deque(maxlen=1024)
        self.global_tool_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1024))
        
        self.stats = {
            "ttl_selections": 0,
            "tool_based_selections": 0,
            "idle_gap_based_selections": 0,
            "default_selections": 0,
        }
    
    def reset(self):
        self.program_stats.clear()
        self.global_idle_gaps.clear()
        self.global_tool_durations.clear()
        self.stats = {"ttl_selections": 0, "tool_based_selections": 0, 
                      "idle_gap_based_selections": 0, "default_selections": 0}
    
    def _compute_cdf(self, tool_name: str, tau: float) -> float:
        """计算 P(τ, f) - 工具执行时间 ≤ τ 的概率"""
        samples = list(self.global_tool_durations.get(tool_name, []))
        if not samples:
            return 0.5  # 默认 50%
        
        count_le = sum(1 for s in samples if s <= tau)
        return count_le / len(samples)
    
    def _compute_benefit(self, memory_usage: float) -> float:
        """计算 Benefit = T·η + Prefill-Reload"""
        # T × η + Prefill-Reload(r)
        queue_benefit = self.avg_queue_delay * self.memoryfulness
        return queue_benefit + self.prefill_time
    
    def _compute_cost(self, tau: float, memory_usage: float, total_memory: float) -> float:
        """计算 Cost = (MemUsage/M) × τ"""
        if total_memory <= 0:
            return 0.0
        return (memory_usage / total_memory) * tau
    
    def select_dynamic_ttl(
        self,
        program_id: str,
        tool_name: Optional[str] = None,
        memory_usage: float = 0.0,
        total_memory: float = 1.0,
    ) -> float:
        """
        计算最优 TTL - 论文公式
        
        τ* = argmax_τ P(τ, f) × (T·η + Prefill-Reload) - (MemUsage/M) × τ
        """
        self.stats["ttl_selections"] += 1
        
        if not self.enable_adaptive_ttl:
            self.stats["default_selections"] += 1
            return self.default_ttl
        
        if tool_name:
            benefit = self._compute_benefit(memory_usage)
            
            # 搜索最优 τ
            best_ttl = self.default_ttl
            best_utility = float('-inf')
            
            for tau in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
                p = self._compute_cdf(tool_name, tau)
                cost = self._compute_cost(tau, memory_usage, total_memory)
                utility = p * benefit - cost
                
                if utility > best_utility:
                    best_utility = utility
                    best_ttl = tau
            
            # 检查历史样本
            samples = list(self.program_stats.get(program_id, ProgramStats()).tool_durations.get(tool_name, []))
            if len(samples) >= self.history_threshold:
                self.stats["tool_based_selections"] += 1
                return min(self.max_ttl, max(self.min_ttl, best_ttl))
        
        self.stats["default_selections"] += 1
        return self.default_ttl
    
    def record_tool_execution(self, program_id: str, tool_name: str, duration: float):
        """记录工具执行时间 - OnRequestArrive 时调用"""
        stats = self.program_stats.setdefault(program_id, ProgramStats())
        stats.tool_durations[tool_name].append(duration)
        self.global_tool_durations[tool_name].append(duration)
        
        if duration > 0:
            stats.idle_gaps.append(duration)
            self.global_idle_gaps.append(duration)
    
    def get_stats(self) -> dict:
        return {**self.stats, "programs_tracked": len(self.program_stats)}


# ============================================================================
# 时间预测器
# ============================================================================

class TimePredictor:
    def __init__(
        self,
        prefill_per_token_ms: float = 0.1,
        decode_per_token_ms: float = 5.0,
        cache_hit_speedup: float = 10.0,
    ):
        self.prefill_per_token_ms = prefill_per_token_ms
        self.decode_per_token_ms = decode_per_token_ms
        self.cache_hit_speedup = cache_hit_speedup
    
    def predict_time(self, miss_tokens: int, output_tokens: int, is_cache_hit: bool) -> Tuple[float, float]:
        if is_cache_hit:
            ttft_ms = 0.0
            decode_time_ms = output_tokens * self.decode_per_token_ms / self.cache_hit_speedup
            total_ms = decode_time_ms
        else:
            prefill_time_ms = miss_tokens * self.prefill_per_token_ms
            decode_time_ms = output_tokens * self.decode_per_token_ms
            ttft_ms = prefill_time_ms
            total_ms = prefill_time_ms + decode_time_ms
        
        return ttft_ms, total_ms


# ============================================================================
# 完整调度器模拟器 - 论文对齐版
# ============================================================================

class CompleteSchedulerSimulator:
    """
    完整调度器模拟器 - 论文对齐版
    
    关键修改 (对照论文 Algorithm 1):
    1. PIN 改为程序级别 (program_id)
    2. PIN 在 OnRequestFinish 时设置
    3. PIN 在 Schedule 时释放
    4. 优先级: PIN 程序 > FCFS
    """
    
    def __init__(
        self,
        mode: str,
        max_tokens: int,
        max_host_tokens: int = 0,
        ttl_simulator: Optional[ContinuumTTLSimulator] = None,
        time_predictor: Optional[TimePredictor] = None,
        max_batch_size: int = 16,
    ):
        self.mode = mode
        self.max_tokens = max_tokens
        self.max_host_tokens = max_host_tokens if max_host_tokens > 0 else max_tokens * 10
        self.max_batch_size = max_batch_size
        
        self.time_predictor = time_predictor or TimePredictor()
        self.ttl_simulator = ttl_simulator or ContinuumTTLSimulator()
        
        # 缓存
        self.l0_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l0_tokens = 0
        self.l1_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l1_tokens = 0
        
        # PIN 缓存 - 论文风格: program_id → expiration_timestamp
        self.pinned_programs: Dict[str, float] = {}  # program_id → expires_at
        
        # 请求队列
        self.waiting_queue: deque = deque()
        self.completed_requests: List[QueuedRequest] = []
        
        # 记录程序是否已完成（用于 PIN 释放）
        self.finished_programs: Set[str] = set()
        
        self.current_time = 0.0
        
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "pins": 0,
            "unpins": 0,
            "evicted_tokens": 0,
        }
        
        self.bytes_per_token = 128
    
    def reset(self):
        self.l0_cache.clear()
        self.l1_cache.clear()
        self.pinned_programs.clear()
        self.waiting_queue.clear()
        self.completed_requests.clear()
        self.finished_programs.clear()
        self.current_time = 0.0
        self.ttl_simulator.reset()
        self.stats = {"requests_processed": 0, "l0_hits": 0, "l1_hits": 0, "misses": 0,
                      "pins": 0, "unpins": 0, "evicted_tokens": 0}
    
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
    ) -> str:
        rid = f"{program_id}_T{turn_index}_{int(arrival_time * 1000)}"
        request = QueuedRequest(
            rid=rid, program_id=program_id, turn_index=turn_index,
            arrival_time=arrival_time, input_tokens=input_tokens,
            output_tokens=output_tokens, is_tool_call=is_tool_call,
            tool_name=tool_name, tool_duration=tool_duration,
        )
        self.waiting_queue.append(request)
        return rid
    
    def _find_cache_hit(self, request: QueuedRequest) -> Tuple[CacheHitResult, bool]:
        """查找缓存命中"""
        # L0 查找
        l0_match, l0_key = self._find_best_match(
            request.program_id, request.turn_index, request.input_tokens, self.l0_cache)
        
        if l0_match > 0:
            self.stats["l0_hits"] += 1
            self.l0_cache[l0_key] = (self.l0_cache[l0_key][0], self.current_time)
            return CacheHitResult(l0_tokens=l0_match, miss_tokens=request.input_tokens - l0_match), True
        
        # L1 查找
        l1_match, _ = self._find_best_match(
            request.program_id, request.turn_index, request.input_tokens, self.l1_cache)
        
        if l1_match > 0:
            self.stats["l1_hits"] += 1
            return CacheHitResult(l1_tokens=l1_match, miss_tokens=request.input_tokens - l1_match), True
        
        self.stats["misses"] += 1
        return CacheHitResult(miss_tokens=request.input_tokens), False
    
    def _find_best_match(
        self, program_id: str, turn_index: int, input_tokens: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]]
    ) -> Tuple[int, Optional[Tuple[str, int]]]:
        best_match, best_key = 0, None
        for key, (cached_tokens, _) in cache.items():
            prog_id, cached_turn = key
            if prog_id != program_id or cached_turn >= turn_index:
                continue
            # 假设 100% 匹配率：完全复用已有缓存
            matched = min(cached_tokens, input_tokens)
            if matched > best_match:
                best_match, best_key = matched, key
        return best_match, best_key
    
    def _evict_l0_for(self, tokens_needed: int) -> int:
        evicted = 0
        if tokens_needed > self.max_tokens:
            return 0
        
        while self.current_l0_tokens + tokens_needed > self.max_tokens and self.l0_cache:
            # 论文: 淘汰最老的程序
            oldest_key = None
            oldest_time = float('inf')
            
            for key, (_, last_access) in self.l0_cache.items():
                prog_id = key[0]
                # 跳过 PIN 的程序
                if prog_id in self.pinned_programs and self.current_time <= self.pinned_programs[prog_id]:
                    continue
                if last_access < oldest_time:
                    oldest_time = last_access
                    oldest_key = key
            
            if oldest_key is None:
                break
            
            tokens, _ = self.l0_cache.pop(oldest_key)
            self.current_l0_tokens -= tokens
            evicted += tokens
            
            if self.current_l1_tokens + tokens <= self.max_host_tokens:
                self.l1_cache[oldest_key] = (tokens, oldest_time)
                self.current_l1_tokens += tokens
        
        return evicted
    
    def _calculate_h2d_time(self, tokens: int) -> float:
        if tokens <= 0: return 0.0
        return (tokens * self.bytes_per_token / (1024 ** 3)) / 50.0
    
    def _expire_pins(self):
        """论文: Expire old pins"""
        expired = []
        for pid, expires_at in self.pinned_programs.items():
            # 论文: if now > pinned[pid] and pid not in Q.programs:
            if self.current_time > expires_at and pid not in self.waiting_queue:
                expired.append(pid)
        
        for pid in expired:
            del self.pinned_programs[pid]
            self.stats["unpins"] += 1
    
    def _is_pinned(self, program_id: str) -> bool:
        """检查程序是否被 PIN 且未过期"""
        if program_id not in self.pinned_programs:
            return False
        return self.current_time <= self.pinned_programs[program_id]
    
    def _schedule_batch(self) -> List[QueuedRequest]:
        """调度批次 - 论文 Algorithm 1"""
        # 论文: Expire old pins
        self._expire_pins()
        
        if not self.waiting_queue:
            return []
        
        # 获取可调度的请求
        available = [r for r in self.waiting_queue if r.arrival_time <= self.current_time]
        if not available:
            self.current_time = min(r.arrival_time for r in self.waiting_queue)
            return self._schedule_batch()
        
        # 论文: Pick highest priority
        # 优先级: PIN 程序 > FCFS
        def priority_key(req: QueuedRequest):
            if self._is_pinned(req.program_id):
                return (0, req.arrival_time)  # PIN 程序最高优先级
            return (1, req.arrival_time)  # FCFS
        
        available.sort(key=priority_key)
        
        # 组成批次
        batch = []
        batch_tokens = 0
        for req in available:
            if len(batch) >= self.max_batch_size:
                break
            tokens_needed = req.input_tokens + req.output_tokens
            if tokens_needed > self.max_tokens:
                continue
            batch.append(req)
            batch_tokens += tokens_needed
        
        return batch
    
    def step(self) -> List[ScheduleResult]:
        """执行一个时间步 - 论文 Algorithm 1"""
        results = []
        batch = self._schedule_batch()
        if not batch:
            # 推进时间到下一个请求到达
            if self.waiting_queue:
                self.current_time = min(r.arrival_time for r in self.waiting_queue)
            return results
        
        batch_start_time = self.current_time
        
        # 阶段 1: 缓存查找和淘汰（按顺序）
        request_times = []
        for request in batch:
            cache_hit, is_hit = self._find_cache_hit(request)
            request.cached_tokens = cache_hit.l0_tokens + cache_hit.l1_tokens
            request.miss_tokens = cache_hit.miss_tokens
            
            tokens_needed = cache_hit.miss_tokens + request.output_tokens
            evicted = self._evict_l0_for(tokens_needed)
            self.stats["evicted_tokens"] += evicted
            
            # 推理时间
            _, total_ms = self.time_predictor.predict_time(
                request.miss_tokens, request.output_tokens, is_hit)
            request_times.append((request, cache_hit, evicted, is_hit, total_ms))
            
            # 添加到缓存
            if request.output_tokens > 0:
                total = cache_hit.miss_tokens + request.output_tokens
                if self.current_l0_tokens + total <= self.max_tokens:
                    self.l0_cache[(request.program_id, request.turn_index)] = (total, batch_start_time)
                    self.current_l0_tokens += total
        
        # 阶段 2: Batch 并行处理 - 总时间 = max(各请求时间)
        batch_time_ms = max(t[4] for t in request_times)
        batch_time_s = batch_time_ms / 1000.0
        
        # 更新时间
        self.current_time += batch_time_s
        
        # 阶段 3: 处理结果和 PIN
        for request, cache_hit, evicted, is_hit, total_ms in request_times:
            was_pinned = self._is_pinned(request.program_id)
            queue_time = batch_start_time - request.arrival_time
            
            self.stats["requests_processed"] += 1
            
            # 调度时释放 PIN
            if request.program_id in self.pinned_programs:
                del self.pinned_programs[request.program_id]
            
            self.completed_requests.append(request)
            self.waiting_queue.remove(request)
            
            results.append(ScheduleResult(
                rid=request.rid, program_id=request.program_id, turn_index=request.turn_index,
                cache_hit=cache_hit, queue_time=queue_time, h2d_time=0.0,
                inference_time=total_ms / 1000.0, total_time=batch_time_s,
                was_pinned=was_pinned, evicted_tokens=evicted,
            ))
        
        # 论文: OnRequestFinish - 为未完成的程序设置 PIN
        for request, cache_hit, evicted, is_hit, total_ms in request_times:
            if not request.is_tool_call or request.tool_duration == 0:
                continue
            
            # 记录工具执行
            self.ttl_simulator.record_tool_execution(
                request.program_id, request.tool_name, request.tool_duration)
            
            # 工具执行时间
            self.current_time += request.tool_duration
            
            # 计算 TTL
            memory_usage = request.input_tokens + request.output_tokens
            ttl = self.ttl_simulator.select_dynamic_ttl(
                request.program_id, request.tool_name, memory_usage, self.max_tokens)
            
            # 设置 PIN（只在 Continuum 模式下）
            # Baseline 模式: ttl_sim.enable_adaptive_ttl=False, select_dynamic_ttl 返回 default_ttl
            # 所以需要额外检查 enable_adaptive_ttl
            if ttl > 0 and request.program_id not in self.pinned_programs and self.ttl_simulator.enable_adaptive_ttl:
                self.pinned_programs[request.program_id] = self.current_time + ttl
                self.stats["pins"] += 1
        
        return results
    
    def run_until_complete(self, max_iterations: int = 100000) -> List[ScheduleResult]:
        all_results = []
        for _ in range(max_iterations):
            if not self.waiting_queue:
                # 论文: 处理剩余的 PIN
                self._expire_pins()
                break
            results = self.step()
            all_results.extend(results)
            if not results and not self.waiting_queue:
                break
        return all_results
    
    def get_summary(self) -> Dict[str, Any]:
        total_l0, total_l1, total_miss = self.stats["l0_hits"], self.stats["l1_hits"], self.stats["misses"]
        total_cache = total_l0 + total_l1
        total_requests = total_cache + total_miss
        hit_rate = total_cache / max(1, total_requests)
        
        return {
            "mode": self.mode,
            "total_requests": self.stats["requests_processed"],
            "wall_time": self.current_time,
            "cache_stats": {
                "l0_hits": total_l0, "l1_hits": total_l1,
                "misses": total_miss, "hit_rate": hit_rate,
            },
            "ttl_stats": self.ttl_simulator.get_stats(),
            "pin_stats": {"total_pins": self.stats["pins"], "total_unpins": self.stats["unpins"]},
            "evicted_tokens": self.stats["evicted_tokens"],
            "active_pins": len(self.pinned_programs),
        }


def create_scheduler_simulator(mode: str = "baseline", max_tokens: int = 50000, **kwargs):
    predictor = TimePredictor(
        prefill_per_token_ms=kwargs.get("prefill_per_token_ms", 0.1),
        decode_per_token_ms=kwargs.get("decode_per_token_ms", 5.0),
        cache_hit_speedup=kwargs.get("cache_hit_speedup", 10.0),
    )
    ttl_sim = ContinuumTTLSimulator(
        default_ttl=kwargs.get("default_ttl", 3.0),
        history_threshold=kwargs.get("history_threshold", 1),
        enable_adaptive_ttl=kwargs.get("enable_dttl", True),
    )
    return CompleteSchedulerSimulator(
        mode=mode, max_tokens=max_tokens,
        ttl_simulator=ttl_sim, time_predictor=predictor,
        max_batch_size=kwargs.get("max_batch_size", 16),
    )
