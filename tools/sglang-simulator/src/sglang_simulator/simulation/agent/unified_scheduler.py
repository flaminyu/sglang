#!/usr/bin/env python3
"""
Unified KVCache TTL Scheduler - 论文对齐版

根据 Continuum 论文 (arXiv:2511.02230) 实现

论文核心算法:
    τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - τ

关键修改:
1. PIN 逻辑：请求完成时，如果是非最后一轮且有工具调用，则设置 PIN
2. TTL 计算：基于历史 CDF 的 cost-benefit 模型
3. 调度优先级：PIN 请求 > FCFS
4. 死锁预防：空间不足时选择性 unpin
5. PIN 解除：TTL 过期且下一轮未在队列中
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_simulator.simulation.agent.dataset import AgentProgram, TurnSpec


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CacheHitResult:
    """缓存命中结果"""
    l0_tokens: int = 0  # GPU (HBM) 直接命中
    l1_tokens: int = 0  # Host 命中，需要 H2D 传输
    miss_tokens: int = 0  # 未命中

    @property
    def total_hit_tokens(self) -> int:
        return self.l0_tokens + self.l1_tokens


@dataclass
class QueuedRequest:
    """排队的请求"""
    rid: str
    program_id: str
    turn_index: int
    arrival_time: float  # 到达时间
    input_tokens: int
    output_tokens: int
    is_tool_call: bool = False
    tool_name: Optional[str] = None
    tool_duration: float = 0.0
    is_last_turn: bool = False  # 是否是最后一轮
    program_first_arrival: float = 0.0  # 程序首次到达时间

    @property
    def total_tokens(self) -> int:
        """总 tokens (输入 + 输出)"""
        return self.input_tokens + self.output_tokens


@dataclass
class ScheduleResult:
    """调度结果"""
    rid: str
    program_id: str
    turn_index: int
    arrival_time: float
    completion_time: float  # 完成时间
    cache_hit: CacheHitResult
    queue_time: float  # 排队等待时间
    h2d_time: float  # Host -> GPU 传输时间
    prefill_time: float  # Prefill 时间
    decode_time: float  # Decode 时间
    tool_time: float  # 工具执行时间
    total_time: float  # 处理时间 (不含 queue_time)
    was_pinned: bool  # 是否被 PIN
    pinned_ttl: float  # PIN 的 TTL 值
    evicted_tokens: int  # 从 GPU 淘汰到 Host 的 tokens


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool_name: str
    duration: float
    timestamp: float


# ============================================================================
# Continuum TTL 模拟器 - 论文对齐版
# ============================================================================

class ContinuumTTLSimulator:
    """
    Continuum TTL 模拟器 - 论文对齐版

    论文公式:
    τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) - τ

    Where:
    - P(τ, f) = CDF of tool f execution time ≤ τ
    - T = average queueing delay per unit memory
    - η = memoryfulness factor
    - Prefill-Reload = prefill or CPU reload time
    """

    def __init__(
        self,
        default_ttl: float = 3.0,
        min_ttl: float = 0.1,
        max_ttl: float = 10.0,
        history_threshold: int = 10,
        # 论文参数
        avg_queue_delay: float = 1.0,  # T
        memoryfulness: float = 0.8,  # η
        prefill_time: float = 0.1,  # Prefill-Reload
        reload_time: float = 0.05,  # CPU reload
        enable_adaptive_ttl: bool = True,
    ):
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.history_threshold = history_threshold

        # 论文参数
        self.avg_queue_delay = avg_queue_delay  # T
        self.memoryfulness = memoryfulness  # η
        self.prefill_time = prefill_time  # Prefill-Reload
        self.reload_time = reload_time  # CPU reload

        self.enable_adaptive_ttl = enable_adaptive_ttl

        # 工具执行历史：tool_name -> list of durations
        self.tool_history: Dict[str, List[float]] = defaultdict(list)
        # 程序工具历史：program_id -> tool_name -> list of durations
        self.program_tool_history: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # 程序空闲间隔：program_id -> list of idle gaps
        self.program_idle_gaps: Dict[str, List[float]] = defaultdict(list)
        # 全局空闲间隔
        self.global_idle_gaps: List[float] = []
        # 请求完成时间记录：program_id -> last_completion_time
        self.program_last_completion: Dict[str, float] = {}

        # 当前平均排队延迟（滑动窗口）
        self.queue_delay_history: List[float] = []
        self.queue_delay_maxlen = 100
        self.current_avg_queue_delay: float = avg_queue_delay

        # 统计信息
        self.stats = {
            "ttl_selections": 0,
            "tool_cdf_selections": 0,
            "global_cdf_selections": 0,
            "default_selections": 0,
            "cold_start_selections": 0,
        }

    def reset(self):
        """重置模拟器状态"""
        self.tool_history.clear()
        self.program_tool_history.clear()
        self.program_idle_gaps.clear()
        self.global_idle_gaps.clear()
        self.program_last_completion.clear()
        self.queue_delay_history = []
        self.current_avg_queue_delay = self.avg_queue_delay
        self.stats = {
            "ttl_selections": 0,
            "tool_cdf_selections": 0,
            "global_cdf_selections": 0,
            "default_selections": 0,
            "cold_start_selections": 0,
        }

    def _compute_cdf_probability(self, samples: List[float], tau: float) -> float:
        """
        计算 P(τ) = P(tool_duration ≤ τ)
        即工具执行时间 ≤ τ 的概率（累积分布函数）
        """
        if not samples:
            return 0.5  # 默认 50%
        count_le = sum(1 for s in samples if s <= tau)
        return count_le / len(samples)

    def _compute_benefit(self) -> float:
        """
        计算 Benefit = T·η + Prefill-Reload

        论文中：
        - T: 平均排队延迟
        - η: memoryfulness factor (0.8 表示较强的内存感知)
        - Prefill-Reload: KV 重建时间
        """
        return self.current_avg_queue_delay * self.memoryfulness + self.prefill_time

    def record_tool_execution(
        self,
        program_id: str,
        tool_name: str,
        duration: float,
        timestamp: float = 0.0,
    ):
        """记录工具执行时间"""
        # 记录到全局历史
        self.tool_history[tool_name].append(duration)
        # 保持历史记录在合理范围内
        if len(self.tool_history[tool_name]) > 1000:
            self.tool_history[tool_name] = self.tool_history[tool_name][-500:]

        # 记录到程序历史
        self.program_tool_history[program_id][tool_name].append(duration)

        # 更新程序最后完成时间
        self.program_last_completion[program_id] = timestamp

    def record_request_completion(
        self,
        program_id: str,
        completion_time: float,
    ):
        """记录请求完成时间"""
        self.program_last_completion[program_id] = completion_time

    def record_idle_gap(
        self,
        program_id: str,
        gap: float,
    ):
        """记录空闲间隔（工具执行开始到下一轮到达的时间）"""
        if gap <= 0:
            return

        # 记录到程序历史
        self.program_idle_gaps[program_id].append(gap)
        if len(self.program_idle_gaps[program_id]) > 1000:
            self.program_idle_gaps[program_id] = self.program_idle_gaps[program_id][-500:]

        # 记录到全局历史
        self.global_idle_gaps.append(gap)
        if len(self.global_idle_gaps) > 1000:
            self.global_idle_gaps = self.global_idle_gaps[-500:]

    def record_queue_delay(self, queue_delay: float):
        """记录排队延迟"""
        if queue_delay > 0:
            self.queue_delay_history.append(queue_delay)
            if len(self.queue_delay_history) > self.queue_delay_maxlen:
                self.queue_delay_history = self.queue_delay_history[-50:]
            if len(self.queue_delay_history) > 10:
                self.current_avg_queue_delay = statistics.mean(self.queue_delay_history)

    def select_ttl(
        self,
        program_id: str,
        tool_name: str,
    ) -> Tuple[float, str]:
        """
        论文公式计算最优 TTL

        τ* = argmax_τ P(τ,f) × (T·η + Prefill-Reload) − τ

        枚举所有候选 τ 值，选择使收益最大化的
        """
        self.stats["ttl_selections"] += 1

        if not self.enable_adaptive_ttl:
            self.stats["default_selections"] += 1
            return self.default_ttl, "fixed"

        # 获取候选 τ 值
        candidates = self._get_ttl_candidates(program_id, tool_name)

        if not candidates:
            # 冷启动：使用默认 TTL
            self.stats["cold_start_selections"] += 1
            return self.default_ttl, "cold_start"

        benefit = self._compute_benefit()
        best_ttl = self.default_ttl
        best_score = float('-inf')
        strategy = "default"

        # 获取工具执行时间的统计信息
        tool_durations = self.tool_history.get(tool_name, [])
        tool_mean = statistics.mean(tool_durations) if tool_durations else self.default_ttl

        # 获取程序空闲间隔统计
        idle_gaps = list(self.program_idle_gaps.get(program_id, [])) + list(self.global_idle_gaps)
        idle_mean = statistics.mean(idle_gaps) if idle_gaps else self.avg_queue_delay

        for tau in candidates:
            # P(τ, f) = 工具执行时间 ≤ τ 的概率
            p_tau = self._compute_cdf_probability(tool_durations, tau)
            
            # TTL 选择策略：
            # 1. 如果 τ 太短 (小于 idle_mean)，即使缓存失效后下次请求也难以命中
            # 2. 如果 τ 太长，会占用过多 GPU 缓存空间
            # 3. 理想 TTL 应该略大于 idle_mean，让下一轮可以复用但不会长期占用
            
            # 计算 τ 与 idle_mean 的接近度
            if idle_mean > 0:
                # τ 应该略大于 idle_mean，但不能超过太多
                if tau < idle_mean * 0.5:
                    # 太短了，下次请求可能还没来
                    idle_proximity = tau / (idle_mean * 0.5)
                elif tau <= idle_mean * 1.5:
                    # 理想范围
                    idle_proximity = 1.0
                else:
                    # 太长了
                    idle_proximity = max(0.1, idle_mean * 1.5 / tau)
            else:
                idle_proximity = 0.5
            
            # 分数 = P(τ,f) × benefit × idle_proximity - τ × memory_cost
            # memory_cost 是持有缓存的机会成本（高压场景下应该更高）
            memory_cost = 0.1  # 增加机会成本，限制长期占用
            score = p_tau * benefit * idle_proximity - tau * memory_cost

            if score > best_score:
                best_score = score
                best_ttl = tau
                strategy = "adaptive"

        # 限制在合理范围内
        best_ttl = min(self.max_ttl, max(self.min_ttl, best_ttl))

        # 确定使用的策略
        if strategy == "adaptive":
            if len(tool_durations) >= self.history_threshold:
                self.stats["tool_cdf_selections"] += 1
            elif len(self.global_idle_gaps) >= self.history_threshold:
                self.stats["global_cdf_selections"] += 1
        else:
            self.stats["default_selections"] += 1

        return best_ttl, strategy

    def _get_ttl_candidates(
        self,
        program_id: str,
        tool_name: str,
    ) -> List[float]:
        """
        获取 TTL 候选值列表

        论文：枚举所有唯一的工具执行时间作为候选
        """
        candidates = set()

        # 1. 工具历史执行时间
        if tool_name in self.tool_history:
            candidates.update(self.tool_history[tool_name])

        # 2. 程序特定的工具历史
        if program_id in self.program_tool_history:
            if tool_name in self.program_tool_history[program_id]:
                candidates.update(self.program_tool_history[program_id][tool_name])

        # 3. 全局工具历史
        for name, durations in self.tool_history.items():
            if name != tool_name:
                candidates.update(durations[:5])  # 只取前5个

        # 4. 程序空闲间隔
        if program_id in self.program_idle_gaps:
            candidates.update(self.program_idle_gaps[program_id])

        # 5. 全局空闲间隔
        candidates.update(self.global_idle_gaps)

        # 添加一些标准值
        candidates.update([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

        # 转换为列表并排序
        return sorted(list(candidates))

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "avg_queue_delay": self.current_avg_queue_delay,
            "tools_tracked": len(self.tool_history),
            "programs_tracked": len(self.program_tool_history),
        }


# ============================================================================
# 时间预测器
# ============================================================================

class TimePredictor:
    """推理时间预测器"""

    def __init__(
        self,
        prefill_per_token_ms: float = 0.1,
        decode_per_token_ms: float = 5.0,
        cache_hit_speedup: float = 10.0,
        l1_reload_penalty: float = 0.3,
    ):
        self.prefill_per_token_ms = prefill_per_token_ms
        self.decode_per_token_ms = decode_per_token_ms
        self.cache_hit_speedup = cache_hit_speedup
        self.l1_reload_penalty = l1_reload_penalty

    def predict_time(
        self,
        miss_tokens: int,
        output_tokens: int,
        l0_tokens: int = 0,
        l1_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """
        预测推理时间

        Returns: (prefill_time_ms, decode_time_ms, total_ms)
        """
        # L0命中: 完全跳过 prefill
        if l0_tokens >= miss_tokens:
            prefill_time_ms = 0.0
            decode_time_ms = output_tokens * self.decode_per_token_ms / self.cache_hit_speedup
            total_ms = decode_time_ms
        # L1命中: 部分 prefill 开销
        elif l1_tokens > 0:
            actual_miss = max(0, miss_tokens - l1_tokens)
            prefill_time_ms = actual_miss * self.prefill_per_token_ms * self.l1_reload_penalty
            decode_time_ms = output_tokens * self.decode_per_token_ms / self.cache_hit_speedup
            total_ms = prefill_time_ms + decode_time_ms
        # Miss: 完整 prefill
        else:
            prefill_time_ms = miss_tokens * self.prefill_per_token_ms
            decode_time_ms = output_tokens * self.decode_per_token_ms
            total_ms = prefill_time_ms + decode_time_ms

        return prefill_time_ms, decode_time_ms, total_ms


# ============================================================================
# 统一调度器模拟器 - 论文对齐版
# ============================================================================

class UnifiedSchedulerSimulator:
    """
    统一调度器模拟器 - 论文对齐版

    根据 Continuum 论文 Algorithm 1 实现：

    Global state:
        Q: waiting queue
        P: TTL map (records pinned programs and their TTLs)

    OnRequestArrive(request r):
        Q ← Q ∪ {r}
        id ← Program ID of r
        If id is a seen program then
            (f, t) ← Tool-call information from r
            Record tool execution time

    OnRequestFinish(request r):
        If r is the last request of its program then
            Free KV cache used by r
        else
            f ← Next tool to be called after finishing r
            id ← Program ID of r
            P[id] ← CalcTTL(r, S[f])

    Schedule():
        While Q is not empty do
            For each id in P.keys do
                If current time > P[id] and id ∉ Q.programs then
                    Free KV cache used by id's last request
                    P ← P \ (id, P[id])
            r ← argmax r′∈Q CalcPriority(r′, P)
            If r cannot fit into memory then
                # 死锁预防：选择性 unpin
                break
            else
                Q ← Q \ {r}
                Issue r to running
                id ← Program ID of r
                If id ∈ P.keys then
                    P ← P \ (id, P[id])
    """

    def __init__(
        self,
        mode: str,
        max_tokens: int,
        max_host_tokens: int = 0,
        ttl_simulator: Optional[ContinuumTTLSimulator] = None,
        time_predictor: Optional[TimePredictor] = None,
        h2d_bandwidth_gb: float = 50.0,
    ):
        self.mode = mode
        self.max_tokens = max_tokens
        self.max_host_tokens = max_host_tokens if max_host_tokens > 0 else max_tokens * 10
        self.h2d_bandwidth_gb = h2d_bandwidth_gb

        self.time_predictor = time_predictor or TimePredictor()
        self.ttl_simulator = ttl_simulator or ContinuumTTLSimulator(
            enable_adaptive_ttl=(mode == "continuum")
        )

        # 两级缓存
        self.l0_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l0_tokens = 0

        self.l1_cache: Dict[Tuple[str, int], Tuple[int, float]] = {}
        self.current_l1_tokens = 0

        # PIN 缓存 - 论文风格: program_id -> (expires_at, ttl_value)
        self.pinned_programs: Dict[str, Tuple[float, float]] = {}

        # 请求完成记录: (program_id, turn_index) -> completion_time
        self.request_completion: Dict[Tuple[str, int], float] = {}

        # 程序追踪
        self.program_info: Dict[str, Dict] = {}  # program_id -> {first_arrival, total_turns, ...}
        self.program_request_count: Dict[str, int] = {}  # program_id -> 已完成的轮次数

        # 请求队列
        self.waiting_queue: deque = deque()
        self.completed_requests: List[ScheduleResult] = []
        self.results: List[ScheduleResult] = []

        # 时间
        self.current_time = 0.0
        self._request_counter = 0

        # 统计
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "pins": 0,
            "unpins": 0,
            "evicted_tokens": 0,
            "deadlock_preventions": 0,
        }

        # 每 token 字节数
        self.bytes_per_token = 128

    def reset(self):
        """重置模拟器"""
        self.l0_cache.clear()
        self.l1_cache.clear()
        self.pinned_programs.clear()
        self.request_completion.clear()
        self.program_info.clear()
        self.program_request_count.clear()
        self.waiting_queue.clear()
        self.completed_requests.clear()
        self.results.clear()
        self.current_time = 0.0
        self._request_counter = 0
        self.ttl_simulator.reset()
        self.stats = {
            "requests_processed": 0,
            "l0_hits": 0,
            "l1_hits": 0,
            "misses": 0,
            "pins": 0,
            "unpins": 0,
            "evicted_tokens": 0,
            "deadlock_preventions": 0,
        }

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
        is_last_turn: bool = False,
    ) -> str:
        """添加请求到队列"""
        self._request_counter += 1
        rid = f"{program_id}_T{turn_index}_{self._request_counter}"

        # 获取程序首次到达时间
        first_arrival = self.program_info.get(program_id, {}).get(
            "first_arrival", arrival_time
        )

        request = QueuedRequest(
            rid=rid,
            program_id=program_id,
            turn_index=turn_index,
            arrival_time=arrival_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_tool_call=is_tool_call,
            tool_name=tool_name,
            tool_duration=tool_duration,
            is_last_turn=is_last_turn,
            program_first_arrival=first_arrival,
        )
        self.waiting_queue.append(request)

        # 追踪程序信息
        if program_id not in self.program_info:
            self.program_info[program_id] = {
                "first_arrival": arrival_time,
                "total_turns": turn_index + 1,
            }
            self.program_request_count[program_id] = 0
        else:
            self.program_info[program_id]["total_turns"] = max(
                self.program_info[program_id]["total_turns"], turn_index + 1
            )

        return rid

    def _find_cache_hit(self, request: QueuedRequest) -> Tuple[CacheHitResult, bool]:
        """查找两级缓存命中"""
        # L0 查找
        l0_match, l0_key = self._find_best_match(
            request.program_id, request.turn_index, request.input_tokens, self.l0_cache
        )

        if l0_match > 0:
            self.stats["l0_hits"] += 1
            if l0_key:
                self.l0_cache[l0_key] = (self.l0_cache[l0_key][0], self.current_time)
            return CacheHitResult(
                l0_tokens=l0_match, miss_tokens=request.input_tokens - l0_match
            ), True

        # L1 查找
        l1_match, l1_key = self._find_best_match(
            request.program_id, request.turn_index, request.input_tokens, self.l1_cache
        )

        if l1_match > 0:
            self.stats["l1_hits"] += 1
            if l1_key:
                self.l1_cache[l1_key] = (self.l1_cache[l1_key][0], self.current_time)
            return CacheHitResult(
                l1_tokens=l1_match, miss_tokens=request.input_tokens - l1_match
            ), True

        self.stats["misses"] += 1
        return CacheHitResult(miss_tokens=request.input_tokens), False

    def _find_best_match(
        self,
        program_id: str,
        turn_index: int,
        input_tokens: int,
        cache: Dict[Tuple[str, int], Tuple[int, float]],
    ) -> Tuple[int, Optional[Tuple[str, int]]]:
        """在缓存中查找最佳匹配"""
        best_match, best_key = 0, None

        for key, (cached_tokens, _) in cache.items():
            prog_id, cached_turn = key
            if prog_id != program_id or cached_turn >= turn_index:
                continue

            matched = min(cached_tokens, input_tokens)
            if matched > best_match:
                best_match, best_key = matched, key

        return best_match, best_key

    def _calculate_h2d_time(self, tokens: int) -> float:
        """计算 H2D 传输时间"""
        if tokens <= 0:
            return 0.0
        total_bytes = tokens * self.bytes_per_token
        gb = total_bytes / (1024 ** 3)
        return gb / self.h2d_bandwidth_gb

    def _evict_l0_for(self, tokens_needed: int) -> int:
        """
        为新请求腾出 GPU 空间 - LRU 淘汰到 Host

        论文逻辑：PIN 的条目不会被淘汰
        """
        evicted = 0

        if tokens_needed > self.max_tokens:
            return 0

        while self.current_l0_tokens + tokens_needed > self.max_tokens and self.l0_cache:
            # 找最老的未 PIN 条目
            oldest_key = None
            oldest_time = float('inf')

            for key, (cached_tokens, last_access) in self.l0_cache.items():
                prog_id = key[0]
                # 如果程序被 PIN，跳过
                if self._is_pinned(prog_id):
                    continue
                if last_access < oldest_time:
                    oldest_time = last_access
                    oldest_key = key

            if oldest_key is None:
                # 所有条目都是 PIN，无法驱逐
                break

            tokens, _ = self.l0_cache.pop(oldest_key)
            self.current_l0_tokens -= tokens
            evicted += tokens

            # 移到 Host Cache
            if self.current_l1_tokens + tokens <= self.max_host_tokens:
                self.l1_cache[oldest_key] = (tokens, oldest_time)
                self.current_l1_tokens += tokens

        return evicted

    def _unpin_expired(self):
        """
        论文 Schedule() 函数的一部分：

        For each id in P.keys do
            If current time > P[id] and id ∉ Q.programs then
                Free KV cache used by id's last request
                P ← P \ (id, P[id])
        """
        expired = []
        for pid in list(self.pinned_programs.keys()):
            expires_at, _ = self.pinned_programs[pid]
            # 检查是否过期
            if self.current_time > expires_at:
                # 检查程序是否还有等待的请求
                has_pending = any(r.program_id == pid for r in self.waiting_queue)
                if not has_pending:
                    expired.append(pid)

        for pid in expired:
            del self.pinned_programs[pid]
            self.stats["unpins"] += 1

    def _is_pinned(self, program_id: str) -> bool:
        """检查程序是否被 PIN 且未过期"""
        if program_id not in self.pinned_programs:
            return False
        expires_at, _ = self.pinned_programs[program_id]
        return self.current_time <= expires_at

    def _get_pinned_ttl(self, program_id: str) -> float:
        """获取程序的 PIN TTL 值"""
        if program_id not in self.pinned_programs:
            return 0.0
        return self.pinned_programs[program_id][1]

    def _schedule_one(self) -> Optional[QueuedRequest]:
        """
        论文 Schedule() 函数:

        r ← argmax r′∈Q CalcPriority(r′, P)
        """
        # 论文：先清理过期的 PIN
        self._unpin_expired()

        if not self.waiting_queue:
            return None

        # 获取可调度的请求（到达时间 <= 当前时间）
        available = [r for r in self.waiting_queue if r.arrival_time <= self.current_time]
        if not available:
            # 推进时间到下一个请求到达
            self.current_time = min(r.arrival_time for r in self.waiting_queue)
            return self._schedule_one()

        # 论文优先级：
        # 1. PIN 状态（TTL 窗口内）优先
        # 2. Program-level FCFS
        def priority_key(req: QueuedRequest):
            is_pinned = self._is_pinned(req.program_id)
            # PIN 程序优先，然后按程序首次到达时间排序
            return (0 if is_pinned else 1, req.program_first_arrival, req.turn_index)

        available.sort(key=priority_key)
        return available[0]

    def _try_schedule_with_deadlock_prevention(self) -> Optional[QueuedRequest]:
        """
        尝试调度请求，包含死锁预防机制

        论文：
        If r cannot fit into memory then
            # 选择 PIN 中到达时间最晚的作为受害者
            break
        """
        # 先尝试调度
        request = self._schedule_one()
        if not request:
            return None

        tokens_needed = request.total_tokens

        # 检查是否能放入内存
        max_eviction_attempts = 100  # 防止无限循环
        eviction_attempts = 0

        while self.current_l0_tokens + tokens_needed > self.max_tokens and eviction_attempts < max_eviction_attempts:
            eviction_attempts += 1

            # 找可以驱逐的最老条目
            evictable_keys = [
                k for k in self.l0_cache.keys()
                if not self._is_pinned(k[0])
            ]

            if not evictable_keys:
                # 没有可驱逐的条目，进入死锁预防
                self._deadlock_prevention()
                # 再次尝试
                if self.current_l0_tokens + tokens_needed <= self.max_tokens:
                    break
                # 仍然空间不足，检查是否有未到达的请求
                if self.waiting_queue:
                    future_requests = [r for r in self.waiting_queue if r.arrival_time > self.current_time]
                    if future_requests:
                        # 推进时间到下一个请求到达
                        self.current_time = min(r.arrival_time for r in future_requests)
                        continue
                return None

            # 驱逐最老的条目
            oldest_key = min(evictable_keys, key=lambda k: self.l0_cache[k][1])
            tokens, _ = self.l0_cache.pop(oldest_key)
            self.current_l0_tokens -= tokens
            self.stats["evicted_tokens"] += tokens

            # 移到 Host
            if self.current_l1_tokens + tokens <= self.max_host_tokens:
                self.l1_cache[oldest_key] = (tokens, self.current_time)
                self.current_l1_tokens += tokens

        return request

    def _deadlock_prevention(self):
        """
        死锁预防机制

        论文：
        When the scheduling logic fails to schedule a new request to execute,
        we iteratively selects victims from pinned_requests with the latest
        program arrival time to unpin and free the space until the first
        request can be scheduled to run.
        """
        self.stats["deadlock_preventions"] += 1

        # 选择 PIN 中到达时间最晚的作为受害者
        pinned_programs = list(self.pinned_programs.keys())
        if not pinned_programs:
            return

        # 按程序首次到达时间排序，选择最晚的
        victims = sorted(
            pinned_programs,
            key=lambda pid: self.program_info.get(pid, {}).get("first_arrival", 0),
            reverse=True,
        )

        for victim_pid in victims:
            if victim_pid in self.pinned_programs:
                del self.pinned_programs[victim_pid]
                self.stats["unpins"] += 1

                # 检查是否腾出了足够空间
                # 找该程序最老的缓存条目并驱逐
                victim_keys = [
                    k for k in self.l0_cache.keys() if k[0] == victim_pid
                ]
                if victim_keys:
                    oldest_key = min(victim_keys, key=lambda k: self.l0_cache[k][1])
                    tokens, _ = self.l0_cache.pop(oldest_key)
                    self.current_l0_tokens -= tokens
                    self.stats["evicted_tokens"] += tokens

                    # 移到 Host
                    if self.current_l1_tokens + tokens <= self.max_host_tokens:
                        self.l1_cache[oldest_key] = (tokens, self.current_time)
                        self.current_l1_tokens += tokens

                # 再次检查是否足够空间
                # 找到当前队列中需要最少空间的请求
                if self.waiting_queue:
                    min_tokens = min(r.total_tokens for r in self.waiting_queue)
                    if self.current_l0_tokens + min_tokens <= self.max_tokens:
                        break

    def step(self) -> List[ScheduleResult]:
        """执行一个时间步"""
        results = []

        # 调度一个请求
        request = self._try_schedule_with_deadlock_prevention()
        if not request:
            return results

        request_start_time = self.current_time

        # ===== 阶段 1: 缓存查找 =====
        cache_hit, is_hit = self._find_cache_hit(request)

        # ===== 阶段 2: 缓存驱逐（为当前请求腾出空间）=====
        tokens_needed = cache_hit.miss_tokens + request.output_tokens
        self._evict_l0_for(tokens_needed)

        # ===== 阶段 3: 推理时间计算 =====
        prefill_ms, decode_ms, total_ms = self.time_predictor.predict_time(
            cache_hit.miss_tokens,
            request.output_tokens,
            l0_tokens=cache_hit.l0_tokens,
            l1_tokens=cache_hit.l1_tokens,
        )

        # H2D 时间
        h2d_time = self._calculate_h2d_time(cache_hit.l1_tokens)

        # ===== 阶段 4: 记录工具执行和设置 PIN =====
        tool_duration = request.tool_duration if request.is_tool_call else 0.0
        pinned_ttl = 0.0

        # 论文 OnRequestFinish:
        # If r is the last request of its program then
        #     Free KV cache used by r
        # else
        #     f ← Next tool to be called after finishing r
        #     id ← Program ID of r
        #     P[id] ← CalcTTL(r, S[f])

        if not request.is_last_turn and request.is_tool_call:
            # 非最后一轮且有工具调用，设置 PIN

            # 记录工具执行历史
            self.ttl_simulator.record_tool_execution(
                request.program_id,
                request.tool_name or "unknown",
                request.tool_duration,
                self.current_time,
            )

            if self.ttl_simulator.enable_adaptive_ttl:
                # 计算 TTL
                ttl, strategy = self.ttl_simulator.select_ttl(
                    request.program_id,
                    request.tool_name or "unknown",
                )

                # 设置 PIN
                expires_at = self.current_time + ttl
                self.pinned_programs[request.program_id] = (expires_at, ttl)
                self.stats["pins"] += 1
                pinned_ttl = ttl

        # ===== 阶段 5: 添加到缓存 =====
        if request.output_tokens > 0:
            total = cache_hit.miss_tokens + request.output_tokens
            key = (request.program_id, request.turn_index)

            if self.current_l0_tokens + total <= self.max_tokens:
                self.l0_cache[key] = (total, self.current_time)
                self.current_l0_tokens += total

        # ===== 阶段 6: 更新时间 =====
        request_processing_time = total_ms / 1000.0 + h2d_time
        completion_time = self.current_time + request_processing_time
        self.current_time = completion_time

        # 记录请求完成
        self.request_completion[(request.program_id, request.turn_index)] = completion_time
        self.ttl_simulator.record_request_completion(
            request.program_id, completion_time
        )

        # 更新程序请求计数
        self.program_request_count[request.program_id] = (
            self.program_request_count.get(request.program_id, 0) + 1
        )

        # ===== 阶段 7: 记录 idle_gap =====
        # 计算空闲间隔：上一轮完成到这一轮开始的时间
        if request.turn_index > 0 and tool_duration > 0:
            prev_completion = self.request_completion.get(
                (request.program_id, request.turn_index - 1), 0
            )
            if prev_completion > 0:
                idle_gap = request.arrival_time - prev_completion - tool_duration
                if idle_gap > 0:
                    self.ttl_simulator.record_idle_gap(request.program_id, idle_gap)

        # 记录排队延迟
        queue_time = request_start_time - request.arrival_time
        if queue_time > 0:
            self.ttl_simulator.record_queue_delay(queue_time)

        # ===== 阶段 8: 移除 PIN（如果下一轮已到达）=====
        # 论文: id ∈ P.keys then P ← P \ (id, P[id])
        if request.program_id in self.pinned_programs:
            # 检查是否需要移除 PIN
            has_next_turn = any(
                r.program_id == request.program_id
                and r.turn_index == request.turn_index + 1
                for r in self.waiting_queue
            )
            if has_next_turn:
                # 下一轮已在队列中，移除 PIN（因为会立即使用缓存）
                del self.pinned_programs[request.program_id]

        # ===== 阶段 9: 生成结果 =====
        was_pinned = self._is_pinned(request.program_id)
        self.stats["requests_processed"] += 1

        result = ScheduleResult(
            rid=request.rid,
            program_id=request.program_id,
            turn_index=request.turn_index,
            arrival_time=request.arrival_time,
            completion_time=completion_time,
            cache_hit=cache_hit,
            queue_time=queue_time,
            h2d_time=h2d_time,
            prefill_time=prefill_ms / 1000.0,
            decode_time=decode_ms / 1000.0,
            tool_time=tool_duration,
            total_time=request_processing_time,
            was_pinned=was_pinned,
            pinned_ttl=pinned_ttl,
            evicted_tokens=self.stats["evicted_tokens"],
        )

        self.completed_requests.append(result)
        self.results.append(result)
        self.waiting_queue.remove(request)

        results.append(result)
        return results

    def run_until_complete(self, max_iterations: int = 100000) -> List[ScheduleResult]:
        """运行直到所有请求完成"""
        self.results = []
        for _ in range(max_iterations):
            if not self.waiting_queue:
                self._unpin_expired()
                break
            results = self.step()
            self.results.extend(results)
            self.completed_requests.extend(results)
            if not results and not self.waiting_queue:
                break
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """获取模拟摘要"""
        total_l0, total_l1, total_miss = (
            self.stats["l0_hits"],
            self.stats["l1_hits"],
            self.stats["misses"],
        )
        total_cache = total_l0 + total_l1
        total_requests = total_cache + total_miss
        hit_rate = total_cache / max(1, total_requests)

        # 计算总时间
        total_queue_time = sum(r.queue_time for r in self.completed_requests)
        total_inference_time = sum(r.prefill_time + r.decode_time for r in self.completed_requests)
        total_wall_time = self.current_time

        # 计算总工具时间
        total_tool_time = sum(r.tool_time for r in self.completed_requests)

        # E2E 时间 = 推理时间 + 排队时间（不含工具执行）
        # 这是实际消耗的 GPU 计算资源时间
        e2e_time = total_inference_time + total_queue_time

        return {
            "mode": self.mode,
            "total_requests": self.stats["requests_processed"],
            "wall_time": total_wall_time,
            "e2e_time": e2e_time,
            "total_queue_time": total_queue_time,
            "total_inference_time": total_inference_time,
            "total_tool_time": total_tool_time,
            "cache_stats": {
                "l0_hits": total_l0,
                "l1_hits": total_l1,
                "misses": total_miss,
                "hit_rate": hit_rate,
            },
            "ttl_stats": self.ttl_simulator.get_stats(),
            "pin_stats": {
                "total_pins": self.stats["pins"],
                "total_unpins": self.stats["unpins"],
                "active_pins": len(self.pinned_programs),
                "deadlock_preventions": self.stats["deadlock_preventions"],
            },
            "evicted_tokens": self.stats["evicted_tokens"],
        }

    def get_full_state(self) -> Dict[str, Any]:
        """获取当前完整状态用于可视化"""
        def format_cache_entry(key, value):
            prog_id, turn = key
            tokens, last_access = value
            return {
                "program_id": prog_id,
                "turn_index": turn,
                "tokens": tokens,
                "last_access": last_access,
            }

        l0_entries = [format_cache_entry(k, v) for k, v in self.l0_cache.items()]
        l1_entries = [format_cache_entry(k, v) for k, v in self.l1_cache.items()]

        pinned_entries = []
        for prog_id, (expires_at, ttl) in self.pinned_programs.items():
            pinned_entries.append({
                "program_id": prog_id,
                "expires_at": expires_at,
                "ttl": ttl,
                "remaining": max(0, expires_at - self.current_time),
            })

        return {
            "time": self.current_time,
            "mode": self.mode,
            "cache": {
                "l0_total_tokens": self.current_l0_tokens,
                "l0_max_tokens": self.max_tokens,
                "l0_entries": l0_entries,
                "l1_total_tokens": self.current_l1_tokens,
                "l1_max_tokens": self.max_host_tokens,
                "l1_entries": l1_entries,
                "pinned_entries": pinned_entries,
            },
            "queue": [
                {
                    "rid": r.rid,
                    "program_id": r.program_id,
                    "turn_index": r.turn_index,
                    "arrival_time": r.arrival_time,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "is_tool_call": r.is_tool_call,
                    "is_last_turn": r.is_last_turn,
                }
                for r in self.waiting_queue
            ],
            "queue_length": len(self.waiting_queue),
            "results_count": len(self.completed_requests),
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_scheduler_simulator(
    mode: str = "baseline",
    max_tokens: int = 50000,
    max_host_tokens: int = 0,
    **kwargs,
) -> UnifiedSchedulerSimulator:
    """创建调度器模拟器"""
    ttl_sim = ContinuumTTLSimulator(
        default_ttl=kwargs.get("default_ttl", 3.0),
        min_ttl=kwargs.get("min_ttl", 0.1),
        max_ttl=kwargs.get("max_ttl", 10.0),
        history_threshold=kwargs.get("history_threshold", 10),
        avg_queue_delay=kwargs.get("avg_queue_delay", 1.0),
        memoryfulness=kwargs.get("memoryfulness", 0.8),
        prefill_time=kwargs.get("prefill_time", 0.1),
        reload_time=kwargs.get("reload_time", 0.05),
        enable_adaptive_ttl=(mode == "continuum"),
    )

    predictor = TimePredictor(
        prefill_per_token_ms=kwargs.get("prefill_per_token_ms", 0.1),
        decode_per_token_ms=kwargs.get("decode_per_token_ms", 5.0),
        cache_hit_speedup=kwargs.get("cache_hit_speedup", 10.0),
        l1_reload_penalty=kwargs.get("l1_reload_penalty", 0.3),
    )

    return UnifiedSchedulerSimulator(
        mode=mode,
        max_tokens=max_tokens,
        max_host_tokens=max_host_tokens,
        ttl_simulator=ttl_sim,
        time_predictor=predictor,
        h2d_bandwidth_gb=kwargs.get("h2d_bandwidth_gb", 50.0),
    )


# ============================================================================
# 兼容性别名（用于旧代码兼容）
# ============================================================================

# ProgramStats 和 TTLNode 现在是内部使用，不对外导出
# 保留别名用于旧代码兼容
ProgramStats = None  # type: ignore
TTLNode = None  # type: ignore
