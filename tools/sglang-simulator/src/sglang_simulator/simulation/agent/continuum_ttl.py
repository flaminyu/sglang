#!/usr/bin/env python3
"""
DTTL (Dynamic TTL) 策略模拟器

模拟 Continuum 的 Dynamic TTL 策略，用于评估 DTTL 对缓存命中率和内存使用的影响。

核心概念：
- DTTL 是自适应的，不是固定值！
- TTL 根据历史数据动态计算
- default_ttl 只是没有历史数据时的回退值

DTTL 策略选择顺序：
1. 工具执行历史 (tool_based) - 最准确，基于特定工具的执行时间
2. 程序空闲间隔历史 (idle_gap_based) - 基于该程序的空闲间隔
3. 全局空闲间隔历史 (global_based) - 基于全局空闲间隔
4. 默认 TTL (default) - 没有足够历史数据时的回退值

Usage:
    ttl_sim = ContinuumTTLSimulator(config)
    
    # 选择 TTL (动态计算，不是固定值)
    ttl, strategy = ttl_sim.select_dynamic_ttl(program_id, tool_name)
    # strategy 可能是: "tool_history", "idle_gap_history", 
    #                 "global_idle_gap", "default"
    
    # 记录工具执行
    ttl_sim.record_tool_execution(program_id, tool_name, duration)
    
    # 检查过期
    expired = ttl_sim.check_expired_nodes(current_time)
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ProgramStats:
    """程序的统计信息"""
    tool_durations: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=1024)))
    idle_gaps: deque = field(default_factory=lambda: deque(maxlen=1024))
    turn_count: int = 0
    cached_turns: List[int] = field(default_factory=list)  # 被缓存的轮次索引
    ttl_values: List[float] = field(default_factory=list)  # 使用的 TTL 值
    memory_pressure: float = 0.5  # 0.0-1.0
    
    def mean_tool_duration(self, tool_name: str) -> Optional[float]:
        """获取指定工具的平均执行时间"""
        durations = self.tool_durations.get(tool_name)
        if durations and len(durations) > 0:
            return statistics.mean(durations)
        return None
    
    def mean_idle_gap(self) -> Optional[float]:
        """获取平均空闲间隔"""
        if len(self.idle_gaps) > 0:
            return statistics.mean(self.idle_gaps)
        return None
    
    def recent_idle_gaps(self, n: int = 10) -> List[float]:
        """获取最近的 n 个空闲间隔"""
        return list(self.idle_gaps)[-n:]
    
    def recent_tool_durations(self, tool_name: str, n: int = 10) -> List[float]:
        """获取指定工具最近的 n 个执行时间"""
        durations = self.tool_durations.get(tool_name)
        if durations:
            return list(durations)[-n:]
        return []


@dataclass
class TTLNode:
    """TTL 节点，表示一个带 TTL 的缓存条目"""
    program_id: str
    turn_index: int
    ttl_sec: float
    created_at: float
    last_accessed: float
    tokens: int  # 占用的 token 数
    
    @property
    def expires_at(self) -> float:
        """过期时间点"""
        return self.created_at + self.ttl_sec
    
    def is_expired(self, current_time: float) -> bool:
        """检查是否已过期"""
        return current_time > self.expires_at
    
    def remaining_ttl(self, current_time: float) -> float:
        """剩余 TTL 时间"""
        return max(0, self.expires_at - current_time)


# ============================================================================
# Continuum TTL 模拟器
# ============================================================================

class ContinuumTTLSimulator:
    """
    DTTL (Dynamic TTL) 策略模拟器

    DTTL 是自适应的，基于历史数据动态计算 TTL：
    1. 工具执行历史 → 动态计算 TTL = avg_time * 1.5
    2. 程序空闲间隔历史 → 动态计算 TTL = avg_idle_gap * 2.0
    3. 内存压力感知的 TTL 调整
    4. TTL 过期后的节点管理

    重要：default_ttl 只是没有历史数据时的回退值，不是固定值！
    """
    
    def __init__(
        self,
        default_ttl: float = 3.0,
        min_ttl: float = 0.1,
        max_ttl: float = 10.0,
        history_threshold: int = 10,
        memory_pressure_penalty: float = 0.5,
        enable_adaptive_ttl: bool = True,
    ):
        """
        Args:
            default_ttl: 默认 TTL（秒）
                       **重要**: 这是没有历史数据时的回退值，不是固定值！
                       DTTL 会根据历史数据动态计算 TTL。
            min_ttl: 最小 TTL（秒）
            max_ttl: 最大 TTL（秒）
            history_threshold: 历史样本阈值（达到此数量后才使用历史数据）
            memory_pressure_penalty: 内存压力惩罚系数
            enable_adaptive_ttl: 是否启用自适应 TTL
        """
        self.default_ttl = default_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.history_threshold = history_threshold
        self.memory_pressure_penalty = memory_pressure_penalty
        self.enable_adaptive_ttl = enable_adaptive_ttl
        
        # 全局统计
        self.program_stats: Dict[str, ProgramStats] = {}
        self.global_idle_gaps: deque = deque(maxlen=1024)
        self.global_tool_durations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1024))
        
        # TTL 节点管理
        self.active_nodes: Dict[Tuple[str, int], TTLNode] = {}  # (program_id, turn_index) -> TTLNode
        self.expired_nodes: List[TTLNode] = []  # 已过期的节点
        
        # 当前内存压力
        self.current_memory_pressure: float = 0.5
        self.memory_history: deque = deque(maxlen=100)
        
        # 统计信息
        self.stats = {
            "ttl_selections": 0,
            "tool_based_selections": 0,
            "idle_gap_based_selections": 0,
            "global_based_selections": 0,
            "default_selections": 0,
            "expired_nodes_checked": 0,
            "expired_nodes_removed": 0,
        }
    
    def reset(self):
        """重置模拟器状态"""
        self.program_stats.clear()
        self.global_idle_gaps.clear()
        self.global_tool_durations.clear()
        self.active_nodes.clear()
        self.expired_nodes.clear()
        self.current_memory_pressure = 0.5
        self.memory_history.clear()
        self.stats = {
            "ttl_selections": 0,
            "tool_based_selections": 0,
            "idle_gap_based_selections": 0,
            "global_based_selections": 0,
            "default_selections": 0,
            "expired_nodes_checked": 0,
            "expired_nodes_removed": 0,
        }
    
    # ========================================================================
    # TTL 选择策略
    # ========================================================================
    
    def select_dynamic_ttl(
        self,
        program_id: str,
        tool_name: Optional[str] = None,
        queue_time: Optional[float] = None,
    ) -> Tuple[float, str]:
        """
        动态选择 TTL 值

        重要：这是动态计算，不是返回固定值！
        策略优先级（按准确性排序）：
        1. 工具执行历史（最准确）→ TTL = avg_tool_time * 1.5
        2. 程序空闲间隔历史 → TTL = avg_idle_gap * 2.0
        3. 全局空闲间隔历史 → TTL = avg_global * 2.0
        4. 队列时间调整 → TTL = default_ttl + queue_component
        5. 默认 TTL（没有历史数据时的回退）→ TTL = default_ttl

        Args:
            program_id: 程序 ID
            tool_name: 工具名称（如果有）
            queue_time: 当前排队时间（秒）

        Returns:
            (ttl_sec, strategy_name): 动态计算的 TTL 值和使用的策略名称
        """
        self.stats["ttl_selections"] += 1
        
        stats = self.program_stats.setdefault(program_id, ProgramStats())
        
        # 1. 基于工具执行历史
        if tool_name and self.enable_adaptive_ttl:
            tool_samples = stats.recent_tool_durations(tool_name)
            if len(tool_samples) >= self.history_threshold:
                avg_tool_time = statistics.mean(tool_samples)
                # TTL 应该大于平均工具执行时间
                ttl = min(self.max_ttl, max(self.min_ttl, avg_tool_time * 1.5))
                self._apply_memory_pressure(ttl)
                self.stats["tool_based_selections"] += 1
                return ttl, "tool_history"
            
            global_samples = list(self.global_tool_durations.get(tool_name, []))
            if len(global_samples) >= self.history_threshold:
                avg_tool_time = statistics.mean(global_samples)
                ttl = min(self.max_ttl, max(self.min_ttl, avg_tool_time * 1.5))
                self._apply_memory_pressure(ttl)
                self.stats["tool_based_selections"] += 1
                return ttl, "global_tool_history"
        
        # 2. 基于程序空闲间隔历史
        idle_gaps = stats.recent_idle_gaps()
        if len(idle_gaps) >= self.history_threshold:
            avg_idle_gap = statistics.mean(idle_gaps)
            ttl = min(self.max_ttl, max(self.min_ttl, avg_idle_gap * 2.0))
            self._apply_memory_pressure(ttl)
            self.stats["idle_gap_based_selections"] += 1
            return ttl, "idle_gap_history"
        
        # 3. 基于全局空闲间隔历史
        if len(self.global_idle_gaps) >= self.history_threshold:
            avg_global = statistics.mean(self.global_idle_gaps)
            ttl = min(self.max_ttl, max(self.min_ttl, avg_global * 2.0))
            self._apply_memory_pressure(ttl)
            self.stats["global_based_selections"] += 1
            return ttl, "global_idle_gap"
        
        # 4. 队列时间调整
        if queue_time is not None and queue_time > 0:
            queue_component = queue_time * 0.5  # 队列时间的一半
            queue_component = min(queue_component, self.default_ttl)
            ttl = min(self.max_ttl, max(self.min_ttl, self.default_ttl + queue_component))
            self._apply_memory_pressure(ttl)
            self.stats["default_selections"] += 1
            return ttl, "queue_adjusted"
        
        # 5. 默认 TTL
        ttl = self._apply_memory_pressure(self.default_ttl)
        self.stats["default_selections"] += 1
        return ttl, "default"
    
    def _apply_memory_pressure(self, ttl: float) -> float:
        """应用内存压力调整"""
        if self.current_memory_pressure > 0.5:
            # 内存压力大时，缩短 TTL
            penalty = 1.0 - self.memory_pressure_penalty * (self.current_memory_pressure - 0.5) * 2
            return max(self.min_ttl, ttl * penalty)
        return ttl
    
    # ========================================================================
    # 记录和更新
    # ========================================================================
    
    def record_tool_execution(
        self,
        program_id: str,
        tool_name: str,
        duration: float,
        idle_gap: Optional[float] = None,
    ):
        """
        记录工具执行和空闲间隔
        
        Args:
            program_id: 程序 ID
            tool_name: 工具名称
            duration: 工具执行时间（秒）
            idle_gap: 工具执行前的空闲间隔（秒）
        """
        stats = self.program_stats.setdefault(program_id, ProgramStats())
        
        # 记录工具执行时间
        stats.tool_durations[tool_name].append(duration)
        
        # 记录全局工具执行时间
        self.global_tool_durations[tool_name].append(duration)
        
        # 记录空闲间隔
        if idle_gap is not None and idle_gap > 0:
            stats.idle_gaps.append(idle_gap)
            self.global_idle_gaps.append(idle_gap)
    
    def record_idle_gap(self, program_id: str, gap: float):
        """
        记录空闲间隔
        
        Args:
            program_id: 程序 ID
            gap: 空闲间隔（秒）
        """
        if gap <= 0:
            return
        
        stats = self.program_stats.setdefault(program_id, ProgramStats())
        stats.idle_gaps.append(gap)
        self.global_idle_gaps.append(gap)
    
    def update_memory_pressure(self, allocated: float, total: float):
        """
        更新内存压力
        
        Args:
            allocated: 已分配的内存
            total: 总内存
        """
        if total > 0:
            self.current_memory_pressure = min(1.0, max(0.0, allocated / total))
            self.memory_history.append(self.current_memory_pressure)
    
    def add_node(
        self,
        program_id: str,
        turn_index: int,
        ttl_sec: float,
        tokens: int,
        current_time: Optional[float] = None,
    ):
        """
        添加 TTL 节点
        
        Args:
            program_id: 程序 ID
            turn_index: 轮次索引
            ttl_sec: TTL 值（秒）
            tokens: 占用的 token 数
            current_time: 当前时间（秒）
        """
        if current_time is None:
            current_time = time.time()
        
        node = TTLNode(
            program_id=program_id,
            turn_index=turn_index,
            ttl_sec=ttl_sec,
            created_at=current_time,
            last_accessed=current_time,
            tokens=tokens,
        )
        key = (program_id, turn_index)
        self.active_nodes[key] = node
    
    def access_node(
        self,
        program_id: str,
        turn_index: int,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        访问 TTL 节点（刷新 TTL）
        
        Args:
            program_id: 程序 ID
            turn_index: 轮次索引
            current_time: 当前时间（秒）
        
        Returns:
            是否成功访问（节点存在且未过期）
        """
        if current_time is None:
            current_time = time.time()
        
        key = (program_id, turn_index)
        node = self.active_nodes.get(key)
        
        if node is None:
            return False
        
        if node.is_expired(current_time):
            return False
        
        # 刷新 TTL
        node.last_accessed = current_time
        return True
    
    def check_expired_nodes(
        self,
        current_time: Optional[float] = None,
    ) -> List[Tuple[str, int, int]]:
        """
        检查并返回所有过期的节点
        
        Args:
            current_time: 当前时间（秒）
        
        Returns:
            List of (program_id, turn_index, tokens): 已过期的节点
        """
        if current_time is None:
            current_time = time.time()
        
        expired = []
        expired_keys = []
        
        for key, node in self.active_nodes.items():
            self.stats["expired_nodes_checked"] += 1
            if node.is_expired(current_time):
                expired.append((node.program_id, node.turn_index, node.tokens))
                expired_keys.append(key)
                self.expired_nodes.append(node)
                self.stats["expired_nodes_removed"] += 1
        
        # 移除过期节点
        for key in expired_keys:
            del self.active_nodes[key]
        
        return expired
    
    # ========================================================================
    # 查询接口
    # ========================================================================
    
    def get_node(self, program_id: str, turn_index: int) -> Optional[TTLNode]:
        """获取 TTL 节点"""
        return self.active_nodes.get((program_id, turn_index))
    
    def is_node_active(self, program_id: str, turn_index: int) -> bool:
        """检查节点是否活跃（存在且未过期）"""
        node = self.get_node(program_id, turn_index)
        if node is None:
            return False
        return not node.is_expired(time.time())
    
    def get_active_nodes_count(self) -> int:
        """获取活跃节点数量"""
        return len(self.active_nodes)
    
    def get_active_tokens(self) -> int:
        """获取活跃节点占用的总 token 数"""
        return sum(node.tokens for node in self.active_nodes.values())
    
    def get_program_stats(self, program_id: str) -> Optional[ProgramStats]:
        """获取程序的统计信息"""
        return self.program_stats.get(program_id)
    
    # ========================================================================
    # 统计和调试
    # ========================================================================
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            "active_nodes": len(self.active_nodes),
            "expired_nodes": len(self.expired_nodes),
            "active_tokens": self.get_active_tokens(),
            "memory_pressure": self.current_memory_pressure,
            "programs_tracked": len(self.program_stats),
        }
    
    def get_strategy_distribution(self) -> Dict[str, int]:
        """获取 TTL 选择策略分布"""
        return {
            "tool_based": self.stats["tool_based_selections"],
            "idle_gap_based": self.stats["idle_gap_based_selections"],
            "global_based": self.stats["global_based_selections"],
            "default": self.stats["default_selections"],
        }
    
    def print_summary(self):
        """打印模拟器摘要"""
        stats = self.get_stats()
        strategy_dist = self.get_strategy_distribution()
        total_selections = sum(strategy_dist.values())
        
        print("\n" + "=" * 60)
        print("ContinuumTTLSimulator Summary")
        print("=" * 60)
        print(f"Active nodes: {stats['active_nodes']}")
        print(f"Expired nodes: {stats['expired_nodes']}")
        print(f"Active tokens: {stats['active_tokens']:,}")
        print(f"Memory pressure: {stats['memory_pressure']:.2%}")
        print(f"Programs tracked: {stats['programs_tracked']}")
        print(f"\nTTL Selections: {stats['ttl_selections']}")
        if total_selections > 0:
            print(f"  - Tool-based: {strategy_dist['tool_based']} ({strategy_dist['tool_based']/total_selections:.1%})")
            print(f"  - Idle gap-based: {strategy_dist['idle_gap_based']} ({strategy_dist['idle_gap_based']/total_selections:.1%})")
            print(f"  - Global-based: {strategy_dist['global_based']} ({strategy_dist['global_based']/total_selections:.1%})")
            print(f"  - Default: {strategy_dist['default']} ({strategy_dist['default']/total_selections:.1%})")
        print("=" * 60)


# ============================================================================
# 便捷函数
# ============================================================================

def create_ttl_simulator(config: dict) -> ContinuumTTLSimulator:
    """从配置字典创建 TTL 模拟器"""
    return ContinuumTTLSimulator(
        default_ttl=config.get("ttl_default_sec", 3.0),
        min_ttl=config.get("ttl_min_sec", 0.1),
        max_ttl=config.get("ttl_max_sec", 10.0),
        history_threshold=config.get("ttl_history_threshold", 10),
        memory_pressure_penalty=config.get("memory_pressure_penalty", 0.5),
        enable_adaptive_ttl=config.get("enable_adaptive_ttl", True),
    )
