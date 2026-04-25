"""
Agent Simulation Module for SGLang Simulator

支持多轮 Agent 场景的 KV Cache 策略模拟。

主要组件：
- dataset: 多轮 Agent 数据集
- unified_scheduler: 统一调度器（整合 Baseline + Continuum）
- continuum_ttl: Continuum TTL 策略模拟
- state_manager: 多级缓存状态追踪

Usage:
    from sglang_simulator.simulation.agent import (
        MultiTurnAgentDataset,
        ContinuumTTLSimulator,
        AgentStateManager,
        UnifiedSchedulerSimulator,
        create_scheduler_simulator,
        run_experiment,
    )
"""
from __future__ import annotations

from sglang_simulator.simulation.agent.dataset import (
    AgentDatasetConfig,
    AgentProgram,
    MultiTurnAgentDataset,
    TurnSpec,
)

from sglang_simulator.simulation.agent.continuum_ttl import (
    ContinuumTTLSimulator,
    ProgramStats,
    TTLNode,
    create_ttl_simulator,
)

from sglang_simulator.simulation.agent.state_manager import (
    AgentStateManager,
    CacheHitStats,
    LatencyStats,
    RequestStats,
)

from sglang_simulator.simulation.agent.scheduler_simulator import (
    BaselineScheduler,
    ContinuumScheduler,
    CacheHitResult,
    QueuedRequest,
    PinnedRequest,
    ScheduleResult,
    SchedulerSimulator,
)

# 导入统一调度器模块
try:
    from sglang_simulator.simulation.agent.unified_scheduler import (
        UnifiedSchedulerSimulator,
        TimePredictor,
        CacheHitResult as UnifiedCacheHitResult,
        QueuedRequest as UnifiedQueuedRequest,
        ScheduleResult as UnifiedScheduleResult,
        create_scheduler_simulator,
    )
    # ProgramStats 和 TTLNode 在新版 unified_scheduler 中内部使用
    ProgramStats = None
    TTLNode = None
except ImportError:
    UnifiedSchedulerSimulator = None
    TimePredictor = None
    UnifiedCacheHitResult = None
    UnifiedQueuedRequest = None
    UnifiedScheduleResult = None
    create_scheduler_simulator = None
    ProgramStats = None
    TTLNode = None

from sglang_simulator.simulation.agent.experiment import run_experiment

__all__ = [
    # Dataset
    "AgentDatasetConfig",
    "AgentProgram",
    "MultiTurnAgentDataset",
    "TurnSpec",
    # TTL
    "ContinuumTTLSimulator",
    "ProgramStats",
    "TTLNode",
    "create_ttl_simulator",
    # State
    "AgentStateManager",
    "CacheHitStats",
    "LatencyStats",
    "RequestStats",
    # Scheduler (Legacy)
    "BaselineScheduler",
    "ContinuumScheduler",
    "CacheHitResult",
    "QueuedRequest",
    "PinnedRequest",
    "ScheduleResult",
    "SchedulerSimulator",
    # Unified Scheduler (Recommended)
    "UnifiedSchedulerSimulator",
    "UnifiedCacheHitResult",
    "UnifiedQueuedRequest",
    "UnifiedScheduleResult",
    "TimePredictor",
    "create_scheduler_simulator",
    # Experiment
    "run_experiment",
]
