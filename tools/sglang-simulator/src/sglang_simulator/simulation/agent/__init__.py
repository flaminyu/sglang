"""
Agent Simulation Module for SGLang Simulator

支持多轮 Agent 场景的 KV Cache 策略模拟。

主要组件：
- dataset: 多轮 Agent 数据集
- continuum_ttl: Continuum TTL 策略模拟
- state_manager: 多级缓存状态追踪
- scheduler_simulator: 请求调度模拟

Usage:
    from sglang_simulator.simulation.agent import (
        MultiTurnAgentDataset,
        ContinuumTTLSimulator,
        AgentStateManager,
        SchedulerSimulator,
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
    # Scheduler
    "BaselineScheduler",
    "ContinuumScheduler",
    "CacheHitResult",
    "QueuedRequest",
    "PinnedRequest",
    "ScheduleResult",
    "SchedulerSimulator",
    # Experiment
    "run_experiment",
]
