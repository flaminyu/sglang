#!/usr/bin/env python3
"""
实验运行脚本

支持运行 Baseline 和 DTTL (Dynamic TTL) 策略的对比实验。

NOTE: DTTL 是动态 TTL 机制，根据历史工具执行数据自动计算最优 TTL。
使用 --ttl-sec 参数仅用于设置默认值（当没有历史数据时使用）。

Usage:
    # 运行完整实验
    python run_agent_experiment.py --config configs/agent_sim_config.json

    # 运行快速测试
    python run_agent_experiment.py --num-programs 10 --turns 5 --seed 42

    # 运行 DTTL 实验
    python run_agent_experiment.py --ttl-default 3.0 --enable-dttl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang_simulator.simulation.agent import (
    AgentDatasetConfig,
    AgentProgram,
    ContinuumTTLSimulator,
    SchedulerSimulator,
    MultiTurnAgentDataset,
)


# ============================================================================
# 实验配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 程序参数
    num_programs: int = 20
    turns_per_program: int = 7
    arrival_rate: float = 0.3
    target_tokens: int = 20000
    input_growth: int = 1000
    output_tokens: int = 16
    tool_ratio: float = 0.7
    tool_time_mean: float = 0.5
    tool_time_std: float = 0.2
    seed: int = 42
    
    # 缓存参数
    max_tokens: int = 50000  # KV pool 大小
    
    # 带宽参数
    l1_bandwidth_gb: float = 50.0  # Host <-> GPU 带宽
    l2_bandwidth_gb: float = 5.0   # Host <-> Disk 带宽
    
    # TTL 参数（仅 Continuum）
    ttl_default: float = 3.0
    ttl_min: float = 0.1
    ttl_max: float = 10.0
    ttl_history_threshold: int = 10
    memory_pressure_penalty: float = 0.5
    
    # 输出
    output_dir: str = "simulator_experiment/results"
    save_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_programs": self.num_programs,
            "turns_per_program": self.turns_per_program,
            "arrival_rate": self.arrival_rate,
            "target_tokens": self.target_tokens,
            "input_growth": self.input_growth,
            "output_tokens": self.output_tokens,
            "tool_ratio": self.tool_ratio,
            "tool_time_mean": self.tool_time_mean,
            "tool_time_std": self.tool_time_std,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "l1_bandwidth_gb": self.l1_bandwidth_gb,
            "l2_bandwidth_gb": self.l2_bandwidth_gb,
            "ttl_default": self.ttl_default,
            "ttl_min": self.ttl_min,
            "ttl_max": self.ttl_max,
        }


@dataclass
class ExperimentResult:
    """实验结果"""
    mode: str
    config: ExperimentConfig
    programs: List[AgentProgram]
    
    # 时间统计
    total_time: float = 0.0
    total_queue_time: float = 0.0
    total_l1_time: float = 0.0
    total_l2_time: float = 0.0
    total_inference_time: float = 0.0
    
    # 缓存统计
    l0_hit_tokens: int = 0
    l1_hit_tokens: int = 0
    l2_hit_tokens: int = 0
    miss_tokens: int = 0
    evicted_tokens: int = 0
    
    # 请求统计
    total_requests: int = 0
    pinned_requests: int = 0
    cache_hit_rate: float = 0.0
    
    # 调度统计
    scheduler_stats: Dict[str, Any] = field(default_factory=dict)
    ttl_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 每个程序的统计
    program_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        total_cache = self.l0_hit_tokens + self.l1_hit_tokens + self.l2_hit_tokens
        total_tokens = total_cache + self.miss_tokens
        
        return {
            "mode": self.mode,
            "config": self.config.to_dict(),
            "timing": {
                "total_time": self.total_time,
                "total_queue_time": self.total_queue_time,
                "total_l1_time": self.total_l1_time,
                "total_l2_time": self.total_l2_time,
                "total_inference_time": self.total_inference_time,
            },
            "cache": {
                "l0_hit_tokens": self.l0_hit_tokens,
                "l1_hit_tokens": self.l1_hit_tokens,
                "l2_hit_tokens": self.l2_hit_tokens,
                "miss_tokens": self.miss_tokens,
                "evicted_tokens": self.evicted_tokens,
                "cache_hit_rate": self.cache_hit_rate,
            },
            "requests": {
                "total_requests": self.total_requests,
                "pinned_requests": self.pinned_requests,
            },
            "scheduler_stats": self.scheduler_stats,
            "ttl_stats": self.ttl_stats,
            "program_stats": self.program_stats,
        }


# ============================================================================
# 实验运行
# ============================================================================

def run_single_experiment(
    mode: str,
    programs: List[AgentProgram],
    config: ExperimentConfig,
) -> ExperimentResult:
    """
    运行单个实验
    
    Args:
        mode: "baseline" 或 "continuum"
        programs: 程序列表
        config: 实验配置
    
    Returns:
        ExperimentResult
    """
    print(f"\n{'='*70}")
    print(f"Running {mode.upper()} experiment")
    print(f"{'='*70}")
    print(f"Programs: {len(programs)}")
    print(f"Total requests: {sum(len(p.turns) for p in programs)}")
    print(f"Target tokens: {config.target_tokens}")
    print(f"Tool ratio: {config.tool_ratio:.1%}")
    
    # 创建 TTL 模拟器
    ttl_sim = ContinuumTTLSimulator(
        default_ttl=config.ttl_default,
        min_ttl=config.ttl_min,
        max_ttl=config.ttl_max,
        history_threshold=config.ttl_history_threshold,
        memory_pressure_penalty=config.memory_pressure_penalty,
    )
    
    # 创建调度器
    scheduler = SchedulerSimulator(
        mode=mode,
        max_tokens=config.max_tokens,
        max_host_tokens=config.max_tokens * 10,
        ttl_simulator=ttl_sim,
        h2d_bandwidth_gb=config.l1_bandwidth_gb,
        inference_speed=0.0001,
    )
    
    # 添加所有请求
    current_time = 0.0
    program_request_map = defaultdict(list)  # program_id -> [rids]
    
    for program in programs:
        for turn in program.turns:
            rid = scheduler.add_request(
                program_id=program.program_id,
                turn_index=turn.turn_index,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                arrival_time=program.arrival_time,
                is_tool_call=turn.is_tool_call,
                tool_name=turn.tool_name,
                tool_duration=turn.tool_duration,
            )
            program_request_map[program.program_id].append(rid)
    
    # 运行模拟
    start_time = time.time()
    scheduler.run_until_complete()
    end_time = time.time()
    
    # 获取结果
    summary = scheduler.get_summary()
    
    # 构建结果
    result = ExperimentResult(
        mode=mode,
        config=config,
        programs=programs,
        total_time=summary["total_time"],
        total_queue_time=summary["total_queue_time"],
        total_l1_time=summary["total_l1_time"],
        total_l2_time=summary["total_l2_time"],
        total_inference_time=summary["total_inference_time"],
        l0_hit_tokens=summary["cache_stats"]["l0_hit_tokens"],
        l1_hit_tokens=summary["cache_stats"]["l1_hit_tokens"],
        l2_hit_tokens=summary["cache_stats"]["l2_hit_tokens"],
        miss_tokens=summary["cache_stats"]["miss_tokens"],
        evicted_tokens=summary["total_evicted_tokens"],
        total_requests=summary["total_requests"],
        pinned_requests=summary["pinned_requests"],
        cache_hit_rate=summary["cache_stats"]["hit_rate"],
        scheduler_stats=summary["scheduler_stats"],
    )
    
    if mode == "continuum":
        result.ttl_stats = ttl_sim.get_stats()
    
    # 计算每个程序的统计
    for program in programs:
        req_stats = []
        for rid in program_request_map[program.program_id]:
            req = scheduler.state_manager.get_request_stats(rid)
            if req:
                req_stats.append(req.to_dict())
        
        # 汇总程序统计
        if req_stats:
            prog_total_time = sum(s["total_time"] for s in req_stats)
            prog_cache_hit = sum(s["l0_hit_tokens"] + s["l1_hit_tokens"] + s["l2_hit_tokens"] for s in req_stats)
            prog_total = prog_cache_hit + sum(s["miss_tokens"] for s in req_stats)
            prog_hit_rate = prog_cache_hit / max(1, prog_total)
            prog_pinned = sum(1 for s in req_stats if s["is_tool_call"])
            
            result.program_stats.append({
                "program_id": program.program_id,
                "turns": len(program.turns),
                "tool_calls": program.num_tool_calls,
                "total_tokens": program.total_tokens,
                "wall_time": prog_total_time,
                "cache_hit_rate": prog_hit_rate,
                "pinned_turns": prog_pinned,
                "request_stats": req_stats,
            })
    
    # 打印摘要
    print_result_summary(result)
    
    return result


def print_result_summary(result: ExperimentResult):
    """打印结果摘要"""
    print(f"\n{'='*70}")
    print(f"{result.mode.upper()} Results")
    print(f"{'='*70}")
    print(f"Total Time: {result.total_time:.4f}s")
    print(f"  - Queue Time: {result.total_queue_time:.4f}s")
    print(f"  - L1 Load Time: {result.total_l1_time:.4f}s")
    print(f"  - L2 Load Time: {result.total_l2_time:.4f}s")
    print(f"  - Inference Time: {result.total_inference_time:.4f}s")
    print(f"\nCache Hit Stats:")
    print(f"  L0 (GPU) Hit: {result.l0_hit_tokens:,} tokens")
    print(f"  L1 (Host) Hit: {result.l1_hit_tokens:,} tokens")
    print(f"  L2 (Disk) Hit: {result.l2_hit_tokens:,} tokens")
    print(f"  Miss: {result.miss_tokens:,} tokens")
    print(f"  Overall Hit Rate: {result.cache_hit_rate:.1%}")
    print(f"  Evicted Tokens: {result.evicted_tokens:,}")
    print(f"\nRequests:")
    print(f"  Total: {result.total_requests}")
    print(f"  Pinned: {result.pinned_requests}")
    
    if result.ttl_stats:
        print(f"\nTTL Stats:")
        ttl_strat = result.ttl_stats.get("ttl_selections", 0)
        if ttl_strat > 0:
            print(f"  Tool-based: {result.ttl_stats.get('tool_based_selections', 0)}")
            print(f"  Idle-gap-based: {result.ttl_stats.get('idle_gap_based_selections', 0)}")
            print(f"  Global-based: {result.ttl_stats.get('global_based_selections', 0)}")
            print(f"  Default: {result.ttl_stats.get('default_selections', 0)}")


def compare_results(baseline: ExperimentResult, continuum: ExperimentResult) -> Dict[str, Any]:
    """
    对比 Baseline 和 Continuum 的结果
    
    Returns:
        对比分析的字典
    """
    # 计算改进
    time_improvement = (baseline.total_time - continuum.total_time) / baseline.total_time * 100
    queue_improvement = (baseline.total_queue_time - continuum.total_queue_time) / baseline.total_queue_time * 100
    
    l0_baseline = baseline.l0_hit_tokens + baseline.l1_hit_tokens + baseline.l2_hit_tokens
    l0_continuum = continuum.l0_hit_tokens + continuum.l1_hit_tokens + continuum.l2_hit_tokens
    hit_improvement = (l0_continuum - l0_baseline) / max(1, l0_baseline) * 100
    
    evicted_improvement = (baseline.evicted_tokens - continuum.evicted_tokens) / max(1, baseline.evicted_tokens) * 100
    
    return {
        "time_improvement_pct": time_improvement,
        "queue_time_improvement_pct": queue_improvement,
        "cache_hit_improvement_pct": hit_improvement,
        "evicted_improvement_pct": evicted_improvement,
        "baseline_total_time": baseline.total_time,
        "continuum_total_time": continuum.total_time,
        "baseline_cache_hit_rate": baseline.cache_hit_rate,
        "continuum_cache_hit_rate": continuum.cache_hit_rate,
        "baseline_evicted": baseline.evicted_tokens,
        "continuum_evicted": continuum.evicted_tokens,
    }


def run_experiment(config: ExperimentConfig, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    运行完整实验（Baseline + Continuum）
    
    Args:
        config: 实验配置
        output_path: 输出路径
    
    Returns:
        实验结果字典
    """
    # 设置随机种子
    random.seed(config.seed)
    
    # 生成程序
    print("\n" + "="*70)
    print("Generating Programs")
    print("="*70)
    
    programs = MultiTurnAgentDataset.generate_programs(
        num_programs=config.num_programs,
        turns_per_program=config.turns_per_program,
        arrival_rate=config.arrival_rate,
        target_tokens=config.target_tokens,
        input_growth=config.input_growth,
        output_tokens=config.output_tokens,
        tool_ratio=config.tool_ratio,
        tool_time_mean=config.tool_time_mean,
        tool_time_std=config.tool_time_std,
        seed=config.seed,
    )
    
    print(f"Generated {len(programs)} programs")
    print(f"Total requests: {sum(len(p.turns) for p in programs)}")
    print(f"Total tokens: {sum(p.total_tokens for p in programs):,}")
    print(f"Tool call ratio: {sum(p.num_tool_calls for p in programs) / max(1, sum(len(p.turns) for p in programs)):.1%}")
    
    # 运行 Baseline 实验
    baseline_result = run_single_experiment("baseline", programs, config)
    
    # 运行 Continuum 实验
    continuum_result = run_single_experiment("continuum", programs, config)
    
    # 对比结果
    comparison = compare_results(baseline_result, continuum_result)
    
    # 打印对比摘要
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"Time Improvement: {comparison['time_improvement_pct']:+.2f}%")
    print(f"Queue Time Improvement: {comparison['queue_time_improvement_pct']:+.2f}%")
    print(f"Cache Hit Improvement: {comparison['cache_hit_improvement_pct']:+.2f}%")
    print(f"Evicted Reduction: {comparison['evicted_improvement_pct']:+.2f}%")
    print(f"\nCache Hit Rate:")
    print(f"  Baseline: {comparison['baseline_cache_hit_rate']:.1%}")
    print(f"  Continuum: {comparison['continuum_cache_hit_rate']:.1%}")
    print(f"\nEvicted Tokens:")
    print(f"  Baseline: {comparison['baseline_evicted']:,}")
    print(f"  Continuum: {comparison['continuum_evicted']:,}")
    print("="*70)
    
    # 保存结果
    if config.save_results and output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        baseline_path = output_path / f"baseline_result.json"
        continuum_path = output_path / f"continuum_result.json"
        comparison_path = output_path / f"comparison.json"
        
        baseline_path.write_text(json.dumps(baseline_result.to_dict(), indent=2))
        continuum_path.write_text(json.dumps(continuum_result.to_dict(), indent=2))
        comparison_path.write_text(json.dumps(comparison, indent=2))
        
        print(f"\nResults saved to: {output_path}")
    
    return {
        "baseline": baseline_result.to_dict(),
        "continuum": continuum_result.to_dict(),
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Agent Simulation Experiment")
    
    # 实验参数
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON)")
    parser.add_argument("--output-dir", type=str, default="simulator_experiment/results",
                       help="输出目录")
    
    # 程序参数
    parser.add_argument("--num-programs", type=int, default=20, help="程序数量")
    parser.add_argument("--turns", type=int, default=7, help="每程序轮数")
    parser.add_argument("--arrival-rate", type=float, default=0.3, help="Poisson 到达率")
    parser.add_argument("--target-tokens", type=int, default=20000, help="目标 tokens")
    parser.add_argument("--input-growth", type=int, default=1000, help="每轮输入增长")
    parser.add_argument("--output-tokens", type=int, default=16, help="每轮输出 tokens")
    parser.add_argument("--tool-ratio", type=float, default=0.7, help="工具调用比例")
    parser.add_argument("--tool-time-mean", type=float, default=0.5, help="工具时间均值")
    parser.add_argument("--tool-time-std", type=float, default=0.2, help="工具时间标准差")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 缓存参数
    parser.add_argument("--max-tokens", type=int, default=50000, help="KV pool 大小")
    
    # TTL 参数
    parser.add_argument("--ttl-default", type=float, default=3.0, help="默认 TTL")
    parser.add_argument("--ttl-min", type=float, default=0.1, help="最小 TTL")
    parser.add_argument("--ttl-max", type=float, default=10.0, help="最大 TTL")
    
    args = parser.parse_args()
    
    # 加载配置文件或使用命令行参数
    if args.config:
        config_dict = json.loads(Path(args.config).read_text())
        config = ExperimentConfig(**config_dict)
    else:
        config = ExperimentConfig(
            num_programs=args.num_programs,
            turns_per_program=args.turns,
            arrival_rate=args.arrival_rate,
            target_tokens=args.target_tokens,
            input_growth=args.input_growth,
            output_tokens=args.output_tokens,
            tool_ratio=args.tool_ratio,
            tool_time_mean=args.tool_time_mean,
            tool_time_std=args.tool_time_std,
            seed=args.seed,
            max_tokens=args.max_tokens,
            ttl_default=args.ttl_default,
            ttl_min=args.ttl_min,
            ttl_max=args.ttl_max,
            output_dir=args.output_dir,
        )
    
    # 运行实验
    output_path = Path(config.output_dir)
    results = run_experiment(config, output_path)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Output directory: {output_path}")
    print(f"Baseline result: {output_path}/baseline_result.json")
    print(f"Continuum result: {output_path}/continuum_result.json")
    print(f"Comparison: {output_path}/comparison.json")


if __name__ == "__main__":
    main()
