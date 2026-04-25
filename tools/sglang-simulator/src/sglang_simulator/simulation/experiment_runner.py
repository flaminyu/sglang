#!/usr/bin/env python3
"""
Full SGLang Simulator TTL Experiment Runner

Uses the full sglang simulator with hardware-aware timing to run
Baseline vs Continuum TTL experiments with time breakdown analysis.

Usage:
    python experiment_runner.py --hardware h100_sxm --mode ttl

    # Run multiple scenarios
    python experiment_runner.py --hardware h100_sxm --mode comparison

    # Run with custom TTL settings
    python experiment_runner.py --hardware a100_sxm --ttl-default 5.0 --ttl-min 1.0 --ttl-max 15.0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add sglang-simulator to path
SGLANG_SIM_PATH = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SGLANG_SIM_PATH))

from sglang_simulator.spec import AcceleratorInfo
from sglang_simulator.simulation.agent.dataset import (
    MultiTurnAgentDataset,
    AgentDatasetConfig,
)


# ============================================================================
# Hardware Configurations
# ============================================================================

HARDWARE_CONFIGS = {
    "h100_sxm": {
        "name": "h100_sxm",
        "hbm_bandwidth_gb": 3.35,
        "hbm_capacity_gb": 80,
        "tflops": 3958,
        "display_name": "NVIDIA H100 SXM",
    },
    "a100_sxm": {
        "name": "a100_sxm",
        "hbm_bandwidth_gb": 2.0,
        "hbm_capacity_gb": 80,
        "tflops": 1560,
        "display_name": "NVIDIA A100 SXM",
    },
    "h20": {
        "name": "h20",
        "hbm_bandwidth_gb": 0.9,
        "hbm_capacity_gb": 80,
        "tflops": 148,
        "display_name": "NVIDIA H20 (China)",
    },
    "a800": {
        "name": "a800",
        "hbm_bandwidth_gb": 2.0,
        "hbm_capacity_gb": 80,
        "tflops": 624,
        "display_name": "NVIDIA A800",
    },
}


# ============================================================================
# Experiment Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Hardware
    hardware: str = "h100_sxm"

    # Model
    model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Workload
    num_programs: int = 20
    turns_per_program: int = 7
    arrival_rate: float = 0.3
    target_tokens: int = 20000
    input_growth: int = 1000
    output_tokens: int = 16
    tool_ratio: float = 0.7
    tool_time_mean: float = 0.5
    tool_time_std: float = 0.2

    # Cache
    cache_ratio: float = 0.15  # Cache size relative to total tokens

    # TTL Settings
    ttl_default: float = 3.0
    ttl_min: float = 0.1
    ttl_max: float = 10.0
    ttl_history: int = 10
    ttl_penalty: float = 0.5
    ttl_enabled: bool = True

    # Output
    output_dir: str = "results"
    seed: int = 42

    def to_dict(self) -> Dict:
        return {
            "hardware": self.hardware,
            "model": self.model,
            "workload": {
                "num_programs": self.num_programs,
                "turns_per_program": self.turns_per_program,
                "arrival_rate": self.arrival_rate,
                "target_tokens": self.target_tokens,
                "tool_ratio": self.tool_ratio,
                "tool_time_mean": self.tool_time_mean,
            },
            "cache_ratio": self.cache_ratio,
            "ttl": {
                "enabled": self.ttl_enabled,
                "default": self.ttl_default,
                "min": self.ttl_min,
                "max": self.ttl_max,
            },
        }


@dataclass
class ExperimentResult:
    """Result from a single experiment run"""
    mode: str
    config: ExperimentConfig

    # Time breakdown (in seconds)
    total_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    tool_time: float = 0.0
    queue_time: float = 0.0
    h2d_time: float = 0.0
    l2_backup_time: float = 0.0

    # Cache stats
    l0_hits: int = 0
    l1_hits: int = 0
    misses: int = 0
    cache_hit_rate: float = 0.0
    evicted_tokens: int = 0

    # TTL stats
    pinned_count: int = 0
    ttl_strategy_dist: Dict[str, int] = field(default_factory=dict)

    # Metadata
    total_requests: int = 0
    hardware_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        total = max(0.001, self.total_time)
        return {
            "mode": self.mode,
            "hardware": self.config.hardware,
            "time": {
                "total": self.total_time,
                "prefill": self.prefill_time,
                "decode": self.decode_time,
                "tool": self.tool_time,
                "queue": self.queue_time,
                "h2d": self.h2d_time,
                "l2_backup": self.l2_backup_time,
                "prefill_pct": self.prefill_time / total * 100,
                "decode_pct": self.decode_time / total * 100,
                "tool_pct": self.tool_time / total * 100,
                "queue_pct": self.queue_time / total * 100,
            },
            "cache": {
                "l0_hits": self.l0_hits,
                "l1_hits": self.l1_hits,
                "misses": self.misses,
                "hit_rate": self.cache_hit_rate,
                "evicted_tokens": self.evicted_tokens,
            },
            "ttl": {
                "pinned_count": self.pinned_count,
                "strategy_dist": self.ttl_strategy_dist,
            },
            "requests": self.total_requests,
            "hardware_info": self.hardware_info,
        }


# ============================================================================
# Simulation Config Generator
# ============================================================================

def generate_sim_config(config: ExperimentConfig) -> Dict:
    """Generate simulation config JSON for the full simulator"""

    hw = HARDWARE_CONFIGS.get(config.hardware, HARDWARE_CONFIGS["h100_sxm"])

    sim_config = {
        "platform": {
            "accelerator": {
                "name": hw["name"],
                "vendor": "NVIDIA",
                "hbm_bandwidth_gb": hw["hbm_bandwidth_gb"],
                "hbm_capacity_gb": hw["hbm_capacity_gb"],
                "tflops": hw["tflops"],
            },
            "disk_read_bandwidth_gb": 8.0,
            "disk_write_bandwidth_gb": 8.0,
            "memory_read_bandwidth_gb": 64.0,
            "memory_write_bandwidth_gb": 64.0,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "aiconfigurator",  # Use AIConfigurator for realistic timing if available
            "database_mode": "SILICON",
            "prefill_scale_factor": 1.0,
            "decode_scale_factor": 1.0,
        },
        "scheduler": {
            "tp_size": 1,
            "ep_size": 1,
            "dp_size": 1,
            "backend_version": "0.5.9",
        },
        "continuum": {
            "enabled": config.ttl_enabled,
            "default_ttl": config.ttl_default,
            "min_ttl": config.ttl_min,
            "max_ttl": config.ttl_max,
            "history_threshold": config.ttl_history,
            "memory_pressure_penalty": config.ttl_penalty,
        },
    }

    return sim_config


# ============================================================================
# Workload Generator
# ============================================================================

def generate_workload(config: ExperimentConfig) -> List[Dict]:
    """Generate multi-turn agent workload"""

    dataset_config = AgentDatasetConfig(
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

    programs = MultiTurnAgentDataset.generate_programs(dataset_config)

    # Convert to simulation request format
    requests = []
    for program in programs:
        for turn in program.turns:
            requests.append({
                "program_id": program.program_id,
                "turn_index": turn.turn_index,
                "arrival_time": program.arrival_time,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "is_tool_call": turn.is_tool_call,
                "tool_name": turn.tool_name,
                "tool_duration": turn.tool_duration,
            })

    return requests


# ============================================================================
# Standalone Simulator (Fallback)
# ============================================================================

def run_standalone_simulation(
    config: ExperimentConfig,
    mode: str,
) -> ExperimentResult:
    """
    Run simulation using the unified UnifiedSchedulerSimulator
    when the full simulator is not available.
    """
    from sglang_simulator.simulation.agent.unified_scheduler import (
        UnifiedSchedulerSimulator,
        ContinuumTTLSimulator,
        TimePredictor,
    )

    # Set TTL enabled based on mode
    ttl_enabled = mode == "continuum" or mode == "ttl"

    # Create TTL simulator
    ttl_sim = ContinuumTTLSimulator(
        default_ttl=config.ttl_default,
        min_ttl=config.ttl_min,
        max_ttl=config.ttl_max,
        history_threshold=config.ttl_history,
        enable_adaptive_ttl=ttl_enabled,
    )

    # Create time predictor with hardware-aware settings
    hw = HARDWARE_CONFIGS.get(config.hardware, HARDWARE_CONFIGS["h100_sxm"])

    # Calculate realistic per-token timing based on hardware
    # Prefill: based on TFLOPS (tokens/sec = TFLOPS / params * batch_size)
    # These are simplified estimates
    prefill_per_token_ms = 0.1 / (hw["tflops"] / 1000) * 10  # Normalized
    decode_per_token_ms = 5.0 / (hw["hbm_bandwidth_gb"] / 3.35)  # Based on bandwidth

    predictor = TimePredictor(
        prefill_per_token_ms=prefill_per_token_ms,
        decode_per_token_ms=decode_per_token_ms,
        cache_hit_speedup=10.0,
    )

    # Calculate max_tokens based on cache_ratio
    programs = generate_workload(config)
    total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in programs)
    max_tokens = int(total_tokens * config.cache_ratio)

    # Create simulator
    sim = UnifiedSchedulerSimulator(
        mode=mode,
        max_tokens=max_tokens,
        ttl_simulator=ttl_sim,
        time_predictor=predictor,
    )

    # Add requests
    for req in programs:
        sim.add_request(
            program_id=req["program_id"],
            turn_index=req["turn_index"],
            input_tokens=req["input_tokens"],
            output_tokens=req["output_tokens"],
            arrival_time=req["arrival_time"],
            is_tool_call=req["is_tool_call"],
            tool_name=req["tool_name"],
            tool_duration=req["tool_duration"],
        )

    # Run simulation
    results = sim.run_until_complete()
    summary = sim.get_summary()

    # Calculate time breakdown
    total_prefill = 0.0
    total_decode = 0.0
    total_tool = 0.0
    total_queue = 0.0

    for r in results:
        total_prefill += getattr(r, 'prefill_time', 0.0)
        total_decode += getattr(r, 'decode_time', 0.0)
        total_tool += r.tool_time if hasattr(r, 'tool_time') else 0.0
        total_queue += r.queue_time

    result = ExperimentResult(
        mode=mode,
        config=config,
        total_time=summary.get("wall_time", 0),
        prefill_time=total_prefill,
        decode_time=total_decode,
        tool_time=total_tool,
        queue_time=total_queue,
        l0_hits=summary.get("cache_stats", {}).get("l0_hits", 0),
        l1_hits=summary.get("cache_stats", {}).get("l1_hits", 0),
        misses=summary.get("cache_stats", {}).get("misses", 0),
        cache_hit_rate=summary.get("cache_stats", {}).get("hit_rate", 0),
        evicted_tokens=summary.get("evicted_tokens", 0),
        pinned_count=summary.get("pin_stats", {}).get("total_pins", 0),
        total_requests=summary.get("total_requests", 0),
        hardware_info=hw,
    )

    return result


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    mode: str,
) -> ExperimentResult:
    """Run a single experiment"""

    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} experiment")
    print(f"Hardware: {HARDWARE_CONFIGS[config.hardware]['display_name']}")
    print(f"Workload: {config.num_programs} programs, {config.turns_per_program} turns")
    print(f"Cache ratio: {config.cache_ratio:.0%}")
    print(f"TTL enabled: {config.ttl_enabled}")
    if config.ttl_enabled:
        print(f"TTL settings: default={config.ttl_default}s, range=[{config.ttl_min}, {config.ttl_max}]")
    print(f"{'='*60}")

    # Use standalone simulator (more reliable for experiments)
    result = run_standalone_simulation(config, mode)

    # Print time breakdown
    print(f"\nTime Breakdown:")
    print(f"  Prefill:   {result.prefill_time:>8.3f}s ({result.prefill_time/max(0.001,result.total_time)*100:5.1f}%)")
    print(f"  Decode:    {result.decode_time:>8.3f}s ({result.decode_time/max(0.001,result.total_time)*100:5.1f}%)")
    print(f"  Tool:      {result.tool_time:>8.3f}s ({result.tool_time/max(0.001,result.total_time)*100:5.1f}%)")
    print(f"  Queue:     {result.queue_time:>8.3f}s ({result.queue_time/max(0.001,result.total_time)*100:5.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:     {result.total_time:>8.3f}s (100.0%)")

    print(f"\nCache Performance:")
    print(f"  L0 hits:     {result.l0_hits:>6}")
    print(f"  L1 hits:     {result.l1_hits:>6}")
    print(f"  Misses:      {result.misses:>6}")
    print(f"  Hit rate:    {result.cache_hit_rate:>6.1%}")
    print(f"  Evicted:     {result.evicted_tokens:>6} tokens")

    if result.pinned_count > 0:
        print(f"\nTTL Performance:")
        print(f"  Pinned:      {result.pinned_count:>6}")

    return result


def run_comparison_experiment(
    config: ExperimentConfig,
) -> Dict:
    """Run baseline vs TTL comparison"""

    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON: Baseline vs Continuum TTL")
    print("="*70)

    # Baseline experiment
    baseline_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": False})
    baseline_result = run_single_experiment(baseline_config, "baseline")

    # TTL experiment
    ttl_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": True})
    ttl_result = run_single_experiment(ttl_config, "continuum")

    # Calculate comparison metrics
    speedup = baseline_result.total_time / max(0.001, ttl_result.total_time)
    hit_rate_improvement = (ttl_result.cache_hit_rate - baseline_result.cache_hit_rate) * 100
    evict_reduction = (baseline_result.evicted_tokens - ttl_result.evicted_tokens) / max(1, baseline_result.evicted_tokens) * 100

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nOverall:")
    print(f"  Speedup:           {speedup:.3f}x")
    print(f"  Hit rate improve:  +{hit_rate_improvement:.1f}%")
    print(f"  Eviction reduce:   -{evict_reduction:.1f}%")

    print(f"\nTime saved:")
    print(f"  Prefill:  {baseline_result.prefill_time - ttl_result.prefill_time:>+8.3f}s")
    print(f"  Decode:   {baseline_result.decode_time - ttl_result.decode_time:>+8.3f}s")
    print(f"  Tool:     {baseline_result.tool_time - ttl_result.tool_time:>+8.3f}s")
    print(f"  Queue:    {baseline_result.queue_time - ttl_result.queue_time:>+8.3f}s")

    return {
        "baseline": baseline_result.to_dict(),
        "continuum": ttl_result.to_dict(),
        "comparison": {
            "speedup": speedup,
            "hit_rate_improvement_pct": hit_rate_improvement,
            "eviction_reduction_pct": evict_reduction,
        },
    }


def run_stress_experiment(
    base_config: ExperimentConfig,
) -> List[Dict]:
    """Run stress tests with different pressures"""

    print("\n" + "="*70)
    print("STRESS TEST: Different Cache Pressures")
    print("="*70)

    results = []

    for cache_ratio in [0.3, 0.2, 0.15, 0.1, 0.08]:
        print(f"\n{'─'*60}")
        print(f"Cache ratio: {cache_ratio:.0%} (pressure: {1/cache_ratio:.1f}x)")

        config = ExperimentConfig(**{**base_config.__dict__, "cache_ratio": cache_ratio})

        # Baseline
        baseline_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": False})
        baseline = run_single_experiment(baseline_config, "baseline")

        # TTL
        ttl_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": True})
        ttl = run_single_experiment(ttl_config, "continuum")

        speedup = baseline.total_time / max(0.001, ttl.total_time)

        results.append({
            "cache_ratio": cache_ratio,
            "pressure": 1 / cache_ratio,
            "baseline": baseline.to_dict(),
            "continuum": ttl.to_dict(),
            "speedup": speedup,
        })

    return results


def run_hardware_comparison(
    base_config: ExperimentConfig,
) -> List[Dict]:
    """Run experiments on different hardware"""

    print("\n" + "="*70)
    print("HARDWARE COMPARISON: H100 vs A100 vs H20")
    print("="*70)

    results = []

    for hardware in ["h100_sxm", "a100_sxm", "h20"]:
        print(f"\n{'─'*60}")
        print(f"Hardware: {HARDWARE_CONFIGS[hardware]['display_name']}")

        config = ExperimentConfig(**{**base_config.__dict__, "hardware": hardware})

        # Baseline
        baseline_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": False})
        baseline = run_single_experiment(baseline_config, "baseline")

        # TTL
        ttl_config = ExperimentConfig(**{**config.__dict__, "ttl_enabled": True})
        ttl = run_single_experiment(ttl_config, "continuum")

        speedup = baseline.total_time / max(0.001, ttl.total_time)

        results.append({
            "hardware": hardware,
            "hardware_info": HARDWARE_CONFIGS[hardware],
            "baseline": baseline.to_dict(),
            "continuum": ttl.to_dict(),
            "speedup": speedup,
        })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full SGLang Simulator TTL Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline comparison
  python experiment_runner.py --mode comparison

  # Run on different hardware
  python experiment_runner.py --mode hardware

  # Run stress test
  python experiment_runner.py --mode stress

  # Custom settings
  python experiment_runner.py --mode comparison --hardware a100_sxm --ttl-default 5.0
        """,
    )

    # Experiment mode
    parser.add_argument(
        "--mode",
        type=str,
        default="comparison",
        choices=["comparison", "stress", "hardware", "baseline", "ttl"],
        help="Experiment mode",
    )

    # Hardware
    parser.add_argument(
        "--hardware",
        type=str,
        default="h100_sxm",
        choices=list(HARDWARE_CONFIGS.keys()),
        help="Hardware configuration",
    )

    # Workload
    parser.add_argument("--num-programs", type=int, default=20, help="Number of programs")
    parser.add_argument("--turns", type=int, default=7, help="Turns per program")
    parser.add_argument("--arrival-rate", type=float, default=0.3, help="Arrival rate")
    parser.add_argument("--tool-ratio", type=float, default=0.7, help="Tool call ratio")
    parser.add_argument("--tool-time", type=float, default=0.5, help="Mean tool execution time")

    # Cache
    parser.add_argument("--cache-ratio", type=float, default=0.15, help="Cache size ratio")

    # TTL
    parser.add_argument("--ttl-default", type=float, default=3.0, help="Default TTL (seconds)")
    parser.add_argument("--ttl-min", type=float, default=0.1, help="Minimum TTL (seconds)")
    parser.add_argument("--ttl-max", type=float, default=10.0, help="Maximum TTL (seconds)")

    # Output
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        hardware=args.hardware,
        num_programs=args.num_programs,
        turns_per_program=args.turns,
        arrival_rate=args.arrival_rate,
        tool_ratio=args.tool_ratio,
        tool_time_mean=args.tool_time,
        cache_ratio=args.cache_ratio,
        ttl_default=args.ttl_default,
        ttl_min=args.ttl_min,
        ttl_max=args.ttl_max,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Create output directory
    output_dir = Path(__file__).parent / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment based on mode
    timestamp = int(time.time())

    if args.mode == "comparison":
        result = run_comparison_experiment(config)
        output_file = output_dir / f"comparison_{timestamp}.json"
    elif args.mode == "stress":
        result = run_stress_experiment(config)
        output_file = output_dir / f"stress_{timestamp}.json"
    elif args.mode == "hardware":
        result = run_hardware_comparison(config)
        output_file = output_dir / f"hardware_{timestamp}.json"
    elif args.mode == "baseline":
        config.ttl_enabled = False
        result = run_single_experiment(config, "baseline").to_dict()
        output_file = output_dir / f"baseline_{timestamp}.json"
    elif args.mode == "ttl":
        config.ttl_enabled = True
        result = run_single_experiment(config, "continuum").to_dict()
        output_file = output_dir / f"ttl_{timestamp}.json"

    # Save results
    output_data = {
        "config": config.to_dict(),
        "timestamp": timestamp,
        "results": result,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
