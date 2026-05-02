#!/usr/bin/env python3
"""
Pure Simulator Unified Experiment Script

统一实验脚本，支持多种测试场景和参数配置。

Usage:
    # 基准测试: Baseline vs Adaptive TTL
    python run_experiment.py --mode baseline

    # 异常场景测试
    python run_experiment.py --mode anomaly

    # 工具执行时间影响测试
    python run_experiment.py --mode tool_time

    # 分布突变测试
    python run_experiment.py --mode shift

    # 自定义参数
    python run_experiment.py --mode baseline --l1 3000 --l2 4500 --programs 10 --turns 30

    # 参数扫描
    python run_experiment.py --mode sweep --sweep l1 --l1-range 1000 5000 1000

Examples:
    python run_experiment.py --mode baseline --verbose
    python run_experiment.py --mode anomaly --exception timeout --rate 0.3
    python run_experiment.py --mode sweep --sweep programs --programs-range 5 20 5
"""

import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'programs': 10,
    'turns': 15,
    'l1': 3000,
    'l2_ratio': 1.5,  # L2 = L1 * ratio
    'prefill': 200,
    'first_turn_multiplier': 5,
    'tool_time': 2.0,
    'seed': 42,
}

HARDWARE_CONFIG = {
    'prefill_latency_per_token_ms': 2.0,
    'decode_latency_per_token_ms': 20.0,
    'ttft_overhead_ms': 100.0,
    'queue_delay_per_request_ms': 50.0,
    'bytes_per_token': 512.0,
    'h2d_bandwidth_GBps': 32.0,
    'bandwidth_efficiency': 0.85,
    'h2d_overhead_us': 6.67,
    'd2h_overhead_us': 4.0,
    'memory_bandwidth_GBps': 64.0,
}


# ============================================================================
# Request Generation
# ============================================================================

def generate_requests(
    programs: int,
    turns: int,
    prefill: int,
    tool_time: float,
    first_turn_multiplier: int = 5,
    arrival_time: float = 0.0,
    seed: int = 42
) -> List[Dict]:
    """Generate requests for simulation."""
    random.seed(seed)
    requests = []

    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []

        for turn in range(turns):
            if turn == 0:
                current_prefill = int(prefill * first_turn_multiplier)
            elif turn == 1:
                current_prefill = int(prefill * 3)
            elif turn == 2:
                current_prefill = int(prefill * 2)
            else:
                current_prefill = prefill

            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 1000000 + prev_total,
                prog_idx * 1000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': arrival_time,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': tool_time,
            })
            accumulated.extend(new_tokens)

    return requests


def generate_staggered_requests(
    programs: int,
    turns: int,
    prefill: int,
    tool_time: float,
    first_turn_multiplier: int = 5,
    seed: int = 42
) -> List[Dict]:
    """Generate requests with staggered arrival times."""
    random.seed(seed)
    requests = []
    current_time = 0.0

    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []

        for turn in range(turns):
            if turn == 0:
                current_time = prog_idx * 0.5
            else:
                current_time += tool_time + 0.1

            if turn == 0:
                current_prefill = int(prefill * first_turn_multiplier)
            elif turn == 1:
                current_prefill = int(prefill * 3)
            elif turn == 2:
                current_prefill = int(prefill * 2)
            else:
                current_prefill = prefill

            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 1000000 + prev_total,
                prog_idx * 1000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': current_time,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': tool_time,
            })
            accumulated.extend(new_tokens)

    return requests


def apply_exceptions(
    requests: List[Dict],
    exception_type: str,
    rate: float,
    base_duration: float = 2.0,
    seed: int = 123
) -> List[Dict]:
    """Apply tool exceptions to requests."""
    random.seed(seed)
    modified = []

    for req in requests:
        if not req['is_tool_call']:
            modified.append(req.copy())
            continue

        new_req = req.copy()
        if random.random() < rate:
            if exception_type == 'timeout':
                new_req['actual_tool_duration'] = base_duration * 5.0
            elif exception_type == 'retry':
                new_req['actual_tool_duration'] = base_duration * 2.0
            elif exception_type == 'hang':
                new_req['actual_tool_duration'] = base_duration * 10.0
            elif exception_type == 'early_return':
                new_req['actual_tool_duration'] = base_duration * 0.1
            elif exception_type == 'mixed':
                r = random.random()
                if r < 0.33:
                    new_req['actual_tool_duration'] = base_duration * 5.0  # timeout
                elif r < 0.66:
                    new_req['actual_tool_duration'] = base_duration * 2.0  # retry
                else:
                    new_req['actual_tool_duration'] = base_duration * 10.0  # hang
        else:
            new_req['actual_tool_duration'] = base_duration
        modified.append(new_req)

    return modified


# ============================================================================
# Simulator
# ============================================================================

def create_simulator(
    requests: List[Dict],
    enable_ttl: bool,
    l1: int,
    l2: int,
    tool_time: float = 2.0,
    write_policy: str = "write_through",
    history_threshold: int = 100,
) -> KVCacheSimulator:
    """Create and run simulator."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        sim = KVCacheSimulator(
            cache_capacity=l1,
            l2_capacity=l2,
            l2_enabled=l2 > 0,
            l2_reload_penalty=0.3,
            default_ttl=5.0,
            min_ttl=0.5,
            max_ttl=60.0,
            enable_ttl=enable_ttl,
            enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven',
            poisson_lambda=1.0 / 0.3,
            tool_execution_time=tool_time,
            random_seed=42,
            hardware_config=hw,
            timing_config=timing_calc,
            write_policy=write_policy,
            history_threshold=history_threshold,
        )

        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999

        sim.run(verbose=False)
        return sim
    finally:
        os.unlink(temp_path)


def analyze_result(sim: KVCacheSimulator) -> Dict[str, Any]:
    """Analyze simulation result."""
    log = sim.scheduler.scheduling_log

    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_time = log[-1]['finish_time'] if log else 0

    # Time breakdown
    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_decode_ms = sum(e.get('decode_ms', 0) for e in log)
    total_h2d_ms = sum(e.get('h2d_ms', 0) for e in log)
    total_queue_delay_ms = sum(e.get('queue_delay_ms', 0) for e in log)
    total_ttft_ms = sum(e.get('ttft_ms', 0) for e in log)

    l2_spills = 0
    if hasattr(sim, 'l2_cache') and sim.l2_cache:
        stats = sim.l2_cache.get_stats()
        l2_spills = stats.get('l1_to_l2_spills', 0)

    return {
        'time': total_time,
        'ttl_hits': ttl_hits,
        'l1_hits': l1_hits,
        'l2_hits': l2_hits,
        'misses': misses,
        'l2_spills': l2_spills,
        # Time breakdown (seconds)
        'prefill_s': total_prefill_ms / 1000,
        'decode_s': total_decode_ms / 1000,
        'h2d_s': total_h2d_ms / 1000,
        'queue_delay_s': total_queue_delay_ms / 1000,
        'ttft_s': total_ttft_ms / 1000,
    }


# ============================================================================
# Test Modes
# ============================================================================

def run_baseline_test(args) -> List[Dict]:
    """Run baseline comparison: Baseline + L2 vs Adaptive TTL + L2."""
    l2 = int(args.l1 * args.l2_ratio)
    requests = generate_requests(
        args.programs, args.turns, args.prefill, args.tool_time,
        args.first_turn_multiplier, seed=args.seed
    )
    history_threshold = getattr(args, 'history_threshold', 3)

    configs = [
        ("Baseline + L2", False, l2),
        ("Adaptive TTL + L2", True, l2),
    ]

    results = []
    for name, enable_ttl, l2_size in configs:
        print(f"Running: {name}...", end=" ", flush=True)
        sim = create_simulator(requests, enable_ttl, args.l1, l2_size, args.tool_time,
                               history_threshold=history_threshold)
        r = analyze_result(sim)
        print(f"Time={r['time']:.1f}s | TTL={r['ttl_hits']} L1={r['l1_hits']} L2={r['l2_hits']} Miss={r['misses']}")
        results.append({'name': name, 'enable_ttl': enable_ttl, 'l2_size': l2_size, **r})

    return results


def run_anomaly_test(args) -> List[Dict]:
    """Run anomaly scenario test: Timeout, Retry, Hang, Early Return."""
    l2 = int(args.l1 * args.l2_ratio)
    base_requests = generate_requests(
        args.programs, args.turns, args.prefill, args.tool_time,
        args.first_turn_multiplier, seed=args.seed
    )
    history_threshold = getattr(args, 'history_threshold', 3)

    scenarios = [
        ("正常", None, 0.0),
        ("超时(10%)", "timeout", 0.10),
        ("超时(30%)", "timeout", 0.30),
        ("超时(50%)", "timeout", 0.50),
        ("重试(10%)", "retry", 0.10),
        ("重试(30%)", "retry", 0.30),
        ("重试(50%)", "retry", 0.50),
        ("挂起(10%)", "hang", 0.10),
        ("挂起(30%)", "hang", 0.30),
        ("提前返回(10%)", "early_return", 0.10),
        ("提前返回(30%)", "early_return", 0.30),
        ("混合异常(各15%)", "mixed", 0.15),
    ]

    results = []
    for scenario_name, exception_type, rate in scenarios:
        # Apply exceptions
        if exception_type == 'mixed':
            reqs = apply_exceptions(base_requests, 'timeout', rate / 3, args.tool_time)
            reqs = apply_exceptions(reqs, 'retry', rate / 3, args.tool_time)
            reqs = apply_exceptions(reqs, 'hang', rate / 3, args.tool_time)
        elif exception_type:
            reqs = apply_exceptions(base_requests, exception_type, rate, args.tool_time)
        else:
            reqs = base_requests

        # Calculate average tool time
        tool_times = [r['actual_tool_duration'] for r in reqs if r['is_tool_call']]
        avg_tool_time = sum(tool_times) / len(tool_times) if tool_times else args.tool_time

        # Baseline
        print(f"[{scenario_name}] Baseline...", end=" ", flush=True)
        sim_bl = create_simulator(reqs, False, args.l1, l2, avg_tool_time,
                                  history_threshold=history_threshold)
        r_bl = analyze_result(sim_bl)
        print(f"{r_bl['time']:.1f}s")

        # Adaptive TTL
        print(f"[{scenario_name}] AdaptiveTTL...", end=" ", flush=True)
        sim_ad = create_simulator(reqs, True, args.l1, l2, avg_tool_time,
                                  history_threshold=history_threshold)
        r_ad = analyze_result(sim_ad)
        print(f"{r_ad['time']:.1f}s")

        speedup = r_bl['time'] / r_ad['time'] if r_ad['time'] > 0 else 0

        results.append({
            'scenario': scenario_name,
            'avg_tool_time': avg_tool_time,
            'baseline': r_bl,
            'adaptive': r_ad,
            'speedup': speedup,
        })

    return results


def run_tool_time_test(args) -> List[Dict]:
    """Run tool execution time sensitivity test."""
    l2 = int(args.l1 * args.l2_ratio)
    tool_times = [0.5, 1.0, 2.0, 5.0, 10.0]
    tool_names = ["极短(0.5s)", "短(1.0s)", "正常(2.0s)", "长(5.0s)", "超长(10.0s)"]
    history_threshold = getattr(args, 'history_threshold', 3)

    results = []
    for tool_name, tool_time in zip(tool_names, tool_times):
        requests = generate_staggered_requests(
            args.programs, args.turns, args.prefill, tool_time,
            args.first_turn_multiplier, seed=args.seed
        )

        # Baseline
        print(f"[{tool_name}] Baseline...", end=" ", flush=True)
        sim_bl = create_simulator(requests, False, args.l1, l2, tool_time,
                                 history_threshold=history_threshold)
        r_bl = analyze_result(sim_bl)
        print(f"{r_bl['time']:.1f}s")

        # Adaptive TTL
        print(f"[{tool_name}] AdaptiveTTL...", end=" ", flush=True)
        sim_ad = create_simulator(requests, True, args.l1, l2, tool_time,
                                 history_threshold=history_threshold)
        r_ad = analyze_result(sim_ad)
        print(f"{r_ad['time']:.1f}s")

        speedup = r_bl['time'] / r_ad['time'] if r_ad['time'] > 0 else 0

        results.append({
            'tool_name': tool_name,
            'tool_time': tool_time,
            'baseline': r_bl,
            'adaptive': r_ad,
            'speedup': speedup,
        })

    return results


def run_distribution_shift_test(args) -> List[Dict]:
    """Run distribution shift test: history vs test tool times differ."""
    l2 = int(args.l1 * args.l2_ratio)
    history_threshold = getattr(args, 'history_threshold', 3)
    scenarios = [
        ("短→长 (2s→10s)", 2.0, 10.0),
        ("长→短 (10s→2s)", 10.0, 2.0),
        ("历史稳定→测试噪声", 2.0, None),  # None = random
        ("短历史→长对话", 2.0, 5.0),
    ]

    results = []
    for name, history_time, test_time in scenarios:
        print(f"\n[{name}]")

        # Generate requests with different history vs test times
        history_turns = min(5, args.turns - 1)
        test_turns = args.turns - history_turns

        history_reqs = generate_requests(
            args.programs, history_turns, args.prefill, history_time,
            args.first_turn_multiplier, seed=args.seed
        )

        test_reqs = []
        for prog_idx in range(args.programs):
            program_id = f'prog_{prog_idx:03d}'
            accumulated = []
            for turn in range(history_turns, args.turns):
                current_prefill = args.prefill
                actual_time = test_time if test_time else random.uniform(0.5, 5.0)
                prev_total = len(accumulated)
                new_tokens = list(range(
                    prog_idx * 1000000 + prev_total + history_turns * args.prefill * 5,
                    prog_idx * 1000000 + prev_total + history_turns * args.prefill * 5 + current_prefill
                ))
                tokens = accumulated + new_tokens
                test_reqs.append({
                    'rid': f'{program_id}_turn_{turn}',
                    'program_id': program_id,
                    'turn_index': turn,
                    'token_ids': tokens,
                    'input_len': len(tokens),
                    'output_len': 64,
                    'arrival_time': 0.0,
                    'is_tool_call': turn < args.turns - 1,
                    'tool_name': f'tool_{turn % 3}',
                    'extra_key': program_id,
                    'ttl_sec': None,
                    'actual_tool_duration': actual_time,
                })
                accumulated.extend(new_tokens)

        all_reqs = history_reqs + test_reqs

        # Baseline
        print(f"  Baseline...", end=" ", flush=True)
        sim_bl = create_simulator(all_reqs, False, args.l1, l2, history_time,
                                  history_threshold=history_threshold)
        r_bl = analyze_result(sim_bl)
        print(f"{r_bl['time']:.1f}s")

        # Adaptive TTL
        print(f"  AdaptiveTTL...", end=" ", flush=True)
        sim_ad = create_simulator(all_reqs, True, args.l1, l2, history_time,
                                 history_threshold=history_threshold)
        r_ad = analyze_result(sim_ad)
        print(f"{r_ad['time']:.1f}s")

        speedup = r_bl['time'] / r_ad['time'] if r_ad['time'] > 0 else 0

        results.append({
            'name': name,
            'history_time': history_time,
            'test_time': test_time,
            'baseline': r_bl,
            'adaptive': r_ad,
            'speedup': speedup,
        })

    return results


def run_sweep_test(args) -> List[Dict]:
    """Run parameter sweep test."""
    l2 = int(args.l1 * args.l2_ratio)
    param_name = args.sweep
    param_range = args.sweep_range
    history_threshold = getattr(args, 'history_threshold', 3)

    results = []
    for param_value in param_range:
        # Apply parameter
        if param_name == 'l1':
            l1_val = param_value
            l2_val = int(param_value * args.l2_ratio)
        elif param_name == 'programs':
            l1_val = args.l1
            l2_val = int(args.l1 * args.l2_ratio)
        elif param_name == 'turns':
            l1_val = args.l1
            l2_val = l2
        elif param_name == 'prefill':
            l1_val = args.l1
            l2_val = l2
        else:
            l1_val = args.l1
            l2_val = l2

        # Generate requests with updated params
        if param_name == 'programs':
            requests = generate_requests(param_value, args.turns, args.prefill, args.tool_time,
                                       args.first_turn_multiplier, seed=args.seed)
        elif param_name == 'turns':
            requests = generate_requests(args.programs, param_value, args.prefill, args.tool_time,
                                       args.first_turn_multiplier, seed=args.seed)
        elif param_name == 'prefill':
            requests = generate_requests(args.programs, args.turns, param_value, args.tool_time,
                                       args.first_turn_multiplier, seed=args.seed)
        else:
            requests = generate_requests(args.programs, args.turns, args.prefill, args.tool_time,
                                       args.first_turn_multiplier, seed=args.seed)

        print(f"[{param_name}={param_value}] Baseline...", end=" ", flush=True)
        sim_bl = create_simulator(requests, False, l1_val, l2_val, args.tool_time,
                                 history_threshold=history_threshold)
        r_bl = analyze_result(sim_bl)
        print(f"{r_bl['time']:.1f}s")

        print(f"[{param_name}={param_value}] AdaptiveTTL...", end=" ", flush=True)
        sim_ad = create_simulator(requests, True, l1_val, l2_val, args.tool_time,
                                 history_threshold=history_threshold)
        r_ad = analyze_result(sim_ad)
        print(f"{r_ad['time']:.1f}s")

        speedup = r_bl['time'] / r_ad['time'] if r_ad['time'] > 0 else 0

        results.append({
            'param': param_name,
            'value': param_value,
            'l1': l1_val,
            'baseline': r_bl,
            'adaptive': r_ad,
            'speedup': speedup,
        })

    return results


# ============================================================================
# Output
# ============================================================================

def print_baseline_summary(results: List[Dict]):
    """Print baseline test summary."""
    print("\n" + "=" * 100)
    print("SUMMARY: Baseline + L2 vs Adaptive TTL + L2")
    print("=" * 100)

    if not results:
        return

    baseline = results[0]['time']

    print(f"\n{'Config':<25} | {'Time':>8} | {'vs Base':>8} | {'Prefill':>8} | {'Decode':>8} | {'H2D':>6} | {'Queue':>8} | {'TTL':>5} | {'L2':>5} | {'Miss':>5}")
    print("-" * 100)

    for r in results:
        speedup = baseline / r['time']
        vs_base = f"{speedup:.2f}x" if speedup >= 1 else f"{-1/speedup:.2f}x"
        print(f"{r['name']:<25} | {r['time']:>7.1f}s | {vs_base:>8} | {r['prefill_s']:>7.1f}s | {r['decode_s']:>7.1f}s | {r['h2d_s']:>5.1f}s | {r['queue_delay_s']:>7.1f}s | {r['ttl_hits']:>5} | {r['l2_hits']:>5} | {r['misses']:>5}")

    # Find TTL+L2 result
    ttl_l2 = next((r for r in results if 'TTL' in r['name']), None)
    baseline_l2 = next((r for r in results if 'Baseline' in r['name']), None)

    if baseline_l2 and ttl_l2:
        speedup = baseline_l2['time'] / ttl_l2['time']
        print(f"\nTTL + L2 vs Baseline + L2: {speedup:.2f}x speedup")


def print_anomaly_summary(results: List[Dict]):
    """Print anomaly test summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Anomaly Scenarios")
    print("=" * 80)

    print(f"\n{'场景':<20} | {'工具时间':>8} | {'Baseline':>10} | {'AdaptiveTTL':>12} | {'加速':>8}")
    print("-" * 70)

    for r in results:
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        print(f"{r['scenario']:<20} | {r['avg_tool_time']:>6.1f}s | {r['baseline']['time']:>8.1f}s | {r['adaptive']['time']:>10.1f}s | {sp_str:>8}")

    worse = [r for r in results if r['speedup'] < 0.95]
    if worse:
        print(f"\nTTL加速效果下降的场景:")
        for r in worse:
            print(f"  - {r['scenario']}: {r['speedup']:.2f}x")


def print_tool_time_summary(results: List[Dict]):
    """Print tool time test summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Tool Execution Time Sensitivity")
    print("=" * 80)

    print(f"\n{'场景':<20} | {'工具时间':>8} | {'Baseline':>10} | {'AdaptiveTTL':>12} | {'加速':>8}")
    print("-" * 70)

    for r in results:
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        print(f"{r['tool_name']:<20} | {r['tool_time']:>6.1f}s | {r['baseline']['time']:>8.1f}s | {r['adaptive']['time']:>10.1f}s | {sp_str:>8}")


def print_shift_summary(results: List[Dict]):
    """Print distribution shift summary."""
    print("\n" + "=" * 80)
    print("SUMMARY: Distribution Shift")
    print("=" * 80)

    print(f"\n{'场景':<25} | {'历史':>6} | {'测试':>6} | {'Baseline':>10} | {'AdaptiveTTL':>12} | {'效果':>8}")
    print("-" * 80)

    for r in results:
        test_str = f"{r['test_time']:.1f}s" if r['test_time'] else "随机"
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        print(f"{r['name']:<25} | {r['history_time']:>5.1f}s | {test_str:>6} | {r['baseline']['time']:>8.1f}s | {r['adaptive']['time']:>10.1f}s | {sp_str:>8}")


def print_sweep_summary(results: List[Dict]):
    """Print sweep test summary."""
    print("\n" + "=" * 120)
    print(f"SUMMARY: Parameter Sweep ({results[0]['param']}) - Baseline + L2 vs Adaptive TTL + L2")
    print("=" * 120)

    param = results[0]['param']
    print(f"\n{'L1':>6} | {'P':>4} | {'Config':<15} | {'Time':>8} | {'vs Base':>8} | {'Prefill':>8} | {'Decode':>8} | {'H2D':>6} | {'Queue':>8} | {'TTL':>5} | {'L2':>5} | {'Miss':>5}")
    print("-" * 120)

    for r in results:
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x"
        
        for config_name, config_data in [('Baseline', r['baseline']), ('TTL', r['adaptive'])]:
            is_baseline = config_name == 'Baseline'
            l1 = r.get('l1', '?')
            speedup_str = "1.00x" if is_baseline else sp_str
            print(f"{l1:>6} | {r['value']:>4} | {config_name + ' + L2':<15} | {config_data['time']:>7.1f}s | {speedup_str:>8} | {config_data['prefill_s']:>7.1f}s | {config_data['decode_s']:>7.1f}s | {config_data['h2d_s']:>5.1f}s | {config_data['queue_delay_s']:>7.1f}s | {config_data['ttl_hits']:>5} | {config_data['l2_hits']:>5} | {config_data['misses']:>5}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pure Simulator Unified Experiment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  baseline      Baseline comparison (Baseline vs Adaptive TTL)
  anomaly       Anomaly scenarios (timeout, retry, hang, early_return)
  tool_time     Tool execution time sensitivity
  shift         Distribution shift (history vs test)
  sweep         Parameter sweep

Examples:
  python run_experiment.py --mode baseline
  python run_experiment.py --mode anomaly --exception timeout --rate 0.3
  python run_experiment.py --mode sweep --sweep l1 --sweep-range 1000 5000 1000
        """
    )

    parser.add_argument('--mode', type=str, default='baseline',
                       choices=['baseline', 'anomaly', 'tool_time', 'shift', 'sweep'],
                       help='Test mode')
    parser.add_argument('--programs', type=int, default=DEFAULT_CONFIG['programs'],
                       help='Number of programs')
    parser.add_argument('--turns', type=int, default=DEFAULT_CONFIG['turns'],
                       help='Number of turns per program')
    parser.add_argument('--l1', type=int, default=DEFAULT_CONFIG['l1'],
                       help='L1 cache capacity (tokens)')
    parser.add_argument('--l2-ratio', type=float, default=DEFAULT_CONFIG['l2_ratio'],
                       help='L2/L1 ratio (default: 1.5)')
    parser.add_argument('--prefill', type=int, default=DEFAULT_CONFIG['prefill'],
                       help='Prefill tokens per turn')
    parser.add_argument('--first-turn-multiplier', type=int, default=DEFAULT_CONFIG['first_turn_multiplier'],
                       help='First turn multiplier')
    parser.add_argument('--tool-time', type=float, default=DEFAULT_CONFIG['tool_time'],
                       help='Tool execution time (seconds)')
    parser.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                       help='Random seed')

    # Anomaly options
    parser.add_argument('--exception', type=str, default=None,
                       choices=['timeout', 'retry', 'hang', 'early_return', 'mixed'],
                       help='Exception type (for anomaly mode)')
    parser.add_argument('--rate', type=float, default=0.3,
                       help='Exception rate (0.0-1.0)')

    # Sweep options
    parser.add_argument('--sweep', type=str, default='programs',
                       choices=['l1', 'programs', 'turns', 'prefill'],
                       help='Parameter to sweep')
    parser.add_argument('--sweep-range', type=int, nargs=3, default=[1000, 5000, 1000],
                       metavar=('START', 'STOP', 'STEP'),
                       help='Sweep range (start, stop, step)')
    parser.add_argument('--history-threshold', type=int, default=3,
                       help='History threshold for adaptive TTL CDF (default: 3)')

    # Other options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    print("=" * 80)
    print(f"Pure Simulator Experiment: {args.mode.upper()}")
    print("=" * 80)
    print(f"Config: P={args.programs}, T={args.turns}, L1={args.l1}, L2={int(args.l1 * args.l2_ratio)}")
    print(f"         prefill={args.prefill}, tool_time={args.tool_time}s")
    print()

    results = []

    if args.mode == 'baseline':
        results = run_baseline_test(args)
        print_baseline_summary(results)

    elif args.mode == 'anomaly':
        results = run_anomaly_test(args)
        print_anomaly_summary(results)

    elif args.mode == 'tool_time':
        results = run_tool_time_test(args)
        print_tool_time_summary(results)

    elif args.mode == 'shift':
        results = run_distribution_shift_test(args)
        print_shift_summary(results)

    elif args.mode == 'sweep':
        results = run_sweep_test(args)
        print_sweep_summary(results)

    return results


if __name__ == "__main__":
    main()
