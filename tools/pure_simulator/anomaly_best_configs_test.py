#!/usr/bin/env python3
"""
Anomaly Test for Best TTL Configurations

Tests TTL vs Baseline behavior under various tool anomalies (timeout, retry, hang, early_return)
using the best configurations found from sweep tests.

Best configurations from comprehensive_ttl_test.py:
1. L1=6000, P=80, T=15, Prefill=200 → 1.29x
2. L1=6000, P=70, T=15, Prefill=200 → 1.28x
3. L1=8000, P=80, T=15, Prefill=200 → 1.26x
4. L1=8000, P=70, T=15, Prefill=200 → 1.24x
5. L1=8000, P=50, T=15, Prefill=200, Tool=3s → 1.23x
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any, Tuple
from itertools import product

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator


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


# Best configurations from sweep tests
BEST_CONFIGS = [
    {'name': 'Best-1', 'l1': 6000, 'programs': 80, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-2', 'l1': 6000, 'programs': 70, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-3', 'l1': 8000, 'programs': 80, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-4', 'l1': 8000, 'programs': 70, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-5', 'l1': 8000, 'programs': 50, 'turns': 15, 'prefill': 200, 'tool_time': 3.0},
]


def generate_requests(
    programs: int, turns: int, prefill: int, tool_time: float,
    arrival_time: float = 0.0, seed: int = 42
) -> List[Dict]:
    """Generate requests for simulation."""
    random.seed(seed)
    requests = []

    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []

        for turn in range(turns):
            if turn == 0:
                current_prefill = prefill * 5
            elif turn == 1:
                current_prefill = prefill * 3
            elif turn == 2:
                current_prefill = prefill * 2
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
                # 5x longer tool execution
                new_req['actual_tool_duration'] = base_duration * 5.0
            elif exception_type == 'retry':
                # 2x longer (simulating retry overhead)
                new_req['actual_tool_duration'] = base_duration * 2.0
            elif exception_type == 'hang':
                # 10x longer (simulating hang)
                new_req['actual_tool_duration'] = base_duration * 10.0
            elif exception_type == 'early_return':
                # 10x shorter (tool returns immediately)
                new_req['actual_tool_duration'] = base_duration * 0.1
            elif exception_type == 'timeout_extreme':
                # 20x longer
                new_req['actual_tool_duration'] = base_duration * 20.0
        else:
            new_req['actual_tool_duration'] = base_duration
        modified.append(new_req)

    return modified


def create_simulator(requests, enable_ttl, l1, l2, tool_time=2.0, history_threshold=3, ttl_config=None):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        ttl_cfg = ttl_config or {}
        default_ttl = ttl_cfg.get('default_ttl', 5.0)
        min_ttl = ttl_cfg.get('min_ttl', 0.5)
        max_ttl = ttl_cfg.get('max_ttl', 60.0)

        sim = KVCacheSimulator(
            cache_capacity=l1, l2_capacity=l2, l2_enabled=l2 > 0,
            l2_reload_penalty=0.3,
            default_ttl=default_ttl, min_ttl=min_ttl, max_ttl=max_ttl,
            enable_ttl=enable_ttl, enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven', poisson_lambda=1.0 / 0.3,
            tool_execution_time=tool_time, random_seed=42,
            hardware_config=hw, timing_config=timing_calc, write_policy='write_through',
            history_threshold=history_threshold, ttl_grid_points=50,
        )
        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999
        sim.run(verbose=False)
        return sim
    finally:
        os.unlink(temp_path)


def analyze_result(sim):
    log = sim.scheduler.scheduling_log
    if not log:
        return {'time': 0, 'ttl_hits': 0, 'l1_hits': 0, 'l2_hits': 0, 'misses': 0}

    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_time = log[-1]['finish_time'] if log else 0

    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_queue_ms = sum(e.get('queue_wait_ms', 0) for e in log)

    return {
        'time': total_time,
        'ttl_hits': ttl_hits, 'l1_hits': l1_hits, 'l2_hits': l2_hits, 'misses': misses,
        'prefill_s': total_prefill_ms / 1000,
        'queue_s': total_queue_ms / 1000,
    }


def test_anomaly_scenario(
    config: Dict,
    scenario_name: str,
    exception_type: str,
    rate: float,
    seed: int = 42
) -> Dict[str, Any]:
    """Test a single anomaly scenario for a configuration."""
    l1 = config['l1']
    l2 = int(l1 * 1.5)
    base_tool_time = config['tool_time']

    # Generate base requests
    requests = generate_requests(
        config['programs'], config['turns'], config['prefill'], base_tool_time,
        seed=seed
    )

    # Apply exceptions
    if exception_type:
        requests = apply_exceptions(requests, exception_type, rate, base_tool_time, seed=seed + 100)

    # Calculate actual average tool time
    tool_times = [r['actual_tool_duration'] for r in requests if r['is_tool_call']]
    avg_tool_time = sum(tool_times) / len(tool_times) if tool_times else base_tool_time

    # Baseline
    sim_bl = create_simulator(requests, False, l1, l2, avg_tool_time)
    r_bl = analyze_result(sim_bl)

    # TTL
    sim_ttl = create_simulator(requests, True, l1, l2, avg_tool_time)
    r_ttl = analyze_result(sim_ttl)

    speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
    time_diff = r_ttl['time'] - r_bl['time']

    return {
        'config_name': config['name'],
        'l1': l1,
        'programs': config['programs'],
        'turns': config['turns'],
        'prefill': config['prefill'],
        'scenario': scenario_name,
        'exception_type': exception_type,
        'rate': rate,
        'avg_tool_time': avg_tool_time,
        'baseline': r_bl,
        'ttl': r_ttl,
        'speedup': speedup,
        'time_diff': time_diff,
    }


def run_comprehensive_anomaly_tests():
    """Run comprehensive anomaly tests on best configurations."""
    print("=" * 120)
    print("ANOMALY TEST: TTL vs Baseline under Tool Exceptions")
    print("=" * 120)
    print()

    # Define anomaly scenarios
    scenarios = [
        ('正常 (无异常)', None, 0.0),
        ('超时 10%', 'timeout', 0.10),
        ('超时 30%', 'timeout', 0.30),
        ('超时 50%', 'timeout', 0.50),
        ('重试 10%', 'retry', 0.10),
        ('重试 30%', 'retry', 0.30),
        ('重试 50%', 'retry', 0.50),
        ('挂起 10%', 'hang', 0.10),
        ('挂起 30%', 'hang', 0.30),
        ('提前返回 10%', 'early_return', 0.10),
        ('提前返回 30%', 'early_return', 0.30),
        ('极端超时 10%', 'timeout_extreme', 0.10),
    ]

    all_results = []

    # Test each best configuration
    for config in BEST_CONFIGS:
        print("=" * 120)
        print(f"CONFIG: {config['name']} (L1={config['l1']}, P={config['programs']}, T={config['turns']}, Prefill={config['prefill']}, Tool={config['tool_time']}s)")
        print("=" * 120)
        print()

        tokens_per_prog = sum([
            config['prefill'] * 5,
            config['prefill'] * 3,
            config['prefill'] * 2,
            config['prefill'] * (config['turns'] - 3)
        ])
        print(f"Tokens per program: {tokens_per_prog}")
        print(f"L1 can fit: {config['l1'] / tokens_per_prog:.1f} programs")
        print()

        for scenario_name, exception_type, rate in scenarios:
            print(f"  {scenario_name}...", end=" ", flush=True)
            result = test_anomaly_scenario(config, scenario_name, exception_type, rate)
            all_results.append(result)
            speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] >= 1 else f"{-1/result['speedup']:.2f}x慢"
            print(f"BL={result['baseline']['time']:.1f}s TTL={result['ttl']['time']:.1f}s ({speedup_str}) | Miss_BL={result['baseline']['misses']} Miss_TTL={result['ttl']['misses']}")

        print()

    # =========================================================================
    # Summary Analysis
    # =========================================================================
    print("=" * 120)
    print("SUMMARY: Anomaly Impact Analysis")
    print("=" * 120)
    print()

    # 1. Normal case comparison
    print("-" * 120)
    print("1. NORMAL CASE (No Anomaly) - Best Configurations")
    print("-" * 120)
    print()
    print(f"{'Config':<12} | {'L1':>5} | {'P':>4} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'Miss减少':>10}")
    print("-" * 80)

    normal_results = [r for r in all_results if r['scenario'] == '正常 (无异常)']
    normal_results.sort(key=lambda x: x['speedup'], reverse=True)
    for r in normal_results:
        miss_reduction = r['baseline']['misses'] - r['ttl']['misses']
        miss_str = f"{miss_reduction:+d}" if miss_reduction != 0 else "0"
        print(f"{r['config_name']:<12} | {r['l1']:>5} | {r['programs']:>4} | {r['baseline']['time']:>8.1f}s | {r['ttl']['time']:>8.1f}s | {r['speedup']:>7.2f}x | {miss_str:>10}")

    print()

    # 2. Timeout impact
    print("-" * 120)
    print("2. TIMEOUT ANOMALY - Impact on TTL vs Baseline")
    print("-" * 120)
    print()
    print(f"{'Config':<12} | {'场景':<15} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'TTL变化':>10}")
    print("-" * 90)

    timeout_results = [r for r in all_results if '超时' in r['scenario'] or (r.get('exception_type') and 'extreme' in r.get('exception_type', ''))]
    timeout_results.sort(key=lambda x: (x['config_name'], -x['rate']))

    prev_config = None
    for r in timeout_results:
        if r['config_name'] != prev_config:
            prev_config = r['config_name']
            normal_r = next(x for x in normal_results if x['config_name'] == r['config_name'])
            ttl_normal = normal_r['ttl']['time']
        else:
            ttl_normal = None

        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        ttl_change = f"{r['ttl']['time'] - ttl_normal:+.1f}s" if ttl_normal else "-"
        print(f"{r['config_name']:<12} | {r['scenario']:<15} | {r['baseline']['time']:>8.1f}s | {r['ttl']['time']:>8.1f}s | {speedup_str:>8} | {ttl_change:>10}")

    print()

    # 3. Early return impact
    print("-" * 120)
    print("3. EARLY RETURN ANOMALY - TTL Degradation Analysis")
    print("-" * 120)
    print()
    print(f"{'Config':<12} | {'场景':<15} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'vs正常':>10}")
    print("-" * 90)

    early_return_results = [r for r in all_results if '提前返回' in r['scenario']]

    for r in early_return_results:
        normal_r = next(x for x in normal_results if x['config_name'] == r['config_name'])
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        vs_normal = r['speedup'] - normal_r['speedup']
        vs_normal_str = f"{vs_normal:+.2f}" if vs_normal != 0 else "0"
        print(f"{r['config_name']:<12} | {r['scenario']:<15} | {r['baseline']['time']:>8.1f}s | {r['ttl']['time']:>8.1f}s | {speedup_str:>8} | {vs_normal_str:>10}")

    print()

    # 4. Hang impact
    print("-" * 120)
    print("4. HANG ANOMALY - TTL Robustness Analysis")
    print("-" * 120)
    print()
    print(f"{'Config':<12} | {'场景':<15} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'Miss_BL':>8} | {'Miss_TTL':>8}")
    print("-" * 100)

    hang_results = [r for r in all_results if '挂起' in r['scenario']]
    for r in hang_results:
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x慢"
        print(f"{r['config_name']:<12} | {r['scenario']:<15} | {r['baseline']['time']:>8.1f}s | {r['ttl']['time']:>8.1f}s | {speedup_str:>8} | {r['baseline']['misses']:>8} | {r['ttl']['misses']:>8}")

    print()

    # 5. Anomaly type comparison
    print("-" * 120)
    print("5. ANOMALY TYPE COMPARISON - Which hurts TTL the most?")
    print("-" * 120)
    print()

    # Calculate average speedup for each anomaly type across all configs
    anomaly_types = [
        ('正常 (无异常)', '正常 (无异常)'),
        ('超时 30%', 'timeout'),
        ('重试 30%', 'retry'),
        ('挂起 30%', 'hang'),
        ('提前返回 30%', 'early_return'),
    ]

    print(f"{'异常类型':<20} | {'平均Baseline':>12} | {'平均TTL':>12} | {'平均加速':>10} | {'TTL退化':>10}")
    print("-" * 80)

    for anomaly_name, anomaly_type in anomaly_types:
        if anomaly_type:
            results_for_type = [r for r in all_results if r['scenario'] == anomaly_name]
        else:
            results_for_type = [r for r in all_results if r['scenario'] == anomaly_name]

        if results_for_type:
            avg_bl = sum(r['baseline']['time'] for r in results_for_type) / len(results_for_type)
            avg_ttl = sum(r['ttl']['time'] for r in results_for_type) / len(results_for_type)
            avg_speedup = sum(r['speedup'] for r in results_for_type) / len(results_for_type)

            normal_avg_speedup = sum(r['speedup'] for r in normal_results) / len(normal_results)
            degradation = avg_speedup - normal_avg_speedup
            deg_str = f"{degradation:+.2f}" if degradation != 0 else "0"

            speedup_str = f"{avg_speedup:.2f}x" if avg_speedup >= 1 else f"{-1/avg_speedup:.2f}x慢"
            print(f"{anomaly_name:<20} | {avg_bl:>10.1f}s | {avg_ttl:>10.1f}s | {speedup_str:>10} | {deg_str:>10}")

    print()

    # 6. Key insights
    print("=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    print()

    # Find scenarios where TTL is worse than baseline
    worse_scenarios = [r for r in all_results if r['speedup'] < 0.95]
    if worse_scenarios:
        print("TTL 效果变差的场景:")
        for r in worse_scenarios:
            print(f"  - {r['config_name']} + {r['scenario']}: {r['speedup']:.2f}x")

    print()

    # Find scenarios where TTL helps even with anomalies
    helpful_scenarios = [r for r in all_results if r['speedup'] >= 1.10]
    if helpful_scenarios:
        print("即使有异常，TTL 仍有显著加速的场景:")
        for r in helpful_scenarios:
            print(f"  - {r['config_name']} + {r['scenario']}: {r['speedup']:.2f}x")

    print()

    # Analyze degradation patterns
    print("异常对 TTL 的影响分析:")
    for config in BEST_CONFIGS:
        config_results = [r for r in all_results if r['config_name'] == config['name']]
        normal_r = next(r for r in config_results if r['scenario'] == '正常 (无异常)')
        worst_r = min(config_results, key=lambda x: x['speedup'])

        print(f"  {config['name']}:")
        print(f"    正常时: {normal_r['speedup']:.2f}x")
        print(f"    最差时: {worst_r['scenario']} → {worst_r['speedup']:.2f}x")
        print(f"    退化: {normal_r['speedup'] - worst_r['speedup']:.2f}")
        print()


def run_single_config_test(config_name: str, scenario_name: str):
    """Run a single configuration and scenario for detailed analysis."""
    config = next(c for c in BEST_CONFIGS if c['name'] == config_name)

    scenarios = [
        ('正常 (无异常)', None, 0.0),
        ('超时 30%', 'timeout', 0.30),
        ('提前返回 30%', 'early_return', 0.30),
    ]

    scenario = next(s for s in scenarios if s[0] == scenario_name)
    result = test_anomaly_scenario(config, *scenario)

    print(f"\nDetailed analysis for {config_name} + {scenario_name}:")
    print(f"  Baseline time: {result['baseline']['time']:.1f}s")
    print(f"  TTL time: {result['ttl']['time']:.1f}s")
    print(f"  Speedup: {result['speedup']:.2f}x")
    print(f"  Baseline misses: {result['baseline']['misses']}")
    print(f"  TTL misses: {result['ttl']['misses']}")
    print(f"  Miss reduction: {result['baseline']['misses'] - result['ttl']['misses']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly test for best TTL configurations")
    parser.add_argument('--config', type=str, default=None, help='Specific config to test')
    parser.add_argument('--scenario', type=str, default=None, help='Specific scenario to test')
    args = parser.parse_args()

    if args.config and args.scenario:
        run_single_config_test(args.config, args.scenario)
    else:
        run_comprehensive_anomaly_tests()
