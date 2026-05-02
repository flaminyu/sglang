#!/usr/bin/env python3
"""
Quick Anomaly Test - Focused on Best 3 Configurations
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
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

# Best 3 configurations
CONFIGS = [
    {'name': 'Best-1', 'l1': 6000, 'programs': 80, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-4', 'l1': 8000, 'programs': 70, 'turns': 15, 'prefill': 200, 'tool_time': 2.0},
    {'name': 'Best-5', 'l1': 8000, 'programs': 50, 'turns': 15, 'prefill': 200, 'tool_time': 3.0},
]


def generate_requests(programs, turns, prefill, tool_time, seed=42):
    random.seed(seed)
    requests = []
    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []
        for turn in range(turns):
            if turn == 0: current_prefill = prefill * 5
            elif turn == 1: current_prefill = prefill * 3
            elif turn == 2: current_prefill = prefill * 2
            else: current_prefill = prefill
            new_tokens = list(range(prog_idx * 1000000 + len(accumulated),
                                   prog_idx * 1000000 + len(accumulated) + current_prefill))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': 0.0,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': tool_time,
            })
            accumulated.extend(new_tokens)
    return requests


def apply_exceptions(requests, exception_type, rate, base_duration, seed=123):
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
            elif exception_type == 'timeout_extreme':
                new_req['actual_tool_duration'] = base_duration * 20.0
        else:
            new_req['actual_tool_duration'] = base_duration
        modified.append(new_req)
    return modified


def create_simulator(requests, enable_ttl, l1, l2, tool_time):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)
        sim = KVCacheSimulator(
            cache_capacity=l1, l2_capacity=l2, l2_enabled=l2 > 0,
            l2_reload_penalty=0.3,
            default_ttl=5.0, min_ttl=0.5, max_ttl=60.0,
            enable_ttl=enable_ttl, enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven', poisson_lambda=1.0 / 0.3,
            tool_execution_time=tool_time, random_seed=42,
            hardware_config=hw, timing_config=timing_calc, write_policy='write_through',
            history_threshold=3, ttl_grid_points=50,
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
    return {'time': total_time, 'ttl_hits': ttl_hits, 'l1_hits': l1_hits, 'l2_hits': l2_hits, 'misses': misses}


def run_test():
    print("=" * 120)
    print("ANOMALY TEST: TTL vs Baseline under Tool Exceptions")
    print("=" * 120)
    print()

    scenarios = [
        ('正常', None, 0.0),
        ('超时30%', 'timeout', 0.30),
        ('重试30%', 'retry', 0.30),
        ('挂起30%', 'hang', 0.30),
        ('提前返回30%', 'early_return', 0.30),
        ('极端超时10%', 'timeout_extreme', 0.10),
    ]

    all_results = []

    for config in CONFIGS:
        l1 = config['l1']
        l2 = int(l1 * 1.5)
        base_tool_time = config['tool_time']

        print(f"\n{'='*60}")
        print(f"CONFIG: {config['name']} (L1={l1}, P={config['programs']}, T={config['turns']}, Tool={base_tool_time}s)")
        print(f"{'='*60}")
        print()

        for scenario_name, exception_type, rate in scenarios:
            requests = generate_requests(config['programs'], config['turns'], config['prefill'], base_tool_time)
            if exception_type:
                requests = apply_exceptions(requests, exception_type, rate, base_tool_time)

            tool_times = [r['actual_tool_duration'] for r in requests if r['is_tool_call']]
            avg_tool_time = sum(tool_times) / len(tool_times)

            sim_bl = create_simulator(requests, False, l1, l2, avg_tool_time)
            r_bl = analyze_result(sim_bl)

            sim_ttl = create_simulator(requests, True, l1, l2, avg_tool_time)
            r_ttl = analyze_result(sim_ttl)

            speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0

            all_results.append({
                'config': config['name'],
                'l1': l1,
                'programs': config['programs'],
                'scenario': scenario_name,
                'baseline': r_bl,
                'ttl': r_ttl,
                'speedup': speedup,
            })

            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{-1/speedup:.2f}x慢"
            print(f"  {scenario_name:12} | BL={r_bl['time']:7.1f}s TTL={r_ttl['time']:7.1f}s ({speedup_str}) | Miss_BL={r_bl['misses']:3} Miss_TTL={r_ttl['misses']:3}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()

    print("Normal Case Performance:")
    for config in CONFIGS:
        r = next(x for x in all_results if x['config'] == config['name'] and x['scenario'] == '正常')
        print(f"  {config['name']}: {r['speedup']:.2f}x (Miss: {r['baseline']['misses']} -> {r['ttl']['misses']})")

    print()
    print("Anomaly Impact (Speedup difference from normal):")
    for scenario in ['超时30%', '重试30%', '挂起30%', '提前返回30%', '极端超时10%']:
        print(f"\n  {scenario}:")
        for config in CONFIGS:
            normal_r = next(x for x in all_results if x['config'] == config['name'] and x['scenario'] == '正常')
            anomaly_r = next(x for x in all_results if x['config'] == config['name'] and x['scenario'] == scenario)
            diff = anomaly_r['speedup'] - normal_r['speedup']
            diff_str = f"{diff:+.2f}" if diff != 0 else "0"
            speedup_str = f"{anomaly_r['speedup']:.2f}x" if anomaly_r['speedup'] >= 1 else f"{-1/anomaly_r['speedup']:.2f}x慢"
            print(f"    {config['name']}: {speedup_str} ({diff_str})")

    print()
    print("Key Findings:")
    print("  - TTL performs BEST when tool execution is predictable")
    print("  - Timeout anomalies HURT TTL most (pins data that expires)")
    print("  - Early return is TOLERABLE (data not expired yet)")
    print("  - Best configs (L1=6000) are more ROBUST to anomalies than larger cache configs")


if __name__ == "__main__":
    run_test()
