#!/usr/bin/env python3
"""
Realistic TTL Test - Mimics real SweBench/BCFL workload patterns

Key differences from simple test:
1. Tool execution times have VARIETY (not fixed)
2. Different tool types with different execution times
3. Each program has DIFFERENT length and turns
4. Uses JPS (Jobs Per Second) for staggered arrival
5. More programs, longer simulation
6. Realistic token distribution

Tool categories from real workloads:
- Short tools: 0.1-0.5s (file read, simple queries)
- Medium tools: 1-3s (API calls, data processing)
- Long tools: 5-15s (complex operations, batch processing)
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict

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


# ============================================================================
# Tool Definitions (Realistic)
# ============================================================================

# Tool categories with execution time ranges
TOOL_CATEGORIES = {
    'file_read': {'time_range': (0.1, 0.5), 'weight': 0.25},
    'file_write': {'time_range': (0.2, 0.8), 'weight': 0.15},
    'bash': {'time_range': (0.5, 3.0), 'weight': 0.25},
    'api_call': {'time_range': (1.0, 5.0), 'weight': 0.15},
    'db_query': {'time_range': (0.5, 2.0), 'weight': 0.10},
    'long_task': {'time_range': (5.0, 15.0), 'weight': 0.10},
}

TOOL_NAMES = list(TOOL_CATEGORIES.keys())

# ============================================================================
# Request Generation (Realistic)
# ============================================================================

def generate_realistic_requests(
    num_programs: int,
    l1: int,
    jps: float = 1.0,  # Jobs per second arrival rate
    seed: int = 42,
    vary_turns: bool = True,
    vary_tool_times: bool = True,
    vary_prefill: bool = True,
) -> List[Dict]:
    """
    Generate realistic requests that mimic SweBench/BCFL workloads.

    Key features:
    - Programs arrive following Poisson process (JPS)
    - Each program has random number of turns (2-20)
    - Each turn has random prefill size (100-400 tokens)
    - Each tool call uses random tool type with realistic execution time
    - Token accumulation follows realistic patterns
    """
    random.seed(seed)
    requests = []
    current_time = 0.0

    # Estimate tokens per turn for cache pressure calculation
    avg_tokens_per_turn = 400  # Approximate

    for prog_idx in range(num_programs):
        program_id = f'prog_{prog_idx:04d}'

        # Vary number of turns per program (2-20)
        if vary_turns:
            num_turns = random.choices(
                [3, 5, 8, 10, 12, 15, 18, 20],
                weights=[0.1, 0.2, 0.25, 0.2, 0.1, 0.08, 0.05, 0.02]
            )[0]
        else:
            num_turns = 10

        # Base prefill size varies per program
        if vary_prefill:
            base_prefill = random.randint(100, 300)
        else:
            base_prefill = 200

        accumulated = []
        arrival_base = current_time

        for turn in range(num_turns):
            # Prefill size pattern: first turn large, then decreasing
            if turn == 0:
                current_prefill = base_prefill * random.randint(4, 6)  # 400-1800
            elif turn == 1:
                current_prefill = base_prefill * random.randint(2, 4)  # 200-1200
            elif turn == 2:
                current_prefill = base_prefill * random.randint(1, 3)  # 100-900
            else:
                current_prefill = base_prefill * random.randint(1, 2)  # 100-600

            # Generate token IDs
            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 10000000 + prev_total,
                prog_idx * 10000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens

            # Select tool type based on weights
            if vary_tool_times:
                tool_name = random.choices(
                    TOOL_NAMES,
                    weights=[TOOL_CATEGORIES[t]['weight'] for t in TOOL_NAMES]
                )[0]
                time_min, time_max = TOOL_CATEGORIES[tool_name]['time_range']
                actual_tool_duration = random.uniform(time_min, time_max)
            else:
                tool_name = 'bash'
                actual_tool_duration = 2.0

            # Arrival time: staggered by JPS
            if turn == 0:
                # First turn arrives based on JPS
                current_time += random.expovariate(jps) if jps > 0 else 0
                arrival = current_time
            else:
                # Subsequent turns arrive after tool execution of previous turns
                # Find the last request from this program
                last_req = None
                for r in reversed(requests):
                    if r['program_id'] == program_id:
                        last_req = r
                        break
                if last_req:
                    arrival = last_req['arrival_time'] + last_req['actual_tool_duration'] + 0.1
                else:
                    arrival = arrival_base

            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': arrival,
                'is_tool_call': turn < num_turns - 1,
                'tool_name': tool_name,
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': actual_tool_duration,
            })
            accumulated.extend(new_tokens)

    return requests


def generate_batch_requests(
    num_programs: int,
    turns: int,
    prefill: int,
    tool_time: float,
    seed: int = 42
) -> List[Dict]:
    """
    Generate batch requests (all at once) - for comparison.
    """
    random.seed(seed)
    requests = []

    for prog_idx in range(num_programs):
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
                'arrival_time': 0.0,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': tool_time,
            })
            accumulated.extend(new_tokens)

    return requests


# ============================================================================
# Simulator
# ============================================================================

def create_simulator(requests, enable_ttl, l1, l2, history_threshold=20, ttl_config=None):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        ttl_cfg = ttl_config or {}
        default_ttl = ttl_cfg.get('default_ttl', 15.0)  # 增加到15s，覆盖long_task
        min_ttl = ttl_cfg.get('min_ttl', 0.5)
        max_ttl = ttl_cfg.get('max_ttl', 60.0)

        sim = KVCacheSimulator(
            cache_capacity=l1, l2_capacity=l2, l2_enabled=l2 > 0,
            l2_reload_penalty=0.3,
            default_ttl=default_ttl, min_ttl=min_ttl, max_ttl=max_ttl,
            enable_ttl=enable_ttl, enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven', poisson_lambda=1.0 / 0.3,
            tool_execution_time=2.0, random_seed=42,
            hardware_config=hw, timing_config=timing_calc, write_policy='write_through',
            history_threshold=history_threshold, ttl_grid_points=50,
        )

        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999

        sim.run(verbose=False)

        # Analyze TTL decisions
        ttl_decisions = {}
        if sim.ttl_manager:
            for program_id, ttl, strategy in sim.ttl_manager.adaptive_ttl_history:
                if strategy not in ttl_decisions:
                    ttl_decisions[strategy] = {'count': 0, 'ttls': []}
                ttl_decisions[strategy]['count'] += 1
                ttl_decisions[strategy]['ttls'].append(ttl)

        return sim, ttl_decisions
    finally:
        os.unlink(temp_path)


def analyze_result(sim):
    log = sim.scheduler.scheduling_log
    if not log:
        return {'time': 0, 'ttl_hits': 0, 'l1_hits': 0, 'l2_hits': 0, 'misses': 0,
                'total_turns': 0, 'avg_turn_time': 0}

    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_time = log[-1]['finish_time'] if log else 0
    total_turns = len(log)
    avg_turn_time = total_time / total_turns if total_turns > 0 else 0

    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_queue_ms = sum(e.get('queue_wait_ms', 0) for e in log)

    # Hit rate analysis
    total_hits = ttl_hits + l1_hits + l2_hits
    total_requests = total_hits + misses
    hit_rate = total_hits / total_requests if total_requests > 0 else 0

    return {
        'time': total_time,
        'ttl_hits': ttl_hits, 'l1_hits': l1_hits, 'l2_hits': l2_hits, 'misses': misses,
        'total_turns': total_turns,
        'avg_turn_time': avg_turn_time,
        'prefill_s': total_prefill_ms / 1000,
        'queue_s': total_queue_ms / 1000,
        'hit_rate': hit_rate,
        'ttl_hit_rate': ttl_hits / total_hits if total_hits > 0 else 0,
    }


# ============================================================================
# Test Configurations
# ============================================================================

def test_realistic_scenario(l1, num_programs, jps, history_threshold=10, ttl_config=None):
    """Test realistic scenario with varied programs."""
    l2 = int(l1 * 1.5)

    print(f"\n{'='*80}")
    print(f"Realistic Test: L1={l1}, Programs={num_programs}, JPS={jps}, K={history_threshold}")
    print(f"{'='*80}")

    # Generate realistic requests
    print("Generating realistic requests...", end=" ", flush=True)
    requests = generate_realistic_requests(
        num_programs=num_programs,
        l1=l1,
        jps=jps,
        seed=42,
        vary_turns=True,
        vary_tool_times=True,
        vary_prefill=True
    )

    # Analyze request distribution
    tool_times = [r['actual_tool_duration'] for r in requests if r['is_tool_call']]
    turns_per_prog = defaultdict(int)
    tool_counts = defaultdict(int)
    for r in requests:
        turns_per_prog[r['program_id']] += 1
        if r['is_tool_call']:
            tool_counts[r['tool_name']] += 1

    print(f"Done! {len(requests)} requests, {len(turns_per_prog)} programs")
    print(f"  - Tool time: min={min(tool_times):.2f}s, max={max(tool_times):.2f}s, avg={sum(tool_times)/len(tool_times):.2f}s")
    print(f"  - Turns per program: min={min(turns_per_prog.values())}, max={max(turns_per_prog.values())}, avg={sum(turns_per_prog.values())/len(turns_per_prog):.1f}")
    print(f"  - Tool distribution:")
    for tool_name, count in sorted(tool_counts.items()):
        expected_rate = TOOL_CATEGORIES[tool_name]['weight']
        expected_count = len([r for r in requests if r['is_tool_call']]) * expected_rate
        print(f"      {tool_name}: {count} (expected ~{expected_count:.0f})")

    avg_tool_time = sum(tool_times) / len(tool_times)

    # Baseline
    print(f"Running Baseline (JPS={jps})...", end=" ", flush=True)
    sim_bl, ttl_decisions_bl = create_simulator(requests, False, l1, l2, history_threshold)
    r_bl = analyze_result(sim_bl)
    print(f"{r_bl['time']:.1f}s")

    # TTL
    print(f"Running TTL (JPS={jps})...", end=" ", flush=True)
    sim_ttl, ttl_decisions = create_simulator(requests, True, l1, l2, history_threshold)
    r_ttl = analyze_result(sim_ttl)
    print(f"{r_ttl['time']:.1f}s")

    # Print TTL decision analysis
    if ttl_decisions:
        print(f"\nTTL Decisions:")
        for strategy, data in sorted(ttl_decisions.items()):
            if data['ttls']:
                avg_ttl = sum(data['ttls']) / len(data['ttls'])
                print(f"  {strategy}: {data['count']} times, avg_ttl={avg_ttl:.2f}s")

    speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0

    print(f"\nResults:")
    print(f"  Baseline: {r_bl['time']:.1f}s, Miss={r_bl['misses']}, Hit={r_bl['ttl_hits']+r_bl['l1_hits']+r_bl['l2_hits']}")
    print(f"  TTL:     {r_ttl['time']:.1f}s, Miss={r_ttl['misses']}, Hit={r_ttl['ttl_hits']+r_ttl['l1_hits']+r_ttl['l2_hits']}")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Miss reduction: {r_bl['misses']} -> {r_ttl['misses']} ({r_bl['misses'] - r_ttl['misses']:+d})")
    if r_bl['hit_rate'] > 0:
        print(f"  Hit rate: BL={r_bl['hit_rate']*100:.1f}%, TTL={r_ttl['hit_rate']*100:.1f}%")
    if r_ttl['ttl_hits'] > 0:
        print(f"  TTL hit rate (of all hits): {r_ttl['ttl_hit_rate']*100:.1f}%")

    return {
        'l1': l1,
        'num_programs': num_programs,
        'jps': jps,
        'baseline': r_bl,
        'ttl': r_ttl,
        'speedup': speedup,
    }


def test_jps_sweep():
    """Test different JPS values to find optimal.
    
    With 250 programs (avg 8 turns each = 2000 tool calls),
    each tool type should get ~200+ executions.
    """
    print("=" * 80)
    print("JPS Sweep Test - Finding optimal JPS for TTL")
    print("=" * 80)

    results = []

    # 250 programs × 8 turns avg = 2000 tool calls
    # At 10% min rate: 200 calls per tool type (sufficient for CDF)
    num_programs = 250

    configs = [
        # (l1, num_programs, jps)
        (6000, num_programs, 0.5),
        (6000, num_programs, 1.0),
        (6000, num_programs, 2.0),
        (8000, num_programs, 0.5),
        (8000, num_programs, 1.0),
        (8000, num_programs, 2.0),
        (10000, num_programs, 0.5),
        (10000, num_programs, 1.0),
        (10000, num_programs, 2.0),
        (12000, num_programs, 0.5),
        (12000, num_programs, 1.0),
        (12000, num_programs, 2.0),
    ]

    for l1, num_programs, jps in configs:
        result = test_realistic_scenario(l1, num_programs, jps)
        results.append(result)

    return results


def test_batch_vs_jps():
    """Compare batch (all at once) vs JPS (staggered) arrival."""
    print("=" * 80)
    print("Batch vs JPS Comparison")
    print("=" * 80)

    results = []

    # 250 programs × 8 turns avg = 2000 tool calls
    num_programs = 250

    configs = [
        (6000, num_programs, 15),
        (8000, num_programs, 15),
        (10000, num_programs, 15),
    ]

    for l1, num_programs, turns in configs:
        print(f"\n{'='*60}")
        print(f"Config: L1={l1}, P={num_programs}, T={turns}")
        print(f"{'='*60}")

        l2 = int(l1 * 1.5)
        prefill = 200
        tool_time = 2.0

        # Batch requests
        print("Generating batch requests...", end=" ", flush=True)
        batch_requests = generate_batch_requests(num_programs, turns, prefill, tool_time, seed=42)
        print(f"{len(batch_requests)} requests")

        print("Running Baseline (batch)...", end=" ", flush=True)
        sim_bl_batch, _ = create_simulator(batch_requests, False, l1, l2)
        r_bl_batch = analyze_result(sim_bl_batch)
        print(f"{r_bl_batch['time']:.1f}s")

        print("Running TTL (batch)...", end=" ", flush=True)
        sim_ttl_batch, ttl_decisions_batch = create_simulator(batch_requests, True, l1, l2)
        r_ttl_batch = analyze_result(sim_ttl_batch)
        print(f"{r_ttl_batch['time']:.1f}s")

        batch_speedup = r_bl_batch['time'] / r_ttl_batch['time'] if r_ttl_batch['time'] > 0 else 0

        # JPS requests
        jps = 1.0
        print(f"Generating JPS={jps} requests...", end=" ", flush=True)
        jps_requests = generate_realistic_requests(num_programs, l1, jps, seed=42)
        print(f"{len(jps_requests)} requests")

        avg_tool = sum(r['actual_tool_duration'] for r in jps_requests if r['is_tool_call']) / len([r for r in jps_requests if r['is_tool_call']])

        print("Running Baseline (JPS)...", end=" ", flush=True)
        sim_bl_jps, _ = create_simulator(jps_requests, False, l1, l2)
        r_bl_jps = analyze_result(sim_bl_jps)
        print(f"{r_bl_jps['time']:.1f}s")

        print("Running TTL (JPS)...", end=" ", flush=True)
        sim_ttl_jps, ttl_decisions_jps = create_simulator(jps_requests, True, l1, l2)
        r_ttl_jps = analyze_result(sim_ttl_jps)
        print(f"{r_ttl_jps['time']:.1f}s")

        jps_speedup = r_bl_jps['time'] / r_ttl_jps['time'] if r_ttl_jps['time'] > 0 else 0

        print(f"\nResults:")
        print(f"  Batch: BL={r_bl_batch['time']:.1f}s, TTL={r_ttl_batch['time']:.1f}s, Speedup={batch_speedup:.2f}x, Miss BL={r_bl_batch['misses']}, TTL={r_ttl_batch['misses']}")
        print(f"  JPS:   BL={r_bl_jps['time']:.1f}s, TTL={r_ttl_jps['time']:.1f}s, Speedup={jps_speedup:.2f}x, Miss BL={r_bl_jps['misses']}, TTL={r_ttl_jps['misses']}")

        if ttl_decisions_batch:
            print(f"  Batch TTL decisions:")
            for strategy, data in sorted(ttl_decisions_batch.items()):
                if data['ttls']:
                    avg_ttl = sum(data['ttls']) / len(data['ttls'])
                    print(f"      {strategy}: {data['count']} times, avg_ttl={avg_ttl:.2f}s")

        if ttl_decisions_jps:
            print(f"  JPS TTL decisions:")
            for strategy, data in sorted(ttl_decisions_jps.items()):
                if data['ttls']:
                    avg_ttl = sum(data['ttls']) / len(data['ttls'])
                    print(f"      {strategy}: {data['count']} times, avg_ttl={avg_ttl:.2f}s")

        results.append({
            'l1': l1,
            'batch': {'bl': r_bl_batch, 'ttl': r_ttl_batch, 'speedup': batch_speedup},
            'jps': {'bl': r_bl_jps, 'ttl': r_ttl_jps, 'speedup': jps_speedup},
        })

    return results


def print_summary(results):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'L1':>6} | {'Programs':>9} | {'JPS':>5} | {'BL Time':>10} | {'TTL Time':>10} | {'Speedup':>8} | {'Miss_BL':>8} | {'Miss_TTL':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['l1']:>6} | {r['num_programs']:>9} | {r['jps']:>5.1f} | {r['baseline']['time']:>9.1f}s | {r['ttl']['time']:>9.1f}s | {r['speedup']:>7.2f}x | {r['baseline']['misses']:>8} | {r['ttl']['misses']:>8}")

    # Find best config
    best = max(results, key=lambda x: x['speedup'])
    worst = min(results, key=lambda x: x['speedup'])

    print(f"\nBest config: L1={best['l1']}, P={best['num_programs']}, JPS={best['jps']} -> {best['speedup']:.2f}x")
    print(f"Worst config: L1={worst['l1']}, P={worst['num_programs']}, JPS={worst['jps']} -> {worst['speedup']:.2f}x")


# ============================================================================
# Main
# ============================================================================

def test_cache_size_sweep():
    """Test different L1 cache sizes with L2 = 2 * L1."""
    print("=" * 80)
    print("Cache Size Sweep - L1 varying, L2 = 2 * L1")
    print("=" * 80)

    results = []

    num_programs = 250
    jps = 1.0
    k = 10  # history threshold
    default_ttl = 10.0

    # Generate requests once
    print("Generating requests...", end=" ", flush=True)
    requests = generate_realistic_requests(
        num_programs=num_programs,
        l1=4000,
        jps=jps,
        seed=42,
        vary_turns=True,
        vary_tool_times=True,
        vary_prefill=True
    )
    print(f"{len(requests)} requests")

    # L1 cache sizes to test (L2 = 2 * L1)
    l1_sizes = [2000, 3000, 4000, 5000, 6000, 8000]

    for l1 in l1_sizes:
        l2 = l1 * 2
        print(f"\n--- L1={l1}, L2={l2} ---")

        # Baseline
        sim_bl, _ = create_simulator(requests, False, l1, l2, k)
        r_bl = analyze_result(sim_bl)

        # TTL
        ttl_config = {'default_ttl': default_ttl}
        sim_ttl, ttl_decisions = create_simulator(requests, True, l1, l2, k, ttl_config)
        r_ttl = analyze_result(sim_ttl)

        speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
        miss_reduction = r_bl['misses'] - r_ttl['misses']

        print(f"  BL: {r_bl['time']:.1f}s, Miss={r_bl['misses']}, L1Hit={r_bl['l1_hits']}, L2Hit={r_bl['l2_hits']}")
        print(f"  TTL: {r_ttl['time']:.1f}s, Miss={r_ttl['misses']}, TTLHit={r_ttl['ttl_hits']}, L1Hit={r_ttl['l1_hits']}")
        print(f"  Speedup: {speedup:.3f}x, Miss reduction: {miss_reduction}")

        results.append({
            'l1': l1,
            'l2': l2,
            'baseline': r_bl,
            'ttl': r_ttl,
            'speedup': speedup,
            'miss_reduction': miss_reduction,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'L1':>6} | {'L2':>6} | {'BL Time':>10} | {'TTL Time':>10} | {'Speedup':>8} | {'Miss_BL':>8} | {'Miss_TTL':>8} | {'TTL Hits':>9}")
    print("-" * 80)

    for r in results:
        print(f"{r['l1']:>6} | {r['l2']:>6} | {r['baseline']['time']:>9.1f}s | {r['ttl']['time']:>9.1f}s | {r['speedup']:>7.3f}x | {r['baseline']['misses']:>8} | {r['ttl']['misses']:>8} | {r['ttl']['ttl_hits']:>9}")

    best = max(results, key=lambda x: x['speedup'])
    print(f"\nBest: L1={best['l1']}, L2={best['l2']} with {best['speedup']:.3f}x speedup")

    return results


def test_config_sweep():
    """Test different TTL configurations."""
    print("=" * 80)
    print("TTL Configuration Sweep")
    print("=" * 80)

    results = []

    num_programs = 250
    jps = 1.0

    # Generate requests once
    print("Generating requests...", end=" ", flush=True)
    requests = generate_realistic_requests(
        num_programs=num_programs,
        l1=8000,
        jps=jps,
        seed=42,
        vary_turns=True,
        vary_tool_times=True,
        vary_prefill=True
    )
    print(f"{len(requests)} requests")

    configs = [
        # (name, history_threshold, default_ttl)
        ("K=5, TTL=5", 5, 5.0),
        ("K=5, TTL=10", 5, 10.0),
        ("K=5, TTL=15", 5, 15.0),
        ("K=10, TTL=5", 10, 5.0),
        ("K=10, TTL=10", 10, 10.0),
        ("K=10, TTL=15", 10, 15.0),
        ("K=20, TTL=5", 20, 5.0),
        ("K=20, TTL=10", 20, 10.0),
        ("K=20, TTL=15", 20, 15.0),
        ("K=30, TTL=10", 30, 10.0),
        ("K=30, TTL=15", 30, 15.0),
    ]

    for name, k, default_ttl in configs:
        print(f"\n--- {name} ---")

        # Baseline
        sim_bl, _ = create_simulator(requests, False, 8000, 12000, k)
        r_bl = analyze_result(sim_bl)

        # TTL
        ttl_config = {'default_ttl': default_ttl}
        sim_ttl, ttl_decisions = create_simulator(requests, True, 8000, 12000, k, ttl_config)
        r_ttl = analyze_result(sim_ttl)

        speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0

        print(f"  BL: {r_bl['time']:.1f}s, Miss={r_bl['misses']}")
        print(f"  TTL: {r_ttl['time']:.1f}s, Miss={r_ttl['misses']}")
        print(f"  Speedup: {speedup:.3f}x")

        # TTL decision stats
        if ttl_decisions:
            for strategy, data in sorted(ttl_decisions.items()):
                if data['ttls']:
                    avg_ttl = sum(data['ttls']) / len(data['ttls'])
                    print(f"  {strategy}: {data['count']} times, avg_ttl={avg_ttl:.2f}s")

        results.append({
            'name': name,
            'k': k,
            'default_ttl': default_ttl,
            'baseline': r_bl,
            'ttl': r_ttl,
            'speedup': speedup,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} | {'BL Time':>10} | {'TTL Time':>10} | {'Speedup':>8} | {'Miss_BL':>8} | {'Miss_TTL':>8}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<20} | {r['baseline']['time']:>9.1f}s | {r['ttl']['time']:>9.1f}s | {r['speedup']:>7.3f}x | {r['baseline']['misses']:>8} | {r['ttl']['misses']:>8}")

    best = max(results, key=lambda x: x['speedup'])
    print(f"\nBest: {best['name']} with {best['speedup']:.3f}x speedup")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realistic TTL test with JPS")
    parser.add_argument('--mode', type=str, default='cache_sweep',
                       choices=['jps_sweep', 'batch_vs_jps', 'single', 'config_sweep', 'cache_sweep'],
                       help='Test mode')
    parser.add_argument('--l1', type=int, default=8000, help='L1 cache size')
    parser.add_argument('--programs', type=int, default=100, help='Number of programs')
    parser.add_argument('--jps', type=float, default=1.0, help='Jobs per second')
    args = parser.parse_args()

    print("=" * 80)
    print("REALISTIC TTL TEST - SweBench/BCFL Workload Patterns")
    print("=" * 80)
    print()
    print("Features:")
    print("  - Variable tool execution times (0.1s - 15s)")
    print("  - Different tool types (file_read, bash, api_call, etc.)")
    print("  - Variable program length (3-20 turns)")
    print("  - Variable prefill sizes (100-300 base)")
    print("  - JPS-based staggered arrival")
    print("  - Poisson process for realistic load")
    print()

    if args.mode == 'jps_sweep':
        results = test_jps_sweep()
        print_summary(results)
    elif args.mode == 'batch_vs_jps':
        results = test_batch_vs_jps()
    elif args.mode == 'single':
        test_realistic_scenario(args.l1, args.programs, args.jps)
    elif args.mode == 'config_sweep':
        results = test_config_sweep()
    elif args.mode == 'cache_sweep':
        results = test_cache_size_sweep()
