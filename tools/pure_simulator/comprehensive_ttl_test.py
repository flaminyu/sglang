#!/usr/bin/env python3
"""
Comprehensive TTL Speedup Test - Explore various parameter combinations to maximize TTL acceleration.
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any
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


def generate_requests(
    programs: int, turns: int, prefill: int, tool_time: float,
    jps: float = 0.0, arrival_mode: str = 'simultaneous', seed: int = 42
) -> List[Dict]:
    """Generate requests for simulation."""
    random.seed(seed)
    
    def calc_turn_tokens(turn):
        if turn == 0: return prefill * 5
        elif turn == 1: return prefill * 3
        elif turn == 2: return prefill * 2
        else: return prefill
    
    # Generate all program data first
    program_data = []
    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []
        turn_data = []
        for turn in range(turns):
            nt = calc_turn_tokens(turn)
            new_tokens = list(range(prog_idx * 1000000 + len(accumulated),
                                   prog_idx * 1000000 + len(accumulated) + nt))
            tokens = accumulated + new_tokens
            turn_data.append({
                'tokens': tokens,
                'new_tokens': new_tokens,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
            })
            accumulated.extend(new_tokens)
        program_data.append({
            'program_id': program_id,
            'turn_data': turn_data,
            'extra_key': program_id,
        })
    
    # Assign arrival times based on arrival mode
    requests = []
    if arrival_mode == 'simultaneous':
        current_time = 0.0
        for prog in program_data:
            for turn_idx, turn_data in enumerate(prog['turn_data']):
                requests.append({
                    'rid': f'{prog["program_id"]}_turn_{turn_idx}',
                    'program_id': prog['program_id'],
                    'turn_index': turn_idx,
                    'token_ids': turn_data['tokens'],
                    'input_len': len(turn_data['tokens']),
                    'output_len': 64,
                    'arrival_time': current_time,
                    'is_tool_call': turn_data['is_tool_call'],
                    'tool_name': turn_data['tool_name'],
                    'extra_key': prog['extra_key'],
                    'ttl_sec': None,
                    'actual_tool_duration': tool_time,
                })
    elif arrival_mode == 'batch_poisson':
        if jps > 0:
            mean_inter_arrival = 1.0 / jps
        else:
            mean_inter_arrival = float('inf')
        
        current_time = 0.0
        for prog in program_data:
            for turn_idx, turn_data in enumerate(prog['turn_data']):
                requests.append({
                    'rid': f'{prog["program_id"]}_turn_{turn_idx}',
                    'program_id': prog['program_id'],
                    'turn_index': turn_idx,
                    'token_ids': turn_data['tokens'],
                    'input_len': len(turn_data['tokens']),
                    'output_len': 64,
                    'arrival_time': current_time,
                    'is_tool_call': turn_data['is_tool_call'],
                    'tool_name': turn_data['tool_name'],
                    'extra_key': prog['extra_key'],
                    'ttl_sec': None,
                    'actual_tool_duration': tool_time,
                })
            if jps > 0:
                current_time += random.expovariate(jps)
    
    return requests


def create_simulator(requests, enable_ttl, l1, l2, tool_time=2.0, history_threshold=3, ttl_config=None):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)
        
        # TTL configuration
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


def test_config(l1, programs, turns, prefill, tool_time, jps, ttl_config=None, l2_ratio=1.5):
    """Test a single configuration."""
    l2 = int(l1 * l2_ratio)
    
    requests = generate_requests(
        programs, turns, prefill, tool_time,
        jps=jps, arrival_mode='batch_poisson' if jps > 0 else 'simultaneous'
    )
    
    # Baseline
    sim_bl = create_simulator(requests, False, l1, l2, tool_time, ttl_config=ttl_config)
    r_bl = analyze_result(sim_bl)
    
    # TTL
    sim_ttl = create_simulator(requests, True, l1, l2, tool_time, ttl_config=ttl_config)
    r_ttl = analyze_result(sim_ttl)
    
    speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
    
    return {
        'l1': l1, 'l2': l2, 'programs': programs, 'turns': turns,
        'prefill': prefill, 'tool_time': tool_time, 'jps': jps,
        'baseline': r_bl, 'ttl': r_ttl, 'speedup': speedup
    }


def run_extensive_tests():
    """Run extensive tests to find optimal TTL configuration."""
    print("=" * 100)
    print("EXTENSIVE TTL SPEEDUP TEST - Finding Optimal Configuration")
    print("=" * 100)
    print()
    
    results = []
    
    # =============================================
    # Test 1: Vary programs (50-100) with fixed L1
    # =============================================
    print("=" * 100)
    print("TEST 1: Programs Sweep (50-100) - Fixed L1=8000, Turns=15, Prefill=200")
    print("=" * 100)
    print()
    
    L1_FIXED = 8000
    TURNS_FIXED = 15
    PREFILL_FIXED = 200
    TOOL_TIME_FIXED = 2.0
    
    tokens_per_prog = sum([
        PREFILL_FIXED * 5,  # turn 0
        PREFILL_FIXED * 3,  # turn 1
        PREFILL_FIXED * 2,  # turn 2
        PREFILL_FIXED * 1 * (TURNS_FIXED - 3)  # remaining turns
    ])
    print(f"Tokens per program: {tokens_per_prog}")
    print(f"L1={L1_FIXED} can fit ~{L1_FIXED / tokens_per_prog:.1f} programs")
    print()
    
    for P in [50, 60, 70, 80, 90, 100]:
        print(f"Testing P={P}...", end=" ", flush=True)
        r = test_config(L1_FIXED, P, TURNS_FIXED, PREFILL_FIXED, TOOL_TIME_FIXED, 0.0)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 2: Vary L1/cache size with more programs
    # =============================================
    print("=" * 100)
    print("TEST 2: L1 Sweep (6000-16000) - Fixed P=70, Turns=15, Prefill=200")
    print("=" * 100)
    print()
    
    P_FIXED = 70
    
    for L1 in [6000, 8000, 10000, 12000, 14000, 16000]:
        print(f"Testing L1={L1}...", end=" ", flush=True)
        r = test_config(L1, P_FIXED, TURNS_FIXED, PREFILL_FIXED, TOOL_TIME_FIXED, 0.0)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 3: Vary turns (10-25) - affects token reuse
    # =============================================
    print("=" * 100)
    print("TEST 3: Turns Sweep (10-25) - Fixed L1=8000, P=50, Prefill=200")
    print("=" * 100)
    print()
    
    L1_TEST3 = 8000
    P_TEST3 = 50
    
    for T in [10, 15, 20, 25]:
        tokens = sum([
            PREFILL_FIXED * 5,
            PREFILL_FIXED * 3,
            PREFILL_FIXED * 2,
            PREFILL_FIXED * 1 * max(0, T - 3)
        ])
        print(f"Testing T={T} (tokens/prog={tokens})...", end=" ", flush=True)
        r = test_config(L1_TEST3, P_TEST3, T, PREFILL_FIXED, TOOL_TIME_FIXED, 0.0)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 4: Vary prefill (100-400) - affects token size
    # =============================================
    print("=" * 100)
    print("TEST 4: Prefill Sweep (100-400) - Fixed L1=8000, P=50, Turns=15")
    print("=" * 100)
    print()
    
    for prefill in [100, 150, 200, 300, 400]:
        tokens = sum([
            prefill * 5,
            prefill * 3,
            prefill * 2,
            prefill * 1 * 12  # 15 turns - 3
        ])
        print(f"Testing prefill={prefill} (tokens/prog={tokens})...", end=" ", flush=True)
        r = test_config(L1_TEST3, P_TEST3, 15, prefill, TOOL_TIME_FIXED, 0.0)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 5: Vary tool_time (0.5-5.0) - affects TTL expiry
    # =============================================
    print("=" * 100)
    print("TEST 5: Tool Time Sweep (0.5-5.0s) - Fixed L1=8000, P=50, Turns=15, Prefill=200")
    print("=" * 100)
    print()
    
    for tool_time in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"Testing tool_time={tool_time}s...", end=" ", flush=True)
        r = test_config(L1_TEST3, P_TEST3, 15, PREFILL_FIXED, tool_time, 0.0)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 6: TTL Configuration Sweep
    # =============================================
    print("=" * 100)
    print("TEST 6: TTL Configuration Sweep - Fixed L1=8000, P=50, Turns=15")
    print("=" * 100)
    print()
    
    ttl_configs = [
        {'default_ttl': 2.0, 'min_ttl': 0.5, 'max_ttl': 30.0, 'name': 'TTL=2s'},
        {'default_ttl': 5.0, 'min_ttl': 0.5, 'max_ttl': 60.0, 'name': 'TTL=5s'},
        {'default_ttl': 10.0, 'min_ttl': 1.0, 'max_ttl': 60.0, 'name': 'TTL=10s'},
        {'default_ttl': 30.0, 'min_ttl': 5.0, 'max_ttl': 120.0, 'name': 'TTL=30s'},
        {'default_ttl': 60.0, 'min_ttl': 10.0, 'max_ttl': 180.0, 'name': 'TTL=60s'},
    ]
    
    for ttl_cfg in ttl_configs:
        print(f"Testing {ttl_cfg['name']}...", end=" ", flush=True)
        r = test_config(L1_TEST3, P_TEST3, 15, PREFILL_FIXED, TOOL_TIME_FIXED, 0.0, ttl_config=ttl_cfg)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 7: L2 Ratio Sweep
    # =============================================
    print("=" * 100)
    print("TEST 7: L2 Ratio Sweep (1.0-2.5) - Fixed L1=8000, P=50, Turns=15")
    print("=" * 100)
    print()
    
    for l2_ratio in [1.0, 1.5, 2.0, 2.5]:
        print(f"Testing L2 ratio={l2_ratio}...", end=" ", flush=True)
        r = test_config(L1_TEST3, P_TEST3, 15, PREFILL_FIXED, TOOL_TIME_FIXED, 0.0, l2_ratio=l2_ratio)
        results.append(r)
        print(f"BL={r['baseline']['time']:.1f}s TTL={r['ttl']['time']:.1f}s Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']} Miss_TTL={r['ttl']['misses']}")
    
    print()
    
    # =============================================
    # Test 8: Combined optimal search
    # =============================================
    print("=" * 100)
    print("TEST 8: Combined Optimal Search")
    print("=" * 100)
    print()
    
    # Try different combinations that should maximize TTL benefit
    test_combos = [
        # (L1, P, T, prefill, tool_time)
        (6000, 80, 15, 200, 2.0),
        (8000, 100, 15, 200, 2.0),
        (10000, 80, 20, 200, 2.0),
        (8000, 60, 20, 200, 2.0),
        (12000, 100, 15, 200, 2.0),
        (8000, 80, 15, 150, 2.0),
        (8000, 80, 15, 300, 2.0),
        (6000, 100, 15, 200, 1.0),
        (6000, 100, 15, 200, 3.0),
        (10000, 100, 15, 200, 2.0),
        (15000, 100, 15, 200, 2.0),
        (20000, 100, 15, 200, 2.0),
        (10000, 120, 15, 200, 2.0),
        (8000, 120, 15, 200, 2.0),
    ]
    
    for L1, P, T, prefill, tool_time in test_combos:
        tokens = sum([
            prefill * 5,
            prefill * 3,
            prefill * 2,
            prefill * 1 * max(0, T - 3)
        ])
        print(f"L1={L1:5d}, P={P:3d}, T={T:2d}, prefill={prefill:3d} (tokens={tokens:5d})...", end=" ", flush=True)
        r = test_config(L1, P, T, prefill, tool_time, 0.0)
        results.append(r)
        print(f"Speedup={r['speedup']:.2f}x Miss_BL={r['baseline']['misses']:3d} Miss_TTL={r['ttl']['misses']:3d}")
    
    print()
    
    # =============================================
    # Summary
    # =============================================
    print("=" * 100)
    print("SUMMARY - TOP 10 SPEEDUP CONFIGURATIONS")
    print("=" * 100)
    print()
    
    # Sort by speedup
    sorted_results = sorted(results, key=lambda x: x['speedup'], reverse=True)
    
    print(f"{'Rank':>4} | {'L1':>6} | {'P':>4} | {'T':>3} | {'Prefill':>6} | {'Tool':>5} | {'Speedup':>8} | {'Miss_BL':>7} | {'Miss_TTL':>8}")
    print("-" * 90)
    
    for i, r in enumerate(sorted_results[:10], 1):
        miss_reduction = r['baseline']['misses'] - r['ttl']['misses']
        miss_str = f"{miss_reduction:+d}" if miss_reduction > 0 else str(miss_reduction)
        print(f"{i:>4} | {r['l1']:>6} | {r['programs']:>4} | {r['turns']:>3} | {r['prefill']:>6} | {r['tool_time']:>5.1f} | {r['speedup']:>7.2f}x | {r['baseline']['misses']:>7} | {r['ttl']['misses']:>8}")
    
    print()
    print("=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print()
    
    # Best configuration
    best = sorted_results[0]
    print(f"Best configuration: L1={best['l1']}, P={best['programs']}, T={best['turns']}, prefill={best['prefill']}")
    print(f"Speedup: {best['speedup']:.2f}x")
    print(f"Baseline time: {best['baseline']['time']:.1f}s, TTL time: {best['ttl']['time']:.1f}s")
    print(f"Misses: {best['baseline']['misses']} -> {best['ttl']['misses']} (reduction: {best['baseline']['misses'] - best['ttl']['misses']})")
    
    print()
    print("Key insights:")
    
    # Analyze patterns
    high_speedup = [r for r in results if r['speedup'] >= 1.10]
    if high_speedup:
        avg_programs = sum(r['programs'] for r in high_speedup) / len(high_speedup)
        avg_l1 = sum(r['l1'] for r in high_speedup) / len(high_speedup)
        print(f"  - High speedup (>1.10x) configs: {len(high_speedup)}")
        print(f"  - Average programs: {avg_programs:.1f}")
        print(f"  - Average L1: {avg_l1:.0f}")
    
    low_speedup = [r for r in results if r['speedup'] < 1.01]
    if low_speedup:
        avg_programs_low = sum(r['programs'] for r in low_speedup) / len(low_speedup)
        avg_l1_low = sum(r['l1'] for r in low_speedup) / len(low_speedup)
        print(f"  - Low speedup (<1.01x) configs: {len(low_speedup)}")
        print(f"  - Average programs: {avg_programs_low:.1f}")
        print(f"  - Average L1: {avg_l1_low:.0f}")


if __name__ == "__main__":
    run_extensive_tests()
