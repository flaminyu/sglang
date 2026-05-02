#!/usr/bin/env python3
"""
Sweep test for L1 cache size vs programs with Poisson arrival and JPS sweep.
Adds arrival pattern dimension to understand how TTL performs under different load levels.
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
import math
from typing import List, Dict, Any
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


def generate_requests_poisson(
    programs: int,
    turns: int,
    prefill: int,
    tool_time: float,
    jps: float,
    arrival_mode: str = 'poisson',
    seed: int = 42
) -> List[Dict]:
    """
    Generate requests with Poisson arrival pattern.
    
    Args:
        programs: Number of programs
        turns: Number of turns per program
        prefill: Base prefill tokens per turn
        tool_time: Tool execution time
        jps: Jobs per second (arrival rate for Poisson process)
        arrival_mode: 'poisson' for exponential inter-arrival times, 'uniform' for evenly spaced
        seed: Random seed
    """
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
        # All first turns arrive at time 0
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
    
    elif arrival_mode == 'poisson':
        # Poisson arrival: inter-arrival time ~ Exp(lambda) where lambda = jps
        # JPS = jobs per second, so mean inter-arrival = 1/jps seconds
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
            # Only the first turn of each program arrives according to Poisson process
            # Subsequent turns are triggered by tool completion
            if jps > 0:
                current_time += random.expovariate(jps)
    
    elif arrival_mode == 'batch_poisson':
        # Batch Poisson: programs arrive according to Poisson process
        # All turns of a program arrive at the same time (when program "starts")
        if jps > 0:
            mean_inter_arrival = 1.0 / jps
        else:
            mean_inter_arrival = float('inf')
        
        current_time = 0.0
        for prog in program_data:
            # All turns of this program arrive at current_time
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


def generate_requests_original(programs: int, turns: int, prefill: int, tool_time: float, seed: int = 42) -> List[Dict]:
    """Generate requests with simultaneous arrival (original behavior)."""
    return generate_requests_poisson(programs, turns, prefill, tool_time, jps=0, 
                                     arrival_mode='simultaneous', seed=seed)


def create_simulator(requests, enable_ttl, l1, l2, tool_time=2.0, history_threshold=3, jps=0.0):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)
        
        # Poisson lambda for inter-arrival times
        poisson_lambda = jps if jps > 0 else 1e9  # Very high = essentially simultaneous
        
        sim = KVCacheSimulator(
            cache_capacity=l1, l2_capacity=l2, l2_enabled=l2 > 0,
            l2_reload_penalty=0.3, default_ttl=5.0, min_ttl=0.5, max_ttl=60.0,
            enable_ttl=enable_ttl, enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven', 
            poisson_lambda=poisson_lambda,
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
        return {
            'time': 0, 'ttl_hits': 0, 'l1_hits': 0, 'l2_hits': 0, 'misses': 0,
            'prefill_s': 0, 'decode_s': 0, 'h2d_s': 0, 'queue_delay_s': 0,
        }
    
    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    
    # Total time = finish time of last request
    total_time = log[-1]['finish_time'] if log else 0
    
    # First request start time (when workload actually begins)
    first_start = log[0]['start_time'] if log else 0
    
    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_decode_ms = sum(e.get('decode_ms', 0) for e in log)
    total_h2d_ms = sum(e.get('h2d_ms', 0) for e in log)
    total_queue_delay_ms = sum(e.get('queue_wait_ms', 0) for e in log)
    
    return {
        'time': total_time,
        'first_request_time': first_start,
        'ttl_hits': ttl_hits, 'l1_hits': l1_hits, 'l2_hits': l2_hits, 'misses': misses,
        'prefill_s': total_prefill_ms / 1000, 'decode_s': total_decode_ms / 1000,
        'h2d_s': total_h2d_ms / 1000, 'queue_delay_s': total_queue_delay_ms / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description='Sweep test with Poisson arrival and JPS dimension')
    parser.add_argument('--l1-range', type=int, nargs=3, default=[6000, 6000, 2000])
    parser.add_argument('--programs-range', type=int, nargs=3, default=[10, 30, 10])
    parser.add_argument('--jps-range', type=float, nargs=3, default=[0.5, 2.0, 0.5],
                        help='Jobs per second range: start, end, step')
    parser.add_argument('--turns', type=int, default=15)
    parser.add_argument('--prefill', type=int, default=200)
    parser.add_argument('--l2-ratio', type=float, default=1.5)
    parser.add_argument('--tool-time', type=float, default=2.0)
    parser.add_argument('--history-threshold', type=int, default=3)
    parser.add_argument('--arrival-mode', type=str, default='batch_poisson',
                        choices=['simultaneous', 'poisson', 'batch_poisson'],
                        help='Arrival mode: simultaneous (all at once), poisson (each turn), batch_poisson (per program)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    l1_values = range(args.l1_range[0], args.l1_range[1] + 1, args.l1_range[2])
    programs_values = range(args.programs_range[0], args.programs_range[1] + 1, args.programs_range[2])
    
    # JPS values
    if args.jps_range[2] > 0:
        jps_values = [args.jps_range[0] + i * args.jps_range[2] 
                      for i in range(int((args.jps_range[1] - args.jps_range[0]) / args.jps_range[2]) + 1)]
        jps_values = [round(j, 2) for j in jps_values if j >= 0]
    else:
        jps_values = [args.jps_range[0]]
    jps_values = sorted(set(jps_values))

    def calc_turn_tokens(turn):
        if turn == 0: return args.prefill * 5
        elif turn == 1: return args.prefill * 3
        elif turn == 2: return args.prefill * 2
        else: return args.prefill
    tokens_per_prog = sum(calc_turn_tokens(t) for t in range(args.turns))

    print("=" * 160)
    print(f"SWEEP: L1 + Programs + JPS (Poisson Arrival) - Baseline vs Adaptive TTL")
    print("=" * 160)
    print(f"Config: T={args.turns}, prefill={args.prefill}, tool={args.tool_time}s, hist={args.history_threshold}")
    print(f"Arrival: {args.arrival_mode}, Tokens/prog={tokens_per_prog}")
    print(f"L1: {args.l1_range[0]}-{args.l1_range[1]}, Programs: {args.programs_range[0]}-{args.programs_range[1]}, JPS: {[f'{j:.1f}' for j in jps_values]}")
    print()

    all_results = []

    for l1 in l1_values:
        l2 = int(l1 * args.l2_ratio)
        for P in programs_values:
            for jps in jps_values:
                print(f"L1={l1}, P={P}, JPS={jps:.1f}...", end=" ", flush=True)

                # Generate requests with Poisson arrival
                requests = generate_requests_poisson(
                    P, args.turns, args.prefill, args.tool_time,
                    jps=jps, arrival_mode=args.arrival_mode, seed=args.seed
                )

                # Baseline
                sim_bl = create_simulator(requests, False, l1, l2, args.tool_time, args.history_threshold, jps)
                r_bl = analyze_result(sim_bl)
                print(f"BL={r_bl['time']:.1f}s", end=" ", flush=True)

                # TTL
                sim_ttl = create_simulator(requests, True, l1, l2, args.tool_time, args.history_threshold, jps)
                r_ttl = analyze_result(sim_ttl)
                print(f"TTL={r_ttl['time']:.1f}s", end=" ", flush=True)

                speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
                print(f"({speedup:.2f}x)")
                
                all_results.append({
                    'l1': l1, 'l2': l2, 'programs': P, 'jps': jps,
                    'baseline': r_bl, 'ttl': r_ttl, 'speedup': speedup
                })

    # Print summary table
    print()
    print("=" * 160)
    print("SUMMARY TABLE")
    print("=" * 160)
    print()
    print(f"Config: T={args.turns}, prefill={args.prefill}, tool={args.tool_time}s, hist={args.history_threshold}, arrival={args.arrival_mode}")
    print()

    # Pivot table: rows=L1×P, columns=JPS
    print("\n--- Speedup by L1, Programs, JPS ---")
    print(f"{'L1':>6} | {'P':>4} | {'JPS':>5} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'Miss_BL':>7} | {'Miss_TTL':>7}")
    print("-" * 80)
    
    for r in all_results:
        bl = r['baseline']
        ttl = r['ttl']
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x"
        print(f"{r['l1']:>6} | {r['programs']:>4} | {r['jps']:>5.1f} | {bl['time']:>9.1f}s | {ttl['time']:>9.1f}s | {sp_str:>8} | {bl['misses']:>7} | {ttl['misses']:>7}")

    # Grouped analysis by JPS
    print()
    print("=" * 80)
    print("KEY INSIGHTS BY JPS:")
    print("-" * 80)
    for jps in jps_values:
        subset = [r for r in all_results if r['jps'] == jps]
        if subset:
            avg_speedup = sum(r['speedup'] for r in subset) / len(subset)
            best = max(subset, key=lambda x: x['speedup'])
            worst = min(subset, key=lambda x: x['speedup'])
            print(f"JPS={jps:.1f}: Avg={avg_speedup:.2f}x, Best=L1={best['l1']}/P={best['programs']} ({best['speedup']:.2f}x), Worst=L1={worst['l1']}/P={worst['programs']} ({worst['speedup']:.2f}x)")

    print()
    print("=" * 80)
    print("KEY INSIGHTS BY Programs:")
    print("-" * 80)
    for P in programs_values:
        subset = [r for r in all_results if r['programs'] == P]
        if subset:
            avg_speedup = sum(r['speedup'] for r in subset) / len(subset)
            best = max(subset, key=lambda x: x['speedup'])
            worst = min(subset, key=lambda x: x['speedup'])
            best_jps = best['jps']
            worst_jps = worst['jps']
            print(f"P={P}: Avg={avg_speedup:.2f}x, Best=L1={best['l1']}/JPS={best_jps:.1f} ({best['speedup']:.2f}x), Worst=L1={worst['l1']}/JPS={worst_jps:.1f} ({worst['speedup']:.2f}x)")

    print()
    print("=" * 80)
    print("TTL EFFECTIVENESS HEATMAP (Speedup):")
    print("-" * 80)
    
    # Create heatmap: JPS vs Programs
    header = f"{'JPS':>6}"
    for P in programs_values:
        header += f" | {'P='+str(P):>10}"
    print(header)
    print("-" * (8 + 13 * len(programs_values)))
    
    for jps in jps_values:
        row = f"{jps:>6.1f}"
        for P in programs_values:
            # Find best L1 for this JPS/P combination
            subset = [r for r in all_results if r['jps'] == jps and r['programs'] == P]
            if subset:
                best = max(subset, key=lambda x: x['speedup'])
                speedup = best['speedup']
                if speedup >= 1.0:
                    row += f" | {speedup:>10.2f}x"
                else:
                    row += f" | {speedup:>10.2f}x*"
            else:
                row += f" | {'N/A':>10}"
        print(row)


if __name__ == "__main__":
    main()
