#!/usr/bin/env python3
"""
Sweep test for L1 cache size vs programs, comparing Baseline + L2 vs Adaptive TTL + L2.
"""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any

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

def generate_requests(programs: int, turns: int, prefill: int, tool_time: float, seed: int = 42) -> List[Dict]:
    """Generate requests for simulation."""
    random.seed(seed)
    requests = []
    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []
        for turn in range(turns):
            if turn == 0:
                current_prefill = int(prefill * 5)
            elif turn == 1:
                current_prefill = int(prefill * 3)
            elif turn == 2:
                current_prefill = int(prefill * 2)
            else:
                current_prefill = prefill
            prev_total = len(accumulated)
            new_tokens = list(range(prog_idx * 1000000 + prev_total, prog_idx * 1000000 + prev_total + current_prefill))
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


def create_simulator(requests, enable_ttl, l1, l2, tool_time=2.0, history_threshold=3):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)
        sim = KVCacheSimulator(
            cache_capacity=l1, l2_capacity=l2, l2_enabled=l2 > 0,
            l2_reload_penalty=0.3, default_ttl=5.0, min_ttl=0.5, max_ttl=60.0,
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
    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_time = log[-1]['finish_time'] if log else 0
    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_decode_ms = sum(e.get('decode_ms', 0) for e in log)
    total_h2d_ms = sum(e.get('h2d_ms', 0) for e in log)
    # Use queue_wait_ms (actual wait time from start_time - queue_start_time)
    # instead of queue_delay_ms (which was incorrectly estimated as num_waiting * per_request)
    total_queue_delay_ms = sum(e.get('queue_wait_ms', 0) for e in log)
    return {
        'time': total_time,
        'ttl_hits': ttl_hits, 'l1_hits': l1_hits, 'l2_hits': l2_hits, 'misses': misses,
        'prefill_s': total_prefill_ms / 1000, 'decode_s': total_decode_ms / 1000,
        'h2d_s': total_h2d_ms / 1000, 'queue_delay_s': total_queue_delay_ms / 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1-range', type=int, nargs=3, default=[2000, 6000, 2000])
    parser.add_argument('--programs-range', type=int, nargs=3, default=[10, 30, 10])
    parser.add_argument('--turns', type=int, default=15)
    parser.add_argument('--prefill', type=int, default=200)
    parser.add_argument('--l2-ratio', type=float, default=1.5)
    parser.add_argument('--tool-time', type=float, default=2.0)
    parser.add_argument('--history-threshold', type=int, default=3)
    args = parser.parse_args()

    l1_values = range(args.l1_range[0], args.l1_range[1] + 1, args.l1_range[2])
    programs_values = range(args.programs_range[0], args.programs_range[1] + 1, args.programs_range[2])

    # Calculate total tokens per program
    def calc_turn_tokens(turn):
        if turn == 0: return args.prefill * 5
        elif turn == 1: return args.prefill * 3
        elif turn == 2: return args.prefill * 2
        else: return args.prefill
    tokens_per_prog = sum(calc_turn_tokens(t) for t in range(args.turns))
    total_tokens_all = tokens_per_prog * args.programs_range[1]  # max programs

    print("=" * 140)
    print(f"SWEEP: L1 Cache Size vs Programs - Baseline + L2 vs Adaptive TTL + L2")
    print("=" * 140)
    print(f"Context: turns={args.turns}, prefill={args.prefill}, tool_time={args.tool_time}s, history_threshold={args.history_threshold}")
    print(f"Tokens: {tokens_per_prog}/program, {total_tokens_all} total (max {args.programs_range[1]} programs)")
    print()

    all_results = []

    for l1 in l1_values:
        l2 = int(l1 * args.l2_ratio)
        for P in programs_values:
            print(f"Testing L1={l1}, L2={l2}, P={P}...", end=" ", flush=True)

            requests = generate_requests(P, args.turns, args.prefill, args.tool_time)

            # Baseline
            sim_bl = create_simulator(requests, False, l1, l2, args.tool_time, args.history_threshold)
            r_bl = analyze_result(sim_bl)
            print(f"BL={r_bl['time']:.1f}s", end=" ", flush=True)

            # TTL
            sim_ttl = create_simulator(requests, True, l1, l2, args.tool_time, args.history_threshold)
            r_ttl = analyze_result(sim_ttl)
            print(f"TTL={r_ttl['time']:.1f}s", end=" ", flush=True)

            speedup = r_bl['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
            print(f"({speedup:.2f}x)")
            all_results.append({
                'l1': l1, 'l2': l2, 'programs': P,
                'baseline': r_bl, 'ttl': r_ttl, 'speedup': speedup
            })

    # Print summary table
    print()
    print("=" * 140)
    print("SUMMARY TABLE")
    print("=" * 140)
    print()
    print(f"Config: T={args.turns}, prefill={args.prefill}, tool={args.tool_time}s, hist={args.history_threshold} | Tokens/prog={tokens_per_prog}")
    print()
    print(f"{'L1':>6} | {'L2':>6} | {'P':>4} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'Miss_BL':>7} | {'Miss_TTL':>7} | {'Prefill_BL':>10} | {'Prefill_TTL':>10} | {'Queue_BL':>9} | {'Queue_TTL':>9}")
    print("-" * 140)

    for r in all_results:
        bl = r['baseline']
        ttl = r['ttl']
        miss_reduction = bl['misses'] - ttl['misses']
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.2f}x"
        print(f"{r['l1']:>6} | {r['l2']:>6} | {r['programs']:>4} | {bl['time']:>9.1f}s | {ttl['time']:>9.1f}s | {sp_str:>8} | {bl['misses']:>7} | {ttl['misses']:>7} | {bl['prefill_s']:>9.1f}s | {ttl['prefill_s']:>9.1f}s | {bl['queue_delay_s']:>8.1f}s | {ttl['queue_delay_s']:>8.1f}s")

    print()
    print("KEY INSIGHTS:")
    print("-" * 50)
    # Group by programs
    for P in programs_values:
        subset = [r for r in all_results if r['programs'] == P]
        best = max(subset, key=lambda x: x['speedup'])
        worst = min(subset, key=lambda x: x['speedup'])
        print(f"P={P}: Best speedup at L1={best['l1']} ({best['speedup']:.2f}x), Worst at L1={worst['l1']} ({worst['speedup']:.2f}x)")


if __name__ == "__main__":
    main()
