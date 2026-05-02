#!/usr/bin/env python3
"""Debug script to verify tool_name tracking."""
import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
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

def generate_simple_requests():
    requests = []
    for prog_idx in range(5):
        program_id = 'prog_{}'.format(prog_idx)
        accumulated = []
        for turn in range(3):
            new_tokens = list(range(prog_idx * 1000 + turn * 100, prog_idx * 1000 + turn * 100 + 200))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': '{}_turn_{}'.format(program_id, turn),
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': 0.0,
                'is_tool_call': turn < 2,
                'tool_name': 'tool_{}'.format(turn),
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': 2.0,
            })
            accumulated.extend(new_tokens)
    return requests

def main():
    requests = generate_simple_requests()
    print("Generated requests:")
    for r in requests:
        if r['is_tool_call']:
            print("  {}: tool={}, duration={}s".format(r['rid'], r['tool_name'], r['actual_tool_duration']))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        os.environ['DEBUG_CDF'] = '1'

        sim = KVCacheSimulator(
            cache_capacity=8000, l2_capacity=12000, l2_enabled=True,
            l2_reload_penalty=0.3,
            default_ttl=10.0, min_ttl=0.5, max_ttl=60.0,
            enable_ttl=True, enable_adaptive_ttl=True,
            simulation_mode='event_driven', poisson_lambda=1.0 / 0.3,
            tool_execution_time=2.0, random_seed=42,
            hardware_config=hw, timing_config=timing_calc, write_policy='write_through',
            history_threshold=5, ttl_grid_points=50,
        )

        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999

        print("\nRunning simulation...")
        sim.run(verbose=False)

        print("\n" + "="*60)
        print("TTL Manager State After Simulation")
        print("="*60)

        if sim.ttl_manager:
            print("\nGlobal tool durations:")
            for tool, durations in sim.ttl_manager.global_tool_durations.items():
                print("  {}: {} samples, values={}".format(tool, len(durations), durations))

            print("\nProgram stats:")
            for prog_id, stats in sim.ttl_manager.program_stats.items():
                print("  {}:".format(prog_id))
                for tool, tool_stats in stats.tool_durations.items():
                    print("    {}: {} samples".format(tool, len(tool_stats.durations)))

            print("\nAdaptive TTL history:")
            for prog_id, ttl, strategy in sim.ttl_manager.adaptive_ttl_history:
                print("  {}: TTL={:.2f}s, strategy={}".format(prog_id, ttl, strategy))

    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    main()
