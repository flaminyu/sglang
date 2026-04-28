#!/usr/bin/env python3
"""
Debug why misses still happen even with TTL protection.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os


def analyze_miss_reasons():
    """Understand why misses happen."""
    print("="*70)
    print("MISS REASON ANALYSIS")
    print("="*70)
    
    args = {
        'prefill': 3000,
        'decode': 64,
        'programs': 5,
        'turns': 4,
        'tool_time': 0.5,
        'cache': 8000,  # Small cache to force evictions
        'ttl': 2.0,
        'seed': 42,
    }
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=2.0,
        decode_latency_per_token_ms=20.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=100.0,
        queue_delay_per_request_ms=100.0,
    )
    
    # Generate requests
    rng = random.Random(args['seed'])
    requests = []
    
    for prog_idx in range(args['programs']):
        program_id = f"prog_{prog_idx:03d}"
        accumulated = []
        
        for turn in range(args['turns']):
            new_tokens = list(range(
                prog_idx * 100000 + turn * args['prefill'],
                prog_idx * 100000 + turn * args['prefill'] + args['prefill']
            ))
            tokens = accumulated + new_tokens
            is_tool = turn < args['turns'] - 1
            
            requests.append({
                "rid": f"{program_id}_turn_{turn}",
                "program_id": program_id,
                "turn_index": turn,
                "token_ids": tokens,
                "input_len": len(tokens),
                "output_len": args['decode'],
                "arrival_time": 0.0,
                "is_tool_call": is_tool,
                "tool_name": f"tool_{turn % 3}",
                "extra_key": program_id,
                "ttl_sec": args['ttl'],
                "actual_tool_duration": rng.expovariate(1.0 / args['tool_time']) if is_tool else 0.0,
            })
            accumulated.extend(new_tokens)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    
    try:
        # Run with TTL
        sim_ttl = KVCacheSimulator(
            cache_capacity=args['cache'],
            default_ttl=args['ttl'],
            enable_ttl=True,
            simulation_mode='event_driven',
            tool_execution_time=args['tool_time'],
            random_seed=args['seed'],
            hardware_config=hw,
        )
        sim_ttl.load_requests(temp_path)
        for pid in sim_ttl.scheduler.program_total_turns:
            sim_ttl.scheduler.program_total_turns[pid] = args['turns']
        sim_ttl.run(verbose=False)
        
        print("\nScheduling log with prefix match info:")
        print(f"\n{'RID':>20} | {'Turn':>5} | {'Total':>8} | {'Matched':>8} | {'New':>8} | {'Hit':>10}")
        print("-" * 70)
        
        for e in sim_ttl.scheduler.scheduling_log:
            total = e['input_len']
            matched = e.get('matched_tokens', 0)
            new = total - matched
            print(f"{e['rid']:>20} | {e['turn_index']:>5} | {total:>8} | {matched:>8} | {new:>8} | {e['hit_type']:>10}")
        
        # Analyze per-program prefix growth
        print(f"\n{'='*70}")
        print("PER-PROGRAM CACHE GROWTH")
        print(f"{'='*70}")
        
        for pid in sorted(sim_ttl.scheduler.program_first_arrival.keys()):
            entries = [e for e in sim_ttl.scheduler.scheduling_log if e['program_id'] == pid]
            
            print(f"\n{pid}:")
            cumulative = 0
            for e in entries:
                matched = e.get('matched_tokens', 0)
                new = e['input_len'] - matched
                cumulative += matched
                print(f"  {e['rid']}: matched={matched}, new={new}, cumulative={cumulative}, hit={e['hit_type']}")
        
        # Cache utilization
        print(f"\n{'='*70}")
        print("CACHE UTILIZATION")
        print(f"{'='*70}")
        
        alloc = sim_ttl.scheduler.radix_tree.allocator
        print(f"Cache capacity: {alloc.capacity}")
        print(f"Current used: {alloc.used()}")
        print(f"Available: {alloc.available()}")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    analyze_miss_reasons()
