#!/usr/bin/env python3
"""
Paper scenario tests with realistic A100 cache sizes.

A100 80GB configuration:
- Model: 8B (FP16) = 16GB
- Remaining for KV Cache: ~64GB
- L2 (DRAM): 100GB for A100

KV cache size per token depends on model:
- Llama 8B: ~512 bytes per token (2K head, 32 layers)
- 64GB = 64 * 1024^3 / 512 ≈ 134M tokens max L1
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os


def run_comparison(name, args):
    """Run TTL vs No TTL comparison."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    
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
                "arrival_time": 0.0,  # All arrive at once
                "is_tool_call": is_tool,
                "tool_name": f"tool_{turn % 3}",
                "extra_key": program_id,
                "ttl_sec": args['ttl'] if args['ttl'] > 0 else None,
                "actual_tool_duration": rng.expovariate(1.0 / args['tool_time']) if is_tool else 0.0,
            })
            accumulated.extend(new_tokens)
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=args['prefill_ms'],
        decode_latency_per_token_ms=args['decode_ms'],
        cache_hit_speedup=10.0,
        ttft_overhead_ms=100.0,
        queue_delay_per_request_ms=100.0,
    )
    
    # Run with TTL
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    
    try:
        sim_ttl = KVCacheSimulator(
            cache_capacity=args['cache'],
            l2_capacity=args.get('l2_cache', 0),
            l2_enabled=args.get('l2_cache', 0) > 0,
            default_ttl=args['ttl'],
            enable_ttl=args['ttl'] > 0,
            simulation_mode='event_driven',
            tool_execution_time=args['tool_time'],
            random_seed=args['seed'],
            hardware_config=hw,
            l2_reload_penalty=args.get('l2_reload_penalty', 0.3),
        )
        sim_ttl.load_requests(temp_path)
        for pid in sim_ttl.scheduler.program_total_turns:
            sim_ttl.scheduler.program_total_turns[pid] = args['turns']
        sim_ttl.run(verbose=False)
        
        # Run without TTL
        sim_no_ttl = KVCacheSimulator(
            cache_capacity=args['cache'],
            l2_capacity=args.get('l2_cache', 0),
            l2_enabled=args.get('l2_cache', 0) > 0,
            enable_ttl=False,  # Disable TTL
            simulation_mode='event_driven',
            tool_execution_time=args['tool_time'],
            random_seed=args['seed'],
            hardware_config=hw,
            l2_reload_penalty=args.get('l2_reload_penalty', 0.3),
        )
        sim_no_ttl.load_requests(temp_path)
        for pid in sim_no_ttl.scheduler.program_total_turns:
            sim_no_ttl.scheduler.program_total_turns[pid] = args['turns']
        sim_no_ttl.run(verbose=False)
        
        # Compute JCT
        ttl_jcts = sim_ttl.collector.compute_program_jct_stats(sim_ttl.scheduler.scheduling_log)
        no_ttl_jcts = sim_no_ttl.collector.compute_program_jct_stats(sim_no_ttl.scheduler.scheduling_log)
        
        # Cache stats
        ttl_result = sim_ttl.scheduler.radix_tree.stats
        no_ttl_result = sim_no_ttl.scheduler.radix_tree.stats
        
        # Time breakdown
        ttl_log = sim_ttl.scheduler.scheduling_log
        no_ttl_log = sim_no_ttl.scheduler.scheduling_log
        
        ttl_prefill = sum(e['prefill_ms'] for e in ttl_log)
        no_ttl_prefill = sum(e['prefill_ms'] for e in no_ttl_log)
        
        ttl_pinned_waits = [e['queue_wait_ms'] for e in ttl_log if e.get('is_pinned')]
        ttl_unpinned_waits = [e['queue_wait_ms'] for e in ttl_log if not e.get('is_pinned')]
        no_ttl_waits = [e['queue_wait_ms'] for e in no_ttl_log]
        
        import statistics
        avg_speedup = no_ttl_jcts['avg_jct'] / ttl_jcts['avg_jct'] if ttl_jcts['avg_jct'] > 0 else 1.0
        
        print(f"\nConfig: {args['programs']} programs × {args['turns']} turns")
        print(f"  Prefill: {args['prefill']} tokens/turn")
        print(f"  Cache: L1={args['cache']:,} ({args['cache']*512/1e9:.2f}GB), L2={args.get('l2_cache', 0):,}")
        print(f"  TTL: {args['ttl']}s")
        
        print(f"\n{'Metric':>25} | {'No TTL':>12} | {'With TTL':>12} | {'Speedup':>10}")
        print("-" * 65)
        print(f"{'Avg JCT (s)':>25} | {no_ttl_jcts['avg_jct']/1000:>12.1f} | {ttl_jcts['avg_jct']/1000:>12.1f} | {avg_speedup:>9.2f}x")
        print(f"{'P95 JCT (s)':>25} | {no_ttl_jcts['p95_jct']/1000:>12.1f} | {ttl_jcts['p95_jct']/1000:>12.1f}")
        
        print(f"\nCache Hits:")
        print(f"  TTL Hits: No={no_ttl_result.ttl_hits}, TTL={ttl_result.ttl_hits}")
        print(f"  L1 Hits: No={no_ttl_result.l1_hits}, TTL={ttl_result.l1_hits}")
        print(f"  L2 Hits: No={no_ttl_result.l2_hits}, TTL={ttl_result.l2_hits}")
        print(f"  Misses: No={no_ttl_result.cache_misses}, TTL={ttl_result.cache_misses}")
        
        print(f"\nPrefill Time:")
        print(f"  No TTL: {no_ttl_prefill/1000:.2f}s")
        print(f"  TTL: {ttl_prefill/1000:.2f}s")
        print(f"  Saved: {(no_ttl_prefill-ttl_prefill)/1000:.2f}s")
        
        print(f"\nQueue Wait (avg ms):")
        print(f"  No TTL: {statistics.mean(no_ttl_waits):.0f}")
        if ttl_pinned_waits:
            print(f"  TTL Pinned: {statistics.mean(ttl_pinned_waits):.0f}")
        if ttl_unpinned_waits:
            print(f"  TTL Unpinned: {statistics.mean(ttl_unpinned_waits):.0f}")
        
        return {
            'name': name,
            'ttl_avg_jct': ttl_jcts['avg_jct'],
            'no_ttl_avg_jct': no_ttl_jcts['avg_jct'],
            'speedup': avg_speedup,
            'ttl_hits': ttl_result.ttl_hits,
            'l1_hits': ttl_result.l1_hits,
            'misses': ttl_result.cache_misses,
        }
    finally:
        os.unlink(temp_path)


def main():
    print("="*70)
    print("PAPER SCENARIO TESTS - A100 80GB Configuration")
    print("="*70)
    print("\nCache sizing:")
    print("  A100 memory: 80GB")
    print("  8B model (FP16): ~16GB")
    print("  Remaining: ~64GB for KV cache")
    print("  KV cache per token: ~512 bytes")
    print("  Max L1 tokens: ~134M tokens")
    print("  L2 (DRAM): 100GB for A100 = ~200M tokens")
    
    results = []
    
    # Scenario 1: Paper SWE-Bench approximation (scaled down)
    results.append(run_comparison("SWE-Bench Small", {
        'prefill': 5000,
        'decode': 64,
        'programs': 15,
        'turns': 8,
        'tool_time': 0.5,
        'cache': 50_000,  # 50K tokens ≈ 26MB
        'l2_cache': 200_000,  # 200K tokens ≈ 100MB
        'ttl': 3.0,
        'prefill_ms': 2.0,
        'decode_ms': 20.0,
        'seed': 42,
        'l2_reload_penalty': 0.3,
    }))
    
    # Scenario 2: Higher pressure
    results.append(run_comparison("High Pressure", {
        'prefill': 3000,
        'decode': 64,
        'programs': 20,
        'turns': 10,
        'tool_time': 0.3,
        'cache': 30_000,  # 30K tokens
        'l2_cache': 150_000,
        'ttl': 2.0,
        'prefill_ms': 2.0,
        'decode_ms': 20.0,
        'seed': 42,
        'l2_reload_penalty': 0.3,
    }))
    
    # Scenario 3: Very high pressure
    results.append(run_comparison("Very High Pressure", {
        'prefill': 2000,
        'decode': 64,
        'programs': 30,
        'turns': 12,
        'tool_time': 0.2,
        'cache': 20_000,  # 20K tokens
        'l2_cache': 100_000,
        'ttl': 1.5,
        'prefill_ms': 2.0,
        'decode_ms': 20.0,
        'seed': 42,
        'l2_reload_penalty': 0.3,
    }))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Scenario':>25} | {'TTL Hits':>10} | {'L1 Hits':>10} | {'Speedup':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:>25} | {r['ttl_hits']:>10} | {r['l1_hits']:>10} | {r['speedup']:>9.2f}x")


if __name__ == "__main__":
    main()
