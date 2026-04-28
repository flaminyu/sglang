#!/usr/bin/env python3
"""
Quick Start Example

Minimal example to get started with the simulator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator import KVCacheSimulator
import json


def main():
    # 1. Create some test requests
    requests = [
        {
            "rid": f"req_{i}",
            "program_id": "my_program",
            "turn_index": i,
            "token_ids": list(range(100 + i * 50)),  # Growing context
            "input_len": 100 + i * 50,
            "output_len": 50,
            "arrival_time": float(i),
            "is_tool_call": i < 9,  # All except last
            "tool_name": "tool" if i < 9 else None,
            "extra_key": "__continuum_prog=my_program__ttl=5.0",
        }
        for i in range(10)
    ]
    
    # 2. Save to file
    with open('/tmp/quick_start_requests.jsonl', 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print("Created 10 requests with growing context")
    
    # 3. Create and run simulator
    sim = KVCacheSimulator(
        cache_capacity=2000,  # 2000 tokens
        default_ttl=5.0,     # 5 second TTL
        eviction_policy="lru",
        enable_ttl=True,
    )
    
    sim.load_requests('/tmp/quick_start_requests.jsonl')
    result = sim.run()
    
    # 4. Print results
    print(f"\nResults:")
    print(f"  Hit Rate: {result.cache_stats.hit_rate:.1%}")
    print(f"  Cache Hits: {result.cache_stats.cache_hits}")
    print(f"  Cache Misses: {result.cache_stats.cache_misses}")
    print(f"  Tokens Cached: {result.cache_stats.tokens_cached}")
    print(f"  TTL Pins: {result.cache_stats.ttl_pins}")
    
    # 5. Compare with baseline (no TTL)
    print("\n--- Baseline (no TTL) ---")
    sim_base = KVCacheSimulator(
        cache_capacity=2000,
        default_ttl=0.0,
        eviction_policy="lru",
        enable_ttl=False,
    )
    sim_base.load_requests('/tmp/quick_start_requests.jsonl')
    result_base = sim_base.run()
    
    print(f"  Hit Rate: {result_base.cache_stats.hit_rate:.1%}")
    print(f"  Cache Hits: {result_base.cache_stats.cache_hits}")
    
    # 6. Summary
    print("\n--- Summary ---")
    ttl_improvement = (result.cache_stats.hit_rate - result_base.cache_stats.hit_rate) * 100
    if ttl_improvement > 0:
        print(f"TTL improves hit rate by {ttl_improvement:.1f}%")
    else:
        print("No improvement in this scenario")


if __name__ == "__main__":
    main()
