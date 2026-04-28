#!/usr/bin/env python3
"""
Trace match for prog_000_turn_1.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
from simulator.radix_key import RadixKey
import json
import tempfile
import random
import os

# Patch match_prefix
original_match = None

def traced_match(self, key):
    result = original_match(self, key)
    print(f"\n  [MATCH] key=({key.extra_key}, [{key.token_ids[0]}...{key.token_ids[-1]}] len={len(key.token_ids)})")
    print(f"  [MATCH] result: hit={result.hit}, matched_tokens={result.matched_tokens}, last_node={result.last_node}")
    if result.last_node:
        print(f"  [MATCH] last_node: id={result.last_node.node_id}, key={result.last_node.key.token_ids[:5]}...")
    return result


def trace_match():
    global original_match
    
    from simulator import radix_tree
    original_match = radix_tree.RadixTree.match_prefix
    radix_tree.RadixTree.match_prefix = traced_match
    
    print("="*70)
    print("MATCH TRACE")
    print("="*70)
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=1.0,
        decode_latency_per_token_ms=5.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=10.0,
        queue_delay_per_request_ms=10.0,
    )
    
    requests = [
        {
            "rid": "prog_000_turn_0",
            "program_id": "prog_000",
            "turn_index": 0,
            "token_ids": list(range(0, 100)),
            "input_len": 100,
            "output_len": 10,
            "arrival_time": 0.0,
            "is_tool_call": True,
            "tool_name": "tool_0",
            "extra_key": "prog_000",
            "ttl_sec": 10.0,
            "actual_tool_duration": 0.5,
        },
        {
            "rid": "prog_001_turn_0",
            "program_id": "prog_001",
            "turn_index": 0,
            "token_ids": list(range(1000, 1100)),
            "input_len": 100,
            "output_len": 10,
            "arrival_time": 0.0,
            "is_tool_call": True,
            "tool_name": "tool_0",
            "extra_key": "prog_001",
            "ttl_sec": 10.0,
            "actual_tool_duration": 0.5,
        },
        {
            "rid": "prog_000_turn_1",
            "program_id": "prog_000",
            "turn_index": 1,
            "token_ids": list(range(0, 200)),
            "input_len": 200,
            "output_len": 10,
            "arrival_time": 0.0,
            "is_tool_call": False,
            "tool_name": None,
            "extra_key": "prog_000",
            "ttl_sec": 10.0,
            "actual_tool_duration": 0.0,
        },
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    
    try:
        sim = KVCacheSimulator(
            cache_capacity=150,
            default_ttl=10.0,
            enable_ttl=True,
            simulation_mode='event_driven',
            tool_execution_time=0.5,
            random_seed=42,
            hardware_config=hw,
        )
        sim.load_requests(temp_path)
        sim.scheduler.program_total_turns["prog_000"] = 2
        sim.scheduler.program_total_turns["prog_001"] = 1
        sim.run(verbose=False)
        
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)
        
        for e in sim.scheduler.scheduling_log:
            print(f"\n{e['rid']}: matched={e.get('matched_tokens', '?')}, hit={e['hit_type']}")
    
    finally:
        radix_tree.RadixTree.match_prefix = original_match
        os.unlink(temp_path)


if __name__ == "__main__":
    trace_match()
