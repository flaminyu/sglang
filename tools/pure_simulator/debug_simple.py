#!/usr/bin/env python3
"""
Simple debug: print scheduling log with detailed info.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os


def simple_debug():
    """Simple debug."""
    print("="*70)
    print("SIMPLE DEBUG")
    print("="*70)
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=1.0,
        decode_latency_per_token_ms=5.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=10.0,
        queue_delay_per_request_ms=10.0,
    )
    
    # Very simple scenario
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
            "rid": "prog_000_turn_1",
            "program_id": "prog_000",
            "turn_index": 1,
            "token_ids": list(range(0, 200)),  # Same prefix + 100 new
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
            cache_capacity=500,
            default_ttl=10.0,
            enable_ttl=True,
            simulation_mode='event_driven',
            tool_execution_time=0.5,
            random_seed=42,
            hardware_config=hw,
        )
        sim.load_requests(temp_path)
        sim.scheduler.program_total_turns["prog_000"] = 2
        sim.run(verbose=False)
        
        print("\nScheduling log:")
        for e in sim.scheduler.scheduling_log:
            print(f"\n  {e['rid']}:")
            print(f"    input_len={e['input_len']}")
            print(f"    matched_tokens={e.get('matched_tokens', 'N/A')}")
            print(f"    hit_type={e['hit_type']}")
            print(f"    prefill_ms={e['prefill_ms']}")
        
        # Check radix tree state
        print("\n\nRadix tree state:")
        radix = sim.scheduler.radix_tree
        print(f"  Node count: {len(radix.nodes)}")
        print(f"  Token count: {radix.token_count}")
        
        for node_id, node in list(radix.nodes.items())[:5]:
            print(f"\n  Node {node_id}:")
            print(f"    key tokens: {node.key.token_ids[:10]}...")
            print(f"    kv_indices: {node.kv_indices[:10]}...")
            print(f"    is_pinned: {node.is_pinned}")
    
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    simple_debug()
