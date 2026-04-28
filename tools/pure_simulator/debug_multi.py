#!/usr/bin/env python3
"""
Debug multi-program scenario.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os


def debug_multi_program():
    """Debug with multiple programs."""
    print("="*70)
    print("MULTI-PROGRAM DEBUG")
    print("="*70)
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=1.0,
        decode_latency_per_token_ms=5.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=10.0,
        queue_delay_per_request_ms=10.0,
    )
    
    # Two programs that will compete for cache
    requests = [
        # Program 0, turn 0
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
        # Program 1, turn 0
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
        # Program 0, turn 1
        {
            "rid": "prog_000_turn_1",
            "program_id": "prog_000",
            "turn_index": 1,
            "token_ids": list(range(0, 200)),  # Same prefix
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
            cache_capacity=150,  # Small cache - can only hold 1.5 programs
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
        
        print("\nScheduling log:")
        for e in sim.scheduler.scheduling_log:
            print(f"\n  {e['rid']}:")
            print(f"    input_len={e['input_len']}")
            print(f"    matched_tokens={e.get('matched_tokens', 'N/A')}")
            print(f"    hit_type={e['hit_type']}")
            print(f"    prefill_ms={e['prefill_ms']}")
        
        # Check radix tree
        print("\n\nRadix tree nodes:")
        radix = sim.scheduler.radix_tree
        for node_id, node in radix.nodes.items():
            if node.key.token_ids:  # Skip root
                print(f"  Node {node_id}: tokens={node.key.token_ids[:5]}... len={len(node.key)}, pinned={node.is_pinned}, program={node.program_id}")
        
        # Check TTL manager
        print("\n\nTTL pinned programs:")
        for pid in sim.ttl_manager.pinned_programs if sim.ttl_manager else []:
            print(f"  {pid}: pinned")
    
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    debug_multi_program()
