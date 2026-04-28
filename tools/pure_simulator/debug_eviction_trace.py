#!/usr/bin/env python3
"""
Trace eviction with more detail.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os

# Patch evict to add debug
original_evict = None

def traced_evict(self, num_tokens, current_time=0.0, current_program_id=None):
    print(f"\n  [EVICT] num_tokens={num_tokens}, current_program={current_program_id}")
    
    # Get state before eviction
    all_leaves = self._get_all_leaf_nodes()
    print(f"  [EVICT] Before: {len(all_leaves)} leaves")
    for n in all_leaves:
        print(f"    Node {n.node_id}: program={n.program_id}, pinned={n.is_pinned}, tokens={len(n.kv_indices)}")
    
    result = original_evict(self, num_tokens, current_time, current_program_id)
    
    # Get state after eviction
    all_leaves_after = self._get_all_leaf_nodes()
    print(f"  [EVICT] After: {len(all_leaves_after)} leaves")
    for n in result:
        print(f"    EVICTED: Node {n.node_id}: program={n.program_id}, pinned={n.is_pinned}")
    
    return result


def trace_eviction():
    """Trace eviction events."""
    global original_evict
    
    from simulator import radix_tree
    original_evict = radix_tree.RadixTree.evict
    radix_tree.RadixTree.evict = traced_evict
    
    print("="*70)
    print("EVICTION TRACE")
    print("="*70)
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=1.0,
        decode_latency_per_token_ms=5.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=10.0,
        queue_delay_per_request_ms=10.0,
    )
    
    # Two programs
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
            cache_capacity=150,  # Small cache
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
        radix_tree.RadixTree.evict = original_evict
        os.unlink(temp_path)


if __name__ == "__main__":
    trace_eviction()
