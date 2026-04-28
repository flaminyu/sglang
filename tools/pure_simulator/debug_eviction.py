#!/usr/bin/env python3
"""
Debug why TTL isn't reducing misses.
Track eviction events to understand what's happening.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
from simulator.tree_node import TreeNode
import json
import tempfile
import random
import os


# Monkey-patch to track eviction events
original_evict = None
eviction_log = []

def patched_evict(self, num_tokens, current_time=0.0, current_program_id=None):
    global eviction_log
    all_leaves = self._get_all_leaf_nodes()
    
    # Log before eviction
    pinned_count = sum(1 for n in all_leaves if n.is_pinned and not n.check_expired(current_time))
    
    result = original_evict(self, num_tokens, current_time, current_program_id)
    
    # Log eviction
    for node in result:
        eviction_log.append({
            'time': current_time,
            'program': node.program_id,
            'tokens': len(node.kv_indices),
            'is_pinned': node.is_pinned,
            'current_program': current_program_id,
        })
    
    return result


def analyze_eviction_pattern():
    """Analyze what gets evicted."""
    print("="*70)
    print("EVICTION PATTERN ANALYSIS")
    print("="*70)
    
    global original_evict, eviction_log
    eviction_log = []
    
    # Patch the evict method
    from simulator import radix_tree
    original_evict = radix_tree.RadixTree.evict
    radix_tree.RadixTree.evict = patched_evict
    
    args = {
        'prefill': 3000,
        'decode': 64,
        'programs': 5,
        'turns': 4,
        'tool_time': 0.5,
        'cache': 8000,  # Small cache
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
        
        # Analyze eviction log
        print(f"\nTotal eviction events: {len(eviction_log)}")
        
        pinned_evicted = [e for e in eviction_log if e['is_pinned']]
        unpinned_evicted = [e for e in eviction_log if not e['is_pinned']]
        
        print(f"  Pinned nodes evicted: {len(pinned_evicted)}")
        print(f"  Unpinned nodes evicted: {len(unpinned_evicted)}")
        
        if pinned_evicted:
            print("\nPinned nodes evicted (first 10):")
            for e in pinned_evicted[:10]:
                print(f"  t={e['time']:.2f}: {e['program']} ({e['tokens']} tokens), current={e['current_program']}")
        
        # Check cache state at key moments
        print(f"\n{'='*70}")
        print("CACHE STATE ANALYSIS")
        print(f"{'='*70}")
        
        # Check which programs had their pinned nodes evicted
        programs_affected = set(e['program'] for e in pinned_evicted)
        print(f"\nPrograms whose pinned nodes were evicted: {programs_affected}")
        
        # Analyze scheduling log for these programs
        for pid in list(programs_affected)[:3]:
            entries = [e for e in sim_ttl.scheduler.scheduling_log if e['program_id'] == pid]
            print(f"\n{pid} hit types:")
            for e in entries:
                print(f"  {e['rid']}: {e['hit_type']}")
        
    finally:
        # Restore original method
        radix_tree.RadixTree.evict = original_evict
        os.unlink(temp_path)


if __name__ == "__main__":
    analyze_eviction_pattern()
