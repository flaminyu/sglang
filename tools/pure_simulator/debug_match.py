#!/usr/bin/env python3
"""
Debug why matched_tokens is 0 but hit_type is TTL.
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


def analyze_match():
    """Analyze match_prefix behavior."""
    print("="*70)
    print("MATCH_PREFIX DEBUG")
    print("="*70)
    
    args = {
        'prefill': 3000,
        'decode': 64,
        'programs': 3,
        'turns': 3,
        'tool_time': 0.5,
        'cache': 6000,  # Small cache
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
        sim = KVCacheSimulator(
            cache_capacity=args['cache'],
            default_ttl=args['ttl'],
            enable_ttl=True,
            simulation_mode='event_driven',
            tool_execution_time=args['tool_time'],
            random_seed=args['seed'],
            hardware_config=hw,
        )
        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = args['turns']
        
        # Run first 5 requests manually to analyze
        print("\nManual match analysis:")
        
        radix = sim.scheduler.radix_tree
        alloc = radix.allocator
        
        for i, req_dict in enumerate(requests[:6]):
            tokens = req_dict['token_ids']
            key = RadixKey(token_ids=tokens, extra_key=req_dict['extra_key'])
            
            print(f"\n{'-'*60}")
            print(f"Request {i}: {req_dict['rid']}")
            print(f"  Token IDs: [{tokens[0]}, ..., {tokens[-1]}] ({len(tokens)} total)")
            print(f"  Extra key: {req_dict['extra_key']}")
            
            # Match
            match = radix.match_prefix(key)
            print(f"  Match result:")
            print(f"    hit={match.hit}")
            print(f"    matched_tokens={match.matched_tokens}")
            print(f"    cached_indices={len(match.cached_indices)} items")
            print(f"    last_node={match.last_node}")
            
            if match.last_node:
                node = match.last_node
                print(f"    Node info:")
                print(f"      node_id={node.node_id}")
                print(f"      kv_indices={node.kv_indices[:10] if node.kv_indices else 'EMPTY'}...")
                print(f"      num_tokens={len(node.key)}")
                print(f"      is_pinned={node.is_pinned}")
            
            # Check allocator
            print(f"  Allocator: used={alloc.used()}, available={alloc.available()}")
            
            # Run one step to update state
            if i < len(requests) - 1:
                sim.scheduler._add_request(Request.from_dict(requests[i]))
                sim.scheduler._process_request(
                    sim.scheduler.pending_queue[0],
                    current_time=0.0,
                    ttl_sec=args['ttl'],
                    tool_name=f"tool_{i % 3}"
                )
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    # Need to import Request
    import sys
    sys.path.insert(0, '.')
    from simulator.request import Request
    analyze_match()
