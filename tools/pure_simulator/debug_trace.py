#!/usr/bin/env python3
"""
Simple debug: trace what happens during match.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
from simulator.radix_key import RadixKey
from simulator.request import Request
import json
import tempfile
import random
import os


def trace_match():
    """Trace match_prefix calls."""
    print("="*70)
    print("MATCH TRACE")
    print("="*70)
    
    args = {
        'prefill': 100,  # Small for debugging
        'decode': 10,
        'programs': 2,
        'turns': 2,
        'tool_time': 0.5,
        'cache': 500,  # Small cache
        'ttl': 10.0,  # Long TTL
        'seed': 42,
    }
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=1.0,
        decode_latency_per_token_ms=5.0,
        cache_hit_speedup=10.0,
        ttft_overhead_ms=10.0,
        queue_delay_per_request_ms=10.0,
    )
    
    # Generate requests
    rng = random.Random(args['seed'])
    requests = []
    
    for prog_idx in range(args['programs']):
        program_id = f"prog_{prog_idx:03d}"
        accumulated = []
        
        for turn in range(args['turns']):
            new_tokens = list(range(
                prog_idx * 1000 + turn * args['prefill'],
                prog_idx * 1000 + turn * args['prefill'] + args['prefill']
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
                "actual_tool_duration": 0.1 if is_tool else 0.0,
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
        
        radix = sim.scheduler.radix_tree
        
        # Manually process requests
        for req_dict in requests:
            req = Request.from_dict(req_dict)
            tokens = req_dict['token_ids']
            key = RadixKey(token_ids=tokens, extra_key=req_dict['extra_key'])
            
            print(f"\n{'-'*50}")
            print(f"Request: {req.rid}")
            print(f"  Tokens: [{tokens[0]}...{tokens[-1]}] len={len(tokens)}")
            
            # Match
            match = radix.match_prefix(key)
            print(f"  Match: hit={match.hit}, matched={match.matched_tokens}")
            
            if match.last_node:
                node = match.last_node
                print(f"  Last node: id={node.node_id}, tokens={len(node.key)}, pinned={node.is_pinned}")
                print(f"    kv_indices: {node.kv_indices[:5]}... len={len(node.kv_indices)}")
            
            # Process
            result = sim.scheduler._process_request(req, current_time=0.0, ttl_sec=args['ttl'], tool_name="debug")
            print(f"  Process result: hit={result.hit}, matched={result.matched_tokens}, new={result.new_tokens}")
            
            # Check TTL
            is_ttl = sim.ttl_manager.is_pinned(req.program_id, 0.0) if sim.ttl_manager else False
            print(f"  TTL pinned: {is_ttl}")
            
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    trace_match()
