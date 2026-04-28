#!/usr/bin/env python3
"""
Trace matched_tokens through the pipeline.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os

# Patch _process_request
original_process = None

def traced_process(self, req, current_time, ttl_sec=None, tool_name=None):
    from simulator.radix_key import RadixKey
    
    req.queue_end_time = current_time
    
    key = RadixKey(token_ids=req.token_ids, extra_key=req.extra_key)
    match_result = self.radix_tree.match_prefix(key)
    
    matched_tokens = match_result.matched_tokens
    new_tokens = len(req.token_ids) - matched_tokens
    
    print(f"\n  [PROCESS] {req.rid}:")
    print(f"    token_ids: [{req.token_ids[0]}...{req.token_ids[-1]}] len={len(req.token_ids)}")
    print(f"    matched_tokens (from match): {matched_tokens}")
    print(f"    new_tokens: {new_tokens}")
    
    result = original_process(self, req, current_time, ttl_sec, tool_name)
    
    print(f"    result.matched_tokens: {result.matched_tokens}")
    print(f"    result.new_tokens: {result.new_tokens}")
    
    return result


def trace_pipeline():
    global original_process
    
    from simulator import scheduler as sched_module
    original_process = sched_module.RequestScheduler._process_request
    sched_module.RequestScheduler._process_request = traced_process
    
    print("="*70)
    print("PIPELINE TRACE")
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
            print(f"\n{e['rid']}:")
            print(f"  matched_tokens (in log): {e.get('matched_tokens', 'N/A')}")
            print(f"  hit_type: {e['hit_type']}")
    
    finally:
        sched_module.RequestScheduler._process_request = original_process
        os.unlink(temp_path)


if __name__ == "__main__":
    trace_pipeline()
