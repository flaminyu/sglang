#!/usr/bin/env python3
"""
Deep analysis: Why doesn't queue wait difference translate to JCT improvement?

The key insight: JCT = finish_time - arrival_time
If finish_time is dominated by prefill time (which is the same), 
then queue wait savings don't matter.
"""

import sys
sys.path.insert(0, '.')

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig
import json
import tempfile
import random
import os


def analyze_jct_components():
    """Analyze what determines JCT."""
    print("="*70)
    print("JCT COMPONENT ANALYSIS")
    print("="*70)
    
    args = {
        'prefill': 3000,
        'decode': 64,
        'programs': 10,
        'turns': 6,
        'tool_time': 0.5,
        'cache': 15_000,
        'ttl': 2.0,
        'prefill_ms': 2.0,
        'decode_ms': 20.0,
        'seed': 42,
    }
    
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
    
    hw = HardwareConfig(
        prefill_latency_per_token_ms=args['prefill_ms'],
        decode_latency_per_token_ms=args['decode_ms'],
        cache_hit_speedup=10.0,
        ttft_overhead_ms=100.0,
        queue_delay_per_request_ms=100.0,
    )
    
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
        
        # Run without TTL
        sim_no_ttl = KVCacheSimulator(
            cache_capacity=args['cache'],
            enable_ttl=False,
            simulation_mode='event_driven',
            tool_execution_time=args['tool_time'],
            random_seed=args['seed'],
            hardware_config=hw,
        )
        sim_no_ttl.load_requests(temp_path)
        for pid in sim_no_ttl.scheduler.program_total_turns:
            sim_no_ttl.scheduler.program_total_turns[pid] = args['turns']
        sim_no_ttl.run(verbose=False)
        
        # Analyze per-request components
        print(f"\n{'='*70}")
        print("PER-PROGRAM BREAKDOWN")
        print(f"{'='*70}")
        
        print(f"\n{'Program':>12} | {'No TTL JCT':>12} | {'TTL JCT':>12} | {'Diff':>10}")
        print("-" * 52)
        
        for pid in sorted(sim_ttl.scheduler.program_first_arrival.keys()):
            ttl_entries = [e for e in sim_ttl.scheduler.scheduling_log if e['program_id'] == pid]
            no_ttl_entries = [e for e in sim_no_ttl.scheduler.scheduling_log if e['program_id'] == pid]
            
            ttl_jct = (sim_ttl.scheduler.program_last_finish[pid] - 
                      sim_ttl.scheduler.program_first_arrival[pid]) * 1000
            no_ttl_jct = (sim_no_ttl.scheduler.program_last_finish[pid] - 
                         sim_no_ttl.scheduler.program_first_arrival[pid]) * 1000
            
            diff = no_ttl_jct - ttl_jct
            print(f"{pid:>12} | {no_ttl_jct:>12.1f} | {ttl_jct:>12.1f} | {diff:>+10.1f}")
        
        # Per-request timing analysis
        print(f"\n{'='*70}")
        print("PER-REQUEST TIMING COMPONENTS (first 10 requests)")
        print(f"{'='*70}")
        
        print(f"\n{'RID':>20} | {'Prefill':>10} | {'Decode':>10} | {'Queue':>10} | {'TTL Hit':>8}")
        print("-" * 65)
        
        for i, (ttl_e, no_e) in enumerate(zip(sim_ttl.scheduler.scheduling_log[:10], 
                                                sim_no_ttl.scheduler.scheduling_log[:10])):
            print(f"{ttl_e['rid']:>20} | {ttl_e['prefill_ms']:>10.0f} | {ttl_e['decode_ms']:>10.0f} | "
                  f"{ttl_e['queue_wait_ms']:>10.0f} | {ttl_e['hit_type']:>8}")
        
        # Total time analysis
        ttl_log = sim_ttl.scheduler.scheduling_log
        no_ttl_log = sim_no_ttl.scheduler.scheduling_log
        
        ttl_total_prefill = sum(e['prefill_ms'] for e in ttl_log)
        ttl_total_decode = sum(e['decode_ms'] for e in ttl_log)
        ttl_total_queue = sum(e['queue_wait_ms'] for e in ttl_log)
        
        no_ttl_total_prefill = sum(e['prefill_ms'] for e in no_ttl_log)
        no_ttl_total_decode = sum(e['decode_ms'] for e in no_ttl_log)
        no_ttl_total_queue = sum(e['queue_wait_ms'] for e in no_ttl_log)
        
        print(f"\n{'='*70}")
        print("AGGREGATE TIME COMPONENTS")
        print(f"{'='*70}")
        
        print(f"\n{'Component':>20} | {'No TTL (s)':>15} | {'TTL (s)':>15} | {'Saved':>12}")
        print("-" * 67)
        print(f"{'Prefill':>20} | {no_ttl_total_prefill/1000:>15.2f} | {ttl_total_prefill/1000:>15.2f} | {(no_ttl_total_prefill-ttl_total_prefill)/1000:>+12.2f}")
        print(f"{'Decode':>20} | {no_ttl_total_decode/1000:>15.2f} | {ttl_total_decode/1000:>15.2f} | {(no_ttl_total_decode-ttl_total_decode)/1000:>+12.2f}")
        print(f"{'Queue Wait':>20} | {no_ttl_total_queue/1000:>15.2f} | {ttl_total_queue/1000:>15.2f} | {(no_ttl_total_queue-ttl_total_queue)/1000:>+12.2f}")
        
        # Why JCT doesn't improve
        print(f"\n{'='*70}")
        print("WHY JCT DOESN'T IMPROVE")
        print(f"{'='*70}")
        
        total_ttl = ttl_total_prefill + ttl_total_decode + ttl_total_queue
        total_no_ttl = no_ttl_total_prefill + no_ttl_total_decode + no_ttl_total_queue
        
        print(f"\nPrefill dominates:")
        print(f"  No TTL: Prefill={no_ttl_total_prefill/total_no_ttl*100:.1f}%, Queue={no_ttl_total_queue/total_no_ttl*100:.1f}%")
        print(f"  TTL: Prefill={ttl_total_prefill/total_ttl*100:.1f}%, Queue={ttl_total_queue/total_ttl*100:.1f}%")
        print(f"\nQueue wait savings: {(no_ttl_total_queue-ttl_total_queue)/1000:.2f}s")
        print(f"But prefills are identical, so total time is nearly unchanged!")
        
        # Check scheduling order
        print(f"\n{'='*70}")
        print("SCHEDULING ORDER COMPARISON")
        print(f"{'='*70}")
        
        ttl_order = [e['rid'] for e in ttl_log]
        no_ttl_order = [e['rid'] for e in no_ttl_log]
        
        diff_count = sum(1 for a, b in zip(ttl_order, no_ttl_order) if a != b)
        print(f"\nScheduling order differs: {diff_count}/{len(ttl_order)} requests")
        
        if diff_count > 0:
            print("\nFirst differing position:")
            for i, (a, b) in enumerate(zip(ttl_order, no_ttl_order)):
                if a != b:
                    print(f"  Position {i}: No TTL={b}, TTL={a}")
                    break

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    analyze_jct_components()
