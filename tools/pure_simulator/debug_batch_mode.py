#!/usr/bin/env python3
"""
Debug Batch Mode Cache Issue

This script demonstrates and tests the cache behavior in batch mode vs single mode.
"""

import sys
sys.path.insert(0, '.')

import json
import tempfile
import os
from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator


HARDWARE_CONFIG = {
    'prefill_latency_per_token_ms': 2.0,
    'decode_latency_per_token_ms': 20.0,
    'ttft_overhead_ms': 100.0,
    'queue_delay_per_request_ms': 50.0,
    'bytes_per_token': 512.0,
    'h2d_bandwidth_GBps': 32.0,
    'bandwidth_efficiency': 0.85,
    'h2d_overhead_us': 6.67,
    'd2h_overhead_us': 4.0,
    'memory_bandwidth_GBps': 64.0,
}


def generate_simple_requests(seed=42):
    """Generate simple requests for testing."""
    import random
    random.seed(seed)
    
    requests = []
    for prog_idx in range(2):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []
        
        for turn in range(4):
            if turn == 0:
                current_prefill = 1000
            else:
                current_prefill = 200
                
            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 1000000 + prev_total,
                prog_idx * 1000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': 0.0,  # All at same time initially
                'is_tool_call': turn < 3,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': 2.0,
            })
            accumulated.extend(new_tokens)
    
    return requests


def generate_staggered_requests(seed=42, gap=2.0):
    """Generate requests with staggered arrival times."""
    import random
    random.seed(seed)
    
    requests = []
    current_time = 0.0
    
    for prog_idx in range(2):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []
        
        for turn in range(4):
            if turn == 0:
                current_prefill = 1000
                arrival = current_time
                current_time += 0.1
            else:
                current_prefill = 200
                arrival = current_time
                current_time += gap
            
            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 1000000 + prev_total,
                prog_idx * 1000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': arrival,
                'is_tool_call': turn < 3,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': 2.0,
            })
            accumulated.extend(new_tokens)
    
    return requests


def run_test(requests, enable_ttl, batch_mode, batch_size, name):
    """Run a test and return results."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name
    
    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)
        
        sim = KVCacheSimulator(
            cache_capacity=3000, l2_capacity=4500, l2_enabled=True,
            enable_ttl=enable_ttl, enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven',
            tool_execution_time=2.0, random_seed=42,
            hardware_config=hw, timing_config=timing_calc,
            history_threshold=3,
        )
        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999
        
        sim.run(enable_batch=batch_mode, batch_size=batch_size, verbose=False)
        
        log = sim.scheduler.scheduling_log
        ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
        l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
        l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
        misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
        total_time = log[-1]['finish_time'] if log else 0
        
        return {
            'name': name,
            'requests': len(log) if log else 0,
            'ttl': ttl_hits,
            'l1': l1_hits,
            'l2': l2_hits,
            'miss': misses,
            'time': total_time,
        }
    finally:
        os.unlink(temp_path)


def main():
    print("=" * 80)
    print("Debugging Batch Mode Cache Behavior")
    print("=" * 80)
    print()
    
    # Test 1: Same arrival time (all requests arrive at once)
    print("Test 1: Requests arrive at the same time (turn 0 only)")
    print("-" * 80)
    requests_same_time = []
    for prog_idx in range(3):
        program_id = f'prog_{prog_idx:03d}'
        requests_same_time.append({
            'rid': f'{program_id}_turn_0',
            'program_id': program_id,
            'turn_index': 0,
            'token_ids': list(range(prog_idx * 1000000, prog_idx * 1000000 + 1000)),
            'input_len': 1000,
            'output_len': 64,
            'arrival_time': 0.0,
            'is_tool_call': True,
            'tool_name': 'tool_0',
            'extra_key': program_id,
            'ttl_sec': None,
            'actual_tool_duration': 2.0,
        })
    
    result = run_test(requests_same_time, True, False, 16, "Same time (single mode)")
    print(f"Single mode: TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    
    result = run_test(requests_same_time, True, True, 16, "Same time (batch mode)")
    print(f"Batch mode:  TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    print()
    
    # Test 2: Staggered arrival (next turn arrives after tool time)
    print("Test 2: Staggered arrival (prog_000, 4 turns)")
    print("-" * 80)
    
    requests_staggered = []
    for turn in range(4):
        if turn == 0:
            tokens = list(range(1000))
        else:
            # Turn N includes all previous tokens
            tokens = list(range(1000 + (turn - 1) * 200))
        
        requests_staggered.append({
            'rid': f'prog_000_turn_{turn}',
            'program_id': 'prog_000',
            'turn_index': turn,
            'token_ids': tokens,
            'input_len': len(tokens),
            'output_len': 64,
            'arrival_time': turn * 2.0,  # 2 seconds apart
            'is_tool_call': turn < 3,
            'tool_name': 'tool_0',
            'extra_key': 'prog_000',
            'ttl_sec': None,
            'actual_tool_duration': 2.0,
        })
    
    result = run_test(requests_staggered, True, False, 16, "Staggered (single mode)")
    print(f"Single mode: TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    
    result = run_test(requests_staggered, True, True, 16, "Staggered (batch mode)")
    print(f"Batch mode:  TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    print()
    
    # Test 3: Full simulation with all programs
    print("Test 3: Full simulation (2 programs, 4 turns each, staggered)")
    print("-" * 80)
    
    requests_full = generate_staggered_requests(seed=42, gap=2.0)
    
    result = run_test(requests_full, False, False, 16, "Full (no TTL, single mode)")
    print(f"No TTL Single:  TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    
    result = run_test(requests_full, True, False, 16, "Full (TTL, single mode)")
    print(f"TTL Single:     TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    
    result = run_test(requests_full, False, True, 16, "Full (no TTL, batch mode)")
    print(f"No TTL Batch:   TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    
    result = run_test(requests_full, True, True, 16, "Full (TTL, batch mode)")
    print(f"TTL Batch:      TTL={result['ttl']} L1={result['l1']} L2={result['l2']} Miss={result['miss']} Time={result['time']:.2f}s")
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Key observations:")
    print("1. When all requests arrive at the same time, there's no cache to hit (expected)")
    print("2. When requests are staggered, cache should be hit in subsequent turns")
    print("3. Batch mode with batch_size > 1 may cause requests to be processed together")
    print("   when they should be processed separately (depends on arrival times)")
    print()


if __name__ == "__main__":
    main()
