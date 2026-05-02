#!/usr/bin/env python3
"""
Test time advancement in batch mode.

This script verifies that time advancement is correct in batch mode by:
1. Creating requests with known arrival times and inference times
2. Processing them in batch mode
3. Verifying that the finish times match expected values
"""

import sys
sys.path.insert(0, '.')

import tempfile
import json
import random
from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator


HARDWARE_CONFIG = {
    'prefill_latency_per_token_ms': 2.0,
    'decode_latency_per_token_ms': 20.0,
    'ttft_overhead_ms': 0.0,  # No overhead for easier calculation
    'queue_delay_per_request_ms': 0.0,
    'bytes_per_token': 512.0,
    'h2d_bandwidth_GBps': 32.0,
    'bandwidth_efficiency': 0.85,
    'h2d_overhead_us': 6.67,
    'd2h_overhead_us': 4.0,
    'memory_bandwidth_GBps': 64.0,
}


def test_time_advancement():
    """Test that time advances correctly in batch mode."""
    print("=" * 80)
    print("TEST: Time Advancement in Batch Mode")
    print("=" * 80)

    # Create 8 requests, arriving at times 0, 0.01, 0.02, 0.03, ...
    # Each request has 100 input tokens, 50 output tokens
    # Pre-fill time = 100 tokens * 2 ms = 200 ms
    # Decode time = 50 tokens * 20 ms = 1000 ms
    # Total time per request = 1200 ms = 1.2s

    requests = []
    for i in range(8):
        req = {
            'rid': f'req_{i}',
            'program_id': f'prog_{i}',
            'turn_index': 0,
            'arrival_time': float(i) * 0.01,  # Arrives at times 0, 0.01, 0.02, ...
            'token_ids': list(range(100 * i, 100 * i + 100)),  # 100 unique tokens
            'input_len': 100,
            'output_len': 50,
            'is_tool_call': False,
            'tool_name': None,
            'extra_key': f'program_{i}',
        }
        requests.append(req)

    print(f"Created {len(requests)} requests")
    print(f"Expected time per request (serial): 1.2s")
    print(f"Expected total time (serial): {1.2 * 8:.1f}s")
    print(f"Expected total time (batch=4): 1.2s + 1.2s = 2.4s (2 batches)")
    print()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        # Test batch size 4
        print("-" * 60)
        print("Batch Size = 4")
        print("-" * 60)

        # FIXED: Use poisson_lambda=1000.0 to make inter-arrival time 0.001s (1ms)
        # This ensures all requests arrive almost at once
        sim = KVCacheSimulator(
            cache_capacity=100000,
            l2_capacity=200000,
            l2_enabled=True,
            l2_reload_penalty=0.3,
            default_ttl=5.0,
            min_ttl=0.5,
            max_ttl=60.0,
            enable_ttl=True,
            enable_adaptive_ttl=True,
            simulation_mode='event_driven',
            poisson_lambda=1000.0,  # FIXED: Inter-arrival = 0.001s (all arrive at once)
            tool_execution_time=2.0,
            random_seed=42,
            hardware_config=hw,
            timing_config=timing_calc,
            write_policy='write_through',
            history_threshold=10,
            ttl_grid_points=50,
        )

        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999

        result = sim.run(verbose=False, enable_batch=True, batch_size=4)

        print(f"Total simulation time: {result.duration:.2f}s")
        print(f"Scheduling log entries: {len(sim.scheduler.scheduling_log)}")

        # Check the finish times in the scheduling log
        if sim.scheduler.scheduling_log:
            print()
            print("Scheduling log (first 10 entries):")
            for i, entry in enumerate(sim.scheduler.scheduling_log[:10]):
                print(f"  [{i}] rid={entry['rid']}, start={entry['start_time']:.2f}s, "
                      f"finish={entry['finish_time']:.2f}s, "
                      f"prefill={entry['prefill_ms']:.0f}ms, decode={entry['decode_ms']:.0f}ms")

        # Verify time monotonicity
        print()
        print("Time monotonicity check:")
        log = sim.scheduler.scheduling_log
        monotonic_ok = True
        for i in range(1, len(log)):
            if log[i]['finish_time'] < log[i-1]['finish_time']:
                print(f"  ERROR: finish_time decreased at index {i}: "
                      f"{log[i]['finish_time']:.3f}s < {log[i-1]['finish_time']:.3f}s")
                monotonic_ok = False

        if monotonic_ok:
            print("  OK: All finish_times are non-decreasing")

        # Check if finish times are reasonable
        max_finish = max(entry['finish_time'] for entry in log)
        print(f"  Max finish time: {max_finish:.2f}s")

        # In batch mode with 8 requests and batch_size=4:
        # - Batch 1 (req 0-3) starts at t=0, finishes at t=1.2
        # - Batch 2 (req 4-7) starts at t=1.2, finishes at t=2.4
        # Total should be ~2.4s
        expected = 2.4
        tolerance = 0.3  # Allow 300ms tolerance

        print()
        print("Validation:")
        if abs(max_finish - expected) < tolerance:
            print(f"  PASS: Total time {max_finish:.2f}s is close to expected {expected:.2f}s")
        else:
            print(f"  WARNING: Total time {max_finish:.2f}s differs from expected {expected:.2f}s")
            print(f"  This may indicate a time advancement issue.")

    finally:
        import os
        os.unlink(temp_path)


def test_time_advancement_large_batch():
    """Test time advancement with large batch sizes (16, 32, 64, 128, 256)."""
    print()
    print("=" * 80)
    print("TEST: Time Advancement with Large Batch Sizes")
    print("=" * 80)

    # Create 512 requests, all arriving at time 0
    requests = []
    for i in range(512):
        req = {
            'rid': f'req_{i}',
            'program_id': f'prog_{i % 32}',  # 32 programs
            'turn_index': 0,
            'arrival_time': 0.0,  # All arrive at once
            'token_ids': list(range(100 * i, 100 * i + 100)),  # 100 unique tokens
            'input_len': 100,
            'output_len': 50,
            'is_tool_call': False,
            'tool_name': None,
            'extra_key': f'program_{i % 32}',
        }
        requests.append(req)

    print(f"Created {len(requests)} requests (all arriving at t=0)")
    print(f"Expected time per request: 1.2s")
    print()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        results = {}

        for batch_size in batch_sizes:
            print(f"Testing batch_size={batch_size}...", end=" ", flush=True)

            sim = KVCacheSimulator(
                cache_capacity=1000000,
                l2_capacity=2000000,
                l2_enabled=True,
                l2_reload_penalty=0.3,
                default_ttl=5.0,
                min_ttl=0.5,
                max_ttl=60.0,
                enable_ttl=True,
                enable_adaptive_ttl=True,
                simulation_mode='event_driven',
                poisson_lambda=10000.0,  # Very small inter-arrival (0.0001s)
                tool_execution_time=2.0,
                random_seed=42,
                hardware_config=hw,
                timing_config=timing_calc,
                write_policy='write_through',
                history_threshold=10,
                ttl_grid_points=50,
            )

            sim.load_requests(temp_path)
            for pid in sim.scheduler.program_total_turns:
                sim.scheduler.program_total_turns[pid] = 999

            result = sim.run(verbose=False, enable_batch=True, batch_size=batch_size)

            max_finish = max(entry['finish_time'] for entry in sim.scheduler.scheduling_log)
            results[batch_size] = {
                'duration': result.duration,
                'max_finish': max_finish,
                'num_batches': (512 + batch_size - 1) // batch_size,
            }

            print(f"{result.duration:.2f}s (max_finish={max_finish:.2f}s)")

        # Print summary
        print()
        print("Summary:")
        print(f"  {'Batch Size':>12} | {'Num Batches':>12} | {'Time (s)':>10} | {'Expected (s)':>12} | {'Speedup':>10}")
        print("  " + "-" * 60)

        baseline = results[1]['duration']
        for bs, r in results.items():
            num_batches = r['num_batches']
            expected_time = num_batches * 1.2  # 1.2s per batch
            speedup = baseline / r['duration'] if r['duration'] > 0 else 0
            diff_pct = abs(r['duration'] - expected_time) / expected_time * 100 if expected_time > 0 else 0

            status = "OK" if diff_pct < 20 else "WARN"

            print(f"  {bs:>12} | {num_batches:>12} | {r['duration']:>10.2f} | "
                  f"{expected_time:>11.1f}s | {speedup:>9.2f}x [{status}]")

    finally:
        import os
        os.unlink(temp_path)


if __name__ == '__main__':
    test_time_advancement()
    test_time_advancement_large_batch()
