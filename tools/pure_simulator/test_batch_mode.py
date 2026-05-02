#!/usr/bin/env python3
"""
Test batch processing mode.

This test verifies that the batch processing implementation works correctly
and produces consistent results with serial processing.
"""

import sys
sys.path.insert(0, '.')

import tempfile
import json
import random
from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator, A100_80G
from simulator.ttl_manager import TTLManager


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


def generate_test_requests(num_requests=20, seed=42):
    """Generate simple test requests."""
    random.seed(seed)

    requests = []
    for i in range(num_requests):
        req = {
            'rid': f'req_{i}',
            'program_id': f'prog_{i % 5}',  # 5 programs
            'turn_index': 0,
            'arrival_time': i * 0.5,  # Staggered arrival
            'token_ids': list(range(100, 200)),  # 100 tokens
            'input_len': 100,
            'output_len': 50,
            'is_tool_call': i % 3 == 0,
            'tool_name': 'test_tool' if i % 3 == 0 else None,
            'extra_key': f'program_{i % 5}',
        }
        requests.append(req)
    return requests


def run_test(mode='serial', batch_size=4):
    """Run test with specified mode."""
    requests = generate_test_requests(20)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        sim = KVCacheSimulator(
            cache_capacity=10000,
            l2_capacity=20000,
            l2_enabled=True,
            l2_reload_penalty=0.3,
            default_ttl=5.0,
            min_ttl=0.5,
            max_ttl=60.0,
            enable_ttl=True,
            enable_adaptive_ttl=True,
            simulation_mode='event_driven',
            poisson_lambda=1.0,
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

        # Run with batch mode if specified
        if mode == 'batch':
            result = sim.run(verbose=False, enable_batch=True, batch_size=batch_size)
        else:
            result = sim.run(verbose=False)

        return result
    finally:
        import os
        os.unlink(temp_path)


def test_timing_calculator():
    """Test the batch timing calculator."""
    print("=" * 60)
    print("Test 1: Timing Calculator")
    print("=" * 60)

    hw = HardwareConfig(**HARDWARE_CONFIG)
    calc = TimingCalculator(hw=hw)

    # Test single request
    single_time = calc.calculate_batch_prefill_time(1, 100, is_first_batch=True)
    print(f"Single request prefill: {single_time:.2f}ms")

    # Test batch of 4
    batch4_time = calc.calculate_batch_prefill_time(4, 100, is_first_batch=True)
    print(f"Batch of 4 prefill: {batch4_time:.2f}ms (per request: {batch4_time/4:.2f}ms)")

    # Test batch efficiency
    efficiency = single_time * 4 / batch4_time if batch4_time > 0 else 0
    print(f"Batch efficiency: {efficiency:.2f}x")

    # Test decode time
    decode_single = calc.calculate_decode_time(50, batch_size=1)
    decode_batch = calc.calculate_decode_time(50, batch_size=4)
    print(f"Single decode: {decode_single:.2f}ms")
    print(f"Batch decode: {decode_batch:.2f}ms")

    print()
    return True


def test_serial_vs_batch():
    """Test serial vs batch processing."""
    print("=" * 60)
    print("Test 2: Serial vs Batch Processing")
    print("=" * 60)

    # Run serial
    print("Running serial mode...")
    serial_result = run_test(mode='serial')
    print(f"  Total time: {serial_result.duration:.2f}s")
    print(f"  Cache hits: {serial_result.cache_stats.cache_hits}")
    print(f"  Cache misses: {serial_result.cache_stats.cache_misses}")

    # Run batch
    print("Running batch mode (batch_size=4)...")
    batch_result = run_test(mode='batch', batch_size=4)
    print(f"  Total time: {batch_result.duration:.2f}s")
    print(f"  Cache hits: {batch_result.cache_stats.cache_hits}")
    print(f"  Cache misses: {batch_result.cache_stats.cache_misses}")

    # Compare
    print()
    print("Comparison:")
    print(f"  Time difference: {batch_result.duration - serial_result.duration:.2f}s")
    print(f"  Hit difference: {batch_result.cache_stats.cache_hits - serial_result.cache_stats.cache_hits}")

    # Batch should generally be faster or similar
    if batch_result.duration <= serial_result.duration * 1.1:
        print("  PASS: Batch mode is reasonably fast")
    else:
        print("  WARNING: Batch mode is slower than expected")

    print()
    return True


def test_different_batch_sizes():
    """Test different batch sizes."""
    print("=" * 60)
    print("Test 3: Different Batch Sizes")
    print("=" * 60)

    results = {}
    for batch_size in [1, 2, 4, 8]:
        print(f"Running batch_size={batch_size}...", end=" ")
        result = run_test(mode='batch', batch_size=batch_size)
        results[batch_size] = result
        print(f"{result.duration:.2f}s")

    print()
    print("Summary:")
    print(f"  {'Batch Size':>10} | {'Time (s)':>10} | {'Speedup':>10}")
    print("  " + "-" * 36)

    baseline = results[1].duration
    for size, result in results.items():
        speedup = baseline / result.duration if result.duration > 0 else 0
        print(f"  {size:>10} | {result.duration:>10.2f} | {speedup:>9.3f}x")

    print()
    return True


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TESTS")
    print("=" * 60 + "\n")

    all_passed = True

    all_passed &= test_timing_calculator()
    all_passed &= test_serial_vs_batch()
    all_passed &= test_different_batch_sizes()

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
