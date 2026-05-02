#!/usr/bin/env python3
import sys
print("Python version:", sys.version)
print("Starting...")

try:
    sys.path.insert(0, '.')
    from simulator.timing import HardwareConfig, TimingCalculator
    print("Import successful")

    HARDWARE_CONFIG = {
        'prefill_latency_per_token_ms': 2.0,
        'decode_latency_per_token_ms': 20.0,
        'bytes_per_token': 512.0,
        'h2d_bandwidth_GBps': 32.0,
        'bandwidth_efficiency': 0.85,
        'h2d_overhead_us': 6.67,
        'd2h_overhead_us': 4.0,
        'memory_bandwidth_GBps': 64.0,
    }

    hw = HardwareConfig(**HARDWARE_CONFIG)
    calc = TimingCalculator(hw=hw)

    single = calc.calculate_batch_prefill_time(1, 100, is_first_batch=True)
    batch4 = calc.calculate_batch_prefill_time(4, 100, is_first_batch=True)

    print(f"Single: {single:.2f}ms")
    print(f"Batch4: {batch4:.2f}ms")
    print(f"Efficiency: {single*4/batch4:.2f}x")
    print("Done!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
