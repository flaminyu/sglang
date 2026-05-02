#!/usr/bin/env python3
import sys
print("Starting test...")
sys.path.insert(0, '.')

import tempfile
import json
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

# Generate simple requests
requests = []
for i in range(10):
    requests.append({
        'rid': f'req_{i}',
        'program_id': f'prog_{i % 3}',
        'turn_index': 0,
        'arrival_time': i * 0.5,
        'token_ids': list(range(100, 200)),
        'input_len': 100,
        'output_len': 50,
        'is_tool_call': False,
        'tool_name': None,
        'extra_key': f'program_{i % 3}',
    })

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for req in requests:
        f.write(json.dumps(req) + '\n')
    temp_path = f.name

print("Created temp file")

try:
    hw = HardwareConfig(**HARDWARE_CONFIG)
    timing_calc = TimingCalculator(hw=hw)

    print("Creating simulator...")
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

    print("Loading requests...")
    sim.load_requests(temp_path)
    for pid in sim.scheduler.program_total_turns:
        sim.scheduler.program_total_turns[pid] = 999

    print("Running serial mode...")
    result_serial = sim.run(verbose=False, enable_batch=False)
    print(f"Serial: {result_serial.duration:.2f}s")

    print("Running batch mode...")
    result_batch = sim.run(verbose=False, enable_batch=True, batch_size=4)
    print(f"Batch: {result_batch.duration:.2f}s")

    print(f"Speedup: {result_serial.duration/result_batch.duration:.3f}x")
    print("DONE!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
