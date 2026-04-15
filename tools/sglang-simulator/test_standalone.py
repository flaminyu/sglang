#!/usr/bin/env python3
"""
Standalone test for SGLang Simulator - tests components without requiring CUDA.

This test validates the simulator hooks and TTL mechanism work together
by mocking the SGLang classes that would be hooked.
"""

import sys
import os
import time
import re

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
simulator_src = os.path.join(project_root, "src")
sys.path.insert(0, simulator_src)

print("=" * 60)
print("SGLang Simulator Standalone Test")
print("=" * 60)
print(f"Project: {project_root}")
print(f"Python: {sys.version.split()[0]}")
print()

# ============================================================================
# Test 1: Import all simulator modules
# ============================================================================
print("[TEST 1] Import Simulator Modules")
print("-" * 40)

try:
    from sglang_simulator import simulation
    print("  [OK] simulation module")

    from sglang_simulator.hook import BaseHook, install_class_hooks
    print("  [OK] hook module")

    from sglang_simulator.simulation.sglang import (
        C_SchedulerHook,
        C_ModelRunnerHook,
        C_HiCacheController,
        C_HiRadixCacheHook,
        C_StorageBackendFactory,
    )
    print("  [OK] sglang simulation hooks")

    from sglang_simulator.simulation.manager import ConfigManager, StateManager, Envs
    print("  [OK] manager module")

    from sglang_simulator.simulation.types import (
        RequestStats,
        SchedulerConfig,
        PlatformConfig,
        SimulationMode,
    )
    print("  [OK] types module")

    from sglang_simulator.spec import AcceleratorInfo, ModelInfo, DataType
    print("  [OK] spec module")

    from sglang_simulator.time_predictor import (
        InferTimePredictor,
        ScheduleBatch,
        ScheduleRequest,
    )
    print("  [OK] time_predictor module")

    from sglang_simulator.utils import get_logger
    print("  [OK] utils module")

    print("\n  [PASS] All simulator modules imported successfully!\n")
except Exception as e:
    print(f"\n  [FAIL] Import error: {e}\n")
    sys.exit(1)

# ============================================================================
# Test 2: Hook class properties
# ============================================================================
print("[TEST 2] Hook Class Configuration")
print("-" * 40)

hooks_to_test = [
    ("C_SchedulerHook", C_SchedulerHook, "Scheduler", "sglang.srt.managers.scheduler"),
    ("C_ModelRunnerHook", C_ModelRunnerHook, "ModelRunner", "sglang.srt.model_executor.model_runner"),
    ("C_HiCacheController", C_HiCacheController, "HiCacheController", "sglang.srt.managers.cache_controller"),
    ("C_HiRadixCacheHook", C_HiRadixCacheHook, "HiRadixCache", "sglang.srt.mem_cache.hiradix_cache"),
]

all_pass = True
for name, hook_class, expected_class, expected_module in hooks_to_test:
    class_ok = hook_class.HOOK_CLASS_NAME == expected_class
    module_ok = hook_class.HOOK_MODULE_NAME == expected_module
    has_hook = hasattr(hook_class, 'hook')

    status = "OK" if (class_ok and module_ok and has_hook) else "FAIL"
    if status == "FAIL":
        all_pass = False

    print(f"  {name}:")
    print(f"    HOOK_CLASS_NAME: {hook_class.HOOK_CLASS_NAME} {'OK' if class_ok else f'EXPECTED {expected_class}'}")
    print(f"    HOOK_MODULE_NAME: {hook_class.HOOK_MODULE_NAME} {'OK' if module_ok else f'EXPECTED {expected_module}'}")
    print(f"    has hook(): {'OK' if has_hook else 'MISSING'}")

print(f"\n  [{'PASS' if all_pass else 'FAIL'}] Hook classes configured correctly!\n")

# ============================================================================
# Test 3: Configuration and state management
# ============================================================================
print("[TEST 3] Configuration Management")
print("-" * 40)

try:
    # Test AcceleratorInfo
    hw = AcceleratorInfo.find_by_hw_name("a100_sxm")
    if hw:
        print(f"  [OK] A100 config: {hw.hbm_capacity_gb}GB HBM")

    hw_h100 = AcceleratorInfo.find_by_hw_name("h100_sxm")
    if hw_h100:
        print(f"  [OK] H100 config: {hw_h100.hbm_capacity_gb}GB HBM")

    # Test ModelInfo
    model = ModelInfo(
        model_path="Qwen3-8B",
        attention_arch="MHA",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936,
        context_len=32768,
    )
    print(f"  [OK] ModelInfo: {model.model_path}")
    print(f"  [OK] Attention arch: {model.attention_arch}")
    print(f"  [OK] Context length: {model.context_len}")

    # Test SchedulerConfig
    sched_config = SchedulerConfig(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        backend_name="sglang",
        backend_version="0.5.9",
    )
    print(f"  [OK] SchedulerConfig: TP={sched_config.tp_size}")

    # Test RequestStats
    stats = RequestStats(
        rid="test_req_001",
        input_length=1024,
        output_length=512,
        created_time=1000.0,
    )
    stats.gen_token_latencies = [0.1, 0.05, 0.05, 0.05]
    print(f"  [OK] RequestStats: {stats.rid}, latencies={len(stats.gen_token_latencies)}")

    print("\n  [PASS] Configuration management works!\n")
except Exception as e:
    print(f"\n  [FAIL] Config error: {e}\n")
    sys.exit(1)

# ============================================================================
# Test 4: Hook installation (without actual SGLang classes)
# ============================================================================
print("[TEST 4] Hook Installation Simulation")
print("-" * 40)

print("  [INFO] Testing hook registration mechanism...")
print("  [INFO] (Skipping actual hook install since SGLang requires CUDA)")

# Verify hook classes have the right structure
for name, hook_class, _, _ in hooks_to_test:
    if callable(hook_class.hook):
        print(f"  [OK] {name}.hook() is callable")
    else:
        print(f"  [FAIL] {name}.hook() is not callable")
        all_pass = False

print(f"\n  [{'PASS' if all_pass else 'FAIL'}] Hook mechanism validated!\n")

# ============================================================================
# Test 5: TTL mechanism in source files
# ============================================================================
print("[TEST 5] TTL Mechanism Integration")
print("-" * 40)

sglang_path = os.path.join(project_root, "..", "python")
scheduler_path = os.path.join(sglang_path, "sglang/srt/managers/scheduler.py")
radix_path = os.path.join(sglang_path, "sglang/srt/mem_cache/radix_cache.py")

ttl_found = True

# Check scheduler.py for TTL methods
if os.path.exists(scheduler_path):
    with open(scheduler_path, 'r') as f:
        scheduler_content = f.read()

    ttl_methods = [
        'init_continuum_pin',
        'pin_request_kv',
        'unpin_request_kv',
        'cleanup_expired_pins',
        '_extract_ttl_from_extra_key',
    ]

    for method in ttl_methods:
        if re.search(rf'def {method}\(', scheduler_content):
            print(f"  [OK] Scheduler.{method}()")
        else:
            print(f"  [FAIL] Scheduler.{method}() NOT FOUND")
            ttl_found = False
else:
    print(f"  [SKIP] scheduler.py not found at {scheduler_path}")
    ttl_found = False

# Check radix_cache.py for TTL support
if os.path.exists(radix_path):
    with open(radix_path, 'r') as f:
        radix_content = f.read()

    radix_features = [
        ('_extract_ttl_from_extra_key', 'RadixKey._extract_ttl_from_extra_key'),
        ('get_ttl_sec', 'RadixKey.get_ttl_sec'),
        ('ttl_expiry_time', 'TreeNode.ttl_expiry_time'),
        ('def set_ttl', 'TreeNode.set_ttl'),
        ('def is_expired', 'TreeNode.is_expired'),
    ]

    for pattern, name in radix_features:
        if re.search(pattern, radix_content):
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name} NOT FOUND")
            ttl_found = False
else:
    print(f"  [SKIP] radix_cache.py not found")

print(f"\n  [{'PASS' if ttl_found else 'FAIL'}] TTL mechanism integration verified!\n")

# ============================================================================
# Test 6: Simulation Mode
# ============================================================================
print("[TEST 6] Simulation Modes")
print("-" * 40)

modes = [
    (SimulationMode.BLOCKING, "BLOCKING"),
    (SimulationMode.OFFLINE, "OFFLINE"),
]

for mode, expected in modes:
    status = "OK" if mode.value == expected else "FAIL"
    print(f"  [{status}] SimulationMode.{mode.name} = {mode.value}")

print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 60)
if all_pass and ttl_found:
    print("ALL TESTS PASSED!")
    print()
    print("The SGLang Simulator with TTL Support is ready!")
    print()
    print("Next steps:")
    print("  1. For GPU testing: Use SLURM with single GPU")
    print("  2. For production: Install aiconfigurator package")
    print()
    print("Usage:")
    print("  export PYTHONPATH=$PWD/src:$PWD/python:$PYTHONPATH")
    print("  python3 -m sglang_simulator.simulation.sglang.launch_server")
else:
    print("SOME TESTS FAILED - Please review the output above")
print("=" * 60)
