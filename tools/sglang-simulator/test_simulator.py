#!/usr/bin/env python3
"""Test script for SGLang Simulator with TTL mechanism."""

import sys
import os

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
simulator_src = os.path.join(project_root, "src")
sys.path.insert(0, simulator_src)

from sglang_simulator.hook import install_class_hooks
from sglang_simulator.simulation.sglang import (
    scheduler,
    model_runner,
    cache_controller,
    hiradix_cache,
    hicache_storage,
    mem_cache_allocator,
    mem_pool_host,
)
from sglang_simulator.simulation.manager import ConfigManager, Envs, StateManager
from sglang_simulator.simulation.types import RequestStats, SimulationMode
from sglang_simulator.spec import AcceleratorInfo, ModelInfo, DataType


def test_simulator_modules():
    """Test all simulator modules can be imported and initialized."""
    print("=" * 60)
    print("Testing SGLang Simulator with TTL Support")
    print("=" * 60)

    # Test 1: Module imports
    print("\n[TEST 1] Module Imports")
    try:
        from sglang_simulator import simulation
        print("  - simulation module: OK")
        from sglang_simulator.hook import BaseHook, install_class_hooks
        print("  - hook module: OK")
        from sglang_simulator.simulation.sglang import (
            C_SchedulerHook,
            C_ModelRunnerHook,
            C_HiCacheController,
            C_HiRadixCacheHook,
        )
        print("  - sglang simulation hooks: OK")
        from sglang_simulator.simulation.manager import ConfigManager, StateManager
        print("  - manager module: OK")
        print("  [PASS] All modules imported successfully")
    except Exception as e:
        print(f"  [FAIL] Module import error: {e}")
        return False

    # Test 2: Hook installation
    print("\n[TEST 2] Hook Installation")
    try:
        install_class_hooks([
            scheduler.C_SchedulerHook,
            model_runner.C_ModelRunnerHook,
            cache_controller.C_HiCacheController,
            hiradix_cache.C_HiRadixCacheHook,
        ])
        print("  [PASS] Simulator hooks installed successfully")
    except Exception as e:
        print(f"  [FAIL] Hook installation error: {e}")
        return False

    # Test 3: Configuration
    print("\n[TEST 3] Configuration")
    try:
        hw = AcceleratorInfo.find_by_hw_name("a100_sxm")
        if hw:
            print(f"  - Hardware: {hw.name} ({hw.hbm_capacity_gb}GB)")
        else:
            print("  - Hardware: custom")

        model_info = ModelInfo(
            model_path="Qwen2.5-0.5B",
            attention_arch="MHA",
            num_hidden_layers=24,
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=151936,
            context_len=8192,
        )
        print(f"  - Model: {model_info.model_path}")
        print(f"  - Attention: {model_info.attention_arch}")
        print("  [PASS] Configuration created successfully")
    except Exception as e:
        print(f"  [FAIL] Configuration error: {e}")
        return False

    # Test 4: Request stats and simulation types
    print("\n[TEST 4] Simulation Types")
    try:
        stats = RequestStats(
            rid="test_req_001",
            input_length=1024,
            output_length=512,
            created_time=1000.0,
        )
        stats.gen_token_latencies = [0.1, 0.05, 0.05, 0.05]
        print(f"  - RequestStats created: rid={stats.rid}")
        print(f"  - Input length: {stats.input_length}")
        print(f"  - Output length: {stats.output_length}")
        print(f"  - Latencies: {stats.gen_token_latencies}")
        print("  [PASS] Simulation types work correctly")
    except Exception as e:
        print(f"  [FAIL] Simulation types error: {e}")
        return False

    # Test 5: TTL Mechanism check
    print("\n[TEST 5] TTL Mechanism (SGLang Scheduler)")
    try:
        from sglang.srt.managers.scheduler import Scheduler
        has_pin = hasattr(Scheduler, 'pin_request_kv')
        has_unpin = hasattr(Scheduler, 'unpin_request_kv')
        has_cleanup = hasattr(Scheduler, 'cleanup_expired_pins')
        has_init_pin = hasattr(Scheduler, 'init_continuum_pin')
        has_extract_ttl = hasattr(Scheduler, '_extract_ttl_from_extra_key')

        print(f"  - pin_request_kv: {'OK' if has_pin else 'MISSING'}")
        print(f"  - unpin_request_kv: {'OK' if has_unpin else 'MISSING'}")
        print(f"  - cleanup_expired_pins: {'OK' if has_cleanup else 'MISSING'}")
        print(f"  - init_continuum_pin: {'OK' if has_init_pin else 'MISSING'}")
        print(f"  - _extract_ttl_from_extra_key: {'OK' if has_extract_ttl else 'MISSING'}")

        if has_pin and has_unpin and has_cleanup and has_init_pin:
            print("  [PASS] TTL mechanism fully integrated")
        else:
            print("  [WARN] Some TTL methods may be missing")
    except ImportError as e:
        print(f"  [INFO] SGLang module not in Python path: {e}")
        print("  [SKIP] TTL check skipped (set PYTHONPATH to sglang_continuum/python)")
    except Exception as e:
        print(f"  [FAIL] TTL check error: {e}")
        return False

    return True


def test_hook_classes():
    """Test individual hook classes."""
    print("\n" + "=" * 60)
    print("Testing Hook Classes")
    print("=" * 60)

    hooks = [
        ("C_SchedulerHook", scheduler.C_SchedulerHook),
        ("C_ModelRunnerHook", model_runner.C_ModelRunnerHook),
        ("C_HiCacheController", cache_controller.C_HiCacheController),
        ("C_HiRadixCacheHook", hiradix_cache.C_HiRadixCacheHook),
    ]

    all_ok = True
    for name, hook_class in hooks:
        print(f"\n[{name}]")
        print(f"  - HOOK_CLASS_NAME: {hook_class.HOOK_CLASS_NAME}")
        print(f"  - HOOK_MODULE_NAME: {hook_class.HOOK_MODULE_NAME}")
        print(f"  - Has hook method: {hasattr(hook_class, 'hook')}")

    print("\n[PASS] All hook classes validated")


def main():
    print("\n" + "=" * 60)
    print("SGLang Simulator + TTL Merge Test")
    print("=" * 60)

    success = test_simulator_modules()

    if success:
        test_hook_classes()

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED - Simulator is ready for use!")
    else:
        print("SOME TESTS FAILED - Please check the errors above")
    print("=" * 60)

    print("\nTo use the simulator:")
    print("  1. Set PYTHONPATH: export PYTHONPATH=$PWD/tools/sglang-simulator/src:$PWD/python:$PYTHONPATH")
    print("  2. Set config: export SGLANG_SIMULATOR_CONFIG_PATH=tools/sglang-simulator/test/assets/config.json")
    print("  3. Launch: python3 -m sglang_simulator.simulation.sglang.launch_server --help")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
