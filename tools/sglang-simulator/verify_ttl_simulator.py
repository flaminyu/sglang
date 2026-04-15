#!/usr/bin/env python3
"""Verify TTL mechanism and Simulator integration in source files."""

import os
import re

project_root = "/data/home/sczd795/run/KVbloking/sglang_continuum"

def check_ttl_methods_in_scheduler():
    """Check TTL methods exist in scheduler.py"""
    scheduler_path = os.path.join(
        project_root,
        "python/sglang/srt/managers/scheduler.py"
    )

    with open(scheduler_path, 'r') as f:
        content = f.read()

    ttl_methods = {
        'init_continuum_pin': r'def init_continuum_pin\(self\):',
        'pin_request_kv': r'def pin_request_kv\(self, req:',
        'unpin_request_kv': r'def unpin_request_kv\(self, req:',
        'cleanup_expired_pins': r'def cleanup_expired_pins\(self\):',
        '_extract_ttl_from_extra_key': r'def _extract_ttl_from_extra_key\(self,',
        'prevent_pinned_deadlock': r'def prevent_pinned_deadlock\(self,',
        'pinned_requests': r'self\.pinned_requests: List\[Req\] = \[\]',
    }

    print("\n[TTL Methods in scheduler.py]")
    all_found = True
    for method, pattern in ttl_methods.items():
        if re.search(pattern, content):
            print(f"  {method}: OK")
        else:
            print(f"  {method}: MISSING")
            all_found = False

    return all_found


def check_ttl_in_radix_cache():
    """Check TTL in radix_cache.py"""
    radix_path = os.path.join(
        project_root,
        "python/sglang/srt/mem_cache/radix_cache.py"
    )

    with open(radix_path, 'r') as f:
        content = f.read()

    ttl_items = {
        'RadixKey._extract_ttl_from_extra_key': r'def _extract_ttl_from_extra_key',
        'RadixKey.get_ttl_sec': r'def get_ttl_sec\(self\)',
        'TreeNode.ttl_expiry_time': r'self\.ttl_expiry_time: Optional\[float\] = None',
        'TreeNode.set_ttl': r'def set_ttl\(self, ttl_sec:',
        'TreeNode.is_expired': r'def is_expired\(self\)',
    }

    print("\n[TTL in radix_cache.py]")
    all_found = True
    for item, pattern in ttl_items.items():
        if re.search(pattern, content):
            print(f"  {item}: OK")
        else:
            print(f"  {item}: MISSING")
            all_found = False

    return all_found


def check_ttl_in_hiradix_cache():
    """Check TTL in hiradix_cache.py"""
    hiradix_path = os.path.join(
        project_root,
        "python/sglang/srt/mem_cache/hiradix_cache.py"
    )

    with open(hiradix_path, 'r') as f:
        content = f.read()

    ttl_items = {
        'insert with ttl': r'params\.ttl_sec|ttl_sec = params\.ttl_sec',
        'evict expired': r'node\.is_expired|is_expired',
        'TTL logging': r'ttl_sec|TTL',
    }

    print("\n[TTL in hiradix_cache.py]")
    all_found = True
    for item, pattern in ttl_items.items():
        if re.search(pattern, content):
            print(f"  {item}: OK")
        else:
            print(f"  {item}: MISSING")
            all_found = False

    return all_found


def check_simulator_hooks():
    """Check simulator hooks exist"""
    simulator_path = os.path.join(
        project_root,
        "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/scheduler.py"
    )

    with open(simulator_path, 'r') as f:
        content = f.read()

    hook_items = {
        'C_SchedulerHook class': r'class C_SchedulerHook',
        'wrapped_init': r'def wrapped_init',
        'wrapped_recv_requests': r'def wrapped_recv_requests',
        'wrapped_run_batch': r'def wrapped_run_batch',
        'profile method': r'def wrapped_profile',
    }

    print("\n[Simulator Scheduler Hooks]")
    all_found = True
    for item, pattern in hook_items.items():
        if re.search(pattern, content):
            print(f"  {item}: OK")
        else:
            print(f"  {item}: MISSING")
            all_found = False

    return all_found


def check_simulator_files():
    """Check all simulator files exist"""
    base_path = os.path.join(project_root, "tools/sglang-simulator/src/sglang_simulator")

    required_files = [
        "simulation/sglang/scheduler.py",
        "simulation/sglang/cache_controller.py",
        "simulation/sglang/hiradix_cache.py",
        "simulation/sglang/launch_server.py",
        "simulation/sglang/model_runner.py",
        "hook/class_hook_entry.py",
        "hook/module_hook_entry.py",
        "simulation/manager/config.py",
        "simulation/manager/state.py",
        "spec/accelerator.py",
        "spec/model.py",
        "time_predictor/__init__.py",
    ]

    print("\n[Simulator Files]")
    all_found = True
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"  {file_path}: OK")
        else:
            print(f"  {file_path}: MISSING")
            all_found = False

    return all_found


def main():
    print("=" * 60)
    print("SGLang Simulator + TTL Verification")
    print("=" * 60)
    print(f"Project root: {project_root}")

    results = []
    results.append(check_ttl_methods_in_scheduler())
    results.append(check_ttl_in_radix_cache())
    results.append(check_ttl_in_hiradix_cache())
    results.append(check_simulator_hooks())
    results.append(check_simulator_files())

    print("\n" + "=" * 60)
    if all(results):
        print("VERIFICATION PASSED: All components verified!")
        print("\nThe merged code is ready for testing.")
        print("\nTo test on GPU cluster:")
        print("  1. Commit changes to simulator-ttl-merge branch")
        print("  2. Submit SLURM job with single GPU")
        print("  3. Run: python3 -m sglang_simulator.simulation.sglang.launch_server")
    else:
        print("VERIFICATION FAILED: Some components are missing!")
    print("=" * 60)


if __name__ == "__main__":
    main()
