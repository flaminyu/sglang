#!/usr/bin/env python3
"""
CPU-only launcher for SGLang Simulator.

This script launches the simulator without requiring CUDA by:
1. Pre-emptively blocking sgl_kernel import errors
2. Using CPU-only inference engine  
3. Loading the simulator hooks

Usage:
    python3 launch_cpu.py --help
"""

import sys
import os
import importlib.abc
import importlib.machinery

# ============================================================================
# CRITICAL: Set environment variables BEFORE any torch/sglang import
# ============================================================================

# Block CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['SGLANG_USE_CPU_ENGINE'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['SGLANG_DISABLE_REDUNDANT_MEM_INIT'] = '1'

# Set default simulator config
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "test", "assets", "config.json")
if os.path.exists(config_path):
    os.environ.setdefault('SGLANG_SIMULATOR_CONFIG_PATH', config_path)

# Add paths
simulator_src = os.path.join(script_dir, "src")
sglang_python = os.path.join(script_dir, "..", "..", "python")
sys.path.insert(0, simulator_src)
sys.path.insert(0, sglang_python)

print("[CPU Simulator] Environment configured")

# ============================================================================
# Patch torch BEFORE it tries to use CUDA
# ============================================================================
print("[CPU Simulator] Patching torch...")

import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.cuda.current_device = lambda: 0

# Also patch torch.cuda module
if hasattr(torch.cuda, 'is_available'):
    torch.cuda.is_available = lambda: False
if hasattr(torch.cuda, 'device_count'):
    torch.cuda.device_count = lambda: 0

print(f"[CPU Simulator] torch.cuda.is_available = {torch.cuda.is_available()}")

# ============================================================================
# Create mock sgl_kernel module
# ============================================================================
print("[CPU Simulator] Creating mock sgl_kernel...")

class MockOps:
    """Mock operations for CPU simulation."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def __call__(self, *args, **kwargs):
        return None

class MockLoadUtils:
    def _load_architecture_specific_ops(self):
        return MockOps()
    def _preload_cuda_library(self):
        pass

class MockSGLKernel:
    """Mock sgl_kernel module."""
    def __init__(self):
        self.common_ops = MockOps()
        self.load_utils = MockLoadUtils()
    
    def __getattr__(self, name):
        # Return mock functions for any attribute access
        return lambda *args, **kwargs: None

# Install mock sgl_kernel in sys.modules
mock_sgl_kernel = MockSGLKernel()
sys.modules['sgl_kernel'] = mock_sgl_kernel
sys.modules['sgl_kernel.load_utils'] = mock_sgl_kernel.load_utils

print("[CPU Simulator] Mock sgl_kernel installed")

# ============================================================================
# Import and launch SGLang
# ============================================================================
print("[CPU Simulator] Installing simulator hooks...")

try:
    import sglang_simulator.hook as sglang_simulator_hook
    from sglang_simulator.simulation.sglang import (
        cache_controller,
        hicache_storage,
        hiradix_cache,
        mem_cache_allocator,
        mem_pool_host,
        model_runner,
        scheduler,
        sgl_kernel_hook,
    )

    # Install module hook for sgl_kernel (will be triggered if sgl_kernel is re-imported)
    sglang_simulator_hook.install_module_hooks([
        sgl_kernel_hook.M_SGLangKernelLoadUtilHook,
    ])

    # Install class hooks
    sglang_simulator_hook.install_class_hooks([
        scheduler.C_SchedulerHook,
        model_runner.C_ModelRunnerHook,
        hicache_storage.C_StorageBackendFactory,
        cache_controller.C_HiCacheController,
        hiradix_cache.C_HiRadixCacheHook,
        mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
        mem_pool_host.C_MHATokenToKVPoolHostHook,
        mem_pool_host.C_HostKVCacheHook,
    ])

    print("[CPU Simulator] Simulator hooks installed successfully")
except Exception as e:
    print(f"[CPU Simulator] Warning: Could not install hooks: {e}")

print("[CPU Simulator] Ready to import SGLang...")

# ============================================================================
# Launch server
# ============================================================================
if __name__ == "__main__":
    import argparse
    import dataclasses
    
    @dataclasses.dataclass
    class SimulationArgs:
        sim_config_path: str = None
        
        @staticmethod
        def add_cli_args(parser):
            parser.add_argument(
                "--sim-config-path",
                type=str,
                default=None,
                help="Path to simulation JSON config"
            )
        
        @classmethod
        def from_cli_args(cls, ns):
            return cls(sim_config_path=ns.sim_config_path)

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import kill_process_tree
    from sglang_simulator.utils import get_logger

    logger = get_logger("sgl_simulator")

    parser = argparse.ArgumentParser(description="SGLang Simulator (CPU Mode)")

    g = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(g)

    g = parser.add_argument_group("simulation")
    SimulationArgs.add_cli_args(g)

    raw_args = parser.parse_args(sys.argv[1:])
    server_args = ServerArgs.from_cli_args(raw_args)
    simulation_args = SimulationArgs.from_cli_args(raw_args)

    if simulation_args.sim_config_path:
        os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = simulation_args.sim_config_path

    logger.info(f"Launching SGLang Simulator (CPU Mode)")
    logger.info(f"Model: {server_args.model_path}")
    logger.info(f"Config: {os.environ.get('SGLANG_SIMULATOR_CONFIG_PATH', 'default')}")

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
