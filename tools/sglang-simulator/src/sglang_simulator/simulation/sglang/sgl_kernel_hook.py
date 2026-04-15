import sys
import os
from sglang_simulator.hook import BaseHook


class M_SGLangKernelLoadUtilHook(BaseHook):
    """Hook to bypass sgl_kernel's CUDA dependency for CPU simulation."""
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sgl_kernel.load_utils"

    @classmethod
    def hook(cls, target):
        # Get the path to mock_common_ops
        simulator_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mock_ops_path = os.path.join(simulator_path, "sglang_simulator", "mock_common_ops.py")

        def override_load_architecture_specific_ops(*args, **kwargs):
            """
            Override _load_architecture_specific_ops to use our mock common_ops.
            This allows CPU-based simulation without CUDA.
            """
            import importlib.util
            from sglang_simulator.mock_common_ops import common_ops as mock_ops
            return mock_ops

        def override_preload_cuda_library(*args, **kwargs):
            """Override CUDA preload - no-op for CPU simulation."""
            pass

        target._load_architecture_specific_ops = override_load_architecture_specific_ops
        target._preload_cuda_library = override_preload_cuda_library

        # Also patch sgl_kernel's __init__ to not raise on import
        import sgl_kernel
        original_init = sgl_kernel.__class__.__init__ if hasattr(sgl_kernel, '__class__') else None
        
        # Replace common_ops in sgl_kernel namespace
        from sglang_simulator.mock_common_ops import common_ops as mock_ops
        sgl_kernel.common_ops = mock_ops

        return mock_ops
