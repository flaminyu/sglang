# Mock common_ops for CPU simulation
# This module provides dummy implementations of sgl_kernel operations

class MockOps:
    """Mock operations for CPU simulation."""
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    
    def __call__(self, *args, **kwargs):
        return None


# Create a singleton mock ops instance
common_ops = MockOps()
