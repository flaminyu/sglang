from dataclasses import dataclass
from typing import Optional


@dataclass
class AcceleratorInfo:
    """Hardware accelerator information."""

    name: str
    vendor: Optional[str] = None
    hbm_bandwidth_gb: Optional[float] = None
    hbm_capacity_gb: Optional[float] = None
    inter_node_bandwidth_gb: Optional[float] = None
    intra_node_bandwidth_gb: Optional[float] = None
    tflops: Optional[float] = None

    # Pre-defined accelerator configurations
    _HW_CONFIGS = {
        "h100_sxm": {
            "name": "h100_sxm",
            "vendor": "NVIDIA",
            "hbm_bandwidth_gb": 3.35,
            "hbm_capacity_gb": 80,
            "tflops": 3958,
        },
        "a100_sxm": {
            "name": "a100_sxm",
            "vendor": "NVIDIA",
            "hbm_bandwidth_gb": 2,
            "hbm_capacity_gb": 80,
            "tflops": 1560,
        },
        "h20": {
            "name": "h20",
            "vendor": "NVIDIA",
            "hbm_bandwidth_gb": 0.9,
            "hbm_capacity_gb": 80,
            "tflops": 148,
        },
        "a800": {
            "name": "a800",
            "vendor": "NVIDIA",
            "hbm_bandwidth_gb": 2,
            "hbm_capacity_gb": 80,
            "tflops": 624,
        },
    }

    @classmethod
    def find_by_hw_name(cls, hw_name: str) -> Optional["AcceleratorInfo"]:
        """Find accelerator info by hardware name."""
        if hw_name in cls._HW_CONFIGS:
            config = cls._HW_CONFIGS[hw_name]
            return cls(**config)
        return None

    @classmethod
    def list_all_hws(cls) -> dict:
        """List all available hardware configurations."""
        return cls._HW_CONFIGS.copy()
