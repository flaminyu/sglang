"""Time prediction module for sglang simulator.

This module provides inference time prediction capabilities.
The AIConfigurator predictor requires the aiconfigurator package.
Install with: pip install aiconfigurator

If aiconfigurator is not available, simulation will use a basic analytical model.
"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.spec.model import ModelInfo


class InferTimePredictor:
    """Base class for inference time prediction."""

    def predict_infer_time(self, batch) -> float:
        """Predict inference time for a batch."""
        raise NotImplementedError()


class ScheduleRequest:
    """Request for scheduling."""

    def __init__(self, extend_length: int = 1, past_kv_length: int = 0):
        self.extend_length = extend_length
        self.past_kv_length = past_kv_length


class ScheduleBatch:
    """Batch for scheduling."""

    def __init__(self, reqs: List[ScheduleRequest] = None):
        self.reqs = reqs or []

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def request_info(self) -> dict:
        """Return request information for logging."""
        return {
            "num_requests": len(self.reqs),
            "total_extend": sum(r.extend_length for r in self.reqs),
        }


def AIConfiguratorTimePredictor(*args, **kwargs):
    """Alias that raises an error indicating the package is not installed."""
    raise ImportError(
        "AIConfiguratorTimePredictor requires the aiconfigurator package. "
        "Please install with: pip install aiconfigurator"
    )


def get_perf_model(*args, **kwargs):
    """Stub function for performance model retrieval."""
    raise NotImplementedError(
        "Performance model requires aiconfigurator package. "
        "Please install with: pip install aiconfigurator"
    )
