"""Time prediction module for sglang simulator.

This module provides inference time prediction capabilities.

Classes:
- InferTimePredictor: Base class for time prediction
- AnalyticalTimePredictor: Simple analytical model (fallback)
- AIConfiguratorTimePredictor: Hardware-aware model using AIConfigurator library
- ScheduleRequest: Request for scheduling
- ScheduleBatch: Batch for scheduling

Usage:
    # Auto-detect best available predictor
    from sglang_simulator.time_predictor import create_time_predictor
    predictor = create_time_predictor("auto")

    # Use analytical predictor explicitly
    from sglang_simulator.time_predictor import AnalyticalTimePredictor
    predictor = AnalyticalTimePredictor(prefill_per_token_ms=0.05, decode_per_token_ms=10.0)

    # Use AIConfigurator predictor (requires aiconfigurator package)
    from sglang_simulator.time_predictor import AIConfiguratorTimePredictor
    predictor = AIConfiguratorTimePredictor(model, hw, config)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.spec.model import ModelInfo
    from sglang_simulator.spec.accelerator import AcceleratorInfo


# Import from aiconfigurator module
from sglang_simulator.time_predictor.aiconfigurator import (
    InferTimePredictor,
    ScheduleRequest,
    ScheduleBatch,
    AnalyticalTimePredictor,
    AIConfiguratorTimePredictor,
    create_time_predictor,
    get_perf_model,
    is_aiconfigurator_available,
    get_supported_hardware,
    estimate_from_spec,
    AI_CONFIGURATOR_AVAILABLE,
)


def create_predictor(
    predictor_type: str = "auto",
    **kwargs
) -> InferTimePredictor:
    """
    Create a time predictor.

    Alias for create_time_predictor for convenience.

    Args:
        predictor_type: Type of predictor ("auto", "aiconfigurator", "analytical")
        **kwargs: Arguments passed to the predictor

    Returns:
        InferTimePredictor instance
    """
    return create_time_predictor(predictor_type, **kwargs)


__all__ = [
    # Classes
    "InferTimePredictor",
    "ScheduleRequest",
    "ScheduleBatch",
    "AnalyticalTimePredictor",
    "AIConfiguratorTimePredictor",
    # Functions
    "create_time_predictor",
    "create_predictor",
    "get_perf_model",
    "is_aiconfigurator_available",
    "get_supported_hardware",
    "estimate_from_spec",
    # Constants
    "AI_CONFIGURATOR_AVAILABLE",
]
