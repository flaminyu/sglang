"""AIConfigurator-based time predictor.

This module provides time prediction using the AIConfigurator library.
The AIConfigurator library provides hardware-aware performance modeling
for LLM inference.

If the library is not available, it falls back to a basic analytical model.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Try to import AIConfigurator
try:
    from aiconfigurator.sdk import models
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.common import (
        CommQuantMode,
        DatabaseMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.perf_database import get_database, get_systems_paths
    AI_CONFIGURATOR_AVAILABLE = True
except ImportError:
    AI_CONFIGURATOR_AVAILABLE = False
    logger.warning("AIConfigurator not available. Using fallback analytical model.")


# ============================================================================
# Type Definitions
# ============================================================================

if TYPE_CHECKING:
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.spec.accelerator import AcceleratorInfo
    from sglang_simulator.spec.model import ModelInfo


# Map common data types to AIConfigurator data types
_DATA_TYPE_MAP = {
    "FP8": "float8",
    "FP16": "float16",
    "BF16": "bfloat16",
    "FP32": "float32",
}

# Quantization mode mappings
# Note: FMHAQuantMode uses lowercase attribute names
_QUANT_MODE_MAP = {
    "fp8": FMHAQuantMode.fp8,
    "fp16": FMHAQuantMode.float16,
    "bf16": None,  # BF16 not supported
    "fp32": None,  # FP32 not supported
    "int8": None,  # INT8 not supported
}


# ============================================================================
# Base Class
# ============================================================================

class InferTimePredictor:
    """Base class for inference time prediction."""

    def predict_infer_time(self, batch) -> float:
        """Predict inference time for a batch."""
        raise NotImplementedError()


class ScheduleRequest:
    """Request for scheduling."""

    def __init__(
        self,
        extend_length: int = 1,
        past_kv_length: int = 0,
        is_cache_hit: bool = False,
    ):
        self.extend_length = extend_length
        self.past_kv_length = past_kv_length
        self.is_cache_hit = is_cache_hit


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


# ============================================================================
# Analytical Fallback Predictor
# ============================================================================

class AnalyticalTimePredictor(InferTimePredictor):
    """
    Analytical time predictor using simple per-token estimates.

    This is the fallback when AIConfigurator is not available.
    Uses profiled or estimated per-token times for prefill and decode.
    """

    def __init__(
        self,
        prefill_per_token_ms: float = 0.05,
        decode_per_token_ms: float = 10.0,
        cache_hit_speedup: float = 10.0,
        l1_reload_penalty: float = 0.3,
        ttft_overhead_ms: float = 50.0,
    ):
        """
        Args:
            prefill_per_token_ms: Time per token for prefill phase
            decode_per_token_ms: Time per token for decode phase
            cache_hit_speedup: Speedup factor for cache hits
            l1_reload_penalty: Penalty for L1 cache hits (H2D transfer)
            ttft_overhead_ms: Time to first token overhead
        """
        self.prefill_per_token_ms = prefill_per_token_ms
        self.decode_per_token_ms = decode_per_token_ms
        self.cache_hit_speedup = cache_hit_speedup
        self.l1_reload_penalty = l1_reload_penalty
        self.ttft_overhead_ms = ttft_overhead_ms

    def predict_time(
        self,
        miss_tokens: int,
        output_tokens: int,
        l0_tokens: int = 0,
        l1_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """
        Predict inference time.

        Returns: (prefill_ms, decode_ms, total_ms)
        """
        total_cached = l0_tokens + l1_tokens
        actual_miss_tokens = max(0, miss_tokens - total_cached)

        # L0 hit: skip prefill
        if l0_tokens >= miss_tokens:
            prefill_ms = 0.0
            decode_ms = output_tokens * self.decode_per_token_ms / self.cache_hit_speedup
        # L1 hit: partial prefill for H2D
        elif l1_tokens > 0:
            prefill_ms = actual_miss_tokens * self.prefill_per_token_ms * self.l1_reload_penalty
            decode_ms = output_tokens * self.decode_per_token_ms / self.cache_hit_speedup
        # Miss: full prefill
        else:
            prefill_ms = actual_miss_tokens * self.prefill_per_token_ms
            decode_ms = output_tokens * self.decode_per_token_ms

        total_ms = prefill_ms + decode_ms + self.ttft_overhead_ms
        return prefill_ms, decode_ms, total_ms

    def predict_infer_time(self, batch) -> float:
        """Predict inference time for a batch."""
        total_prefill = 0.0
        total_decode = 0.0

        for req in batch.reqs:
            prefill, decode, _ = self.predict_time(
                req.extend_length,
                1,  # Assume 1 output token for batch prediction
                past_kv_length=req.past_kv_length,
            )
            total_prefill += prefill
            total_decode += decode

        # Batch time is max of all requests (parallel processing)
        return max(total_prefill, total_decode)


# ============================================================================
# AIConfigurator-based Predictor
# ============================================================================

if AI_CONFIGURATOR_AVAILABLE:

    class AIConfiguratorTimePredictor(InferTimePredictor):
        """
        Time predictor using AIConfigurator library.

        AIConfigurator provides hardware-aware performance modeling
        based on actual profiled data for specific GPU models.
        """

        def __init__(
            self,
            model: "ModelInfo",
            hw: "AcceleratorInfo",
            config: "SchedulerConfig",
            database_path: Optional[str] = None,
            database_mode: str = "SILICON",
            prefill_scale_factor: float = 1.0,
            decode_scale_factor: float = 1.0,
        ):
            """
            Args:
                model: Model information
                hw: Hardware/Accelerator information
                config: Scheduler configuration
                database_path: Path to performance database
                database_mode: Database mode (SILICON, FPGA, etc.)
                prefill_scale_factor: Scale factor for prefill times
                decode_scale_factor: Scale factor for decode times
            """
            self.model = model
            self.hw = hw
            self.config = config
            self.prefill_scale_factor = prefill_scale_factor
            self.decode_scale_factor = decode_scale_factor

            # Performance model
            self.perf_model = None
            self.inference_session = None

            self._init_predictor(database_path, database_mode)

        def _init_predictor(
            self,
            database_path: Optional[str],
            database_mode: str,
        ):
            """Initialize the AIConfigurator predictor."""
            try:
                # Get systems paths
                systems = get_systems_paths()
                logger.info(f"AIConfigurator systems: {systems}")

                # Get or create database
                if database_path:
                    db = get_database(database_path, DatabaseMode[database_mode.upper()])
                else:
                    db = get_database(mode=DatabaseMode[database_mode.upper()])

                # Create model config
                model_config = self._create_model_config()

                # Create runtime config
                runtime_config = self._create_runtime_config()

                # Get performance model
                self.perf_model = db.get_performance_model(
                    model_config=model_config,
                    runtime_config=runtime_config,
                )

                # Create inference session
                self.inference_session = InferenceSession(
                    perf_model=self.perf_model,
                    model_config=model_config,
                )

                logger.info("AIConfiguratorTimePredictor initialized successfully")

            except Exception as e:
                logger.warning(f"Failed to initialize AIConfigurator: {e}")
                logger.info("Falling back to analytical model")
                self.perf_model = None
                self.inference_session = None

        def _create_model_config(self) -> ModelConfig:
            """Create model configuration for AIConfigurator."""
            # Map data type
            dtype = _DATA_TYPE_MAP.get(
                getattr(self.model, "dtype", "FP16"),
                "float16"
            )

            config = ModelConfig(
                model_name=getattr(self.model, "name", "unknown"),
                num_layers=getattr(self.model, "num_layers", 32),
                hidden_size=getattr(self.model, "hidden_size", 4096),
                num_heads=getattr(self.model, "num_heads", 32),
                vocab_size=getattr(self.model, "vocab_size", 32000),
                max_position_embeddings=getattr(
                    self.model, "max_position", 4096
                ),
                data_type=dtype,
            )

            # Add MoE config if available
            if hasattr(self.model, "num_experts") and self.model.num_experts:
                config.num_experts = self.model.num_experts
                config.moe_config = models.MoEConfig(
                    num_experts=self.model.num_experts,
                    top_k=getattr(self.model, "moe_top_k", 2),
                )

            return config

        def _create_runtime_config(self) -> RuntimeConfig:
            """Create runtime configuration for AIConfigurator."""
            config = RuntimeConfig(
                device_type=getattr(self.hw, "device_type", "NVIDIA"),
                device_name=getattr(self.hw, "name", "A100"),
                memory_gb=getattr(self.hw, "memory_gb", 80),
                compute_capability=getattr(self.hw, "compute_capability", "8.0"),
            )

            # Add quantization configs if available
            if hasattr(self.config, "quantization"):
                q = self.config.quantization
                config.fused_mha_quant = _QUANT_MODE_MAP.get(
                    getattr(q, "fused_mha", "fp16"),
                    FMHAQuantMode.FP16
                )
                config.gemm_quant = _QUANT_MODE_MAP.get(
                    getattr(q, "gemm", "fp16"),
                    GEMMQuantMode.FP16
                )
                config.kv_cache_quant = _QUANT_MODE_MAP.get(
                    getattr(q, "kv_cache", "fp16"),
                    KVCacheQuantMode.FP16
                )

            return config

        def predict_infer_time(self, batch) -> float:
            """
            Predict inference time using AIConfigurator.

            Args:
                batch: ScheduleBatch containing requests

            Returns:
                Predicted inference time in milliseconds
            """
            if self.perf_model is None:
                # Fall back to analytical model
                fallback = AnalyticalTimePredictor()
                return fallback.predict_infer_time(batch)

            try:
                # Build batch specification
                batch_spec = models.BatchSpecification(
                    num_requests=len(batch.reqs),
                    total_input_tokens=sum(r.extend_length for r in batch.reqs),
                    total_past_kv_tokens=sum(r.past_kv_length for r in batch.reqs),
                    max_new_tokens=max(
                        (r.extend_length - r.past_kv_length) for r in batch.reqs
                    ) if batch.reqs else 1,
                )

                # Get prediction
                prediction = self.perf_model.predict(batch_spec)

                return (
                    prediction.prefill_time_ms * self.prefill_scale_factor +
                    prediction.decode_time_ms * self.decode_scale_factor
                )

            except Exception as e:
                logger.warning(f"AIConfigurator prediction failed: {e}")
                fallback = AnalyticalTimePredictor()
                return fallback.predict_infer_time(batch)

        def get_hardware_info(self) -> Dict[str, Any]:
            """Get hardware information used by the predictor."""
            return {
                "device_type": getattr(self.hw, "device_type", "NVIDIA"),
                "device_name": getattr(self.hw, "name", "A100"),
                "memory_gb": getattr(self.hw, "memory_gb", 80),
                "compute_capability": getattr(self.hw, "compute_capability", "8.0"),
                "perf_model_available": self.perf_model is not None,
            }

else:
    # Stub class when AIConfigurator is not available
    class AIConfiguratorTimePredictor(InferTimePredictor):
        """
        AIConfigurator predictor stub.

        This class is used when aiconfigurator is not installed.
        All methods delegate to the AnalyticalTimePredictor.
        """

        def __init__(
            self,
            model: Optional["ModelInfo"] = None,
            hw: Optional["AcceleratorInfo"] = None,
            config: Optional["SchedulerConfig"] = None,
            database_path: Optional[str] = None,
            database_mode: str = "SILICON",
            prefill_scale_factor: float = 1.0,
            decode_scale_factor: float = 1.0,
            # Fallback parameters
            prefill_per_token_ms: float = 0.05,
            decode_per_token_ms: float = 10.0,
            cache_hit_speedup: float = 10.0,
            l1_reload_penalty: float = 0.3,
            ttft_overhead_ms: float = 50.0,
        ):
            logger.warning(
                "AIConfigurator not available. Using AnalyticalTimePredictor."
            )
            self._fallback = AnalyticalTimePredictor(
                prefill_per_token_ms=prefill_per_token_ms,
                decode_per_token_ms=decode_per_token_ms,
                cache_hit_speedup=cache_hit_speedup,
                l1_reload_penalty=l1_reload_penalty,
                ttft_overhead_ms=ttft_overhead_ms,
            )
            self.prefill_scale_factor = prefill_scale_factor
            self.decode_scale_factor = decode_scale_factor

        def predict_time(
            self,
            miss_tokens: int,
            output_tokens: int,
            l0_tokens: int = 0,
            l1_tokens: int = 0,
        ) -> Tuple[float, float, float]:
            """Predict time using fallback model."""
            return self._fallback.predict_time(
                miss_tokens, output_tokens, l0_tokens, l1_tokens
            )

        def predict_infer_time(self, batch) -> float:
            """Predict batch time using fallback model."""
            return self._fallback.predict_infer_time(batch) * self.prefill_scale_factor

        def get_hardware_info(self) -> Dict[str, Any]:
            """Get hardware info (fallback values)."""
            return {
                "device_type": "Unknown",
                "device_name": "Unknown",
                "memory_gb": 0,
                "compute_capability": "0.0",
                "perf_model_available": False,
                "using_fallback": True,
            }


# ============================================================================
# Hardware-Aware Time Predictor Factory
# ============================================================================

def create_time_predictor(
    predictor_type: str = "auto",
    **kwargs,
) -> InferTimePredictor:
    """
    Create a time predictor based on available libraries and configuration.

    Args:
        predictor_type: Type of predictor to create
            - "auto": Use AIConfigurator if available, else analytical
            - "aiconfigurator": Force AIConfigurator (may fail if not installed)
            - "analytical": Use analytical model
        **kwargs: Arguments passed to the predictor constructor

    Returns:
        An InferTimePredictor instance
    """
    if predictor_type == "analytical":
        return AnalyticalTimePredictor(**kwargs)

    if predictor_type == "aiconfigurator":
        if not AI_CONFIGURATOR_AVAILABLE:
            raise ImportError(
                "AIConfigurator is not installed. "
                "Please install with: pip install aiconfigurator"
            )
        return AIConfiguratorTimePredictor(**kwargs)

    # Auto mode
    if AI_CONFIGURATOR_AVAILABLE:
        try:
            return AIConfiguratorTimePredictor(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to create AIConfigurator predictor: {e}")
            logger.info("Falling back to analytical model")

    return AnalyticalTimePredictor(**kwargs)


def get_perf_model(config: "SchedulerConfig", model: "ModelInfo"):
    """
    Get performance model from AIConfigurator database.

    Args:
        config: Scheduler configuration
        model: Model information

    Returns:
        Performance model or None if not available
    """
    if not AI_CONFIGURATOR_AVAILABLE:
        return None

    try:
        db = get_database(mode=DatabaseMode.SILICON)

        model_config = ModelConfig(
            model_name=getattr(model, "name", "unknown"),
            num_layers=getattr(model, "num_layers", 32),
            hidden_size=getattr(model, "hidden_size", 4096),
        )

        runtime_config = RuntimeConfig(
            device_type="NVIDIA",
            device_name=getattr(config, "device_name", "A100"),
        )

        return db.get_performance_model(
            model_config=model_config,
            runtime_config=runtime_config,
        )
    except Exception as e:
        logger.warning(f"Failed to get performance model: {e}")
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def is_aiconfigurator_available() -> bool:
    """Check if AIConfigurator is available."""
    return AI_CONFIGURATOR_AVAILABLE


def get_supported_hardware() -> List[str]:
    """Get list of supported hardware."""
    if not AI_CONFIGURATOR_AVAILABLE:
        return []
    try:
        systems = get_systems_paths()
        return list(systems.keys())
    except Exception:
        return []


def estimate_from_spec(
    prefill_per_token_ms: float,
    decode_per_token_ms: float,
    input_tokens: int,
    output_tokens: int,
    is_cache_hit: bool = False,
) -> Dict[str, float]:
    """
    Estimate inference time from specifications.

    Args:
        prefill_per_token_ms: Time per prefill token
        decode_per_token_ms: Time per decode token
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        is_cache_hit: Whether this is a cache hit

    Returns:
        Dictionary with time estimates
    """
    if is_cache_hit:
        prefill_time = 0.0
        speedup = getattr(
            AI_CONFIGuratorTimePredictor.__init__,
            "__defaults__",
            (10.0,),
        )[0]
        decode_time = output_tokens * decode_per_token_ms / speedup
    else:
        prefill_time = input_tokens * prefill_per_token_ms
        decode_time = output_tokens * decode_per_token_ms

    return {
        "prefill_time_ms": prefill_time,
        "decode_time_ms": decode_time,
        "total_time_ms": prefill_time + decode_time,
        "ttft_ms": prefill_time,
        "tpot_ms": decode_time / max(1, output_tokens),
    }
