"""
AIConfigurator adapter for pure_simulator.

This module provides an interface to use AIConfigurator for time prediction,
integrating with the pure_simulator's cache model (L1/L2).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# Try to import AIConfigurator SDK
AI_CONFIGURATOR_AVAILABLE = False
try:
    from aiconfigurator.sdk.config import RuntimeConfig, ModelConfig
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.perf_database import get_database, get_systems_paths, DatabaseMode
    AI_CONFIGURATOR_AVAILABLE = True
except ImportError:
    pass


@dataclass
class ScheduleRequest:
    """A single request in a batch for scheduling."""
    extend_length: int = 0
    past_kv_length: int = 0
    is_cache_hit: bool = False


@dataclass
class ScheduleBatch:
    """A batch of requests for time prediction."""
    reqs: List[ScheduleRequest] = None
    
    def __post_init__(self):
        if self.reqs is None:
            self.reqs = []
    
    def is_empty(self) -> bool:
        return len(self.reqs) == 0
    
    def batch_size(self) -> int:
        return len(self.reqs)


class AIConfiguratorAdapter:
    """
    Adapter to use AIConfigurator SDK with pure_simulator.
    
    This adapter bridges the gap between:
    - pure_simulator's cache model (L1/L2 with StateManager)
    - AIConfigurator's time prediction (RuntimeConfig format)
    
    Note: AIConfigurator's RuntimeConfig only knows about "cached vs not cached".
    The distinction between L1 (GPU) and L2 (CPU) hits is handled by
    the StateManager tracking L2 load/backup durations.
    
    Usage:
        # Initialize
        adapter = AIConfiguratorAdapter(
            model_name="Llama-3.1-8B",
            device_name="A100",
        )
        
        # Predict time for a request
        batch = ScheduleBatch(reqs=[
            ScheduleRequest(extend_length=500, past_kv_length=400)
        ])
        total_time_ms = adapter.predict_infer_time(batch)
    """
    
    def __init__(
        self,
        model_name: str = "Llama-3.1-8B",
        device_name: str = "A100",
        num_layers: int = 32,
        hidden_size: int = 4096,
        database_mode: str = "SOL",
        fallback_to_naive: bool = True,
    ):
        """
        Initialize the AIConfigurator adapter.
        
        Args:
            model_name: Model name (e.g., "Llama-3.1-8B")
            device_name: Device name (e.g., "A100", "H100")
            num_layers: Number of transformer layers
            hidden_size: Hidden size dimension
            database_mode: Database mode ("SOL" or "SILICON")
            fallback_to_naive: If AIConfigurator unavailable, use naive model
        """
        self.model_name = model_name
        self.device_name = device_name
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.database_mode = database_mode
        self.fallback_to_naive = fallback_to_naive
        self._session = None
        self._initialized = False
        
        if AI_CONFIGURATOR_AVAILABLE:
            self._init_session()
        elif fallback_to_naive:
            # Use naive model parameters
            self._naive_prefill_ms = 0.1  # ms per token
            self._naive_decode_ms = 4.5    # ms per token
    
    def _init_session(self):
        """Initialize the AIConfigurator session."""
        try:
            db = get_database(mode=DatabaseMode.SOL if self.database_mode == "SOL" else DatabaseMode.SILICON)
            
            model_config = ModelConfig(
                model_name=self.model_name,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
            )
            
            runtime_config = RuntimeConfig(
                device_type="NVIDIA",
                device_name=self.device_name,
            )
            
            self._session = InferenceSession(model_config, runtime_config)
            self._initialized = True
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize AIConfigurator: {e}. Using naive model.")
            self._initialized = False
    
    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        """
        Predict inference time for a batch.
        
        Args:
            batch: ScheduleBatch with requests
            
        Returns:
            Total inference time in seconds
        """
        if batch.is_empty():
            return 0.0
        
        if not self._initialized or self._session is None:
            return self._naive_predict(batch)
        
        try:
            return self._aiconfigurator_predict(batch)
        except Exception as e:
            if self.fallback_to_naive:
                return self._naive_predict(batch)
            raise e
    
    def _aiconfigurator_predict(self, batch: ScheduleBatch) -> float:
        """Use AIConfigurator for prediction."""
        # Aggregate batch info
        total_tokens = sum(r.extend_length for r in batch.reqs)
        total_past = sum(r.past_kv_length for r in batch.reqs)
        batch_size = len(batch.reqs)
        
        # AIConfigurator uses RuntimeConfig
        # Note: prefix = past_kv_length represents cached tokens
        runtime_config = RuntimeConfig(
            batch_size=batch_size,
            isl=total_tokens,
            prefix=total_past,
            osl=1,  # Per-token prediction
        )
        
        latency_dict = self._session.run_static(runtime_config, mode="static_ctx")
        return sum(latency_dict.values())
    
    def _naive_predict(self, batch: ScheduleBatch) -> float:
        """Fallback naive prediction."""
        total_time_ms = 0.0
        
        for req in batch.reqs:
            if req.is_cache_hit:
                # Cache hit: only decode time
                total_time_ms += self._naive_decode_ms
            else:
                # Cache miss: prefill + decode
                prefill_time = req.extend_length * self._naive_prefill_ms
                decode_time = self._naive_decode_ms
                total_time_ms += prefill_time + decode_time
        
        return total_time_ms / 1000.0  # Convert to seconds
    
    def predict_infer_time_detailed(
        self,
        total_input_tokens: int,
        output_tokens: int,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """
        Get detailed time breakdown.
        
        Args:
            total_input_tokens: Total input tokens
            output_tokens: Output tokens to generate
            l1_tokens: Tokens in L1 cache (GPU, no extra cost)
            l2_tokens: Tokens in L2 cache (CPU, H2D cost tracked separately)
            
        Returns:
            Tuple of (prefill_ms, decode_ms, h2d_ms)
            
        Note: H2D cost for L2 is tracked via StateManager, not returned here.
        This method returns only the inference time.
        """
        cached = l1_tokens + l2_tokens
        
        batch = ScheduleBatch(reqs=[
            ScheduleRequest(
                extend_length=total_input_tokens,
                past_kv_length=cached,
                is_cache_hit=cached > 0,
            )
        ])
        
        total_time_sec = self.predict_infer_time(batch)
        total_time_ms = total_time_sec * 1000
        
        # Simplified split (AIConfigurator would provide better breakdown)
        prefill_ratio = 0.3 if cached == 0 else 0.1
        prefill_ms = total_time_ms * prefill_ratio
        decode_ms = total_time_ms * (1 - prefill_ratio)
        h2d_ms = 0.0  # H2D tracked via StateManager
        
        return prefill_ms, decode_ms, h2d_ms


def create_aiconfigurator_adapter(
    model_name: str = "Llama-3.1-8B",
    device_name: str = "A100",
    **kwargs
) -> Optional[AIConfiguratorAdapter]:
    """
    Factory function to create AIConfigurator adapter.
    
    Returns None if AIConfigurator is not available and fallback is disabled.
    """
    if not AI_CONFIGURATOR_AVAILABLE:
        import warnings
        warnings.warn(
            "AIConfigurator SDK not available. Install with: pip install aiconfigurator"
        )
        return None
    
    return AIConfiguratorAdapter(
        model_name=model_name,
        device_name=device_name,
        **kwargs
    )
