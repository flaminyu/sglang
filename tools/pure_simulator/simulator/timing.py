"""
Timing model for pure simulator.

Based on Continuum paper (arXiv:2511.02230v3) insights.

Key insight from paper:
- TTL mechanism protects KV during tool execution, preventing eviction
- When a program resumes after tool execution, its KV is still in cache (if TTL didn't expire)
- This saves prefill time AND maintains program ordering

The two benefits of TTL (per paper):
1. Prefill/reload overhead saved (faster than L1 hit)
2. No queueing delay - TTL-pinned programs get priority scheduling

Supports two timing modes:
- naive: Simple analytical model (default)
- aiconfigurator: AIConfigurator SDK for hardware-aware prediction
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .aiconfigurator_adapter import AIConfiguratorAdapter

# Import calc_transfer_time from l2_cache for bandwidth amortization model
from .l2_cache import calc_transfer_time


# Model configurations: model_name -> (num_layers, hidden_size, num_heads)
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Llama-3.1-8B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "vocab_size": 128256,
        "estimated_params_b": 8.03,
    },
    "Llama-3.1-70B": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "vocab_size": 128256,
        "estimated_params_b": 70.6,
    },
    "Llama-3.1-405B": {
        "num_layers": 126,
        "hidden_size": 16384,
        "num_heads": 128,
        "vocab_size": 128256,
        "estimated_params_b": 404.0,
    },
    "Llama-3-8B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "vocab_size": 128256,
        "estimated_params_b": 8.0,
    },
    "Llama-3-70B": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "vocab_size": 128256,
        "estimated_params_b": 70.0,
    },
    "Llama-2-7B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "vocab_size": 32000,
        "estimated_params_b": 6.7,
    },
    "Llama-2-13B": {
        "num_layers": 40,
        "hidden_size": 5120,
        "num_heads": 40,
        "vocab_size": 32000,
        "estimated_params_b": 13.0,
    },
    "Llama-2-70B": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "vocab_size": 32000,
        "estimated_params_b": 70.0,
    },
    "Mistral-7B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "vocab_size": 32000,
        "estimated_params_b": 7.3,
    },
    "Mixtral-8x7B": {
        "num_layers": 44,  # per expert
        "hidden_size": 4096,
        "num_heads": 32,
        "vocab_size": 32000,
        "estimated_params_b": 46.7,  # total with MoE
    },
    "Qwen2-7B": {
        "num_layers": 28,
        "hidden_size": 3584,
        "num_heads": 28,
        "vocab_size": 151936,
        "estimated_params_b": 7.6,
    },
    "Qwen2-72B": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "vocab_size": 151936,
        "estimated_params_b": 72.0,
    },
    "Qwen2.5-32B": {
        "num_layers": 48,
        "hidden_size": 5120,
        "num_heads": 40,
        "vocab_size": 151936,
        "estimated_params_b": 31.0,
    },
    "Llama-3.1-32B": {
        "num_layers": 48,
        "hidden_size": 5120,
        "num_heads": 40,
        "vocab_size": 128256,
        "estimated_params_b": 31.2,
    },
}

# Device configurations: device_name -> (memory_bandwidth_GBps, memory_GB, fp16_compute_TFLOPS)
DEVICE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A100": {
        "memory_bandwidth_GBps": 2039,  # 80GB version
        "memory_GB": 80,
        "fp16_compute_TFLOPS": 312,
    },
    "A100-40GB": {
        "memory_bandwidth_GBps": 1555,
        "memory_GB": 40,
        "fp16_compute_TFLOPS": 312,
    },
    "H100": {
        "memory_bandwidth_GBps": 3350,  # SXM5
        "memory_GB": 80,
        "fp16_compute_TFLOPS": 989,
    },
    "H100-80GB": {
        "memory_bandwidth_GBps": 3350,
        "memory_GB": 80,
        "fp16_compute_TFLOPS": 989,
    },
    "A10": {
        "memory_bandwidth_GBps": 600,
        "memory_GB": 24,
        "fp16_compute_TFLOPS": 125,
    },
    "L40S": {
        "memory_bandwidth_GBps": 1000,
        "memory_GB": 48,
        "fp16_compute_TFLOPS": 733,
    },
    "T4": {
        "memory_bandwidth_GBps": 320,
        "memory_GB": 16,
        "fp16_compute_TFLOPS": 65,
    },
}


@dataclass
class HardwareConfig:
    """Hardware performance parameters with model and device configuration."""

    # ============= Timing Parameters =============
    prefill_latency_per_token_ms: float = 0.1   # Prefill time per token (forward pass)
    decode_latency_per_token_ms: float = 4.5     # Decode time per token
    cache_hit_speedup: float = 10.0             # Decode speedup on hit
    ttft_overhead_ms: float = 50.0              # Time to first token overhead
    queue_delay_per_request_ms: float = 100.0    # Extra delay per waiting request

    # ============= Memory Parameters =============
    h2d_bandwidth_GBps: float = 32.0             # PCIe 4.0 x16 bandwidth
    bytes_per_token: float = 512.0               # KV cache size per token

    # ============= Bandwidth Model Parameters (new) =============
    bandwidth_efficiency: float = 0.85           # 85% effective bandwidth utilization
    h2d_overhead_us: float = 6.67                # H2D fixed overhead (microseconds)
    d2h_overhead_us: float = 4.0                 # D2H fixed overhead (microseconds)
    memory_bandwidth_GBps: float = 64.0          # Host memory read bandwidth for segment optimization

    # ============= Model Parameters =============
    model_name: str = "Llama-3.1-8B"
    num_layers: int = 32
    hidden_size: int = 4096
    num_heads: int = 32
    vocab_size: int = 128256
    kv_cache_bytes_per_token: float = 0.0       # Computed from model params

    # ============= Device Parameters =============
    device_name: str = "A100"
    memory_bandwidth_GBps: float = 2039.0       # HBM bandwidth
    memory_GB: float = 80.0                     # Device memory

    def __post_init__(self):
        """Calculate derived parameters if not provided."""
        if self.kv_cache_bytes_per_token == 0.0:
            self.kv_cache_bytes_per_token = self._calculate_kv_cache_bytes()

    @property
    def num_params(self) -> int:
        """Estimated model parameters (in billions)."""
        # Simplified estimation: ~12 * L * H^2 for standard transformer
        # More accurate would account for vocab, attention overhead, etc.
        params = 12 * self.num_layers * (self.hidden_size ** 2)
        # Add FFN: ~4 * L * H^2
        params += 4 * self.num_layers * (self.hidden_size ** 2)
        # Add embeddings: ~2 * V * H (rough)
        params += 2 * self.vocab_size * self.hidden_size
        return params

    @property
    def kv_cache_per_layer_bytes(self) -> float:
        """KV cache bytes per layer per token."""
        # Key and Value: 2 * hidden_size * 2 (fp16 = 2 bytes)
        return 2 * 2 * self.hidden_size

    def _calculate_kv_cache_bytes(self) -> float:
        """Calculate KV cache bytes per token from model config."""
        return 2 * self.num_layers * self.kv_cache_per_layer_bytes

    @property
    def max_tokens_in_cache(self) -> int:
        """Maximum tokens that can fit in GPU cache."""
        bytes_per_token = self.kv_cache_bytes_per_token or self.bytes_per_token
        return int(self.memory_GB * 1e9 / bytes_per_token)

    @property
    def memory_bound_throughput(self) -> float:
        """
        Memory-bound throughput (tokens/sec).

        For LLM inference, throughput is often memory-bandwidth bound:
        throughput = bandwidth / bytes_per_token
        """
        bytes_per_token = self.kv_cache_bytes_per_token or self.bytes_per_token
        return self.memory_bandwidth_GBps * 1e9 / bytes_per_token

    @property
    def prefill_time_per_token(self) -> float:
        """
        Calculate prefill time per token based on model and device.

        This is a simplified model. Real prefill time depends on:
        - Compute FLOPs for attention and FFN
        - Memory bandwidth for loading weights
        - Tensor parallel efficiency
        """
        # For prefill, we need to:
        # 1. Load weights (memory bound)
        # 2. Perform computation (compute bound)

        # Simplified: assume ~50% memory bound, 50% compute bound
        bytes_per_token = self.kv_cache_bytes_per_token or self.bytes_per_token

        # Memory time: load weights + KV cache
        total_bytes = self.num_params * 2 + bytes_per_token  # fp16 = 2 bytes
        memory_time_ms = (total_bytes / (self.memory_bandwidth_GBps * 1e9)) * 1000

        # Rough estimate: prefill is faster than decode due to batching
        return memory_time_ms * 0.5  # Conservative estimate

    def __str__(self) -> str:
        return (f"HardwareConfig(model={self.model_name}, device={self.device_name}, "
                f"layers={self.num_layers}, hidden={self.hidden_size}, "
                f"memory={self.memory_GB}GB, bandwidth={self.memory_bandwidth_GBps}GB/s)")

    @classmethod
    def from_model(
        cls,
        model_name: str,
        device_name: str = "A100",
        override_device_memory_GB: float = None
    ) -> "HardwareConfig":
        """
        Create HardwareConfig from model and device names.

        Args:
            model_name: Model name (e.g., "Llama-3.1-8B")
            device_name: Device name (e.g., "A100", "H100")
            override_device_memory_GB: Override device memory (e.g., for 50GB A100)

        Returns:
            HardwareConfig with calculated parameters
        """
        # Get model config
        if model_name in MODEL_CONFIGS:
            model_cfg = MODEL_CONFIGS[model_name]
        else:
            # Unknown model, use defaults
            model_cfg = {
                "num_layers": 32,
                "hidden_size": 4096,
                "num_heads": 32,
                "vocab_size": 32000,
            }

        # Get device config
        if device_name in DEVICE_CONFIGS:
            device_cfg = DEVICE_CONFIGS[device_name]
        else:
            # Unknown device, use A100 defaults
            device_cfg = DEVICE_CONFIGS["A100"]

        # Calculate KV cache bytes per token
        kv_cache_per_layer = 2 * model_cfg["hidden_size"]  # K or V per layer (fp16 = 2 bytes)
        kv_cache_per_token = 2 * model_cfg["num_layers"] * kv_cache_per_layer  # K + V for all layers

        # Calculate timing
        memory_bandwidth = device_cfg["memory_bandwidth_GBps"]

        # For inference timing, we use empirical scaling factors based on hardware.
        # The actual decode/prefill time depends on both compute and memory,
        # and is best calibrated against real benchmarks.
        #
        # Reference values from common deployments:
        # - A100 80GB: ~10-15 ms per decode token, ~0.1-0.2 ms per prefill token
        # - H100: ~5-8 ms per decode token, ~0.05-0.1 ms per prefill token
        #
        # We scale based on:
        # 1. Memory bandwidth ratio (relative to A100)
        # 2. Model size ratio (relative to 8B)
        
        # Reference: A100 has ~2 TB/s bandwidth, 8B model
        ref_bandwidth = 2039.0  # A100 bandwidth
        ref_model_params = 8.0  # 8B reference model
        
        # Scale decode time based on bandwidth (higher bandwidth = faster)
        bandwidth_factor = ref_bandwidth / memory_bandwidth
        
        # Scale based on model size (larger model = more compute = slower)
        # Approximate: decode time ∝ model_size^0.5 (some parallelism benefits)
        model_params = model_cfg.get("estimated_params_b", 8.0)
        size_factor = (model_params / ref_model_params) ** 0.5
        
        # Reference decode time for 8B on A100 (empirical)
        ref_decode_ms = 10.0  # ms per token
        ref_prefill_ms = 0.1   # ms per token
        
        decode_ms = ref_decode_ms * bandwidth_factor * size_factor
        prefill_ms = ref_prefill_ms * bandwidth_factor * size_factor

        return cls(
            # Model config
            model_name=model_name,
            num_layers=model_cfg["num_layers"],
            hidden_size=model_cfg["hidden_size"],
            num_heads=model_cfg["num_heads"],
            vocab_size=model_cfg.get("vocab_size", 32000),
            kv_cache_bytes_per_token=kv_cache_per_token,

            # Device config
            device_name=device_name,
            memory_bandwidth_GBps=device_cfg["memory_bandwidth_GBps"],
            memory_GB=override_device_memory_GB or device_cfg["memory_GB"],

            # Calculated timing
            prefill_latency_per_token_ms=max(prefill_ms, 0.001),  # At least 0.001ms
            decode_latency_per_token_ms=decode_ms,
            bytes_per_token=kv_cache_per_token,

            # Defaults
            cache_hit_speedup=10.0,
            ttft_overhead_ms=50.0,
            queue_delay_per_request_ms=100.0,
            h2d_bandwidth_GBps=32.0,  # PCIe 4.0 x16
        )


# Pre-defined configurations
A100_80G = HardwareConfig(
    model_name="Llama-3.1-8B",
    device_name="A100",
    prefill_latency_per_token_ms=0.1,
    decode_latency_per_token_ms=4.5,
    cache_hit_speedup=10.0,
    ttft_overhead_ms=50.0,
    queue_delay_per_request_ms=100.0,
    h2d_bandwidth_GBps=32.0,
    bytes_per_token=512.0,
    num_layers=32,
    hidden_size=4096,
    memory_bandwidth_GBps=2039,
    memory_GB=80,
    # Bandwidth model parameters
    bandwidth_efficiency=0.85,
    h2d_overhead_us=6.67,
    d2h_overhead_us=4.0,
)

A100_50G = HardwareConfig.from_model("Llama-3.1-8B", "A100", override_device_memory_GB=50)

H100 = HardwareConfig.from_model("Llama-3.1-8B", "H100")


@dataclass
class TimeBreakdown:
    """Detailed time breakdown for a request."""
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    h2d_ms: float = 0.0  # L2 load time
    queue_delay_ms: float = 0.0
    ttft_ms: float = 50.0
    total_ms: float = 0.0


# Timing modes
TIMING_MODE_NAIVE = "naive"
TIMING_MODE_AICONFIGURATOR = "aiconfigurator"


class TimingCalculator:
    """
    Calculates inference time based on cache hit type.

    Hit Types and Their Costs (per paper):
    - TTL Hit: KV preserved + priority scheduling (same prefill cost as L1)
    - L1 Hit: Fast - KV in GPU cache
    - L2 Hit: Medium - KV on CPU, needs H2D transfer
    - L1+L2 Miss: Slowest - Full prefill + full queueing overhead

    Note: TTL does NOT make inference faster. Its benefits are:
    1. Eviction protection - pinned programs' KV won't be evicted
    2. Priority scheduling - pinned programs get scheduled first

    Supports two timing modes:
    - naive: Simple analytical model (default)
    - aiconfigurator: AIConfigurator SDK for hardware-aware prediction
    """

    def __init__(
        self,
        hw: HardwareConfig = A100_80G,
        mode: str = TIMING_MODE_NAIVE,
        aiconfigurator_adapter: Optional["AIConfiguratorAdapter"] = None,
    ):
        """
        Initialize TimingCalculator.

        Args:
            hw: Hardware configuration
            mode: Timing mode ("naive" or "aiconfigurator")
            aiconfigurator_adapter: Optional AIConfigurator adapter for mode="aiconfigurator"
        """
        self.hw = hw
        self.mode = mode
        self._aiconfigurator_adapter = aiconfigurator_adapter

    def predict_time(
        self,
        total_input_tokens: int,
        output_tokens: int,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """
        Predict inference time components.

        Args:
            total_input_tokens: Total number of input tokens (including cached)
            output_tokens: Number of output tokens to generate
            l1_tokens: Number of tokens found in L1 cache (GPU)
            l2_tokens: Number of tokens found in L2 cache (CPU, needs H2D)

        Returns: (prefill_ms, decode_ms, h2d_ms)

        Time Calculation Formula:
        - L1 Hit: new_tokens × prefill + output × decode
          (e.g., 500 input, L1 hit 400 → 100 × prefill + 200 × decode)
        - L2 Hit: l2_tokens × h2d + new_tokens × prefill + output × decode
          (e.g., 500 input, L2 hit 400 → 400 × h2d + 100 × prefill + 200 × decode)
        - L1+L2 Miss: total_input × prefill + output × decode
          (e.g., 500 input → 500 × prefill + 200 × decode)

        H2D calculation uses GPU bandwidth:
        h2d_ms = l2_tokens × bytes_per_token / (h2d_bandwidth_GBps × 1000) × 1000
               = l2_tokens × bytes_per_token / (h2d_bandwidth_GBps × 10^6)
        """
        if self.mode == TIMING_MODE_AICONFIGURATOR and self._aiconfigurator_adapter:
            return self._predict_with_aiconfigurator(total_input_tokens, output_tokens, l1_tokens, l2_tokens)

        return self._predict_naive(total_input_tokens, output_tokens, l1_tokens, l2_tokens)

    def _predict_naive(
        self,
        total_input_tokens: int,
        output_tokens: int,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """Naive analytical prediction with bandwidth amortization model.
        
        Key insight: total_input_tokens includes ALL tokens that will be in memory
        after this request completes (matched + L2-loaded + new).
        
        For prefill calculation:
        - L1/L2 matched tokens: no prefill needed
        - L2-loaded tokens: need prefill (loaded from CPU to GPU)
        - New tokens: need prefill (never seen before)
        
        The formula (total_input_tokens - l1_tokens) gives tokens needing prefill:
        - L1 hit: total - matched = new tokens ✓
        - L2 hit: total - 0 = matched + l2 + new tokens → need prefill all!
        """
        h2d_ms = 0.0

        if l2_tokens > 0:
            # L2 hit: H2D transfer + prefill only NEW tokens
            # L2 tokens are already computed KV cache, just need H2D transfer
            # Only truly new tokens need prefill computation
            h2d_time = calc_transfer_time(
                num_tokens=l2_tokens,
                bytes_per_token=self.hw.bytes_per_token,
                bandwidth_GBps=self.hw.h2d_bandwidth_GBps,
                overhead_us=self.hw.h2d_overhead_us,
                efficiency=self.hw.bandwidth_efficiency
            )
            h2d_ms = h2d_time * 1000  # Convert seconds to ms
            # Calculate new tokens: total - L1 matched - L2 matched
            new_tokens = total_input_tokens - l1_tokens - l2_tokens
            prefill_ms = new_tokens * self.hw.prefill_latency_per_token_ms
        elif l1_tokens > 0:
            # L1 or TTL hit: prefill only the new tokens
            # TTL and L1 have the SAME prefill cost
            # TTL benefits are eviction protection + priority scheduling
            new_tokens = total_input_tokens - l1_tokens
            prefill_ms = new_tokens * self.hw.prefill_latency_per_token_ms
        else:
            # Miss: prefill all tokens
            prefill_ms = total_input_tokens * self.hw.prefill_latency_per_token_ms

        # Decode is the same regardless of hit type - KV is on GPU
        decode_ms = output_tokens * self.hw.decode_latency_per_token_ms

        return prefill_ms, decode_ms, h2d_ms

    def _predict_with_aiconfigurator(
        self,
        total_input_tokens: int,
        output_tokens: int,
        l1_tokens: int = 0,
        l2_tokens: int = 0,
    ) -> Tuple[float, float, float]:
        """AIConfigurator-based prediction."""
        # AIConfigurator uses "cached" vs "not cached" semantics
        # L1/L2 distinction is handled by StateManager tracking
        cached = l1_tokens + l2_tokens

        batch = self._aiconfigurator_adapter.ScheduleBatch(reqs=[
            self._aiconfigurator_adapter.ScheduleRequest(
                extend_length=total_input_tokens,
                past_kv_length=cached,
                is_cache_hit=cached > 0,
            )
        ])

        total_time_sec = self._aiconfigurator_adapter.predict_infer_time(batch)
        total_time_ms = total_time_sec * 1000

        # Simplified split (AIConfigurator provides better breakdown)
        prefill_ratio = 0.3 if cached == 0 else 0.1
        prefill_ms = total_time_ms * prefill_ratio
        decode_ms = total_time_ms * (1 - prefill_ratio)
        h2d_ms = 0.0  # H2D tracked via StateManager

        return prefill_ms, decode_ms, h2d_ms

    def calculate_time(
        self,
        total_input_tokens: int,
        output_tokens: int,
        hit_type: str = "L1+L2 Miss",
        l2_tokens: int = 0,
    ) -> float:
        """
        Calculate inference time in seconds.

        Args:
            total_input_tokens: Total number of input tokens (including cached)
            output_tokens: Number of output tokens
            hit_type: "L1", "L2", "TTL", or "L1+L2 Miss"
            l2_tokens: Number of tokens found in L2 cache
        """
        l1_tokens = total_input_tokens - l2_tokens if hit_type in ("L1", "TTL") else 0

        prefill_ms, decode_ms, h2d_ms = self.predict_time(
            total_input_tokens=total_input_tokens,
            output_tokens=output_tokens,
            l1_tokens=l1_tokens,
            l2_tokens=l2_tokens if hit_type == "L2" else 0,
        )

        # Pure inference time (prefill + decode)
        # Note: ttft_overhead is not added here - it's conceptual overhead already in prefill
        total_ms = prefill_ms + decode_ms

        # Convert to seconds
        return total_ms / 1000.0

    def get_time_breakdown(
        self,
        total_input_tokens: int,
        output_tokens: int,
        hit_type: str = "L1+L2 Miss",
        l2_tokens: int = 0,
        l1_tokens: int = None,  # Actual number of tokens in L1 cache (matched_tokens)
        num_waiting: int = 0,
    ) -> TimeBreakdown:
        """
        Get detailed time breakdown for a request.

        Args:
            total_input_tokens: Total number of input tokens (including cached)
            output_tokens: Number of output tokens
            hit_type: "L1", "L2", "TTL", or "L1+L2 Miss"
            l2_tokens: Number of tokens found in L2 cache
            l1_tokens: Actual number of tokens in L1 cache (matched_tokens).
                      If None, derived from hit_type for backward compatibility.
            num_waiting: Number of requests waiting in queue (for analysis)

        Returns: TimeBreakdown with detailed components
        """
        is_ttl_hit = hit_type == "TTL"

        # Use provided l1_tokens or derive from hit_type
        # l1_tokens should be the actual number of tokens matched in cache
        if l1_tokens is not None:
            # Use the actual matched tokens from radix tree
            cached_l1_tokens = l1_tokens
        elif hit_type in ("L1", "TTL"):
            # Backward compatibility: assume all tokens are in L1
            cached_l1_tokens = total_input_tokens - l2_tokens
        else:
            cached_l1_tokens = 0

        prefill_ms, decode_ms, h2d_ms = self.predict_time(
            total_input_tokens=total_input_tokens,
            output_tokens=output_tokens,
            l1_tokens=cached_l1_tokens,
            l2_tokens=l2_tokens if hit_type == "L2" else 0,
        )

        # TTL does NOT provide compute speedup - TTL and L1 have the SAME prefill cost.
        # TTL benefits are:
        # 1. Eviction protection - pinned programs' KV won't be evicted
        # 2. Priority scheduling - pinned programs get scheduled first
        # The actual time savings come from cache hits (prefill savings), not from TTL itself.
        #
        # Previous incorrect implementation applied a 0.5x speedup to TTL hits,
        # which overestimated the benefit. TTL just protects data, it doesn't
        # make the GPU compute faster.

        # Queueing delay: calculate from actual start time minus arrival time
        # This is the REAL queue wait time, not an estimate
        # Note: start_time and arrival_time are passed from scheduler, we receive them
        # via the scheduling log in scheduler._log_scheduling()
        # For get_time_breakdown(), we can't compute actual queue delay here
        # because we don't have start_time/arrival_time. The actual queue delay
        # is calculated in scheduler._log_scheduling() as queue_wait_ms.
        # Here we keep queue_delay_ms as 0 - actual value comes from scheduling log.
        queue_delay_ms = 0.0

        # Pure inference time (for advancing simulation time)
        # total_ms = prefill_ms + decode_ms (ttft_overhead is already included in prefill conceptually)
        # Note: ttft_overhead_ms is kept for reporting purposes but does NOT affect simulation timing
        total_ms = prefill_ms + decode_ms

        return TimeBreakdown(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            h2d_ms=h2d_ms,
            queue_delay_ms=queue_delay_ms,
            ttft_ms=self.hw.ttft_overhead_ms,
            total_ms=total_ms,
        )

    # ============= Batch Inference Timing Methods =============

    def calculate_batch_prefill_time(
        self,
        num_requests: int,
        avg_tokens_per_req: float,
        is_first_batch: bool = True,
    ) -> float:
        """
        Calculate prefill time for a batch of requests.

        Batch processing optimization (inspired by sglang-origin):
        - First batch: Need to load model weights, time = total_tokens * prefill_per_token
        - Subsequent batches: Weights already loaded, can parallelize

        For matrix multiplication in transformer:
        - Single request: O(n) for sequence length n
        - Batch B: O(B*n) without optimization
        - With batching optimization: O(B*n) but with better GPU utilization

        Args:
            num_requests: Number of requests in the batch
            avg_tokens_per_req: Average tokens per request
            is_first_batch: Whether this is the first batch (cold start)

        Returns:
            Prefill time in milliseconds
        """
        total_tokens = num_requests * avg_tokens_per_req

        if is_first_batch:
            # First batch: includes weight loading overhead
            base_time = total_tokens * self.hw.prefill_latency_per_token_ms
            # Batch efficiency factor (memory bandwidth saturation)
            # Larger batches achieve better utilization
            import math
            batch_efficiency = 1.0 / (1.0 + 0.1 * math.log(num_requests + 1))
            return base_time * batch_efficiency
        else:
            # Subsequent batches: parallel processing
            # Each request processes independently but shares compute resources
            # Throughput scales sub-linearly with batch size
            import math
            scaling_factor = 1.0 / math.sqrt(num_requests)
            base_time = total_tokens * self.hw.prefill_latency_per_token_ms
            return base_time * scaling_factor

    def calculate_decode_time(
        self,
        num_new_tokens: int,
        batch_size: int = 1,
    ) -> float:
        """
        Calculate decode time for a batch of requests.

        Decoding is typically memory-bandwidth bound:
        - Each token requires loading weights + KV cache access
        - Batch processing allows better utilization

        Args:
            num_new_tokens: Total new tokens to decode (sum across batch)
            batch_size: Number of concurrent requests

        Returns:
            Decode time in milliseconds
        """
        # Parallel decoding: batch size improves throughput
        # But decode is memory-bound, so scaling is sub-linear
        import math
        parallelism_factor = min(batch_size, 8)  # Cap at 8 for memory bandwidth
        effective_decode_time = self.hw.decode_latency_per_token_ms / math.sqrt(parallelism_factor)
        return num_new_tokens * effective_decode_time

    def calculate_batch_l2_load(
        self,
        total_tokens: int,
        batch_size: int = 1,
    ) -> float:
        """
        Calculate L2 (CPU) to L1 (GPU) transfer time for a batch.

        Optimization: Combine multiple L2 loads into a single transfer
        to amortize PCIe overhead.

        Args:
            total_tokens: Total tokens to load from L2
            batch_size: Number of requests contributing to this load

        Returns:
            Transfer time in milliseconds
        """
        if total_tokens == 0:
            return 0.0

        # H2D transfer time
        bytes_per_token = self.hw.kv_cache_bytes_per_token or self.hw.bytes_per_token
        total_bytes = total_tokens * bytes_per_token

        # Transfer time with bandwidth and overhead
        transfer_time_s = calc_transfer_time(
            num_tokens=total_tokens,
            bytes_per_token=bytes_per_token,
            bandwidth_GBps=self.hw.h2d_bandwidth_GBps,
            overhead_us=self.hw.h2d_overhead_us,
            efficiency=self.hw.bandwidth_efficiency,
        )

        # Batch efficiency: larger batches have better bandwidth utilization
        import math
        batch_efficiency = min(1.0, 0.8 + 0.05 * math.log(batch_size + 1))

        return transfer_time_s * 1000 * batch_efficiency  # Convert to ms

    def calculate_batch_time(
        self,
        batch_info: dict,
    ) -> dict:
        """
        Calculate total time for a batch of requests.

        Args:
            batch_info: Dictionary containing:
                - num_requests: Number of requests
                - avg_input_tokens: Average input tokens per request
                - total_output_tokens: Total output tokens across batch
                - l1_cached_tokens: Total tokens cached in L1
                - l2_cached_tokens: Total tokens cached in L2
                - is_first_batch: Whether this is the first batch

        Returns:
            Dictionary with time breakdown:
                - prefill_ms: Prefill time
                - decode_ms: Decode time
                - l2_load_ms: L2 load time
                - total_ms: Total time
        """
        num_requests = batch_info.get('num_requests', 1)
        avg_input_tokens = batch_info.get('avg_input_tokens', 100)
        total_output_tokens = batch_info.get('total_output_tokens', 100)
        l1_cached_tokens = batch_info.get('l1_cached_tokens', 0)
        l2_cached_tokens = batch_info.get('l2_cached_tokens', 0)
        is_first_batch = batch_info.get('is_first_batch', False)

        # Calculate tokens needing prefill
        total_input = num_requests * avg_input_tokens
        new_tokens = total_input - l1_cached_tokens - l2_cached_tokens

        # Prefill time (for tokens not in cache)
        prefill_ms = self.calculate_batch_prefill_time(
            num_requests=num_requests,
            avg_tokens_per_req=max(new_tokens / num_requests, 1) if num_requests > 0 else new_tokens,
            is_first_batch=is_first_batch,
        )

        # Decode time
        decode_ms = self.calculate_decode_time(
            num_new_tokens=total_output_tokens,
            batch_size=num_requests,
        )

        # L2 load time
        l2_load_ms = self.calculate_batch_l2_load(
            total_tokens=l2_cached_tokens,
            batch_size=num_requests,
        )

        return {
            'prefill_ms': prefill_ms,
            'decode_ms': decode_ms,
            'l2_load_ms': l2_load_ms,
            'total_ms': prefill_ms + decode_ms + l2_load_ms,
        }
