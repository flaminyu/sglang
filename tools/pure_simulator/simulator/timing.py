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

from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .aiconfigurator_adapter import AIConfiguratorAdapter


@dataclass
class HardwareConfig:
    """Hardware performance parameters."""
    prefill_latency_per_token_ms: float = 0.1   # Prefill time per token (forward pass)
    decode_latency_per_token_ms: float = 4.5   # Decode time per token
    cache_hit_speedup: float = 10.0             # Decode speedup on hit
    ttft_overhead_ms: float = 50.0              # Time to first token overhead
    # Queueing delay parameter (per paper Section 2.1)
    queue_delay_per_request_ms: float = 100.0   # Extra delay per waiting request
    # L2 (CPU Memory) parameters
    # H2D transfer: same latency as prefill since both need memory access
    # L2 loading cost = 1x prefill_latency (no penalty applied)
    # Reference values for bandwidth calculation
    h2d_bandwidth_GBps: float = 32.0          # PCIe 4.0 x16 bandwidth
    bytes_per_token: float = 512.0              # KV cache size per token


# A100 80GB configuration
A100_80G = HardwareConfig(
    prefill_latency_per_token_ms=0.1,
    decode_latency_per_token_ms=4.5,
    cache_hit_speedup=10.0,
    ttft_overhead_ms=50.0,
    queue_delay_per_request_ms=100.0,
    h2d_bandwidth_GBps=32.0,
    bytes_per_token=512.0,
)


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
        """Naive analytical prediction."""
        h2d_ms = 0.0
        new_tokens = total_input_tokens - l1_tokens - l2_tokens

        if l2_tokens > 0:
            # L2 hit: KV needs to be loaded from CPU to GPU via PCIe
            # H2D time = tokens × bytes_per_token / bandwidth
            h2d_ms = l2_tokens * self.hw.bytes_per_token / (self.hw.h2d_bandwidth_GBps * 1e6)
            # Then prefill the new tokens (not in any cache)
            prefill_ms = h2d_ms + new_tokens * self.hw.prefill_latency_per_token_ms
        elif l1_tokens > 0:
            # L1 or TTL hit: prefill only the new tokens
            # TTL and L1 have the SAME prefill cost
            # TTL benefits are eviction protection + priority scheduling
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

        # Pure inference time (prefill + decode + ttft)
        total_ms = prefill_ms + decode_ms + self.hw.ttft_overhead_ms

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

        # Queueing delay is only recorded for analysis purposes
        # Actual queue delay is captured in JCT = finish_time - arrival_time
        queue_delay_ms = num_waiting * self.hw.queue_delay_per_request_ms if not is_ttl_hit else 0.0

        # Pure inference time (for advancing simulation time)
        total_ms = prefill_ms + decode_ms + self.hw.ttft_overhead_ms

        return TimeBreakdown(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            h2d_ms=h2d_ms,
            queue_delay_ms=queue_delay_ms,
            ttft_ms=self.hw.ttft_overhead_ms,
            total_ms=total_ms,
        )
