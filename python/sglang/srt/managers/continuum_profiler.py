"""
Continuum Offline Profiler: Estimate T (queue delay) and other parameters from hardware specs.

This module provides offline estimation of key Continuum parameters without running workloads:
- T: average queueing delay per unit memory
- Prefill-Reload: time to prefill or reload KV cache
- Memory bandwidth and compute throughput

Based on:
- GPUDojo methodology: https://gpudojo.com/articles/speed-estimation
- Continuum paper: arXiv 2511.02230
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str
    memory_bandwidth_gb_s: float
    fp16_tflops: float
    vram_gb: float
    sm_version: str  # e.g., "8.9" for Ada Lovelace


@dataclass
class ModelInfo:
    """LLM model configuration."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    intermediate_size: int  # FFN hidden size
    max_position_embeddings: int
    quantization: str = "fp16"  # fp16, int8, int4

    @property
    def num_parameters(self) -> int:
        """Estimate total model parameters."""
        # Simplified estimation based on standard LLM architecture
        # embedding = vocab_size * hidden_size
        # attention = 4 * layers * hidden_size^2 (Q, K, V, O)
        # ffn = 3 * layers * hidden_size * intermediate_size
        # approx = 2 * layers * hidden_size^2 + layers * hidden_size * intermediate_size
        embedding_params = self.vocab_size * self.hidden_size
        attention_params = 4 * self.num_layers * self.hidden_size * self.hidden_size
        ffn_params = 2 * self.num_layers * self.hidden_size * self.intermediate_size
        # Add some overhead for embeddings, norms, etc.
        return embedding_params + attention_params + ffn_params + self.hidden_size * 10

    @property
    def model_size_gb(self) -> float:
        """Model size in GB based on quantization."""
        params_bytes = {
            "fp16": 2,
            "int8": 1,
            "int4": 0.5,
        }
        bytes_per_param = params_bytes.get(self.quantization, 2)
        return self.num_parameters * bytes_per_param / (1024**3)

    @property
    def kv_cache_per_token_bytes(self) -> float:
        """KV cache memory per token in bytes."""
        # For each layer: 2 * (hidden_size / num_kv_heads) * 2 (K + V) * 2 (fp16)
        bytes_per_layer = 2 * (self.hidden_size // self.num_key_value_heads) * 2 * 2
        return bytes_per_layer * self.num_layers


@dataclass
class ProfilingResult:
    """Result of offline profiling estimation."""
    # GPU info
    gpu_name: str
    vram_gb: float
    memory_bandwidth_gb_s: float
    fp16_tflops: float

    # Model info
    model_name: str
    num_parameters: int
    model_size_gb: float

    # Estimated throughput (tokens/sec)
    prefill_throughput: float  # tokens/sec during prefill
    decode_throughput: float   # tokens/sec during decode (memory-bandwidth-bound)

    # Estimated latency (seconds)
    prefill_latency_per_token: float   # seconds per token
    decode_latency_per_token: float    # seconds per token

    # Continuum parameters
    t_queue_delay: float  # T: average queueing delay per unit memory (seconds)
    prefill_reload_cost: float  # Prefill/Reload cost per 1K tokens (seconds)
    kv_load_back_speed_gb_s: float  # CPU->GPU load back speed (GB/s)

    # Memory estimation
    max_tokens_in_memory: int  # Max tokens that fit in GPU memory
    kv_bytes_per_token: float  # Actual KV cache bytes per token


def get_gpu_info() -> GPUInfo:
    """Get GPU hardware information from torch.cuda."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device_props = torch.cuda.get_device_properties(0)
    gpu_name = device_props.name
    vram_gb = device_props.total_memory / (1024**3)

    # GPU specifications lookup table
    # SM version format: major.minor (e.g., "8.9" for Ada Lovelace)
    gpu_specs = {
        # Ada Lovelace (L40, RTX 4090, L40S)
        "L40": GPUInfo("L40", 1004.4, 733.0, 48.0, "8.9"),
        "L40S": GPUInfo("L40S", 1004.4, 733.0, 48.0, "8.9"),
        "RTX 4090": GPUInfo("RTX 4090", 1008.0, 1321.0, 24.0, "8.9"),
        "RTX 4090 D": GPUInfo("RTX 4090 D", 1008.0, 1458.0, 24.0, "8.9"),
        # Hopper (H100)
        "H100": GPUInfo("H100", 3350.0, 3957.0, 80.0, "9.0"),
        "H100 SXM": GPUInfo("H100 SXM", 3350.0, 3957.0, 80.0, "9.0"),
        "H100 PCIe": GPUInfo("H100 PCIe", 2000.0, 3957.0, 80.0, "9.0"),
        # Ampere (A100, RTX 3090)
        "A100": GPUInfo("A100", 1935.0, 312.0, 80.0, "8.0"),
        "A100-SXM4-80GB": GPUInfo("A100-SXM4-80GB", 2039.0, 312.0, 80.0, "8.0"),
        "A100-SXM4-40GB": GPUInfo("A100-SXM4-40GB", 1555.0, 312.0, 40.0, "8.0"),
        "RTX 3090": GPUInfo("RTX 3090", 936.2, 285.0, 24.0, "8.6"),
        "RTX 3080": GPUInfo("RTX 3080", 760.3, 285.0, 10.0, "8.6"),
        # Volta (V100)
        "V100": GPUInfo("V100", 900.0, 125.0, 32.0, "7.0"),
        "Tesla V100": GPUInfo("Tesla V100", 900.0, 125.0, 32.0, "7.0"),
    }

    # Try to match GPU name
    for name, spec in gpu_specs.items():
        if name in gpu_name:
            return GPUInfo(
                name=spec.name,
                memory_bandwidth_gb_s=spec.memory_bandwidth_gb_s,
                fp16_tflops=spec.fp16_tflops,
                vram_gb=vram_gb,
                sm_version=spec.sm_version,
            )

    # Fallback: estimate based on compute capability
    sm_version = f"{device_props.major}.{device_props.minor}"
    sm_to_bandwidth = {
        "7.0": 900.0,   # Volta
        "7.5": 1000.0,  # Turing
        "8.0": 2000.0,  # Ampere
        "8.6": 1000.0,  # Ampere (GA10x)
        "8.9": 1000.0,  # Ada Lovelace
        "9.0": 3350.0,  # Hopper
    }
    sm_to_tflops = {
        "7.0": 125.0,
        "7.5": 130.0,
        "8.0": 312.0,
        "8.6": 285.0,
        "8.9": 1321.0,
        "9.0": 3957.0,
    }

    return GPUInfo(
        name=gpu_name,
        memory_bandwidth_gb_s=sm_to_bandwidth.get(sm_version, 1000.0),
        fp16_tflops=sm_to_tflops.get(sm_version, 100.0),
        vram_gb=vram_gb,
        sm_version=sm_version,
    )


def estimate_prefill_throughput(
    gpu_info: GPUInfo,
    model_info: ModelInfo,
    efficiency_factor: float = 0.5,
) -> float:
    """
    Estimate prefill throughput (tokens/sec).

    Prefill is compute-bound:
        tokens/sec = (FP16 TFLOPS × efficiency) / (2 × Parameters × Ops_per_token)

    For transformer: ~2 FLOPs per parameter per token per layer
    """
    ops_per_token_per_layer = 2 * model_info.hidden_size
    ops_per_token = ops_per_token_per_layer * model_info.num_layers
    # + attention projection overhead
    ops_per_token *= 1.2

    tflops_effective = gpu_info.fp16_tflops * efficiency_factor
    flops_per_sec = tflops_effective * 1e12

    throughput = flops_per_sec / (2 * model_info.num_parameters * ops_per_token)

    return max(throughput, 1.0)  # At least 1 token/sec


def estimate_decode_throughput(
    gpu_info: GPUInfo,
    model_info: ModelInfo,
    efficiency_factor: float = 0.55,
) -> float:
    """
    Estimate decode throughput (tokens/sec).

    Decode is memory-bandwidth-bound:
        tokens/sec = (Memory Bandwidth × efficiency) / Model Size
    """
    effective_bandwidth = gpu_info.memory_bandwidth_gb_s * efficiency_factor
    throughput = effective_bandwidth / model_info.model_size_gb

    return max(throughput, 0.1)  # At least 0.1 tokens/sec


def estimate_t_queue_delay(
    gpu_info: GPUInfo,
    model_info: ModelInfo,
    prefill_throughput: float,
    decode_throughput: float,
    avg_batch_size: int = 4,
    system_overhead_sec: float = 0.01,
) -> float:
    """
    Estimate T (average queueing delay per unit memory).

    This estimates the scheduling overhead per token in the queue.
    T represents the cost of keeping KV in GPU vs. letting it spill.

    Based on Continuum paper's offline profiling approach:
    - Measure how long a request waits in queue when others are running
    - This depends on the GPU's ability to switch between requests

    In practice:
    - With batching: higher throughput, lower queueing
    - With context: longer prefill blocks others

    Estimated as:
        T ≈ 1/throughput × batch_penalty × system_overhead
    """
    # Average time to process one batch
    avg_processing_time = 1.0 / prefill_throughput + 1.0 / decode_throughput

    # Queueing delay increases with more concurrent requests
    # Assumes FCFS with some batching
    batch_penalty = max(1.0, avg_batch_size * 0.5)

    # System overhead (scheduler, memory manager, etc.)
    T = avg_processing_time * batch_penalty + system_overhead_sec

    # Cap T at reasonable values (0.001 to 1.0 seconds)
    return max(0.001, min(T, 1.0))


def estimate_kv_load_back_speed(
    gpu_info: GPUInfo,
    pci_gen: int = 4,
    pci_width: int = 16,
) -> float:
    """
    Estimate CPU->GPU KV load back speed (GB/s).

    Based on PCIe bandwidth:
        speed = PCIe_gen × lanes × encoding_rate / 8
    """
    # PCIe bandwidth per lane (effective, with 128/130 encoding)
    pci_bandwidth = {
        3: 0.985,   # PCIe 3.0: 8 GT/s × 1.0 (64b/66b)
        4: 1.969,   # PCIe 4.0: 16 GT/s × 0.98 (128/130b)
        5: 3.938,   # PCIe 5.0: 32 GT/s × 0.98
    }

    # For H100 with NVLink, use NVLink bandwidth
    if "H100" in gpu_info.name:
        # NVLink 4.0: ~50 GB/s per link, 18 links on SXM
        return min(900.0, gpu_info.memory_bandwidth_gb_s * 0.3)

    base_speed = pci_bandwidth.get(pci_gen, 1.969) * pci_width

    # Limit to reasonable values
    return max(10.0, min(base_speed, 100.0))


def run_offline_profiling(
    model_name: str = "unknown",
    model_config: Optional[dict] = None,
    mem_fraction_kv: float = 0.3,
    kv_load_back_speed: Optional[float] = None,
) -> ProfilingResult:
    """
    Run offline profiling to estimate Continuum parameters.

    Args:
        model_name: Name of the model (for logging)
        model_config: Model configuration dict with keys:
            - num_layers
            - hidden_size
            - num_attention_heads
            - num_key_value_heads (optional)
            - vocab_size
            - intermediate_size (optional)
            - max_position_embeddings
        mem_fraction_kv: Fraction of GPU memory for KV cache
        kv_load_back_speed: Override KV load back speed (GB/s)
    """
    # Get GPU info
    gpu_info = get_gpu_info()

    # Create model info
    default_kv_heads = (model_config.get("num_attention_heads", 32)
                       if model_config is None
                       else model_config.get("num_key_value_heads",
                                              model_config.get("num_attention_heads", 32)))
    default_intermediate = (model_config.get("hidden_size", 4096) * 4
                           if model_config is None
                           else model_config.get("intermediate_size",
                                                  model_config.get("hidden_size", 4096) * 4))

    if model_config is None:
        model_config = {}

    model_info = ModelInfo(
        num_layers=model_config.get("num_layers", 32),
        hidden_size=model_config.get("hidden_size", 4096),
        num_attention_heads=model_config.get("num_attention_heads", 32),
        num_key_value_heads=model_config.get("num_key_value_heads", default_kv_heads),
        vocab_size=model_config.get("vocab_size", 32000),
        intermediate_size=model_config.get("intermediate_size", default_intermediate),
        max_position_embeddings=model_config.get("max_position_embeddings", 8192),
        quantization=model_config.get("quantization", "fp16"),
    )

    # Estimate throughput
    prefill_throughput = estimate_prefill_throughput(gpu_info, model_info)
    decode_throughput = estimate_decode_throughput(gpu_info, model_info)

    # Calculate latency
    prefill_latency_per_token = 1.0 / prefill_throughput
    decode_latency_per_token = 1.0 / decode_throughput

    # Estimate T (queue delay per unit memory)
    t_queue_delay = estimate_t_queue_delay(
        gpu_info, model_info, prefill_throughput, decode_throughput
    )

    # Estimate prefill/reload cost per 1K tokens
    prefill_reload_cost = 1000.0 * prefill_latency_per_token

    # Estimate KV load back speed
    if kv_load_back_speed is None:
        pci_gen = int(os.getenv("CONTINUUM_PCIE_GEN", "4"))
        pci_width = int(os.getenv("CONTINUUM_PCIE_WIDTH", "16"))
        kv_load_back_speed = estimate_kv_load_back_speed(gpu_info, pci_gen, pci_width)

    # Estimate memory capacity
    kv_bytes_per_token = model_info.kv_cache_per_token_bytes
    max_tokens_in_memory = int((gpu_info.vram_gb * 1024 * mem_fraction_kv) / kv_bytes_per_token)

    return ProfilingResult(
        gpu_name=gpu_info.name,
        vram_gb=gpu_info.vram_gb,
        memory_bandwidth_gb_s=gpu_info.memory_bandwidth_gb_s,
        fp16_tflops=gpu_info.fp16_tflops,
        model_name=model_name,
        num_parameters=model_info.num_parameters,
        model_size_gb=model_info.model_size_gb,
        prefill_throughput=prefill_throughput,
        decode_throughput=decode_throughput,
        prefill_latency_per_token=prefill_latency_per_token,
        decode_latency_per_token=decode_latency_per_token,
        t_queue_delay=t_queue_delay,
        prefill_reload_cost=prefill_reload_cost,
        kv_load_back_speed_gb_s=kv_load_back_speed,
        max_tokens_in_memory=max_tokens_in_memory,
        kv_bytes_per_token=kv_bytes_per_token,
    )


def print_profiling_result(result: ProfilingResult) -> None:
    """Print profiling result in a readable format."""
    print("=" * 70)
    print("CONTINUUM OFFLINE PROFILING RESULT")
    print("=" * 70)
    print(f"\n[GPU]")
    print(f"  Name:            {result.gpu_name}")
    print(f"  VRAM:            {result.vram_gb:.1f} GB")
    print(f"  Memory BW:       {result.memory_bandwidth_gb_s:.1f} GB/s")
    print(f"  FP16 TFLOPS:     {result.fp16_tflops:.0f}")

    print(f"\n[Model]")
    print(f"  Name:            {result.model_name}")
    print(f"  Parameters:      {result.num_parameters / 1e9:.2f} B")
    print(f"  Model Size:      {result.model_size_gb:.2f} GB")

    print(f"\n[Estimated Throughput]")
    print(f"  Prefill:         {result.prefill_throughput:.1f} tokens/sec")
    print(f"  Decode:          {result.decode_throughput:.2f} tokens/sec")

    print(f"\n[Estimated Latency]")
    print(f"  Prefill/token:   {result.prefill_latency_per_token*1000:.4f} ms")
    print(f"  Decode/token:    {result.decode_latency_per_token*1000:.2f} ms")

    print(f"\n[Continuum Parameters]")
    print(f"  T (queue delay): {result.t_queue_delay*1000:.4f} ms")
    print(f"  Prefill/Reload: {result.prefill_reload_cost:.4f} sec per 1K tokens")
    print(f"  KV Load Back:    {result.kv_load_back_speed_gb_s:.1f} GB/s")

    print(f"\n[Memory Estimation]")
    print(f"  KV bytes/token:  {result.kv_bytes_per_token:.0f} bytes")
    print(f"  Max tokens:      {result.max_tokens_in_memory}")
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Example: Profile for Llama-3.1 8B on typical GPU
    llama_config = {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA
        "vocab_size": 128256,
        "intermediate_size": 14336,
        "max_position_embeddings": 131072,
    }

    result = run_offline_profiling(
        model_name="Meta-Llama-3.1-8B-Instruct",
        model_config=llama_config,
    )
    print_profiling_result(result)
