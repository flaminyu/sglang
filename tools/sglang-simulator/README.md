# SGLang Simulator with TTL Support

This is a merged version of the official SGLang Simulator (PR #22250) with TTL (Time-To-Live) mechanism for KV Cache management in tool calling scenarios.

## Features

- **Hook-based interception**: Non-invasive framework interception using Python's built-in mechanisms
- **TTL Support**: Dynamic TTL mechanism for efficient KV Cache management during tool calls
- **HiCache Integration**: Simulates hierarchical cache (L1 HBM, L2 DRAM, L3 Disk)
- **Hardware Simulation**: Accurate simulation of A100/H100 GPUs with configurable memory and bandwidth
- **launch_server Integration**: Full SGLang server with simulator hooks

## IMPORTANT: Architecture

This project provides **Hook-based simulation** for SGLang. The key distinction:

```
┌─────────────────────────────────────────────────────────────────────┐
│                 sglang-simulator 架构                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  simulation/sglang/          ← Hook 类 (用于 launch_server)          │
│  ├── launch_server.py       ← 入口：启动真实 SGLang + 安装 Hook      │
│  ├── scheduler.py           ← C_SchedulerHook                       │
│  ├── model_runner.py        ← C_ModelRunnerHook                      │
│  ├── cache_controller.py    ← C_HiCacheController                   │
│  ├── hicache_storage.py     ← C_StorageBackendFactory               │
│  ├── hiradix_cache.py      ← C_HiRadixCacheHook                    │
│  └── bench_runner.py        ← 基准测试运行器                         │
│                                                                      │
│  simulation/agent/           ← Pure Simulation (快速验证)              │
│  ├── unified_scheduler.py   ← UnifiedSchedulerSimulator             │
│  ├── continuum_ttl.py      ← ContinuumTTLSimulator                  │
│  └── dataset.py            ← 测试数据生成                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Two Usage Modes

| Mode | Implementation | Use Case | Accuracy |
|------|----------------|----------|----------|
| **launch_server** | Real SGLang + Hooks | Production testing | 100% |
| Pure Simulation | `UnifiedSchedulerSimulator` | Quick validation | Approximate |

## Project Structure

```
tools/sglang-simulator/
├── src/sglang_simulator/
│   ├── hook/                     # Hook 机制
│   │   ├── base_hook.py         # 基类
│   │   ├── class_hook_entry.py  # 类 Hook 安装器
│   │   └── module_hook_entry.py # 模块 Hook 安装器
│   ├── simulation/
│   │   ├── sglang/             # Hook 类 (用于 launch_server)
│   │   │   ├── launch_server.py    # 入口
│   │   │   ├── scheduler.py         # C_SchedulerHook
│   │   │   ├── model_runner.py      # C_ModelRunnerHook
│   │   │   ├── cache_controller.py # C_HiCacheController
│   │   │   ├── hicache_storage.py   # C_StorageBackendFactory
│   │   │   └── hiradix_cache.py   # C_HiRadixCacheHook
│   │   └── agent/               # Pure Simulation (快速验证)
│   │       ├── unified_scheduler.py
│   │       ├── continuum_ttl.py
│   │       └── dataset.py
│   ├── spec/                    # 硬件规格
│   ├── time_predictor/          # 时间预测
│   └── utils/
├── configs/                    # 硬件配置
├── test/
└── README.md
```

## Installation

```bash
cd tools/sglang-simulator
pip install -e .

# For full functionality with AIConfigurator:
pip install aiconfigurator
```

## Usage: launch_server Mode (Recommended)

```bash
# Set environment
export SGLANG_SIMULATOR_CONFIG_PATH="./configs/a100_80g_200g_host.json"
export PYTHONPATH="./src:./python:$PYTHONPATH"

# Start server
python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "/path/to/32b/model" \
  --port 30000 \
  --chunked-prefill-size 4096
```

Then run benchmark:
```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --base-url http://localhost:30000 \
  --model "32b_model" \
  --dataset-name random \
  --num-prompts 100
```

### launch_server Flow

```python
# launch_server.py 核心逻辑
from sglang_simulator.simulation.sglang import (
    scheduler, model_runner, cache_controller, hicache_storage,
    hiradix_cache, mem_cache_allocator, mem_pool_host, sgl_kernel_hook,
)

# 1. Install hooks before starting server
sglang_simulator_hook.install_class_hooks([
    scheduler.C_SchedulerHook,
    model_runner.C_ModelRunnerHook,
    hicache_storage.C_StorageBackendFactory,
    cache_controller.C_HiCacheController,
    hiradix_cache.C_HiRadixCacheHook,
    mem_cache_allocator.C_PagedTokenToKVPoolAllocatorHook,
    mem_pool_host.C_MHATokenToKVPoolHostHook,
    mem_pool_host.C_HostKVCacheHook,
])

# 2. Start real SGLang server (hooks intercept internal calls)
from sglang.srt.entrypoints.http_server import launch_server
launch_server(server_args)
```

## Hook Classes Reference

| Hook Class | Intercepts | Simulates |
|------------|------------|-----------|
| `C_SchedulerHook` | Scheduler | Dynamic TTL decisions |
| `C_ModelRunnerHook` | Model Runner | Inference time prediction |
| `C_HiCacheController` | HiCache Controller | Cache allocation/PIN/UNPIN |
| `C_HiRadixCacheHook` | Radix Cache | LRU eviction policy |
| `C_StorageBackendFactory` | Storage Backend | L1/L2 cache behavior |
| `C_PagedTokenToKVPoolAllocatorHook` | Token Allocator | KV pool management |
| `C_MHATokenToKVPoolHostHook` | Host Memory Pool | Host cache operations |

## Hardware Configuration

### A100 80G + 200G Host Memory

Pre-configured at `configs/a100_80g_200g_host.json`:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100 SXM |
| HBM Capacity | 80 GB |
| HBM Bandwidth | 2.0 TB/s |
| Host Memory | 200 GB |
| H2D Bandwidth | 50 GB/s |
| Compute | 1560 TFLOPS |

### Creating Custom Configurations

```json
{
    "platform": {
        "accelerator": {
            "name": "a100_sxm",
            "vendor": "NVIDIA",
            "hbm_bandwidth_gb": 2.0,
            "hbm_capacity_gb": 80,
            "tflops": 1560
        },
        "memory_capacity_gb": 200,
        "memory_read_bandwidth_gb": 64,
        "memory_write_bandwidth_gb": 64
    },
    "predictor": {
        "name": "simulated",
        "prefill_latency_per_token_ms": 0.1,
        "decode_latency_per_token_ms": 5.0,
        "cache_hit_speedup": 10.0
    },
    "continuum": {
        "enabled": true,
        "ttl_default_sec": 5.0,
        "ttl_min_sec": 1.0,
        "ttl_max_sec": 15.0
    }
}
```

## Configuration Reference

### Platform Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `accelerator.name` | GPU model (a100_sxm, h100_sxm, h20, a800) | required |
| `accelerator.hbm_capacity_gb` | GPU memory in GB | 80 |
| `accelerator.hbm_bandwidth_gb` | GPU memory bandwidth in GB/s | 2.0 |
| `memory_capacity_gb` | Host memory in GB | 200 |
| `memory_read_bandwidth_gb` | Host memory read bandwidth | 64 |
| `memory_write_bandwidth_gb` | Host memory write bandwidth | 64 |

### Predictor Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `predictor.name` | Predictor type (simulated, aiconfigurator) | simulated |
| `prefill_latency_per_token_ms` | Prefill time per token in ms | 0.1 |
| `decode_latency_per_token_ms` | Decode time per token in ms | 5.0 |
| `cache_hit_speedup` | Speedup multiplier for cache hits | 10.0 |

### Continuum TTL Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `continuum.enabled` | Enable TTL mechanism | true |
| `ttl_default_sec` | Default TTL value in seconds | 5.0 |
| `ttl_min_sec` | Minimum TTL value | 1.0 |
| `ttl_max_sec` | Maximum TTL value | 15.0 |
| `history_threshold` | Samples needed for adaptive TTL | 3 |
| `memoryfulness` | Memory awareness factor (0-1) | 0.8 |
| `avg_queue_delay_sec` | Average queue delay for TTL calculation | 2.0 |

## TTL Mechanism

The TTL mechanism allows KV Cache to be pinned during tool execution:

### How It Works

1. **PIN Trigger**: When a request with a tool call completes (not last turn), its KV cache is PIN-protected
2. **TTL Calculation**: Dynamic TTL based on historical tool execution times and idle gaps
3. **Protection**: Pinned entries are immune to LRU eviction
4. **Release**: PIN released when TTL expires or next turn arrives

### TTL Formula

```
tau* = argmax_tau P(tau,f) x (T x eta + Prefill-Reload) - tau
```

Where:
- P(tau,f) = CDF of tool execution time
- T = average queue delay
- eta = memoryfulness factor
- Prefill-Reload = KV reconstruction cost

## Dependencies

- `numpy`
- `scikit-learn`
- `xgboost`
- `aiconfigurator` (optional, for accurate latency prediction)

## References

- SGLang Simulator: https://github.com/sgl-project/sglang/issues/21891
- Original PR: https://github.com/sgl-project/sglang/pull/22250
- TTL Mechanism: See `CONTINUUM_KV_CONTROL.md` in project root
