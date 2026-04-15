# SGLang Simulator with TTL Support

This is a merged version of the official SGLang Simulator (PR #22250) with TTL (Time-To-Live) mechanism for KV Cache management in tool calling scenarios.

## Features

- **CPU-based simulation**: Run inference simulations without GPU hardware
- **TTL Support**: Dynamic TTL mechanism for efficient KV Cache management during tool calls
- **HiCache Integration**: Simulates hierarchical cache (L1 HBM, L2 DRAM, L3 Disk)
- **Hook-based interception**: Non-invasive framework interception using Python's built-in mechanisms

## Project Structure

```
tools/sglang-simulator/
├── src/sglang_simulator/
│   ├── dataset/          # Dataset handling for simulation
│   ├── hook/             # Hook mechanisms for SGLang interception
│   ├── simulation/       # Core simulation logic
│   │   ├── manager/      # Configuration and state management
│   │   ├── sglang/       # SGLang-specific hooks (scheduler, cache, etc.)
│   │   └── benchmark/     # Benchmark utilities
│   ├── spec/             # Model and hardware specifications
│   ├── time_predictor/   # Latency prediction models
│   └── utils/            # Utility functions
├── test/
│   └── assets/
│       └── config.json   # Sample configuration
├── pyproject.toml
├── setup.py
└── README.md
```

## Installation

```bash
cd tools/sglang-simulator
pip install -e .

# For full functionality with AIConfigurator:
pip install aiconfigurator
```

## Usage

### 1. Start Simulation Server

```bash
python3 -m sglang_simulator.simulation.sglang.launch_server \
  --model-path "Qwen/Qwen3-8B" \
  --sim-config-path test/assets/config.json \
  --continuum-ttl-sec 30.0  # Enable TTL with 30s default
```

### 2. Run Benchmark

```bash
python3 -m sglang_simulator.simulation.bench_serving \
  --warmup-requests 0 \
  --model "Qwen/Qwen3-8B" \
  --dataset-name random \
  --request-rate 4 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-prompts 10
```

## Configuration

The simulator uses a JSON configuration file:

```json
{
    "platform": {
        "accelerator": {
            "name": "a100_sxm",
            "hbm_capacity_gb": 80
        },
        "disk_read_bandwidth_gb": 8,
        "disk_write_bandwidth_gb": 8,
        "memory_read_bandwidth_gb": 64,
        "memory_write_bandwidth_gb": 64,
        "num_device_per_node": 8
    },
    "predictor": {
        "name": "aiconfigurator"
    },
    "scheduler": {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "backend_version": "0.5.9"
    }
}
```

## TTL Mechanism

The TTL mechanism allows KV Cache to be pinned during tool execution:

- Set `continuum-ttl-sec` to enable TTL policy
- Requests with `__ttl=<seconds>` in extra_key will have their KV Cache pinned
- Pinned requests are protected from eviction during the TTL period
- After TTL expires, KV Cache follows normal LRU eviction

## Dependencies

- `numpy`
- `scikit-learn`
- `xgboost`
- `aiconfigurator` (optional, for accurate latency prediction)

## References

- SGLang Simulator: https://github.com/sgl-project/sglang/issues/21891
- Original PR: https://github.com/sgl-project/sglang/pull/22250
- TTL Mechanism: See `CONTINUUM_KV_CONTROL.md` in project root
