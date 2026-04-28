# Pure Python KV Cache Simulator

A standalone pure Python simulator that replicates SGLang's KV cache behavior for evaluating the Continuum TTL mechanism. Uses **RadixTree prefix matching** for accurate token sharing simulation.

## Overview

This simulator provides a high-level simulation of KV cache operations without requiring actual LLM inference. It is designed to:

- **Evaluate TTL pinning strategies** for multi-turn agent scenarios
- **Benchmark cache hit rates** under various workloads
- **Test eviction policies** (LRU, LFU, FIFO, Continuum)
- **Analyze cache pressure scenarios** with synthetic request logs
- **Simulate tool exceptions** (timeouts, retries, errors) and their impact on TTL
- **Event-driven simulation** with accurate queue delay modeling

## Quick Start

### Basic Usage

```bash
cd /home/comp/csgfyu/multi-agents/KVBlocking/sglang_continuum/tools/pure_simulator

# Run comprehensive experiment with JCT analysis
python experiments/run_comprehensive_experiment.py

# Generate HTML report
python experiments/generate_html_report.py

# Run three-variable experiment
python experiments/run_three_var_experiment.py

# Run exception scenario experiment
python experiments/run_exception_experiment.py

# Run single simulation
python run_simulation.py requests.jsonl -c 100000 -t 5.0 -e lru -v
```

## Architecture

### Core Components

```
simulator/
├── main.py              # KVCacheSimulator orchestration (Event-driven + Batch modes)
├── cache_allocator.py   # Token slot allocation/deallocation
├── radix_tree.py       # Prefix-based KV cache storage with RadixTree
├── radix_key.py        # Token sequence keys
├── tree_node.py        # Radix tree nodes
├── ttl_manager.py       # TTL pinning logic with forced unpin for deadlock prevention
├── scheduler.py         # Request processing with priority scheduling
├── eviction.py          # Eviction policies
├── request.py          # Request data structures
├── request_log.py      # Log file I/O
├── stats.py            # Statistics collection
└── timing.py          # Timing model (Prefill + Decode + TTFT)
```

### Key Concepts

#### RadixTree Prefix Cache

The simulator uses a **RadixTree** (compressed prefix tree) to store KV cache entries. This allows:

- **Prefix matching**: Common token sequences are shared efficiently
- **Efficient lookup**: O(k) where k is the key length
- **Accurate token sharing**: Token sequence [1,2,3,4] shares prefix with [1,2,3,4,5]
- **Namespace isolation**: Different `extra_key` values prevent cross-contamination

#### TTL Pinning (Continuum)

TTL pinning protects cache entries during tool execution:

1. When a tool call is issued, the matched prefix is "pinned" with a TTL
2. **Pinned entries can be evicted** - TTL only affects scheduling priority, not eviction
3. When TTL expires, the pin is released and the entry becomes evictable
4. On cache eviction, entries are spilled to L2 (Host Memory) instead of being discarded

#### Multi-Level Cache

```
┌─────────────────────────────────────────────────────────┐
│                    Request Stream                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              L2 Cache (GPU HBM - LRU)                   │
│  - Fast access, limited capacity                        │
│  - Eviction → spill to L2                              │
└─────────────────────┬───────────────────────────────────┘
                      │ (evict)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              L2 Cache (Host Memory)                      │
│  - Slower access, larger capacity                       │
│  - Full eviction on capacity exceeded                   │
└─────────────────────────────────────────────────────────┘
```

#### Timing Model

The simulator uses an event-driven timing model that accurately captures queueing behavior:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Time Model Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Prefill Time │ +  │ Decode Time  │ +  │ TTFT Overhead│ =        │
│  │  (miss tokens) │  │ (output tokens)│  │              │         │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
│                           ↓                                        │
│                    inference_time (纯推理时间)                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  JCT = finish_time - arrival_time                            │  │
│  │      = 排队等待时间 + inference_time                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Key Design:                                                        │
│  - Queue delay is NOT manually added to inference_time              │
│  - Queue delay is naturally captured in JCT                          │
│  - TTL requests get priority scheduling (lower JCT)                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
1. **Pure inference time**: `inference_time = prefill + decode + ttft` (no queue delay)
2. **JCT includes everything**: `JCT = finish_time - arrival_time` naturally includes queue wait
3. **GPU availability tracking**: Requests must wait if GPU is busy
4. **TTL priority**: TTL-pinned requests are scheduled first, reducing their JCT

#### L2 Cache (Host Memory)

L2 load-back cost calculation:
- **L2 load cost**: `tokens * prefill_latency * l1_reload_penalty`
- **Full miss cost**: `tokens * prefill_latency`
- **L2 benefit**: Full miss - L2 load = savings per hit

#### Pinned Eviction Fix

A deadlock prevention mechanism that proactively unpins cached KV nodes when:
1. The cache is full (`needed_tokens > available_tokens`)
2. All remaining nodes are pinned (eviction returns empty list)

The fix selects victims based on `last_access_time` (oldest first) to maintain LRU semantics.

## Request Log Format

Requests are stored in JSONL format (one JSON object per line):

```json
{
    "rid": "prog_0000_turn_0",
    "program_id": "prog_0000",
    "turn_index": 0,
    "token_ids": [1, 2, 3, 4, 5, ...],
    "input_len": 500,
    "output_len": 100,
    "arrival_time": 0.0,
    "is_tool_call": true,
    "tool_name": "tool_call",
    "extra_key": "__continuum_prog=prog_0000__ttl=5.0"
}
```

## Experiments

### Comprehensive Experiment (Recommended)

Run multi-parameter experiments with JCT analysis, timeline visualization, and detailed breakdown:

```bash
python experiments/run_comprehensive_experiment.py

# Generate HTML report (saved to results/)
python experiments/generate_html_report.py
```

**Output files:**
- `results/comprehensive_experiment_results.json` - Raw JSON data
- `results/kv_cache_experiment_report.html` - Interactive HTML report

### Three-Variable Experiment

Variables: Concurrency | Turns | Cache Size

```bash
python experiments/run_three_var_experiment.py
```

**Key Results:**

| Metric | Value |
|--------|-------|
| Total Experiments | 192 |
| Average Speedup | 1.43x |
| Best Speedup | 1.79x (c=40, t=20) |
| TTL Hits | Increase with concurrency and turns |

### Exception Scenario Experiment

Tests TTL robustness under tool exceptions (timeouts, errors, retries).

```bash
python experiments/run_exception_experiment.py
```

**Exception Types:**

| Type | Effect | Severity |
|------|--------|----------|
| Timeout | 5x longer execution | Medium-High |
| Error | Short delay then fail | Low |
| Retry | 2x longer (retry) | Medium |
| Hang | 10x longer execution | High |

**Results:**

| Condition | BL Hit | CT Hit | Speedup |
|-----------|--------|--------|---------|
| Normal | 99.6% | 99.6% | 1.73x |
| Timeout (10%) | 99.6% | 99.6% | 1.67x |
| Timeout (30%) | 99.6% | 99.6% | 1.50x |
| Error (10%) | 99.6% | 99.6% | 1.73x |
| Error (30%) | 99.6% | 99.6% | 1.74x |
| Retry (10%) | 99.6% | 99.6% | 1.69x |
| Retry (30%) | 99.6% | 99.6% | 1.57x |

**Key Findings:**

- **Normal case**: TTL provides consistent **1.73x speedup**
- **Exception cases**: TTL speedup varies (1.50x-1.74x) depending on exception type
- **Timeout/Retry**: Speedup decreases as exception rate increases (more timeouts = more LRU evictions)
- **Error**: Minimal impact on TTL effectiveness (short duration)

## Experiment Results (Latest Run)

### Three-Variable Experiment

```
====================================================================================================
ANALYSIS 1: EFFECT OF CONCURRENCY (Fixed: turns=10, l0=10K, l1=50K)
====================================================================================================
Concurrency  Turns    L2 Size      BL Hit     CT Hit     Hit Imp    TTL Hits   Speedup
10           10       10,000       98.0%      98.0%      +0.0%      60         1.5721x
20           10       10,000       99.0%      99.0%      +0.0%      120        1.6060x
30           10       10,000       99.3%      99.3%      +0.0%      180        1.6182x
40           10       10,000       99.5%      99.5%      +0.0%      240        1.6244x

====================================================================================================
ANALYSIS 2: EFFECT OF TURNS (Fixed: concurrency=30, l0=10K, l1=50K)
====================================================================================================
Concurrency  Turns    L2 Size      BL Hit     CT Hit     Hit Imp    TTL Hits   Speedup
30           5        10,000       98.7%      96.7%      -2.0%      90         1.2777x
30           10       10,000       99.3%      99.3%      +0.0%      180        1.6182x
30           15       10,000       99.6%      99.6%      +0.0%      300        1.7309x
30           20       10,000       99.7%      99.7%      +0.0%      420        1.7833x

====================================================================================================
SUMMARY
====================================================================================================
  Total Experiments: 192
  Average Hit Rate Improvement: -0.58%
  Average Speedup: 1.4264x

  Best Speedup: c=40, t=20, l0=5000 -> 1.7877x
```

### TTL Benefit Confirmation

Simple test with cache pressure (small cache, high competition):

| Metric | With TTL | Without TTL | Improvement |
|--------|----------|-------------|-------------|
| Hit Rate | 20.0% | 11.1% | +8.9% |
| Evicted Tokens | 400 | 3200 | -87.5% |
| Duration | 5.48s | 7.08s | **1.29x speedup** |

## API Reference

### KVCacheSimulator

```python
from simulator import KVCacheSimulator

sim = KVCacheSimulator(
    cache_capacity=100000,    # Max tokens in cache
    default_ttl=5.0,          # Default TTL in seconds
    eviction_policy="lru",    # lru, lfu, fifo, continuum
    enable_ttl=True,          # Enable TTL pinning
)

# Load requests
sim.load_requests("requests.jsonl")

# Run simulation
result = sim.run(verbose=True)

# Access results
print(f"Hit Rate: {result.cache_stats.hit_rate:.1%}")
print(f"TTL Hits: {result.cache_stats.ttl_hits}")
print(f"Duration: {result.duration:.2f}s")
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_capacity` | int | 100000 | Maximum tokens in cache |
| `default_ttl` | float | 5.0 | Default TTL in seconds |
| `min_ttl` | float | 0.5 | Minimum TTL value |
| `max_ttl` | float | 15.0 | Maximum TTL value |
| `eviction_policy` | str | "lru" | Eviction policy |
| `enable_ttl` | bool | True | Enable TTL pinning |
| `page_size` | int | 1 | PagedAttention page size |

### Result Fields

```python
result.cache_stats.hit_rate       # Cache hit rate (0.0-1.0)
result.cache_stats.cache_hits     # Number of cache hits
result.cache_stats.ttl_hits       # Number of TTL hits
result.cache_stats.cache_misses   # Number of cache misses
result.cache_stats.tokens_cached  # Total tokens cached
result.cache_stats.tokens_evicted # Total tokens evicted
result.cache_stats.ttl_pins       # Number of TTL pins
result.duration                   # Simulation duration
```

## TTL Benefit Analysis

### When TTL Helps

TTL pinning provides the most benefit when:

1. **Long tool execution times**: TTL protects cache while waiting
2. **High cache pressure**: Other programs can't evict pinned entries
3. **Prefix reuse**: Subsequent turns share prefix with pinned cache
4. **Tool exceptions**: TTL provides resilience against unexpected delays

### Key Metric: JCT (Job Completion Time)

**JCT = finish_time - arrival_time**

This naturally includes:
- Queue waiting time
- Actual processing time (prefill + decode + ttft)

**TTL Advantage:**
- TTL requests: **0ms queue delay** (priority scheduling)
- Non-TTL requests: Higher queue delay (preempted)

### Expected Results

| Scenario | Without TTL | With TTL |
|----------|------------|----------|
| Normal execution | ~95% hit rate | ~95%+ hit rate |
| High cache pressure | ~30-50% hit rate | ~50-70% hit rate |
| Tool exceptions | Variable | More stable |
| TTL JCT | N/A | **~90% reduction** |
| Non-TTL JCT | Baseline | Higher (preempted) |

### Event-Driven vs Batch Mode

| Mode | Description | Use Case |
|------|-------------|----------|
| `event_driven` | Processes by completion order with TTL-aware priority scheduling | **Recommended** - accurate queueing |
| `batch` | Processes requests by arrival_time order | Fast, less accurate timing |

## Troubleshooting

### Low Hit Rate

1. Check if requests have shared prefixes (same `extra_key`)
2. Verify TTL values are reasonable
3. Ensure cache capacity is sufficient
4. Check if requests have correct `arrival_time` ordering

### Unexpected Eviction

1. Verify TTL expiration times
2. Check if nodes are properly unpinned
3. Review eviction policy settings

## License

This simulator is part of the KVBlocking research project.
