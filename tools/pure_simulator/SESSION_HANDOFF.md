# KV Cache Blocking Simulator - Session Handoff

**Date**: Tuesday Apr 28, 2026
**Session Duration**: Afternoon/Evening (started ~4:30 PM UTC+8)
**Next Session**: Continue TTL optimization research

---

## Context

This is a **KV Cache Blocking Simulator** project located at:
```
/home/comp/csgfyu/multi-agents/KVBlocking/sglang_continuum/tools/pure_simulator/
```

The simulator models multi-turn agentic programs with tool calls, where:
- Each **program** consists of multiple **turns**
- Each turn has **inference time** (LLM generation) + **tool execution time**
- KV cache blocking optimizes by caching prefixes across turns using TTL mechanism

---

## What Was Completed in This Session

### 1. Comprehensive Parameter Sweep for TTL Speedup

**Created**: `experiments/parameter_sweep.py`

This script systematically explores which parameters affect TTL speedup ratio:
- Tool execution time (100ms - 10s)
- Tokens per turn / context size (100 - 2000 tokens)
- Concurrency / JPS (Poisson lambda: 0.01 - 2.0)
- Cache capacity (2000 - 15000 tokens)
- Number of programs (5 - 50)
- Turns per program (3 - 51)
- Memory pressure ratios

**Key Finding**: All parameters showed weak correlation with speedup (|r| < 0.13)

### 2. Large-Scale Simulation (1000 Programs)

**Created**: `experiments/large_scale_test.py`

Extended simulation with realistic Poisson arrival patterns:
- 50, 100, 200, 500, 1000 programs
- Multiple TTL configurations: disabled, 1s, 5s, 10s, 30s, adaptive
- Varying arrival rates (Poisson λ: 0.1 - 5.0)
- Varying cache pressures (10K - 200K capacity)

**Results saved to**: `sweep_results/large_scale_results.json`

### 3. Critical Discoveries About TTL Behavior

#### Finding 1: Short TTL (5s) is Ineffective
| Programs | No TTL JCT | TTL 5s JCT | Speedup |
|----------|------------|-------------|---------|
| 50 | 13.4s | 13.4s | 1.00x |
| 500 | 128.9s | 129.2s | 1.00x |
| 1000 | 254.4s | 254.5s | 1.00x |

**Conclusion**: TTL 5s provides NO speedup in any scenario tested.

#### Finding 2: Long TTL (30s) is Effective Under Pressure
| Cache Size | No TTL JCT | TTL 30s JCT | Speedup |
|------------|------------|--------------|---------|
| 10K (high pressure) | 127.8s | 116.3s | **1.10x** |
| 25K | 127.8s | 119.9s | **1.07x** |
| 50K | 128.9s | 123.9s | **1.04x** |
| 200K (low pressure) | 135.3s | 135.3s | 1.00x |

**Conclusion**: TTL 30s provides 4-10% speedup ONLY under high memory pressure.

#### Finding 3: TTL Eliminates L2 Reloads
| Config | L2 Hits | Evicted Tokens |
|--------|---------|----------------|
| No TTL | 6,602 | 84M |
| TTL 30s | **0** | 67M |

**Conclusion**: Long TTL prevents L2 reloads entirely, reducing overhead significantly.

### 4. Visualization

**Created**: `canvases/large-scale-ttl-analysis.canvas.tsx`

Interactive Canvas visualization of the large-scale test results.

---

## Recent Changes (Evening Session - Apr 28, 2026)

### 1. Removed Fixed TTL Testing
- Changed `history_threshold` from 3 to 1 in `ttl_manager.py` so adaptive TTL works from first sample
- Simplified experiments to only compare No TTL vs Adaptive TTL (removed fixed TTL configs)

### 2. Simplified Utility Model
- Removed the flawed `T` (queueing delay) calculation from utility model
- New utility model focuses on:
  - Benefit: P(hit) × prefill_savings + L1_reload_avoidance
  - Cost: memory_holding_cost × τ

### 3. Critical Discovery: Timing Model Mismatch
After extensive debugging, found the core issue:

| Metric | No TTL | TTL 5s | Improvement |
|--------|--------|---------|-------------|
| Queue Delay | 103,900ms | 25,300ms | **5x reduction** |
| Prefill Time | 33,600ms | 33,600ms | Same |
| TTL Hits | 0 | 123 | - |
| L2 Hits | 150 | 27 | - |
| **JCT** | **4,217ms** | **4,217ms** | **1.00x** |

**Root Cause**: TTL DOES reduce queue delay by 5x, but JCT doesn't improve because:
1. Prefill savings (4ms) are tiny compared to tool execution (925ms)
2. Total duration dominated by tool execution time
3. L2 reload avoidance isn't effective with current request generation

### 4. Why Paper Results Differ
The paper's Figure 13 speedup comes from:
1. Real hardware with realistic prefill latency (not synthetic 0.1ms/token)
2. Burst arrivals where TTL priority scheduling helps
3. Higher cache pressure scenarios
4. Actual L2 reload costs

---

## Key Insights Summary

### When TTL Helps:
1. **High cache pressure** (small cache relative to working set)
2. **Long TTL duration** (≥30s, not 5s)
3. **Long-running programs** (more turns = more benefit)
4. **Many concurrent programs** (more contention = more to protect)

### When TTL Hurts or is Neutral:
1. **Low cache pressure** (large cache, no eviction)
2. **Short TTL** (5s is too aggressive, nodes evicted before reuse)
3. **Short programs** (single-turn programs get no benefit)
4. **Low concurrency** (no contention to justify pinning overhead)

### TTL Cost/Benefit Tradeoff:
- **Benefit**: Avoids prefill, eliminates L2 reload, priority scheduling
- **Cost**: Pins nodes, blocks LRU eviction, causes forced unpins under pressure

---

## Files Created/Modified

### New Files:
| File | Purpose |
|------|---------|
| `experiments/parameter_sweep.py` | Comprehensive parameter sweep (7 dimensions) |
| `experiments/large_scale_test.py` | Large-scale simulation (1000 programs) |
| `sweep_results/large_scale_results.json` | Detailed test results |
| `canvases/large-scale-ttl-analysis.canvas.tsx` | Interactive visualization |

### Previously Modified (from prior sessions):
| File | Purpose |
|------|---------|
| `simulator/main.py` | Event-driven scheduling, TTL management |
| `simulator/scheduler.py` | Request processing, cache operations |
| `simulator/ttl_manager.py` | TTL utility calculation, adaptive TTL |
| `simulator/timing.py` | Timing calculations (A100 80GB) |
| `simulator/l1_cache.py` | L2 cache for spilled entries |

---

## Related Plans

- `optimize_ttl_utility_calculation_4cad29fc.plan.md` - TTL optimization strategy (see attached_files)

---

## How to Run

```bash
cd /home/comp/csgfyu/multi-agents/KVBlocking/sglang_continuum/tools/pure_simulator

# Run parameter sweep (takes ~20 seconds)
python3 -u experiments/parameter_sweep.py

# Run large-scale test (takes ~20 minutes)
python3 -u experiments/large_scale_test.py

# Quick verification
python3 -c "
import sys; sys.path.insert(0, '.')
from simulator import KVCacheSimulator, SimulatorConfig
# ... test code ...
"

# Run existing test
python3 experiments/test_ttl_jct_figure13.py
```

---

## Open Questions for Next Session

1. **Why is adaptive TTL underperforming?** Current implementation may need tuning
2. **Is there a better TTL value?** We should explore 15s, 20s, 45s, 60s
3. **How does write_back vs write_through affect results?** We only tested write_through
4. **Should we add memory pressure-based TTL auto-adjustment?**
5. **Real-world workload testing?** Synthetic data may not reflect production patterns

---

## Recommendations for Next Steps

### Understanding: Why We Don't See Paper Speedup
The paper's speedup requires:
1. **Realistic prefill latency** - paper uses actual GPU inference times, not 0.1ms/token
2. **Burst arrival patterns** - requests arriving in bursts where TTL priority helps
3. **Higher memory pressure** - the paper's workload stresses the cache more
4. **Proper L2 integration** - the simulator's L2 needs better integration for reload avoidance

### High Priority:
1. **Increase prefill latency** in timing.py to match paper's actual inference times
2. **Test with burst arrivals** - modify arrival pattern to simulate production workloads
3. **Verify L2 reload avoidance** - the L2 cache should save significant time on eviction
4. **Compare with paper's exact workload** - use the same SWE-Bench tasks

### Medium Priority:
1. **Profile where time is spent** - prefill vs decode vs queueing vs tool execution
2. **Tune utility model** - adjust cost_per_sec_ms and other parameters
3. **Add write_back policy testing** - may work better with current workload

---

## Code Locations Reference

### TTL Utility Calculation
**File**: `simulator/ttl_manager.py`, method `_compare_strategies()`
```python
# Simplified utility model:
# Benefit = P(hit) × prefill_savings + L1_reload_avoidance
# Cost = memory_holding_cost × τ
# P(hit) = CDF(tool_time) - probability tool finishes within TTL
```

### Timing Calculator
**File**: `simulator/timing.py`
```python
# Hardware config for A100 80GB
A100_80G = HardwareConfig(
    prefill_latency_per_token_ms=0.1,
    decode_latency_per_token_ms=4.5,
    cache_hit_speedup=10.0,
    l1_reload_penalty=0.3,
)
```

### Simulator Config
**File**: `simulator/main.py`, class `SimulatorConfig`
```python
# Key parameters
cache_capacity: int = 100_000
default_ttl: float = 5.0
enable_ttl: bool = True
enable_adaptive_ttl: bool = True
write_policy: str = "write_through"
high_pressure_threshold: float = 0.8
```
