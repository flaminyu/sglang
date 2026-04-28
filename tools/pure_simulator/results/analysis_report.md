# TTL Scheduling Analysis Report

**Generated:** 2026-04-27 11:13:26

## Scenario Configuration

| Parameter | Value |
|-----------|-------|
| L0 Cache Size | 20,000 tokens |
| Concurrency | 30 programs |
| Turns per Program | 15 |
| Shared Prefix | 5,000 tokens |
| Tokens per Turn | 2,000 tokens |
| Output Tokens | 64 |
| Default TTL | 5.0s |
| Tool Execution Time | 0.1s |

## Executive Summary

### Overall Speedup

| Metric | Value |
|--------|-------|
| **Baseline Duration** | 760.98s |
| **TTL Duration** | 521.24s |
| **Time Saved** | 239.74s |
| **Speedup** | **1.46x** (46.0% faster) |

### Hit Type Distribution

| Hit Type | Baseline | TTL | Difference | Impact |
|----------|----------|-----|------------|--------|
| **TTL Hit** | 0 | 219 | +219 | Protected by TTL |
| **L0 Hit** | 144 | 55 | -89 | Cached in GPU |
| **L1 Hit** | 68 | 20 | -48 | Loaded from CPU |
| **Miss** | 238 | 156 | -82 | Full prefill needed |

## TTL Benefits Analysis

### Key Insight: TTL Reduces Cache Misses

TTL's primary benefit is **protecting KV cache entries from eviction**, which reduces the number of cache misses.

#### Miss Reduction

```
Baseline Misses: 238
TTL Misses:      156
Miss Reduction:  82 (34.5%)
```

#### L1 Hit Reduction

```
Baseline L1 Hits: 68
TTL L1 Hits:      20
L1 Reduction:     48 (70.6%)
```

### Time Breakdown by Hit Type

#### Baseline
| Hit Type | Avg Prefill (ms) | Avg Decode (ms) | Count |
|----------|------------------|-----------------|-------|
| L0 | 0.00 | 28.80 | 144 |
| L1 | 850.88 | 288.00 | 68 |
| Miss | 2470.59 | 288.00 | 238 |

#### TTL Mode
| Hit Type | Avg Prefill (ms) | Avg Decode (ms) | Count |
|----------|------------------|-----------------|-------|
| TTL | 0.00 | 28.80 | 219 |
| L0 | 0.00 | 28.80 | 55 |
| L1 | 1209.50 | 288.00 | 20 |
| Miss | 2662.82 | 288.00 | 156 |

## Performance Analysis

### Prefill Time Savings

| Mode | Total Prefill Time |
|------|-------------------|
| Baseline | 645.86s |
| TTL | 439.59s |
| **Saved** | **206.27s** (31.9%) |

### Average Time Per Request

| Metric | Baseline | TTL | Difference |
|--------|----------|-----|------------|
| Avg Prefill | 1435.24ms | 976.87ms | -458.38ms |
| Avg Decode | 205.06ms | 130.18ms | -74.88ms |

## Timeline Analysis

### First 20 Requests (TTL Mode)

| # | Time (s) | RID | Hit Type | Pinned | Prefill (ms) | Decode (ms) | Total (ms) |
|---|----------|-----|----------|--------|--------------|-------------|------------|
| 0 | 0.003 | prog_0019_turn_0 | Miss | ✗ | 700.00 | 288.00 | 1038.00 |
| 1 | 1.041 | prog_0001_turn_0 | L0 | ✗ | 0.00 | 28.80 | 78.80 |
| 2 | 1.120 | prog_0012_turn_0 | L0 | ✗ | 0.00 | 28.80 | 78.80 |
| 3 | 1.199 | prog_0019_turn_1 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 4 | 1.278 | prog_0001_turn_1 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 5 | 1.356 | prog_0012_turn_1 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 6 | 1.435 | prog_0019_turn_2 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 7 | 1.514 | prog_0001_turn_2 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 8 | 1.593 | prog_0012_turn_2 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 9 | 1.672 | prog_0019_turn_3 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 10 | 1.750 | prog_0001_turn_3 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 11 | 1.829 | prog_0012_turn_3 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 12 | 1.908 | prog_0019_turn_4 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 13 | 1.987 | prog_0001_turn_4 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 14 | 2.066 | prog_0012_turn_4 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 15 | 2.144 | prog_0019_turn_5 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 16 | 2.223 | prog_0001_turn_5 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 17 | 2.302 | prog_0012_turn_5 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 18 | 2.381 | prog_0019_turn_6 | TTL | ✓ | 0.00 | 28.80 | 78.80 |
| 19 | 2.460 | prog_0001_turn_6 | TTL | ✓ | 0.00 | 28.80 | 78.80 |

### Key Observations

1. **Turn 0 (first request of each program)** typically results in a **Miss** because:
   - No prior KV cache exists for new programs
   - This is expected behavior

2. **Subsequent turns** benefit from TTL:
   - If program is pinned (TTL active), turn hits as **TTL**
   - If TTL expired but KV still in L0, turn hits as **L0**
   - If KV was evicted, turn becomes **Miss**

3. **Pinned programs are protected**:
   - When a program calls a tool (turn N), it's pinned with TTL
   - The next turn (N+1) arrives after tool execution
   - If TTL hasn't expired, turn N+1 hits as **TTL**

## Conclusion

**TTL provides significant speedup ({speedup:.2f}x) primarily through cache protection:**

- **{ttl_hit.get('TTL', 0)} requests** hit as TTL (protected by TTL)
- **{bl_hit.get('Miss', 0) - ttl_hit.get('Miss', 0)} fewer misses** compared to baseline
- **{prefill_saved/1000:.2f}s saved** in prefill time

The timing model shows that:
- **TTL hit = L0 hit in speed** (same cache lookup)
- **TTL benefit = eviction protection** (prevents miss → TTL/L0/L1 conversion)
- **Scheduling priority** provides additional benefits when requests overlap

## Files Generated

- `timeline_analysis.html` - Interactive timeline visualization
- `analysis_report.md` - This report
- `bl_scheduling_log.json` - Baseline scheduling log
- `ttl_scheduling_log.json` - TTL scheduling log
