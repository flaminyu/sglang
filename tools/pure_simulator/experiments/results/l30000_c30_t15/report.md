# TTL Scheduling Detailed Breakdown Report

**Scenario:** L0=30,000, Concurrency=30, Turns=15
**Generated:** 2026-04-27 11:32:41

---

## 1. Scenario Configuration

| Parameter | Value |
|-----------|-------|
| L0 Cache Size | 30,000 tokens |
| Concurrency | 30 programs |
| Turns per Program | 15 |
| Shared Prefix | 5,000 tokens |
| Tokens per Turn | 2,000 tokens |
| Tokens per Program | 35,000 |
| Total Tokens | 1,050,000 |
| **L0 Coverage** | **2.86%** |

---

## 2. Overall Performance

| Metric | Baseline | TTL | Difference |
|--------|----------|-----|------------|
| **Duration** | 408.00s | 410.55s | **-2.55s** |
| Total Requests | 450 | 450 | 0 |

### Speedup Summary

| Metric | Value |
|--------|-------|
| **Speedup** | **0.9938x** |
| **Time Saved** | -2.55s |
| **Improvement** | -0.62% |

---

## 3. Hit Type Distribution (Fine-Grained)

### 3.1 Count Breakdown

| Hit Type | Baseline | TTL | Difference | Change |
|----------|----------|-----|------------|--------|
| **TTL Hit** | 0 | 300 | **+300** | New |
| **L0 Hit** | 360 | 60 | -300 | -83.3% |
| **L1 Hit** | 15 | 15 | +0 | +0.0% |
| **Miss** | 75 | 75 | +0 | +0.0% |

### 3.2 Percentage Breakdown

| Hit Type | Baseline % | TTL % |
|----------|------------|-------|
| TTL Hit | 0.0% | 66.7% |
| L0 Hit | 80.0% | 13.3% |
| L1 Hit | 3.3% | 3.3% |
| Miss | 16.7% | 16.7% |

---

## 4. Detailed Time Breakdown by Hit Type

### 4.1 TTL Hit Analysis

| Metric | Baseline (TTL=0) | TTL Mode |
|--------|------------------|----------|
| Count | 0 | 300 |
| Avg Prefill | N/A | 0.00ms |
| Avg Decode | N/A | 288.00ms |
| Avg H2D | N/A | 0.00ms |
| Avg Queue | N/A | 0.00ms |
| **Avg Total** | N/A | **338.00ms** |

### 4.2 L0 Hit Analysis

| Metric | Baseline | TTL |
|--------|----------|-----|
| Count | 360 | 60 |
| Avg Prefill | 0.00ms | 0.00ms |
| Avg Decode | 288.00ms | 288.00ms |
| Avg H2D | 0.00ms | 0.00ms |
| Avg Queue | 0.00ms | 0.00ms |
| **Avg Total** | **338.00ms** | **338.00ms** |

### 4.3 L1 Hit Analysis

| Metric | Baseline | TTL | Saved |
|--------|----------|-----|-------|
| Count | 15 | 15 | +0 |
| Avg Prefill | 2760.00ms | 2930.00ms | +170.00ms |
| Avg Decode | 288.00ms | 288.00ms | +0.00ms |
| Avg H2D | 60.00ms | 30.00ms | -30.00ms |
| **Avg Total** | **3098.00ms** | **3268.00ms** | +170.00ms |

### 4.4 Miss Analysis

| Metric | Baseline | TTL | Saved |
|--------|----------|-----|-------|
| Count | 75 | 75 | +0 (**-0**) |
| Avg Prefill | 2860.00ms | 2860.00ms | +0.00ms |
| Avg Decode | 288.00ms | 288.00ms | +0.00ms |
| Avg H2D | 0.00ms | 0.00ms | +0.00ms |
| **Avg Total** | **3198.00ms** | **3198.00ms** | +0.00ms |

---

## 5. Aggregated Time Breakdown

### 5.1 Total Time Components

| Component | Baseline | TTL | Saved |
|-----------|----------|-----|-------|
| Prefill | 255.90s | 258.45s | -2.55s |
| Decode | 129.60s | 129.60s | 0.00s |
| H2D Transfer | 0.90s | 0.45s | 0.45s |
| Queue Delay | 0.00s | 0.00s | N/A |
| TTFT Overhead | 22.50s | 22.50s | N/A |
| **Total** | **408.00s** | **410.55s** | **-2.55s** |

### 5.2 Percentage Breakdown

| Component | Baseline % | TTL % |
|-----------|------------|-------|
| Prefill | 62.7% | 63.0% |
| Decode | 31.8% | 31.6% |
| H2D | 0.2% | 0.1% |

---

## 6. TTL Benefit Analysis

### 6.1 Hit Type Conversion

TTL converts cache misses and partial hits into TTL hits:

| From | To | Count |
|------|-----|-------|
| L0 Hit | TTL Hit | 300 |
| L1 Hit | TTL Hit | 0 |
| Miss | TTL Hit | 0 |
| **Total** | | **300** |

### 6.2 Miss Reduction

```
Baseline Misses: 75
TTL Misses:     75
Reduction:      0 (0.0%)
```

### 6.3 L1 Hit Reduction

```
Baseline L1 Hits: 15
TTL L1 Hits:      15
Reduction:        0 (0.0%)
```

---

## 7. Key Insights

1. **TTL does NOT make individual requests faster**:
   - TTL hit timing = L0 hit timing (same cache lookup speed)
   - TTL benefit comes from cache protection

2. **Cache Protection is the Key Benefit**:
   - **300 requests** that would have missed/L1/L0 now hit as TTL
   - **0 fewer misses** (0.0% reduction)

3. **Time Savings Breakdown**:
   - Prefill savings: -2.55s (-1.0%)
   - Decode savings: 0.00s (0.0%)
   - H2D savings: 0.45s (50.0%)

4. **L0 Coverage Impact**:
   - At 2.86% coverage, TTL provides **0.99x** speedup
   - Lower coverage = more eviction = more benefit from TTL protection

---

## 8. Data Files

- `bl_breakdowns.json` - Baseline per-request breakdown
- `ttl_breakdowns.json` - TTL per-request breakdown  
- `summary.json` - Aggregated statistics
- `report.md` - This report
