# TTL Scheduling Detailed Breakdown Report

**Scenario:** L0=20,000, Concurrency=30, Turns=20
**Generated:** 2026-04-27 11:32:49

---

## 1. Scenario Configuration

| Parameter | Value |
|-----------|-------|
| L0 Cache Size | 20,000 tokens |
| Concurrency | 30 programs |
| Turns per Program | 20 |
| Shared Prefix | 5,000 tokens |
| Tokens per Turn | 2,000 tokens |
| Tokens per Program | 45,000 |
| Total Tokens | 1,350,000 |
| **L0 Coverage** | **1.48%** |

---

## 2. Overall Performance

| Metric | Baseline | TTL | Difference |
|--------|----------|-----|------------|
| **Duration** | 1463.70s | 1466.25s | **-2.55s** |
| Total Requests | 600 | 600 | 0 |

### Speedup Summary

| Metric | Value |
|--------|-------|
| **Speedup** | **0.9983x** |
| **Time Saved** | -2.55s |
| **Improvement** | -0.17% |

---

## 3. Hit Type Distribution (Fine-Grained)

### 3.1 Count Breakdown

| Hit Type | Baseline | TTL | Difference | Change |
|----------|----------|-----|------------|--------|
| **TTL Hit** | 0 | 180 | **+180** | New |
| **L0 Hit** | 210 | 30 | -180 | -85.7% |
| **L1 Hit** | 15 | 15 | +0 | +0.0% |
| **Miss** | 375 | 375 | +0 | +0.0% |

### 3.2 Percentage Breakdown

| Hit Type | Baseline % | TTL % |
|----------|------------|-------|
| TTL Hit | 0.0% | 30.0% |
| L0 Hit | 35.0% | 5.0% |
| L1 Hit | 2.5% | 2.5% |
| Miss | 62.5% | 62.5% |

---

## 4. Detailed Time Breakdown by Hit Type

### 4.1 TTL Hit Analysis

| Metric | Baseline (TTL=0) | TTL Mode |
|--------|------------------|----------|
| Count | 0 | 180 |
| Avg Prefill | N/A | 0.00ms |
| Avg Decode | N/A | 288.00ms |
| Avg H2D | N/A | 0.00ms |
| Avg Queue | N/A | 0.00ms |
| **Avg Total** | N/A | **338.00ms** |

### 4.2 L0 Hit Analysis

| Metric | Baseline | TTL |
|--------|----------|-----|
| Count | 210 | 30 |
| Avg Prefill | 0.00ms | 0.00ms |
| Avg Decode | 288.00ms | 288.00ms |
| Avg H2D | 0.00ms | 0.00ms |
| Avg Queue | 0.00ms | 0.00ms |
| **Avg Total** | **338.00ms** | **338.00ms** |

### 4.3 L1 Hit Analysis

| Metric | Baseline | TTL | Saved |
|--------|----------|-----|-------|
| Count | 15 | 15 | +0 |
| Avg Prefill | 1760.00ms | 1930.00ms | +170.00ms |
| Avg Decode | 288.00ms | 288.00ms | +0.00ms |
| Avg H2D | 60.00ms | 30.00ms | -30.00ms |
| **Avg Total** | **2098.00ms** | **2268.00ms** | +170.00ms |

### 4.4 Miss Analysis

| Metric | Baseline | TTL | Saved |
|--------|----------|-----|-------|
| Count | 375 | 375 | +0 (**-0**) |
| Avg Prefill | 3292.00ms | 3292.00ms | +0.00ms |
| Avg Decode | 288.00ms | 288.00ms | +0.00ms |
| Avg H2D | 0.00ms | 0.00ms | +0.00ms |
| **Avg Total** | **3630.00ms** | **3630.00ms** | +0.00ms |

---

## 5. Aggregated Time Breakdown

### 5.1 Total Time Components

| Component | Baseline | TTL | Saved |
|-----------|----------|-----|-------|
| Prefill | 1260.90s | 1263.45s | -2.55s |
| Decode | 172.80s | 172.80s | 0.00s |
| H2D Transfer | 0.90s | 0.45s | 0.45s |
| Queue Delay | 0.00s | 0.00s | N/A |
| TTFT Overhead | 30.00s | 30.00s | N/A |
| **Total** | **1463.70s** | **1466.25s** | **-2.55s** |

### 5.2 Percentage Breakdown

| Component | Baseline % | TTL % |
|-----------|------------|-------|
| Prefill | 86.1% | 86.2% |
| Decode | 11.8% | 11.8% |
| H2D | 0.1% | 0.0% |

---

## 6. TTL Benefit Analysis

### 6.1 Hit Type Conversion

TTL converts cache misses and partial hits into TTL hits:

| From | To | Count |
|------|-----|-------|
| L0 Hit | TTL Hit | 180 |
| L1 Hit | TTL Hit | 0 |
| Miss | TTL Hit | 0 |
| **Total** | | **180** |

### 6.2 Miss Reduction

```
Baseline Misses: 375
TTL Misses:     375
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
   - **180 requests** that would have missed/L1/L0 now hit as TTL
   - **0 fewer misses** (0.0% reduction)

3. **Time Savings Breakdown**:
   - Prefill savings: -2.55s (-0.2%)
   - Decode savings: 0.00s (0.0%)
   - H2D savings: 0.45s (50.0%)

4. **L0 Coverage Impact**:
   - At 1.48% coverage, TTL provides **1.00x** speedup
   - Lower coverage = more eviction = more benefit from TTL protection

---

## 8. Data Files

- `bl_breakdowns.json` - Baseline per-request breakdown
- `ttl_breakdowns.json` - TTL per-request breakdown  
- `summary.json` - Aggregated statistics
- `report.md` - This report
