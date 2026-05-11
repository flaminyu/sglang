#!/usr/bin/env python3
"""
Generate Predictive TTL Oracle comparison data for all anomaly types.

This script runs experiments comparing:
1. Baseline (no TTL)
2. Current TTL (adaptive TTL)
3. Predictive Oracle TTL (with perfect anomaly knowledge)

For all 5 anomaly types at 9 different anomaly rates (0% to 50%).

Output: results/predictive_comparison/full_comparison_raw.json
"""

import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator
from tool_classifier import AnomalyType
from ttl_predictive_patch import apply_ttl_patch_to_simulator, PredictionAccuracy, TTLAdjustmentConfig


# ============================================================================
# Configuration (same as anomaly_ttl_analysis.py)
# ============================================================================

HARDWARE_CONFIG = {
    'prefill_latency_per_token_ms': 2.0,
    'decode_latency_per_token_ms': 20.0,
    'ttft_overhead_ms': 100.0,
    'queue_delay_per_request_ms': 50.0,
    'bytes_per_token': 512.0,
    'h2d_bandwidth_GBps': 32.0,
    'bandwidth_efficiency': 0.85,
    'h2d_overhead_us': 6.67,
    'd2h_overhead_us': 4.0,
    'memory_bandwidth_GBps': 64.0,
}

DEFAULT_CONFIG = {
    'l1': 15000,
    'l2_ratio': 1.5,
    'programs': 50,
    'turns': 20,
    'prefill': 200,
    'first_turn_multiplier': 5,
    'tool_time': 2.0,
    'seed': 42,
}

# Aggressive extension config (best performing from previous tests)
ORACLE_CONFIG = TTLAdjustmentConfig(
    timeout_extension=2.0,
    retry_extension=1.5,
    hang_extension=3.0,
    early_return_shorten=0.5,
    high_anomaly_threshold=0.3,  # Lower threshold to apply more extensions
    low_anomaly_threshold=0.1,
)

# Anomaly rates to test
ANOMALY_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# Anomaly types
ANOMALY_TYPES = ['timeout', 'retry', 'hang', 'early_return', 'mixed']


# ============================================================================
# Request Generation
# ============================================================================

def generate_requests(
    programs: int,
    turns: int,
    prefill: int,
    tool_time: float,
    first_turn_multiplier: int = 5,
    arrival_time: float = 0.0,
    seed: int = 42
) -> List[Dict]:
    """Generate requests for simulation (same as anomaly_ttl_analysis.py)."""
    random.seed(seed)
    requests = []

    for prog_idx in range(programs):
        program_id = f'prog_{prog_idx:03d}'
        accumulated = []

        for turn in range(turns):
            if turn == 0:
                current_prefill = int(prefill * first_turn_multiplier)
            elif turn == 1:
                current_prefill = int(prefill * 3)
            elif turn == 2:
                current_prefill = int(prefill * 2)
            else:
                current_prefill = prefill

            prev_total = len(accumulated)
            new_tokens = list(range(
                prog_idx * 1000000 + prev_total,
                prog_idx * 1000000 + prev_total + current_prefill
            ))
            tokens = accumulated + new_tokens
            requests.append({
                'rid': f'{program_id}_turn_{turn}',
                'program_id': program_id,
                'turn_index': turn,
                'token_ids': tokens,
                'input_len': len(tokens),
                'output_len': 64,
                'arrival_time': arrival_time,
                'is_tool_call': turn < turns - 1,
                'tool_name': f'tool_{turn % 3}',
                'extra_key': program_id,
                'ttl_sec': None,
                'actual_tool_duration': tool_time,
            })
            accumulated.extend(new_tokens)

    return requests


def apply_anomaly_exceptions(
    requests: List[Dict],
    anomaly_type: str,
    rate: float,
    base_duration: float = 2.0,
    seed: int = 123
) -> List[Dict]:
    """Apply anomaly exceptions to requests based on type (same as anomaly_ttl_analysis.py)."""
    random.seed(seed)
    modified = []

    for req in requests:
        if not req['is_tool_call']:
            modified.append(req.copy())
            continue

        new_req = req.copy()
        if random.random() < rate:
            if anomaly_type == 'timeout':
                new_req['actual_tool_duration'] = base_duration * 5.0
            elif anomaly_type == 'retry':
                new_req['actual_tool_duration'] = base_duration * 2.0
            elif anomaly_type == 'hang':
                new_req['actual_tool_duration'] = base_duration * 10.0
            elif anomaly_type == 'early_return':
                new_req['actual_tool_duration'] = base_duration * 0.1
            elif anomaly_type == 'mixed':
                r = random.random()
                if r < 0.333:
                    new_req['actual_tool_duration'] = base_duration * 5.0
                elif r < 0.666:
                    new_req['actual_tool_duration'] = base_duration * 2.0
                else:
                    new_req['actual_tool_duration'] = base_duration * 10.0
        modified.append(new_req)

    return modified


# ============================================================================
# Simulator Creation
# ============================================================================

def create_simulator(
    requests: List[Dict],
    enable_ttl: bool,
    l1: int,
    l2: int,
    tool_time: float = 2.0,
    write_policy: str = "write_through",
    history_threshold: int = 3,
) -> KVCacheSimulator:
    """Create and run simulator (same config as anomaly_ttl_analysis.py)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        temp_path = f.name

    try:
        hw = HardwareConfig(**HARDWARE_CONFIG)
        timing_calc = TimingCalculator(hw=hw)

        sim = KVCacheSimulator(
            cache_capacity=l1,
            l2_capacity=l2,
            l2_enabled=l2 > 0,
            l2_reload_penalty=0.3,
            default_ttl=5.0,
            min_ttl=0.5,
            max_ttl=60.0,
            enable_ttl=enable_ttl,
            enable_adaptive_ttl=enable_ttl,
            simulation_mode='event_driven',
            poisson_lambda=1.0 / 0.3,
            tool_execution_time=tool_time,
            random_seed=42,
            hardware_config=hw,
            timing_config=timing_calc,
            write_policy=write_policy,
            history_threshold=history_threshold,
        )

        sim.load_requests(temp_path)
        for pid in sim.scheduler.program_total_turns:
            sim.scheduler.program_total_turns[pid] = 999

        sim.run(verbose=False)
        return sim
    finally:
        os.unlink(temp_path)


def analyze_result(sim: KVCacheSimulator) -> Dict[str, Any]:
    """Analyze simulation result."""
    log = sim.scheduler.scheduling_log

    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_requests = len(log)
    
    total_time = log[-1]['finish_time'] if log else 0

    ttl_stats = {}
    if sim.ttl_manager:
        ttl_stats = sim.ttl_manager.get_stats()
        ttl_timeline = sim.ttl_manager.get_timeline_data()
        ttl_predictions = sim.ttl_manager.get_prediction_data()
        
        ttl_pins = sum(1 for e in ttl_timeline if e['event'] == 'pin')
        ttl_expires = sum(1 for e in ttl_timeline if e['event'] == 'expire')
        
        prediction_stats = sim.ttl_manager.get_prediction_stats()
        
        ttl_stats.update({
            'pins': ttl_pins,
            'expires': ttl_expires,
            'prediction_stats': prediction_stats,
        })

    return {
        'time': total_time,
        'total_requests': total_requests,
        'ttl_hits': ttl_hits,
        'l1_hits': l1_hits,
        'l2_hits': l2_hits,
        'misses': misses,
        'ttl_hit_rate': ttl_hits / total_requests if total_requests > 0 else 0,
        'l1_hit_rate': l1_hits / total_requests if total_requests > 0 else 0,
        'l2_hit_rate': l2_hits / total_requests if total_requests > 0 else 0,
        'miss_rate': misses / total_requests if total_requests > 0 else 0,
        'cache_hit_rate': (ttl_hits + l1_hits + l2_hits) / total_requests if total_requests > 0 else 0,
        'ttl_stats': ttl_stats,
    }


# ============================================================================
# Run Single Experiment
# ============================================================================

def run_experiment(
    anomaly_type: str,
    anomaly_rate: float,
    l1: int = DEFAULT_CONFIG['l1'],
    l2_ratio: float = DEFAULT_CONFIG['l2_ratio'],
    programs: int = DEFAULT_CONFIG['programs'],
    turns: int = DEFAULT_CONFIG['turns'],
    prefill: int = DEFAULT_CONFIG['prefill'],
    tool_time: float = DEFAULT_CONFIG['tool_time'],
) -> Dict[str, Any]:
    """Run a single experiment comparing Baseline, Current TTL, and Oracle TTL."""
    
    l2 = int(l1 * l2_ratio)
    
    # Generate base requests
    base_requests = generate_requests(
        programs, turns, prefill, tool_time,
        DEFAULT_CONFIG['first_turn_multiplier'], seed=DEFAULT_CONFIG['seed']
    )
    
    # Apply anomalies
    anomaly_requests = apply_anomaly_exceptions(
        base_requests, anomaly_type, anomaly_rate, tool_time
    )
    
    # Calculate actual stats
    tool_calls = [r for r in anomaly_requests if r['is_tool_call']]
    anomalies = [r for r in tool_calls if r['actual_tool_duration'] != tool_time]
    avg_tool_time = sum(r['actual_tool_duration'] for r in tool_calls) / len(tool_calls) if tool_calls else tool_time
    
    # --- Run 1: No TTL (Baseline) ---
    sim_baseline = create_simulator(anomaly_requests, False, l1, l2, avg_tool_time)
    r_baseline = analyze_result(sim_baseline)
    
    # --- Run 2: Current TTL (Adaptive TTL) ---
    sim_current = create_simulator(anomaly_requests, True, l1, l2, avg_tool_time)
    r_current = analyze_result(sim_current)
    
    # --- Run 3: Predictive Oracle TTL ---
    sim_oracle = create_simulator(anomaly_requests, True, l1, l2, avg_tool_time)
    patch = apply_ttl_patch_to_simulator(
        sim_oracle,
        prediction_accuracy=PredictionAccuracy.PERFECT,
        config=ORACLE_CONFIG
    )
    
    # Inject oracle knowledge for anomalous requests
    for r in anomalies:
        ratio = r['actual_tool_duration'] / tool_time
        if ratio > 5.0:
            anomaly = AnomalyType.HANG
        elif ratio > 2.0:
            anomaly = AnomalyType.TIMEOUT
        elif ratio < 0.5:
            anomaly = AnomalyType.EARLY_RETURN
        else:
            anomaly = AnomalyType.RETRY
        patch.inject_oracle_knowledge(r['rid'], anomaly)
    
    sim_oracle.run(verbose=False)
    r_oracle = analyze_result(sim_oracle)
    
    # Calculate speedups (relative to baseline)
    baseline_time = r_baseline['time']
    current_speedup = baseline_time / r_current['time'] if r_current['time'] > 0 else 0
    oracle_speedup = baseline_time / r_oracle['time'] if r_oracle['time'] > 0 else 0
    
    return {
        'anomaly_type': anomaly_type,
        'anomaly_rate': anomaly_rate,
        'actual_rate': len(anomalies) / len(tool_calls) if tool_calls else 0,
        'avg_tool_time': avg_tool_time,
        'baseline_time': baseline_time,
        'current_ttl_time': r_current['time'],
        'oracle_ttl_time': r_oracle['time'],
        'current_ttl_speedup': current_speedup,
        'oracle_ttl_speedup': oracle_speedup,
        'current_ttl_stats': {
            'ttl_hits': r_current['ttl_hits'],
            'ttl_hit_rate': r_current['ttl_hit_rate'],
            'l1_hit_rate': r_current['l1_hit_rate'],
            'l2_hit_rate': r_current['l2_hit_rate'],
            'miss_rate': r_current['miss_rate'],
        },
        'oracle_ttl_stats': {
            'ttl_hits': r_oracle['ttl_hits'],
            'ttl_hit_rate': r_oracle['ttl_hit_rate'],
            'l1_hit_rate': r_oracle['l1_hit_rate'],
            'l2_hit_rate': r_oracle['l2_hit_rate'],
            'miss_rate': r_oracle['miss_rate'],
            'extensions': patch.stats.extensions,
            'shortenings': patch.stats.shortenings,
        },
        'improvement_vs_current': oracle_speedup - current_speedup,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("PREDICTIVE TTL ORACLE DATA GENERATION")
    print("Comparing: Baseline vs Current TTL vs Predictive Oracle TTL")
    print("=" * 70)
    
    output_dir = Path('results/predictive_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'config': {
            'l1': DEFAULT_CONFIG['l1'],
            'l2': int(DEFAULT_CONFIG['l1'] * DEFAULT_CONFIG['l2_ratio']),
            'programs': DEFAULT_CONFIG['programs'],
            'turns': DEFAULT_CONFIG['turns'],
            'prefill': DEFAULT_CONFIG['prefill'],
            'tool_time': DEFAULT_CONFIG['tool_time'],
            'anomaly_rates': ANOMALY_RATES,
            'anomaly_types': ANOMALY_TYPES,
            'oracle_config': {
                'timeout_extension': ORACLE_CONFIG.timeout_extension,
                'retry_extension': ORACLE_CONFIG.retry_extension,
                'hang_extension': ORACLE_CONFIG.hang_extension,
                'early_return_shorten': ORACLE_CONFIG.early_return_shorten,
            },
            'generated_at': datetime.now().isoformat(),
        },
        'results': []
    }
    
    total_experiments = len(ANOMALY_TYPES) * len(ANOMALY_RATES)
    current = 0
    
    for anomaly_type in ANOMALY_TYPES:
        print(f"\n{'='*70}")
        print(f"Testing: {anomaly_type.upper()}")
        print(f"{'='*70}")
        
        for rate in ANOMALY_RATES:
            current += 1
            print(f"\n[{current}/{total_experiments}] {anomaly_type} @ {rate*100:.0f}%")
            
            result = run_experiment(anomaly_type, rate)
            all_results['results'].append(result)
            
            print(f"  Baseline: {result['baseline_time']:.1f}s")
            print(f"  Current TTL: {result['current_ttl_time']:.1f}s (speedup: {result['current_ttl_speedup']:.3f}x)")
            print(f"  Oracle TTL:  {result['oracle_ttl_time']:.1f}s (speedup: {result['oracle_ttl_speedup']:.3f}x)")
            print(f"  Improvement vs Current: {result['improvement_vs_current']:+.3f}x")
    
    # Save results
    output_file = output_dir / 'full_comparison_raw.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"Total experiments: {total_experiments}")
    print(f"{'='*70}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Group by anomaly type
    by_type = {}
    for r in all_results['results']:
        at = r['anomaly_type']
        if at not in by_type:
            by_type[at] = []
        by_type[at].append(r)
    
    for at in ANOMALY_TYPES:
        if at not in by_type:
            continue
        results = by_type[at]
        best_current = max(results, key=lambda x: x['current_ttl_speedup'])
        best_oracle = max(results, key=lambda x: x['oracle_ttl_speedup'])
        oracle_better = sum(1 for r in results if r['oracle_ttl_speedup'] > r['current_ttl_speedup'])
        
        print(f"\n{at.upper()}:")
        print(f"  Current TTL best: {best_current['current_ttl_speedup']:.3f}x @ {best_current['anomaly_rate']*100:.0f}%")
        print(f"  Oracle TTL best:  {best_oracle['oracle_ttl_speedup']:.3f}x @ {best_oracle['anomaly_rate']*100:.0f}%")
        print(f"  Oracle better in {oracle_better}/{len(results)} rates")


if __name__ == "__main__":
    main()
