#!/usr/bin/env python3
"""
Unified Anomaly TTL Analysis Script

Tests how TTL hit/miss ratios change under different anomaly rates.
Supports multiple anomaly types:
1. Retry - Tool takes 2x longer (runs twice)
2. Timeout - Tool takes 5x longer (10s instead of 2s)
3. Hang - Tool takes 10x longer (20s instead of 2s)
4. Early Return - Tool returns early (0.1x = 0.2s)
5. Mixed - Combination of timeout, retry, hang

Each anomaly type generates its own HTML report with detailed analysis.
"""

import sys
sys.path.insert(0, '.')

import os
import json
import tempfile
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import statistics

from simulator import KVCacheSimulator
from simulator.timing import HardwareConfig, TimingCalculator


# ============================================================================
# Configuration (from baseline in anomaly_prediction_analysis.html)
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
    """Generate requests for simulation."""
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
    """Apply anomaly exceptions to requests based on type."""
    random.seed(seed)
    modified = []

    for req in requests:
        if not req['is_tool_call']:
            modified.append(req.copy())
            continue

        new_req = req.copy()
        if random.random() < rate:
            if anomaly_type == 'timeout':
                # Timeout: 5x longer (10s instead of 2s)
                new_req['actual_tool_duration'] = base_duration * 5.0
                new_req['anomaly'] = 'timeout'
            elif anomaly_type == 'retry':
                # Retry: 2x longer (runs twice)
                new_req['actual_tool_duration'] = base_duration * 2.0
                new_req['anomaly'] = 'retry'
            elif anomaly_type == 'hang':
                # Hang: 10x longer (20s instead of 2s)
                new_req['actual_tool_duration'] = base_duration * 10.0
                new_req['anomaly'] = 'hang'
            elif anomaly_type == 'early_return':
                # Early return: 0.1x (0.2s instead of 2s)
                new_req['actual_tool_duration'] = base_duration * 0.1
                new_req['anomaly'] = 'early_return'
            elif anomaly_type == 'mixed':
                # Mixed: 1/3 each of timeout, retry, hang
                r = random.random()
                if r < 0.333:
                    new_req['actual_tool_duration'] = base_duration * 5.0
                    new_req['anomaly'] = 'timeout'
                elif r < 0.666:
                    new_req['actual_tool_duration'] = base_duration * 2.0
                    new_req['anomaly'] = 'retry'
                else:
                    new_req['actual_tool_duration'] = base_duration * 10.0
                    new_req['anomaly'] = 'hang'
        else:
            new_req['actual_tool_duration'] = base_duration
            new_req['anomaly'] = None
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
    """Create and run simulator."""
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


def analyze_result_detailed(sim: KVCacheSimulator) -> Dict[str, Any]:
    """Analyze simulation result with detailed TTL hit/miss tracking."""
    log = sim.scheduler.scheduling_log

    # Basic hit counts
    ttl_hits = sum(1 for e in log if e['hit_type'] == 'TTL')
    l1_hits = sum(1 for e in log if e['hit_type'] == 'L1')
    l2_hits = sum(1 for e in log if e['hit_type'] == 'L2')
    misses = sum(1 for e in log if e['hit_type'] == 'L1+L2 Miss')
    total_requests = len(log)
    
    # Calculate rates
    ttl_hit_rate = ttl_hits / total_requests if total_requests > 0 else 0
    l1_hit_rate = l1_hits / total_requests if total_requests > 0 else 0
    l2_hit_rate = l2_hits / total_requests if total_requests > 0 else 0
    miss_rate = misses / total_requests if total_requests > 0 else 0
    
    # Total time
    total_time = log[-1]['finish_time'] if log else 0

    # Time breakdown
    total_prefill_ms = sum(e.get('prefill_ms', 0) for e in log)
    total_decode_ms = sum(e.get('decode_ms', 0) for e in log)
    total_h2d_ms = sum(e.get('h2d_ms', 0) for e in log)
    total_queue_delay_ms = sum(e.get('queue_delay_ms', 0) for e in log)
    total_ttft_ms = sum(e.get('ttft_ms', 0) for e in log)

    # L2 cache stats
    l2_spills = 0
    l2_evictions = 0
    if hasattr(sim, 'l2_cache') and sim.l2_cache:
        stats = sim.l2_cache.get_stats()
        l2_spills = stats.get('l1_to_l2_spills', 0)
        l2_evictions = stats.get('l2_evictions', 0)

    # TTL Manager stats
    ttl_stats = {}
    if sim.ttl_manager:
        ttl_stats = sim.ttl_manager.get_stats()
        ttl_timeline = sim.ttl_manager.get_timeline_data()
        ttl_predictions = sim.ttl_manager.get_prediction_data()
        
        # Analyze TTL lifecycle
        ttl_pins = sum(1 for e in ttl_timeline if e['event'] == 'pin')
        ttl_expires = sum(1 for e in ttl_timeline if e['event'] == 'expire')
        ttl_unpins = sum(1 for e in ttl_timeline if e['event'] == 'unpin')
        ttl_hits_timeline = sum(1 for e in ttl_timeline if e['event'] == 'hit')
        ttl_misses = sum(1 for e in ttl_timeline if e['event'] == 'miss')
        
        # TTL prediction stats
        prediction_stats = sim.ttl_manager.get_prediction_stats()
        
        ttl_stats.update({
            'pins': ttl_pins,
            'expires': ttl_expires,
            'unpins': ttl_unpins,
            'timeline_hits': ttl_hits_timeline,
            'timeline_misses': ttl_misses,
            'prediction_stats': prediction_stats,
        })

    return {
        'time': total_time,
        'total_requests': total_requests,
        'ttl_hits': ttl_hits,
        'l1_hits': l1_hits,
        'l2_hits': l2_hits,
        'misses': misses,
        'ttl_hit_rate': ttl_hit_rate,
        'l1_hit_rate': l1_hit_rate,
        'l2_hit_rate': l2_hit_rate,
        'miss_rate': miss_rate,
        'cache_hit_rate': (ttl_hits + l1_hits + l2_hits) / total_requests if total_requests > 0 else 0,
        'l2_spills': l2_spills,
        'l2_evictions': l2_evictions,
        # Time breakdown (seconds)
        'prefill_s': total_prefill_ms / 1000,
        'decode_s': total_decode_ms / 1000,
        'h2d_s': total_h2d_ms / 1000,
        'queue_delay_s': total_queue_delay_ms / 1000,
        'ttft_s': total_ttft_ms / 1000,
        'ttl_stats': ttl_stats,
    }


# ============================================================================
# Anomaly-Specific Analysis
# ============================================================================

ANOMALY_CONFIGS = {
    'retry': {
        'name': 'Retry',
        'name_zh': '重试',
        'description': 'Tool takes 2x longer because it runs twice (failed first attempt + successful second)',
        'multiplier': 2.0,
        'color': '#3498db',  # Blue
    },
    'timeout': {
        'name': 'Timeout',
        'name_zh': '超时',
        'description': 'Tool takes 5x longer (10s instead of 2s) due to timeout',
        'multiplier': 5.0,
        'color': '#e74c3c',  # Red
    },
    'hang': {
        'name': 'Hang',
        'name_zh': '挂起',
        'description': 'Tool hangs for 10x longer (20s instead of 2s)',
        'multiplier': 10.0,
        'color': '#9b59b6',  # Purple
    },
    'early_return': {
        'name': 'Early Return',
        'name_zh': '提前返回',
        'description': 'Tool returns early (0.1x = 0.2s instead of 2s)',
        'multiplier': 0.1,
        'color': '#f39c12',  # Yellow/Orange
    },
    'mixed': {
        'name': 'Mixed',
        'name_zh': '混合异常',
        'description': 'Combination of timeout (1/3), retry (1/3), and hang (1/3)',
        'multiplier': None,  # Mixed
        'color': '#1abc9c',  # Teal
    },
}


def run_anomaly_rate_analysis(
    anomaly_type: str,
    l1: int = DEFAULT_CONFIG['l1'],
    l2_ratio: float = DEFAULT_CONFIG['l2_ratio'],
    programs: int = DEFAULT_CONFIG['programs'],
    turns: int = DEFAULT_CONFIG['turns'],
    prefill: int = DEFAULT_CONFIG['prefill'],
    tool_time: float = DEFAULT_CONFIG['tool_time'],
    history_threshold: int = 3,
) -> List[Dict]:
    """Run analysis for a specific anomaly type across different rates."""
    
    l2 = int(l1 * l2_ratio)
    config = ANOMALY_CONFIGS[anomaly_type]
    
    # Rates to test
    if anomaly_type == 'mixed':
        rates = [0.10, 0.15, 0.20, 0.25, 0.30]  # Mixed typically tested at lower rates
    else:
        rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    results = []
    
    print("=" * 100)
    print(f"ANOMALY ANALYSIS: {config['name'].upper()} ({anomaly_type})")
    print("=" * 100)
    print(f"Config: L1={l1}, L2={l2}, Programs={programs}, Turns={turns}, Prefill={prefill}, ToolTime={tool_time}s")
    print(f"Rates: {rates}")
    print()
    
    for rate in rates:
        print(f"\n>>> {config['name']} Rate: {rate*100:.0f}%")
        
        # Generate base requests
        base_requests = generate_requests(
            programs, turns, prefill, tool_time,
            DEFAULT_CONFIG['first_turn_multiplier'], seed=DEFAULT_CONFIG['seed']
        )
        
        # Apply anomaly exceptions
        anomaly_requests = apply_anomaly_exceptions(base_requests, anomaly_type, rate, tool_time)
        
        # Calculate actual average tool time
        tool_times = [r['actual_tool_duration'] for r in anomaly_requests if r['is_tool_call']]
        avg_tool_time = sum(tool_times) / len(tool_times) if tool_times else tool_time
        
        # Count actual anomalies
        actual_anomalies = sum(1 for r in anomaly_requests if r.get('anomaly') and r['is_tool_call'])
        tool_calls = sum(1 for r in anomaly_requests if r['is_tool_call'])
        actual_rate = actual_anomalies / tool_calls if tool_calls > 0 else 0
        
        print(f"  Actual anomaly rate: {actual_rate*100:.1f}% ({actual_anomalies}/{tool_calls} tool calls)")
        print(f"  Avg tool time: {avg_tool_time:.2f}s (base: {tool_time}s)")
        
        # Run Baseline (no TTL)
        print(f"  Running Baseline (no TTL)...", end=" ", flush=True)
        sim_baseline = create_simulator(
            anomaly_requests, False, l1, l2, avg_tool_time,
            history_threshold=history_threshold
        )
        r_baseline = analyze_result_detailed(sim_baseline)
        print(f"Time={r_baseline['time']:.1f}s")
        
        # Run Adaptive TTL
        print(f"  Running Adaptive TTL...", end=" ", flush=True)
        sim_ttl = create_simulator(
            anomaly_requests, True, l1, l2, avg_tool_time,
            history_threshold=history_threshold
        )
        r_ttl = analyze_result_detailed(sim_ttl)
        print(f"Time={r_ttl['time']:.1f}s")
        
        # Calculate speedup
        speedup = r_baseline['time'] / r_ttl['time'] if r_ttl['time'] > 0 else 0
        
        # Print detailed hit breakdown
        print(f"  Baseline: TTL={r_baseline['ttl_hits']}, L1={r_baseline['l1_hits']}, L2={r_baseline['l2_hits']}, Miss={r_baseline['misses']}")
        print(f"  Adaptive: TTL={r_ttl['ttl_hits']}, L1={r_ttl['l1_hits']}, L2={r_ttl['l2_hits']}, Miss={r_ttl['misses']}")
        print(f"  TTL Hit Rate: {r_ttl['ttl_hit_rate']*100:.1f}%")
        print(f"  Speedup: {speedup:.3f}x")
        
        # Get TTL prediction stats
        ttl_pred_stats = r_ttl.get('ttl_stats', {}).get('prediction_stats', {})
        if ttl_pred_stats:
            print(f"  TTL Prediction: avg_pred={ttl_pred_stats.get('avg_predicted_ttl', 0):.2f}s, "
                  f"avg_actual={ttl_pred_stats.get('avg_actual_duration', 0):.2f}s, "
                  f"avg_error={ttl_pred_stats.get('avg_error', 0):.2f}s")
        
        results.append({
            'anomaly_rate': rate,
            'actual_rate': actual_rate,
            'avg_tool_time': avg_tool_time,
            'baseline': r_baseline,
            'adaptive_ttl': r_ttl,
            'speedup': speedup,
            'actual_anomalies': actual_anomalies,
            'tool_calls': tool_calls,
        })
    
    return results


def analyze_results(results: List[Dict]) -> Dict[str, List]:
    """Analyze results to extract key insights."""
    
    analysis = {
        'anomaly_rate': [],
        'avg_tool_time': [],
        'ttl_hit_rate': [],
        'l1_hit_rate': [],
        'l2_hit_rate': [],
        'miss_rate': [],
        'cache_hit_rate': [],
        'speedup': [],
        'baseline_time': [],
        'ttl_time': [],
        'l2_spills': [],
        'ttl_pins': [],
        'ttl_expires': [],
        'ttl_prediction_error': [],
        'ttl_underestimate_rate': [],
    }
    
    for r in results:
        analysis['anomaly_rate'].append(r['anomaly_rate'] * 100)
        analysis['avg_tool_time'].append(r['avg_tool_time'])
        analysis['ttl_hit_rate'].append(r['adaptive_ttl']['ttl_hit_rate'] * 100)
        analysis['l1_hit_rate'].append(r['adaptive_ttl']['l1_hit_rate'] * 100)
        analysis['l2_hit_rate'].append(r['adaptive_ttl']['l2_hit_rate'] * 100)
        analysis['miss_rate'].append(r['adaptive_ttl']['miss_rate'] * 100)
        analysis['cache_hit_rate'].append(r['adaptive_ttl']['cache_hit_rate'] * 100)
        analysis['speedup'].append(r['speedup'])
        analysis['baseline_time'].append(r['baseline']['time'])
        analysis['ttl_time'].append(r['adaptive_ttl']['time'])
        analysis['l2_spills'].append(r['adaptive_ttl']['l2_spills'])
        
        ttl_stats = r['adaptive_ttl'].get('ttl_stats', {})
        analysis['ttl_pins'].append(ttl_stats.get('pins', 0))
        analysis['ttl_expires'].append(ttl_stats.get('expires', 0))
        
        pred_stats = ttl_stats.get('prediction_stats', {})
        analysis['ttl_prediction_error'].append(pred_stats.get('avg_error', 0))
        analysis['ttl_underestimate_rate'].append(pred_stats.get('ttl_underestimate_rate', 0) * 100)
    
    return analysis


def generate_charts(analysis: Dict[str, List], anomaly_type: str, output_dir: Path) -> Tuple[str, str]:
    """Generate visualization charts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping chart generation")
        return "", ""
    
    config = ANOMALY_CONFIGS[anomaly_type]
    anomaly_rates = analysis['anomaly_rate']
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'{config["name"]} Anomaly: TTL Hit/Miss Analysis', fontsize=14, fontweight='bold')
    
    # 1. Speedup vs Anomaly Rate
    ax = axes[0, 0]
    colors = [config['color'] if s >= 1 else '#e74c3c' for s in analysis['speedup']]
    bars = ax.bar(anomaly_rates, analysis['speedup'], color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even')
    ax.set_xlabel('Anomaly Rate (%)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'{config["name"]} Rate vs TTL Speedup')
    ax.set_xticks(anomaly_rates)
    for bar, sp in zip(bars, analysis['speedup']):
        height = bar.get_height()
        ax.annotate(f'{sp:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Cache Hit Rates Breakdown
    ax = axes[0, 1]
    ax.plot(anomaly_rates, analysis['ttl_hit_rate'], 'b-o', label='TTL Hit Rate', linewidth=2, markersize=8)
    ax.plot(anomaly_rates, analysis['l1_hit_rate'], 'g-s', label='L1 Hit Rate', linewidth=2, markersize=8)
    ax.plot(anomaly_rates, analysis['l2_hit_rate'], 'm-^', label='L2 Hit Rate', linewidth=2, markersize=8)
    ax.plot(anomaly_rates, analysis['miss_rate'], 'r-x', label='Miss Rate', linewidth=2, markersize=8)
    ax.set_xlabel('Anomaly Rate (%)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Cache Hit Rate Breakdown')
    ax.legend(loc='best')
    ax.set_xticks(anomaly_rates)
    ax.grid(True, alpha=0.3)
    
    # 3. TTL vs Baseline Time Comparison
    ax = axes[1, 0]
    width = 2
    x = list(range(len(anomaly_rates)))
    ax.bar([i - width/2 for i in x], analysis['baseline_time'], width, label='Baseline', color='steelblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], analysis['ttl_time'], width, label='Adaptive TTL', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{r:.0f}%' for r in anomaly_rates])
    ax.set_xlabel('Anomaly Rate')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Execution Time: Baseline vs TTL')
    ax.legend()
    
    # 4. TTL Prediction Error vs Anomaly Rate
    ax = axes[1, 1]
    ax.plot(anomaly_rates, analysis['ttl_prediction_error'], 'r-o', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(anomaly_rates, 0, analysis['ttl_prediction_error'], 
                    where=[e > 0 for e in analysis['ttl_prediction_error']], 
                    color='green', alpha=0.3, label='Overestimation (+)')
    ax.fill_between(anomaly_rates, 0, analysis['ttl_prediction_error'], 
                    where=[e < 0 for e in analysis['ttl_prediction_error']], 
                    color='red', alpha=0.3, label='Underestimation (-)')
    ax.set_xlabel('Anomaly Rate (%)')
    ax.set_ylabel('Prediction Error (s)')
    ax.set_title('TTL Prediction Error (Predicted - Actual)')
    ax.set_xticks(anomaly_rates)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 5. L2 Spills vs Anomaly Rate
    ax = axes[2, 0]
    ax.bar(anomaly_rates, analysis['l2_spills'], color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Anomaly Rate (%)')
    ax.set_ylabel('L2 Spills')
    ax.set_title('L2 Cache Spills vs Anomaly Rate')
    ax.set_xticks(anomaly_rates)
    for i, v in enumerate(analysis['l2_spills']):
        ax.annotate(f'{int(v)}',
                    xy=(anomaly_rates[i], v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # 6. Average Tool Time vs Speedup
    ax = axes[2, 1]
    scatter = ax.scatter(analysis['avg_tool_time'], analysis['speedup'], 
                        c=anomaly_rates, cmap='viridis', s=100, edgecolors='black')
    for i, rate in enumerate(anomaly_rates):
        ax.annotate(f'{rate:.0f}%', 
                   (analysis['avg_tool_time'][i], analysis['speedup'][i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax.set_xlabel('Average Tool Time (s)')
    ax.set_ylabel('Speedup (x)')
    ax.set_title('Tool Time vs Speedup (Color = Anomaly Rate)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Anomaly Rate (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / f'{anomaly_type}_ttl_analysis.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {chart_path}")
    
    # Additional pie charts for hit composition
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    fig2.suptitle(f'{config["name"]} Anomaly: Cache Hit Composition', fontsize=14, fontweight='bold')
    
    max_plots = min(7, len(anomaly_rates))
    for idx in range(max_plots):
        ax = axes2[idx // 4, idx % 4]
        ttl = analysis['ttl_hit_rate'][idx]
        l1 = analysis['l1_hit_rate'][idx]
        l2 = analysis['l2_hit_rate'][idx]
        miss = analysis['miss_rate'][idx]
        
        sizes = [ttl, l1, l2, miss]
        labels = ['TTL', 'L1', 'L2', 'Miss']
        colors_pie = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
        
        non_zero_sizes = [s for s in sizes if s > 0]
        non_zero_labels = [l for s, l in zip(sizes, labels) if s > 0]
        non_zero_colors = [c for s, c in zip(sizes, colors_pie) if s > 0]
        non_zero_explode = [0.05] + [0] * (len(non_zero_sizes) - 1)
        
        if non_zero_sizes:
            ax.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
                   autopct='%1.1f%%', explode=non_zero_explode, startangle=90)
        ax.set_title(f'{config["name"]}: {anomaly_rates[idx]:.0f}%')
    
    # Hide extra subplots
    for idx in range(max_plots, 8):
        axes2[idx // 4, idx % 4].axis('off')
    
    # Add summary in remaining space
    fig2.text(0.88, 0.15, f'Speedup at each rate:\n' + 
              '\n'.join([f'{anomaly_rates[i]:.0f}%: {analysis["speedup"][i]:.2f}x' 
                        for i in range(max_plots)]),
              fontsize=9, ha='center', va='bottom',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    pie_path = output_dir / f'{anomaly_type}_hit_composition.png'
    plt.savefig(pie_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pie chart saved to: {pie_path}")
    
    return str(chart_path.name), str(pie_path.name)


def generate_html_report(
    results: List[Dict], 
    analysis: Dict[str, List], 
    anomaly_type: str,
    output_dir: Path
) -> str:
    """Generate HTML analysis report for a specific anomaly type."""
    
    config = ANOMALY_CONFIGS[anomaly_type]
    chart_path = output_dir / f'{anomaly_type}_ttl_analysis.png'
    pie_path = output_dir / f'{anomaly_type}_hit_composition.png'
    
    # Find key metrics
    optimal_idx = analysis['speedup'].index(max(analysis['speedup']))
    worst_idx = analysis['speedup'].index(min(analysis['speedup']))
    best_speedup = max(analysis['speedup'])
    worst_speedup = min(analysis['speedup'])
    
    # Calculate RGB for gradient
    r, g, b = int(config['color'][1:3], 16), int(config['color'][3:5], 16), int(config['color'][5:7], 16)
    
    # Determine verdict for each rate
    def get_verdict(speedup):
        if speedup >= 1.05:
            return "TTL Beneficial", "speedup-good"
        elif speedup >= 0.95:
            return "Neutral", "speedup-neutral"
        else:
            return "TTL Harmful", "speedup-bad"
    
    # Build table rows
    table_rows = ""
    for i, r in enumerate(results):
        rate_pct = analysis['anomaly_rate'][i]
        verdict, verdict_class = get_verdict(r['speedup'])
        speedup_class = 'speedup-good' if r['speedup'] >= 1.05 else 'speedup-bad' if r['speedup'] < 0.95 else 'speedup-neutral'
        table_rows += f"""
                    <tr>
                        <td>{rate_pct:.0f}%</td>
                        <td>{r['avg_tool_time']:.2f}s</td>
                        <td>{r['baseline']['time']:.1f}s</td>
                        <td>{r['adaptive_ttl']['time']:.1f}s</td>
                        <td class="{speedup_class}">{r['speedup']:.3f}x</td>
                        <td>{r['adaptive_ttl']['ttl_hit_rate']*100:.1f}%</td>
                        <td>{r['adaptive_ttl']['l1_hit_rate']*100:.1f}%</td>
                        <td>{r['adaptive_ttl']['l2_hit_rate']*100:.1f}%</td>
                        <td>{r['adaptive_ttl']['miss_rate']*100:.1f}%</td>
                        <td>{r['adaptive_ttl']['l2_spills']}</td>
                    </tr>
"""
    
    # Build lifecycle table rows
    lifecycle_rows = ""
    for i, r in enumerate(results):
        rate_pct = analysis['anomaly_rate'][i]
        ttl_stats = r['adaptive_ttl'].get('ttl_stats', {})
        error = analysis['ttl_prediction_error'][i]
        under_rate = analysis['ttl_underestimate_rate'][i]
        verdict, verdict_class = get_verdict(r['speedup'])
        lifecycle_rows += f"""
                    <tr>
                        <td>{rate_pct:.0f}%</td>
                        <td>{ttl_stats.get('pins', 0)}</td>
                        <td>{ttl_stats.get('expires', 0)}</td>
                        <td>{error:.2f}s</td>
                        <td>{under_rate:.1f}%</td>
                        <td class="{verdict_class}">{verdict}</td>
                    </tr>
"""
    
    # Generate insights
    trend_desc = _get_trend_description(analysis, anomaly_type)
    pred_analysis = _get_prediction_analysis(analysis)
    recommendations = _get_recommendations(analysis, config)
    
    miss_trend = "increases" if analysis['miss_rate'][-1] > analysis['miss_rate'][0] else "decreases"
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config['name']} Anomaly TTL Analysis</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-red: #ef4444;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-blue: #3b82f6;
            --accent-purple: #a855f7;
            --anomaly-color: {config['color']};
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1a0a1f 0%, #0f172a 100%);
            padding: 2rem;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        .header h1 {{
            font-size: 2rem;
            background: linear-gradient(135deg, var(--anomaly-color), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{ color: var(--text-secondary); margin-top: 0.5rem; }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
        
        .config-box {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        
        .config-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--anomaly-color);
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
        }}
        
        .config-item {{
            background: var(--bg-card);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        
        .config-item .value {{
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--anomaly-color);
        }}
        
        .config-item .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}
        
        .anomaly-description {{
            background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.15), rgba(0,0,0,0.3));
            border-left: 4px solid var(--anomaly-color);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        
        .summary-card .value {{ font-size: 1.8rem; font-weight: 700; }}
        .summary-card .label {{ color: var(--text-secondary); font-size: 0.8rem; margin-top: 0.25rem; }}
        .summary-card .value.good {{ color: var(--accent-green); }}
        .summary-card .value.bad {{ color: var(--accent-red); }}
        .summary-card .value.neutral {{ color: var(--accent-yellow); }}
        
        .section-title {{
            font-size: 1.3rem;
            font-weight: 600;
            margin: 2.5rem 0 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid var(--anomaly-color);
        }}
        
        .chart-block {{
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        
        .chart-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }}
        .chart-subtitle {{ color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 1rem; }}
        
        .chart-img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            margin-top: 1rem;
        }}
        
        .result-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.85rem;
        }}
        
        .result-table th, .result-table td {{
            padding: 0.75rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .result-table th {{
            background: rgba(0,0,0,0.3);
            font-weight: 600;
        }}
        
        .result-table tr:nth-child(even) {{
            background: rgba(255,255,255,0.02);
        }}
        
        .speedup-good {{ color: var(--accent-green); font-weight: 600; }}
        .speedup-bad {{ color: var(--accent-red); font-weight: 600; }}
        .speedup-neutral {{ color: var(--accent-yellow); font-weight: 600; }}
        
        .insight-box {{
            background: linear-gradient(135deg, rgba({r}, {g}, {b}, 0.1), rgba(168, 85, 247, 0.1));
            border: 1px solid rgba({r}, {g}, {b}, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }}
        
        .insight-title {{
            font-weight: 600;
            color: var(--anomaly-color);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        
        .insight-text {{ color: var(--text-secondary); font-size: 0.85rem; }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid rgba(255,255,255,0.05);
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{config['name']} Anomaly TTL Analysis</h1>
        <p>{config['name_zh']}场景下TTL命中/失效比例变化分析</p>
    </div>
    
    <div class="container">
        <!-- Configuration -->
        <div class="config-box">
            <div class="config-title">Experiment Configuration</div>
            <div class="config-grid">
                <div class="config-item">
                    <div class="value">15,000</div>
                    <div class="label">L1 Cache (tokens)</div>
                </div>
                <div class="config-item">
                    <div class="value">22,500</div>
                    <div class="label">L2 Cache (tokens)</div>
                </div>
                <div class="config-item">
                    <div class="value">50</div>
                    <div class="label">Programs</div>
                </div>
                <div class="config-item">
                    <div class="value">20</div>
                    <div class="label">Turns</div>
                </div>
                <div class="config-item">
                    <div class="value">2.0s</div>
                    <div class="label">Base Tool Time</div>
                </div>
            </div>
            <div class="anomaly-description">
                <strong>Anomaly Description:</strong> {config['description']}
            </div>
        </div>
        
        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="summary-card">
                <div class="value good">{best_speedup:.2f}x</div>
                <div class="label">Best Speedup ({analysis['anomaly_rate'][optimal_idx]:.0f}% {config['name']})</div>
            </div>
            <div class="summary-card">
                <div class="value bad">{worst_speedup:.2f}x</div>
                <div class="label">Worst Speedup ({analysis['anomaly_rate'][worst_idx]:.0f}% {config['name']})</div>
            </div>
            <div class="summary-card">
                <div class="value">{max(analysis['ttl_hit_rate']):.1f}%</div>
                <div class="label">Peak TTL Hit Rate</div>
            </div>
            <div class="summary-card">
                <div class="value">{sum(1 for s in analysis['speedup'] if s >= 1.0)}/{len(analysis['speedup'])}</div>
                <div class="label">Rates Where TTL Helps</div>
            </div>
        </div>
        
        <!-- Detailed Results Table -->
        <div class="section-title">Detailed Results by {config['name']} Rate</div>
        
        <div class="chart-block">
            <table class="result-table">
                <thead>
                    <tr>
                        <th>{config['name']} Rate</th>
                        <th>Avg Tool Time</th>
                        <th>Baseline Time</th>
                        <th>TTL Time</th>
                        <th>Speedup</th>
                        <th>TTL Hit Rate</th>
                        <th>L1 Hit Rate</th>
                        <th>L2 Hit Rate</th>
                        <th>Miss Rate</th>
                        <th>L2 Spills</th>
                    </tr>
                </thead>
                <tbody>
{table_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Charts -->
        <div class="section-title">Visual Analysis</div>
        
        <div class="chart-block">
            <div class="chart-title">TTL Performance vs {config['name']} Rate</div>
            <div class="chart-subtitle">Multi-panel analysis of TTL behavior under {config['name']} conditions</div>
            <img class="chart-img" src="{chart_path.name}" alt="TTL Analysis Chart">
        </div>
        
        <div class="chart-block">
            <div class="chart-title">Cache Hit Composition</div>
            <div class="chart-subtitle">Pie charts showing TTL vs L1 vs L2 vs Miss breakdown at each rate</div>
            <img class="chart-img" src="{pie_path.name}" alt="Hit Composition Chart">
        </div>
        
        <!-- Key Insights -->
        <div class="section-title">Key Findings</div>
        
        <div class="chart-block">
            <div class="insight-box">
                <div class="insight-title">1. TTL Effectiveness vs {config['name']} Rate</div>
                <div class="insight-text">
                    <strong>Optimal {config['name']} rate for TTL:</strong> {analysis['anomaly_rate'][optimal_idx]:.0f}% 
                    (Speedup: {best_speedup:.3f}x)<br><br>
                    <strong>Worst {config['name']} rate for TTL:</strong> {analysis['anomaly_rate'][worst_idx]:.0f}% 
                    (Speedup: {worst_speedup:.3f}x)<br><br>
                    <strong>Trend:</strong> {trend_desc}
                </div>
            </div>
            
            <div class="insight-box" style="margin-top: 1rem;">
                <div class="insight-title">2. Cache Hit Rate Analysis</div>
                <div class="insight-text">
                    <strong>TTL Hit Rate Range:</strong> {min(analysis['ttl_hit_rate']):.1f}% to {max(analysis['ttl_hit_rate']):.1f}%<br><br>
                    <strong>Miss Rate Impact:</strong> {miss_trend} from {analysis['miss_rate'][0]:.1f}% to {analysis['miss_rate'][-1]:.1f}%<br><br>
                    <strong>L2 Cache Role:</strong> L2 acts as fallback when TTL expires early.
                </div>
            </div>
            
            <div class="insight-box" style="margin-top: 1rem;">
                <div class="insight-title">3. TTL Prediction Analysis</div>
                <div class="insight-text">
                    <strong>Prediction Error Range:</strong> {min(analysis['ttl_prediction_error']):.2f}s to {max(analysis['ttl_prediction_error']):.2f}s<br><br>
                    <strong>Underestimate Rate:</strong> {min(analysis['ttl_underestimate_rate']):.1f}% to {max(analysis['ttl_underestimate_rate']):.1f}%<br><br>
                    <strong>Analysis:</strong> {pred_analysis}
                </div>
            </div>
            
            <div class="insight-box" style="margin-top: 1rem;">
                <div class="insight-title">4. Practical Recommendations</div>
                <div class="insight-text">
                    {recommendations}
                </div>
            </div>
        </div>
        
        <!-- TTL Lifecycle Analysis -->
        <div class="section-title">TTL Lifecycle Statistics</div>
        
        <div class="chart-block">
            <table class="result-table">
                <thead>
                    <tr>
                        <th>{config['name']} Rate</th>
                        <th>TTL Pins</th>
                        <th>TTL Expires</th>
                        <th>Avg Prediction Error</th>
                        <th>Underestimate Rate</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
{lifecycle_rows}
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            {config['name']} Anomaly TTL Analysis | Baseline Config: L1=15K, L2=22.5K, Programs=50, Turns=20
        </div>
    </div>
</body>
</html>
"""
    
    return html


def _get_trend_description(analysis: Dict[str, List], anomaly_type: str) -> str:
    """Get trend description based on speedup pattern."""
    speedups = analysis['speedup']
    
    if all(s >= 1.0 for s in speedups):
        return "TTL is BENEFICIAL at ALL rates tested."
    elif all(s < 1.0 for s in speedups):
        return "TTL is HARMFUL at ALL rates tested."
    elif speedups[0] > speedups[-1]:
        return f"TTL effectiveness DECREASES with higher {anomaly_type} rates."
    elif speedups[0] < speedups[-1]:
        return f"TTL effectiveness INCREASES with higher {anomaly_type} rates."
    else:
        return "TTL effectiveness is NON-MONOTONIC with varying rates."


def _get_prediction_analysis(analysis: Dict[str, List]) -> str:
    """Get prediction analysis description."""
    errors = analysis['ttl_prediction_error']
    
    if all(e > 0 for e in errors):
        return "TTL consistently OVERESTIMATES (keeps data longer than needed). Generally beneficial for hit rates."
    elif all(e < 0 for e in errors):
        return "TTL consistently UNDERESTIMATES (expires too early). Causes premature cache eviction."
    else:
        return "TTL prediction accuracy varies by rate, suggesting sensitivity to tool time variance."


def _get_recommendations(analysis: Dict[str, List], config: Dict) -> str:
    """Get practical recommendations."""
    beneficial_count = sum(1 for s in analysis['speedup'] if s >= 1.0)
    total_count = len(analysis['speedup'])
    
    if beneficial_count == total_count:
        return f"""<strong>Recommendation:</strong> Enable TTL for all {config['name']} rates.<br>
                    TTL provides consistent performance improvement regardless of {config['name']} frequency."""
    elif beneficial_count == 0:
        return f"""<strong>Recommendation:</strong> Consider disabling TTL when {config['name']} is expected.<br>
                    TTL causes consistent performance degradation in this scenario."""
    else:
        beneficial_rates = [analysis['anomaly_rate'][i] for i, s in enumerate(analysis['speedup']) if s >= 1.0]
        harmful_rates = [analysis['anomaly_rate'][i] for i, s in enumerate(analysis['speedup']) if s < 1.0]
        return f"""<strong>Recommendation:</strong> TTL is beneficial at {config['name']} rates of {beneficial_rates[0]:.0f}%, 
                    but harmful at {', '.join([f'{r:.0f}%' for r in harmful_rates])}%.<br>
                    Consider adaptive TTL strategies that adjust based on observed anomaly rates."""


def main():
    parser = argparse.ArgumentParser(
        description="Unified Anomaly TTL Hit/Miss Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Anomaly Types:
  retry        - Tool takes 2x longer (runs twice)
  timeout      - Tool takes 5x longer (10s instead of 2s)
  hang         - Tool hangs for 10x longer (20s instead of 2s)
  early_return - Tool returns early (0.1x = 0.2s)
  mixed        - Combination of timeout, retry, hang
  all          - Run all anomaly types

Examples:
  python anomaly_ttl_analysis.py --type retry
  python anomaly_ttl_analysis.py --type timeout
  python anomaly_ttl_analysis.py --type all
        """
    )
    
    parser.add_argument('--type', type=str, default='all',
                       choices=['retry', 'timeout', 'hang', 'early_return', 'mixed', 'all'],
                       help='Anomaly type to analyze')
    parser.add_argument('--l1', type=int, default=DEFAULT_CONFIG['l1'],
                       help='L1 cache capacity (tokens)')
    parser.add_argument('--l2-ratio', type=float, default=DEFAULT_CONFIG['l2_ratio'],
                       help='L2/L1 ratio (default: 1.5)')
    parser.add_argument('--programs', type=int, default=DEFAULT_CONFIG['programs'],
                       help='Number of programs (default: 50)')
    parser.add_argument('--turns', type=int, default=DEFAULT_CONFIG['turns'],
                       help='Number of turns (default: 20)')
    parser.add_argument('--prefill', type=int, default=DEFAULT_CONFIG['prefill'],
                       help='Prefill tokens per turn (default: 200)')
    parser.add_argument('--tool-time', type=float, default=DEFAULT_CONFIG['tool_time'],
                       help='Base tool execution time (default: 2.0s)')
    parser.add_argument('--history-threshold', type=int, default=3,
                       help='History threshold for adaptive TTL (default: 3)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which anomaly types to run
    if args.type == 'all':
        anomaly_types = ['retry', 'timeout', 'hang', 'early_return', 'mixed']
    else:
        anomaly_types = [args.type]
    
    all_results = {}
    
    for anomaly_type in anomaly_types:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {anomaly_type.upper()}")
        print("=" * 80)
        
        # Create subdirectory for this anomaly type
        sub_dir = output_dir / f'{anomaly_type}_analysis'
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        results = run_anomaly_rate_analysis(
            anomaly_type=anomaly_type,
            l1=args.l1,
            l2_ratio=args.l2_ratio,
            programs=args.programs,
            turns=args.turns,
            prefill=args.prefill,
            tool_time=args.tool_time,
            history_threshold=args.history_threshold,
        )
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Save raw results
        raw_output = {
            'config': {
                'anomaly_type': anomaly_type,
                'l1': args.l1,
                'l2': int(args.l1 * args.l2_ratio),
                'programs': args.programs,
                'turns': args.turns,
                'prefill': args.prefill,
                'tool_time': args.tool_time,
                'history_threshold': args.history_threshold,
            },
            'results': [
                {
                    'anomaly_rate': r['anomaly_rate'],
                    'actual_rate': r['actual_rate'],
                    'avg_tool_time': r['avg_tool_time'],
                    'speedup': r['speedup'],
                    'baseline': r['baseline'],
                    'adaptive_ttl': {
                        'time': r['adaptive_ttl']['time'],
                        'total_requests': r['adaptive_ttl']['total_requests'],
                        'ttl_hits': r['adaptive_ttl']['ttl_hits'],
                        'l1_hits': r['adaptive_ttl']['l1_hits'],
                        'l2_hits': r['adaptive_ttl']['l2_hits'],
                        'misses': r['adaptive_ttl']['misses'],
                        'ttl_hit_rate': r['adaptive_ttl']['ttl_hit_rate'],
                        'l1_hit_rate': r['adaptive_ttl']['l1_hit_rate'],
                        'l2_hit_rate': r['adaptive_ttl']['l2_hit_rate'],
                        'miss_rate': r['adaptive_ttl']['miss_rate'],
                        'cache_hit_rate': r['adaptive_ttl']['cache_hit_rate'],
                        'l2_spills': r['adaptive_ttl']['l2_spills'],
                        'ttl_stats': r['adaptive_ttl'].get('ttl_stats', {}),
                    },
                }
                for r in results
            ],
            'analysis': analysis,
        }
        
        raw_path = sub_dir / f'{anomaly_type}_analysis_raw.json'
        with open(raw_path, 'w') as f:
            json.dump(raw_output, f, indent=2, default=str)
        print(f"Raw results saved to: {raw_path}")
        
        # Generate charts
        print("Generating charts...")
        chart_file, pie_file = generate_charts(analysis, anomaly_type, sub_dir)
        
        # Generate HTML report
        print("Generating HTML report...")
        html_content = generate_html_report(results, analysis, anomaly_type, sub_dir)
        html_path = sub_dir / f'{anomaly_type}_analysis.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"HTML report saved to: {html_path}")
        
        all_results[anomaly_type] = {
            'results': results,
            'analysis': analysis,
            'html_path': str(html_path),
        }
        
        # Print summary
        _print_summary(anomaly_type, results, analysis)
    
    # Generate comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: ALL ANOMALY TYPES")
    print("=" * 80)
    _print_comparison_summary(all_results)


def _print_summary(anomaly_type: str, results: List[Dict], analysis: Dict[str, List]):
    """Print summary for a single anomaly type."""
    config = ANOMALY_CONFIGS[anomaly_type]
    
    print(f"\n{'='*60}")
    print(f"{config['name'].upper()} ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Rate':>8} | {'Avg Tool':>10} | {'Baseline':>10} | {'TTL':>10} | {'Speedup':>8} | {'TTL Hit%':>8}")
    print("-" * 80)
    
    for i, r in enumerate(results):
        sp_str = f"{r['speedup']:.3f}x" if r['speedup'] >= 1 else f"{-1/r['speedup']:.3f}x慢"
        print(f"{analysis['anomaly_rate'][i]:>7.0f}% | {r['avg_tool_time']:>9.2f}s | "
              f"{r['baseline']['time']:>9.1f}s | {r['adaptive_ttl']['time']:>9.1f}s | "
              f"{sp_str:>8} | {r['adaptive_ttl']['ttl_hit_rate']*100:>7.1f}%")
    
    optimal_idx = analysis['speedup'].index(max(analysis['speedup']))
    worst_idx = analysis['speedup'].index(min(analysis['speedup']))
    print(f"\nBest: {analysis['anomaly_rate'][optimal_idx]:.0f}% ({max(analysis['speedup']):.3f}x)")
    print(f"Worst: {analysis['anomaly_rate'][worst_idx]:.0f}% ({min(analysis['speedup']):.3f}x)")


def _print_comparison_summary(all_results: Dict):
    """Print comparison summary across all anomaly types."""
    print("\n" + "=" * 100)
    print("CROSS-ANOMALY COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Anomaly':<15} | {'Best Rate':>10} | {'Best Speedup':>12} | {'Worst Rate':>10} | {'Worst Speedup':>12}")
    print("-" * 100)
    
    for anomaly_type, data in all_results.items():
        analysis = data['analysis']
        results = data['results']
        config = ANOMALY_CONFIGS[anomaly_type]
        
        optimal_idx = analysis['speedup'].index(max(analysis['speedup']))
        worst_idx = analysis['speedup'].index(min(analysis['speedup']))
        
        print(f"{config['name']:<15} | {analysis['anomaly_rate'][optimal_idx]:>9.0f}% | "
              f"{max(analysis['speedup']):>11.3f}x | {analysis['anomaly_rate'][worst_idx]:>9.0f}% | "
              f"{min(analysis['speedup']):>11.3f}x")
    
    print("\n" + "=" * 100)
    print("TTL RECOMMENDATION BY ANOMALY TYPE")
    print("=" * 100)
    
    for anomaly_type, data in all_results.items():
        analysis = data['analysis']
        config = ANOMALY_CONFIGS[anomaly_type]
        
        beneficial = sum(1 for s in analysis['speedup'] if s >= 1.0)
        total = len(analysis['speedup'])
        
        if beneficial == total:
            recommendation = "RECOMMEND: Enable TTL"
        elif beneficial == 0:
            recommendation = "RECOMMEND: Disable TTL"
        else:
            recommendation = "RECOMMEND: Adaptive TTL"
        
        print(f"{config['name']:<15} | {beneficial}/{total} rates beneficial | {recommendation}")


if __name__ == "__main__":
    main()
