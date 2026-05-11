#!/usr/bin/env python3
"""
Generate Improved Report Charts for TTL Anomaly Analysis

Fixed:
1. Speedup vs Rate chart - complete x-axis with actual data points
2. All English labels - no encoding issues
3. Improved Error vs History Size visualization
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")
    sys.exit(1)

# Color schemes - all English
COLORS = {
    'timeout': '#e74c3c',
    'retry': '#3498db',
    'early_return': '#f39c12',
    'hang': '#9b59b6',
    'mixed': '#1abc9c',
    'normal': '#2ecc71',
}

TYPE_LABELS = {
    'timeout': 'Timeout',
    'retry': 'Retry',
    'early_return': 'Early Return',
    'hang': 'Hang',
    'mixed': 'Mixed',
    'normal': 'Normal',
}

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_dir(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Chart 1: Speedup vs Rate Line Chart - FIXED AXIS
# =============================================================================

def plot_speedup_vs_rate_line(anomaly_data, output_dir):
    """Generate line chart with FIXED x-axis showing actual exception rates."""
    # Group by exception type
    by_type = {}
    for r in anomaly_data:
        exc_type = r.get('exception_type', 'unknown')
        rate = r.get('exception_rate', 0)
        speedup = r.get('speedup', 1.0)
        
        if exc_type not in by_type:
            by_type[exc_type] = []
        by_type[exc_type].append({'rate': rate, 'speedup': speedup})
    
    # Sort each type by rate
    for exc_type in by_type:
        by_type[exc_type] = sorted(by_type[exc_type], key=lambda x: x['rate'])
    
    # Get ALL unique rates from data
    all_rates = sorted(set(r['rate'] for data in by_type.values() for r in data))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for exc_type, data in sorted(by_type.items()):
        rates = [d['rate'] for d in data]
        speedups = [d['speedup'] for d in data]
        color = COLORS.get(exc_type, '#333333')
        label = TYPE_LABELS.get(exc_type, exc_type)
        
        # Plot with markers
        ax.plot(rates, speedups, 'o-', color=color, linewidth=2.5, 
                markersize=12, label=label, markeredgecolor='black', markeredgewidth=1.5)
        
        # Add value labels on points
        for r, s in zip(rates, speedups):
            offset = 0.08 if s >= 1.0 else -0.15
            ax.annotate(f'{s:.2f}x', (r, s), textcoords="offset points",
                       xytext=(0, 1800 if offset > 0 else -2500), ha='center',
                       fontsize=10, fontweight='bold')
    
    # Add shaded zones - extended to cover 50%
    max_rate = max(all_rates) if all_rates else 0.5
    ax.fill_between([0, max_rate], [1, 1], [0.5, 0.5], alpha=0.15, color='red', label='Harmful')
    ax.fill_between([0, max_rate], [1, 1], [2.2, 2.2], alpha=0.15, color='green', label='Beneficial')
    
    # Baseline
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    
    # X-AXIS: Use ALL actual data points
    ax.set_xlabel('Exception Rate (%)', fontsize=14)
    ax.set_xticks(all_rates)
    ax.set_xticklabels([f'{int(r*100)}%' for r in all_rates], fontsize=12)
    
    ax.set_ylabel('Speedup (TTL vs Baseline)', fontsize=14)
    ax.set_title('TTL Speedup vs Exception Rate by Error Type', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.02, max_rate + 0.05)
    ax.set_ylim(0.5, 2.2)
    ax.legend(loc='upper right', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'speedup_vs_rate_by_type.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Chart 2: Error vs History Size - IMPROVED
# =============================================================================

def plot_error_by_history_heatmap(anomaly_data, prediction_data, output_dir):
    """Generate improved heatmap showing prediction error by history size and scenario."""
    # Parse prediction data
    scenario_errors = {}
    history_sizes = set()
    
    for name, data in prediction_data.items():
        if isinstance(data, dict) and 'predictions' in data:
            exc_type = data.get('exception_type', 'normal')
            if exc_type is None:
                exc_type = 'normal'
            
            errors_by_history = {}
            for p in data['predictions']:
                hs = p.get('cumulative_history_size', 0)
                error = abs(p.get('error', 0))
                if hs not in errors_by_history:
                    errors_by_history[hs] = []
                errors_by_history[hs].append(error)
                history_sizes.add(hs)
            
            # Calculate average error for each history size
            avg_errors = {hs: sum(errs)/len(errs) for hs, errs in errors_by_history.items()}
            scenario_errors[exc_type] = avg_errors
    
    if not history_sizes:
        print("  Warning: No history size data found")
        return
    
    # Create matrix for heatmap
    history_sizes = sorted(list(history_sizes))
    scenarios = sorted(scenario_errors.keys())
    
    # Filter to show meaningful history sizes (0-20)
    history_sizes = [h for h in history_sizes if h <= 20]
    
    matrix = np.zeros((len(scenarios), len(history_sizes)))
    for i, scenario in enumerate(scenarios):
        for j, hs in enumerate(history_sizes):
            if hs in scenario_errors[scenario]:
                matrix[i, j] = scenario_errors[scenario][hs]
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=15)
    
    ax.set_xlabel('History Size (Number of Previous Tool Calls)', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    ax.set_title('Prediction Error vs History Size (Heatmap)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(len(history_sizes)))
    ax.set_xticklabels(history_sizes, fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([TYPE_LABELS.get(s, s) for s in scenarios], fontsize=11)
    
    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(history_sizes)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       fontsize=9, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Prediction Error (seconds)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_by_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def plot_error_trend_by_history(prediction_data, output_dir):
    """Generate line plot showing error trend with history size."""
    scenario_errors = {}
    history_sizes = set()
    
    for name, data in prediction_data.items():
        if isinstance(data, dict) and 'predictions' in data:
            exc_type = data.get('exception_type', 'normal')
            if exc_type is None:
                exc_type = 'normal'
            
            errors_by_history = {}
            for p in data['predictions']:
                hs = p.get('cumulative_history_size', 0)
                error = abs(p.get('error', 0))
                if hs not in errors_by_history:
                    errors_by_history[hs] = []
                errors_by_history[hs].append(error)
                history_sizes.add(hs)
            
            avg_errors = {hs: sum(errs)/len(errs) for hs, errs in errors_by_history.items()}
            scenario_errors[exc_type] = avg_errors
    
    history_sizes = sorted([h for h in history_sizes if h <= 20])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for scenario, errors in sorted(scenario_errors.items()):
        hs_list = sorted([h for h in history_sizes if h in errors])
        error_list = [errors[h] for h in hs_list]
        color = COLORS.get(scenario, '#333333')
        label = TYPE_LABELS.get(scenario, scenario)
        ax.plot(hs_list, error_list, 'o-', color=color, linewidth=2, 
                markersize=8, label=label)
    
    ax.set_xlabel('History Size', fontsize=14)
    ax.set_ylabel('Average Prediction Error (seconds)', fontsize=14)
    ax.set_title('Prediction Error Trend with History Size', fontsize=16, fontweight='bold')
    ax.set_xlim(-0.5, 20.5)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for first-turn anomaly
    ax.annotate('First turn: no history\n(large error expected)', 
               xy=(0, 5), xytext=(3, 12),
               fontsize=10, 
               arrowprops=dict(arrowstyle='->', color='gray'),
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_trend_by_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Chart 3: Error Type Summary
# =============================================================================

def plot_error_type_summary(anomaly_data, prediction_data, output_dir):
    """Generate summary chart for five error types."""
    type_stats = {}
    for r in anomaly_data:
        exc_type = r.get('exception_type', 'unknown')
        rate = r.get('exception_rate', 0)
        speedup = r.get('speedup', 1.0)
        
        if exc_type not in type_stats:
            type_stats[exc_type] = {'speedups': [], 'rates': []}
        type_stats[exc_type]['speedups'].append(speedup)
        type_stats[exc_type]['rates'].append(rate)
    
    pred_stats = {}
    for name, data in prediction_data.items():
        if isinstance(data, dict) and 'predictor_stats' in data:
            exc_type = data.get('exception_type', 'normal')
            if exc_type is None:
                exc_type = 'normal'
            pred_stats[exc_type] = {
                'avg_error_pct': data['predictor_stats'].get('avg_error_pct', 0),
            }
    
    fig = plt.figure(figsize=(16, 10))
    
    types = ['timeout', 'retry', 'early_return', 'hang', 'mixed']
    
    # 1. Speedup comparison
    ax1 = fig.add_subplot(2, 2, 1)
    avg_speedups = [sum(type_stats[t]['speedups'])/len(type_stats[t]['speedups']) 
                   if t in type_stats else 1.0 for t in types]
    min_speedups = [min(type_stats[t]['speedups']) 
                   if t in type_stats else 1.0 for t in types]
    colors = [COLORS.get(t, '#333333') for t in types]
    
    x = range(len(types))
    bars = ax1.bar(x, avg_speedups, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([TYPE_LABELS.get(t, t) for t in types], rotation=30, ha='right', fontsize=11)
    ax1.set_ylabel('Average Speedup', fontsize=12)
    ax1.set_title('Average TTL Speedup by Error Type', fontsize=13, fontweight='bold')
    ax1.set_ylim(0.5, 2.0)
    
    for bar, val in zip(bars, avg_speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Worst case
    ax2 = fig.add_subplot(2, 2, 2)
    bars2 = ax2.bar(x, min_speedups, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([TYPE_LABELS.get(t, t) for t in types], rotation=30, ha='right', fontsize=11)
    ax2.set_ylabel('Worst Case Speedup', fontsize=12)
    ax2.set_title('Worst Case TTL Speedup', fontsize=13, fontweight='bold')
    ax2.set_ylim(0.5, 1.5)
    
    for bar, val in zip(bars2, min_speedups):
        height = bar.get_height()
        color = 'red' if val < 1.0 else 'green'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
    
    # 3. Prediction error
    ax3 = fig.add_subplot(2, 2, 3)
    pred_types = ['normal', 'timeout', 'retry', 'hang', 'mixed']
    pred_errors = [pred_stats.get(t, {}).get('avg_error_pct', 0) for t in pred_types]
    pred_colors = [COLORS.get(t, '#333333') for t in pred_types]
    
    bars3 = ax3.bar(range(len(pred_types)), pred_errors, color=pred_colors, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(pred_types)))
    ax3.set_xticklabels([TYPE_LABELS.get(t, t) for t in pred_types], rotation=30, ha='right', fontsize=11)
    ax3.set_ylabel('Prediction Error (%)', fontsize=12)
    ax3.set_title('TTL Prediction Error by Scenario', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars3, pred_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Summary table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    table_data = [['Error Type', 'Avg Speedup', 'Worst Case', 'Prediction Error', 'Recommendation']]
    recommendations = {
        'timeout': 'Enable TTL (reduces competition)',
        'retry': 'Disable TTL (prediction fails)',
        'early_return': 'Enable TTL (faster = better)',
        'hang': 'Enable TTL (reduces competition)',
        'mixed': 'Disable TTL (prediction fails)',
    }
    
    for i, t in enumerate(types):
        avg_sp = avg_speedups[i]
        worst_sp = min_speedups[i]
        pred_err = pred_errors[pred_types.index(t)] if t in pred_types else 0
        rec = recommendations.get(t, 'Monitor')
        table_data.append([
            TYPE_LABELS.get(t, t),
            f'{avg_sp:.2f}x',
            f'{worst_sp:.2f}x',
            f'{pred_err:.1f}%',
            rec
        ])
    
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.18, 0.15, 0.15, 0.18, 0.34])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for j in range(5):
        table[(0, j)].set_facecolor('#4CAF50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    ax4.set_title('Summary Table', fontsize=13, fontweight='bold', y=0.95)
    
    plt.suptitle('Five Error Types Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_type_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Chart 4: Prediction Accuracy by Speedup Group
# =============================================================================

def plot_prediction_comparison_by_speedup(anomaly_data, prediction_data, output_dir):
    """Generate prediction accuracy comparison for scenarios with different speedup levels."""
    speedup_groups = {'high_speedup': [], 'medium_speedup': [], 'low_speedup': []}
    
    for r in anomaly_data:
        exc_type = r.get('exception_type', 'unknown')
        speedup = r.get('speedup', 1.0)
        
        if speedup > 1.5:
            speedup_groups['high_speedup'].append(exc_type)
        elif speedup >= 1.0:
            speedup_groups['medium_speedup'].append(exc_type)
        else:
            speedup_groups['low_speedup'].append(exc_type)
    
    scenario_pred_stats = {}
    for name, data in prediction_data.items():
        if isinstance(data, dict) and 'predictor_stats' in data:
            exc_type = data.get('exception_type', 'normal')
            if exc_type is None:
                exc_type = 'normal'
            scenario_pred_stats[exc_type] = {
                'avg_error_pct': data['predictor_stats'].get('avg_error_pct', 0),
            }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    group_labels = ['High Speedup (>1.5x)', 'Medium Speedup (1.0x-1.5x)', 'Low Speedup (<1.0x)']
    group_keys = ['high_speedup', 'medium_speedup', 'low_speedup']
    
    for idx, (key, label) in enumerate(zip(group_keys, group_labels)):
        ax = axes[idx]
        types = list(set(speedup_groups[key]))
        
        if not types:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(label, fontsize=12, fontweight='bold')
            continue
        
        errors = [scenario_pred_stats.get(t, {}).get('avg_error_pct', 0) for t in types]
        colors = [COLORS.get(t, '#333333') for t in types]
        labels = [TYPE_LABELS.get(t, t) for t in types]
        
        bars = ax.bar(range(len(errors)), errors, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(errors)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Prediction Error (%)', fontsize=11)
        
        for bar, val in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('TTL Prediction Error: Speedup Groups Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'prediction_by_speedup_group.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Chart 5: Predicted vs Actual by Speedup
# =============================================================================

def plot_predicted_vs_actual_by_speedup(anomaly_data, prediction_data, output_dir):
    """Generate scatter plot comparing predicted vs actual for different speedup levels."""
    speedup_by_type = {}
    for r in anomaly_data:
        exc_type = r.get('exception_type', 'unknown')
        speedup_by_type[exc_type] = r.get('speedup', 1.0)
    
    groups = {
        'high_speedup': {'predicted': [], 'actual': [], 'label': 'High Speedup (>1.5x)'},
        'medium_speedup': {'predicted': [], 'actual': [], 'label': 'Medium Speedup (1.0x-1.5x)'},
        'low_speedup': {'predicted': [], 'actual': [], 'label': 'Low Speedup (<1.0x)'},
    }
    
    for name, data in prediction_data.items():
        if isinstance(data, dict) and 'predictor_stats' in data:
            exc_type = data.get('exception_type', 'normal')
            if exc_type is None:
                exc_type = 'normal'
            
            speedup = speedup_by_type.get(exc_type, 1.0)
            pred = data['predictor_stats'].get('avg_predicted', 0)
            actual = data.get('avg_actual_duration', 0)
            
            if speedup > 1.5:
                groups['high_speedup']['predicted'].append(pred)
                groups['high_speedup']['actual'].append(actual)
            elif speedup >= 1.0:
                groups['medium_speedup']['predicted'].append(pred)
                groups['medium_speedup']['actual'].append(actual)
            else:
                groups['low_speedup']['predicted'].append(pred)
                groups['low_speedup']['actual'].append(actual)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    group_keys = ['high_speedup', 'medium_speedup', 'low_speedup']
    
    for idx, key in enumerate(group_keys):
        ax = axes[idx]
        group = groups[key]
        
        if not group['predicted']:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(group['label'], fontsize=12, fontweight='bold')
            continue
        
        ax.scatter(group['predicted'], group['actual'], alpha=0.7, s=150, 
                   c='#3498db', edgecolors='black', linewidth=2)
        
        max_val = max(max(group['predicted']), max(group['actual'])) + 1
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        errors = [abs(p - a) for p, a in zip(group['predicted'], group['actual'])]
        avg_error_pct = (sum(errors) / len(errors) / (sum(group['actual']) / len(group['actual']))) * 100 if group['actual'] else 0
        
        ax.set_xlabel('Predicted Duration (s)', fontsize=11)
        ax.set_ylabel('Actual Duration (s)', fontsize=11)
        ax.set_title(f"{group['label']}\nAvg Error: {avg_error_pct:.1f}%", fontsize=12, fontweight='bold')
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs Actual Duration by Speedup Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'predicted_vs_actual_by_speedup.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    results_dir = Path(__file__).parent / 'results'
    charts_dir = results_dir / 'charts_latest'
    ensure_dir(charts_dir)
    
    print("=" * 60)
    print("Generating Improved Report Charts")
    print("=" * 60)
    
    anomaly_path = results_dir / 'anomaly_test' / 'anomaly_test_latest.json'
    prediction_path = results_dir / 'prediction_test' / 'prediction_gap_analysis.json'
    
    if anomaly_path.exists():
        anomaly_data = load_json(str(anomaly_path))
        print(f"\nLoaded anomaly data: {len(anomaly_data)} records")
    
    if prediction_path.exists():
        prediction_data = load_json(str(prediction_path))
        print(f"Loaded prediction data: {len(prediction_data)} scenarios")
    
    print("\n[1] Generating Speedup vs Rate Line Chart (Fixed Axis)...")
    plot_speedup_vs_rate_line(anomaly_data, charts_dir)
    
    print("\n[2] Generating Error Type Summary...")
    plot_error_type_summary(anomaly_data, prediction_data, charts_dir)
    
    print("\n[3] Generating Prediction by Speedup Group...")
    plot_prediction_comparison_by_speedup(anomaly_data, prediction_data, charts_dir)
    
    print("\n[4] Generating Predicted vs Actual by Speedup...")
    plot_predicted_vs_actual_by_speedup(anomaly_data, prediction_data, charts_dir)
    
    print("\n[5] Generating Error by History Heatmap (Improved)...")
    plot_error_by_history_heatmap(anomaly_data, prediction_data, charts_dir)
    
    print("\n[6] Generating Error Trend by History (New)...")
    plot_error_trend_by_history(prediction_data, charts_dir)
    
    print("\n" + "=" * 60)
    print(f"Charts saved to: {charts_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
