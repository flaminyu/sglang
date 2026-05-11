#!/usr/bin/env python3
"""
Generate Predictive TTL Oracle comparison charts.

This script generates charts comparing:
1. Baseline (no TTL) - reference line at 1.0x
2. Current TTL (adaptive TTL) - solid lines
3. Predictive Oracle TTL (with perfect anomaly knowledge) - dashed lines

Charts generated:
1. speedup_vs_rate_by_type.png - Line chart showing all 3 series
2. speedup_comparison_with_oracle.png - Faceted chart by anomaly type
3. improvement_summary.png - Bar chart showing improvement over current TTL
"""

import sys
sys.path.insert(0, '.')

import os
import json
from pathlib import Path
from typing import Dict, List, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

# Color schemes (same as generate_report_charts.py)
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


# ============================================================================
# Data Loading
# ============================================================================

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dir(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def load_comparison_data(data_dir: Path) -> List[Dict]:
    """Load the full comparison data from generate_predictive_data.py output."""
    data_file = data_dir / 'predictive_comparison' / 'full_comparison_raw.json'
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = load_json(str(data_file))
    return data.get('results', [])


# ============================================================================
# Chart 1: Speedup vs Rate - All Types with Oracle
# ============================================================================

def plot_speedup_vs_rate_with_oracle(data: List[Dict], output_dir: Path):
    """Generate line chart showing speedup vs rate for all anomaly types, with Current TTL and Oracle TTL."""
    
    # Group by anomaly type
    by_type = {}
    for r in data:
        at = r['anomaly_type']
        if at not in by_type:
            by_type[at] = []
        by_type[at].append({
            'rate': r['anomaly_rate'],
            'current_speedup': r['current_ttl_speedup'],
            'oracle_speedup': r['oracle_ttl_speedup'],
            'baseline_time': r['baseline_time'],
            'current_time': r['current_ttl_time'],
            'oracle_time': r['oracle_ttl_time'],
        })
    
    # Sort each type by rate
    for at in by_type:
        by_type[at] = sorted(by_type[at], key=lambda x: x['rate'])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for at in sorted(by_type.keys()):
        results = by_type[at]
        rates = [d['rate'] * 100 for d in results]  # Convert to percentage
        current_speedups = [d['current_speedup'] for d in results]
        oracle_speedups = [d['oracle_speedup'] for d in results]
        color = COLORS.get(at, '#333333')
        label = TYPE_LABELS.get(at, at)
        
        # Current TTL - solid line with circle markers
        ax.plot(rates, current_speedups, 'o-', color=color, linewidth=2.5, 
                markersize=8, label=f'{label} TTL')
        
        # Oracle TTL - dashed line with square markers
        ax.plot(rates, oracle_speedups, 's--', color=color, linewidth=2.5, 
                markersize=8, label=f'{label} Oracle', alpha=0.8)
    
    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    
    # Shaded zones
    ax.fill_between([0, 55], [1, 1], [0.5, 0.5], alpha=0.1, color='red', label='Harmful Zone')
    ax.fill_between([0, 55], [1, 1], [2.5, 2.5], alpha=0.1, color='green', label='Beneficial Zone')
    
    ax.set_xlabel('Anomaly Rate (%)', fontsize=14)
    ax.set_ylabel('Speedup (vs Baseline)', fontsize=14)
    ax.set_title('TTL Speedup vs Anomaly Rate: Current TTL vs Predictive Oracle', 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(-2, 55)
    ax.set_ylim(0.5, 2.5)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for at in sorted(by_type.keys()):
        results = by_type[at]
        for d in results:
            if d['oracle_speedup'] > 2.0 or d['oracle_speedup'] < 0.7:
                ax.annotate(f'{d["oracle_speedup"]:.2f}x',
                           xy=(d['rate'] * 100, d['oracle_speedup']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', alpha=0.8)
    
    plt.tight_layout()
    output_path = output_dir / 'speedup_vs_rate_by_type.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# ============================================================================
# Chart 2: Faceted Comparison by Type
# ============================================================================

def plot_faceted_comparison(data: List[Dict], output_dir: Path):
    """Generate faceted chart showing comparison for each anomaly type."""
    
    # Group by anomaly type
    by_type = {}
    for r in data:
        at = r['anomaly_type']
        if at not in by_type:
            by_type[at] = []
        by_type[at].append(r)
    
    # Sort each type by rate
    for at in by_type:
        by_type[at] = sorted(by_type[at], key=lambda x: x['anomaly_rate'])
    
    types = sorted(by_type.keys())
    n_types = len(types)
    
    # Create subplot grid
    n_cols = 3
    n_rows = (n_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, at in enumerate(types):
        ax = axes[idx]
        results = by_type[at]
        rates = [r['anomaly_rate'] * 100 for r in results]
        current_speedups = [r['current_ttl_speedup'] for r in results]
        oracle_speedups = [r['oracle_ttl_speedup'] for r in results]
        
        color = COLORS.get(at, '#333333')
        label = TYPE_LABELS.get(at, at)
        
        # Plot both lines
        ax.plot(rates, current_speedups, 'o-', color=color, linewidth=2, 
                markersize=7, label='Current TTL')
        ax.plot(rates, oracle_speedups, 's--', color=color, linewidth=2, 
                markersize=7, label='Oracle TTL', alpha=0.8)
        
        # Reference line
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Anomaly Rate (%)', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title(f'{label}', fontsize=13, fontweight='bold')
        ax.set_xlim(-2, 55)
        ax.set_ylim(0.5, 2.5)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
        
        # Find and annotate best points
        best_current = max(results, key=lambda x: x['current_ttl_speedup'])
        best_oracle = max(results, key=lambda x: x['oracle_ttl_speedup'])
        
        ax.annotate(f'Best TTL: {best_current["current_ttl_speedup"]:.2f}x\n'
                   f'Best Oracle: {best_oracle["oracle_ttl_speedup"]:.2f}x',
                   xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=8, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused axes
    for idx in range(len(types), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Speedup by Anomaly Type: Current TTL vs Predictive Oracle', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'speedup_comparison_with_oracle.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# ============================================================================
# Chart 3: Improvement Summary Bar Chart
# ============================================================================

def plot_improvement_summary(data: List[Dict], output_dir: Path):
    """Generate bar chart showing Oracle improvement over Current TTL."""
    
    # Group by anomaly type and find best rate for each
    by_type = {}
    for r in data:
        at = r['anomaly_type']
        if at not in by_type:
            by_type[at] = []
        by_type[at].append(r)
    
    # Calculate improvement for each rate
    all_improvements = []
    for at, results in by_type.items():
        for r in results:
            improvement = r['oracle_ttl_speedup'] - r['current_ttl_speedup']
            all_improvements.append({
                'type': at,
                'rate': r['anomaly_rate'],
                'improvement': improvement,
                'current_speedup': r['current_ttl_speedup'],
                'oracle_speedup': r['oracle_ttl_speedup'],
            })
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Best improvement by type at optimal rate
    ax1 = axes[0]
    best_improvements = []
    for at in sorted(by_type.keys()):
        results = by_type[at]
        best = max(results, key=lambda x: x['oracle_ttl_speedup'] - x['current_ttl_speedup'])
        best_improvements.append({
            'type': at,
            'improvement': best['oracle_ttl_speedup'] - best['current_ttl_speedup'],
            'current': best['current_ttl_speedup'],
            'oracle': best['oracle_ttl_speedup'],
            'rate': best['anomaly_rate'],
        })
    
    types = [TYPE_LABELS.get(b['type'], b['type']) for b in best_improvements]
    improvements = [b['improvement'] for b in best_improvements]
    colors = [COLORS.get(b['type'], '#333333') for b in best_improvements]
    
    x = range(len(types))
    bars = ax1.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black')
    
    # Color bars based on improvement
    for bar, imp in zip(bars, improvements):
        if imp > 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, rotation=30, ha='right', fontsize=11)
    ax1.set_ylabel('Oracle Improvement (speedup delta)', fontsize=12)
    ax1.set_title('Best Oracle Improvement by Anomaly Type\n(at optimal anomaly rate)', 
                  fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, imp, b in zip(bars, improvements, best_improvements):
        height = bar.get_height()
        color = 'green' if imp > 0 else 'red'
        sign = '+' if imp > 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{sign}{imp:.3f}x\n@{b["rate"]*100:.0f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
    
    # Plot 2: Speedup comparison at best rate for each type
    ax2 = axes[1]
    current_speedups = [b['current'] for b in best_improvements]
    oracle_speedups = [b['oracle'] for b in best_improvements]
    
    x = np.arange(len(types))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, current_speedups, width, label='Current TTL', 
                     color=[COLORS.get(b['type'], '#333333') for b in best_improvements], 
                     alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, oracle_speedups, width, label='Oracle TTL',
                     color=[COLORS.get(b['type'], '#333333') for b in best_improvements], 
                     alpha=0.5, edgecolor='black', hatch='//')
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types, rotation=30, ha='right', fontsize=11)
    ax2.set_ylabel('Speedup (vs Baseline)', fontsize=12)
    ax2.set_title('Current TTL vs Oracle TTL Speedup\n(at best rate for each type)', 
                  fontsize=13, fontweight='bold')
    ax2.set_ylim(0.5, 2.0)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Predictive Oracle TTL: Improvement Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = output_dir / 'improvement_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# ============================================================================
# Chart 4: Rate-specific Improvement Heatmap
# ============================================================================

def plot_improvement_heatmap(data: List[Dict], output_dir: Path):
    """Generate heatmap showing Oracle improvement at different rates for different types."""
    
    # Get unique types and rates
    types = sorted(set(r['anomaly_type'] for r in data))
    rates = sorted(set(r['anomaly_rate'] for r in data))
    
    # Build improvement matrix
    matrix = np.zeros((len(types), len(rates)))
    for r in data:
        type_idx = types.index(r['anomaly_type'])
        rate_idx = rates.index(r['anomaly_rate'])
        matrix[type_idx, rate_idx] = r['oracle_ttl_speedup'] - r['current_ttl_speedup']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=0.5)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Oracle Improvement (speedup delta)', rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(len(rates)))
    ax.set_yticks(np.arange(len(types)))
    ax.set_xticklabels([f'{r*100:.0f}%' for r in rates], fontsize=11)
    ax.set_yticklabels([TYPE_LABELS.get(t, t) for t in types], fontsize=11)
    
    ax.set_xlabel('Anomaly Rate', fontsize=14)
    ax.set_ylabel('Anomaly Type', fontsize=14)
    ax.set_title('Oracle TTL Improvement over Current TTL\n(Green = Oracle Better, Red = Current Better)', 
                 fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(types)):
        for j in range(len(rates)):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.25 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', 
                   fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    results_dir = Path(__file__).parent / 'results'
    output_dir = results_dir / 'predictive_comparison'
    ensure_dir(output_dir)
    
    print("=" * 70)
    print("Generating Predictive TTL Oracle Charts")
    print("=" * 70)
    
    # Load data
    try:
        data = load_comparison_data(results_dir)
        print(f"\nLoaded {len(data)} experiment results")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run generate_predictive_data.py first to generate the comparison data.")
        return
    
    print("\n[1] Generating Speedup vs Rate chart (all types with Oracle)...")
    plot_speedup_vs_rate_with_oracle(data, output_dir)
    
    print("\n[2] Generating faceted comparison chart...")
    plot_faceted_comparison(data, output_dir)
    
    print("\n[3] Generating improvement summary chart...")
    plot_improvement_summary(data, output_dir)
    
    print("\n[4] Generating improvement heatmap...")
    plot_improvement_heatmap(data, output_dir)
    
    print("\n" + "=" * 70)
    print(f"Charts saved to: {output_dir}/")
    print("=" * 70)
    
    print("\nGenerated charts:")
    print("  - speedup_vs_rate_by_type.png (main comparison)")
    print("  - speedup_comparison_with_oracle.png (faceted by type)")
    print("  - improvement_summary.png (bar chart)")
    print("  - improvement_heatmap.png (heatmap)")


if __name__ == '__main__':
    main()
