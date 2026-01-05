#!/usr/bin/env python3
"""
Compare benchmark results from AMD and NVIDIA GPU runs.
Creates visualization and statistical comparison.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_results(results_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load all benchmark results and separate by platform."""
    results_path = Path(results_dir)
    
    cuda_results = []
    rocm_results = []
    
    for json_file in sorted(results_path.glob("benchmark_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if data['platform'] == 'cuda':
                cuda_results.append(data)
            elif data['platform'] == 'rocm':
                rocm_results.append(data)
        except Exception as e:
            print(f"⚠️  Error loading {json_file}: {e}")
    
    return cuda_results, rocm_results


def create_comparison_plot(cuda_data: Dict, rocm_data: Dict, output_file: str = "comparison.png"):
    """Create visual comparison of AMD vs NVIDIA performance."""
    
    # Check if we have per-core data
    has_per_core = (cuda_data['performance_metrics'].get('throughput_per_gpu_core', 0) > 0 and 
                    rocm_data['performance_metrics'].get('throughput_per_gpu_core', 0) > 0)
    
    if has_per_core:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle('AMD vs NVIDIA GPU Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing if we have 2x3 grid
    if has_per_core:
        axes = axes.flatten()
    
    # 1. Average Step Time Comparison
    ax1 = axes[0] if has_per_core else axes[0, 0]
    platforms = ['NVIDIA\n' + cuda_data['gpu_info']['device_name'], 
                 'AMD\n' + rocm_data['gpu_info']['device_name']]
    step_times = [
        cuda_data['performance_metrics']['avg_step_time_seconds'],
        rocm_data['performance_metrics']['avg_step_time_seconds']
    ]
    colors = ['#76B900', '#ED1C24']  # NVIDIA green, AMD red
    bars = ax1.bar(platforms, step_times, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Average Step Time (Lower is Better)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, step_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}s',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Throughput Comparison
    ax2 = axes[1] if has_per_core else axes[0, 1]
    throughputs = [
        cuda_data['performance_metrics']['throughput_steps_per_second'],
        rocm_data['performance_metrics']['throughput_steps_per_second']
    ]
    bars = ax2.bar(platforms, throughputs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Steps per Second', fontweight='bold')
    ax2.set_title('Throughput (Higher is Better)')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Throughput per Core Comparison (if available)
    if has_per_core:
        ax3 = axes[2]
        throughput_per_core = [
            cuda_data['performance_metrics']['throughput_per_gpu_core'],
            rocm_data['performance_metrics']['throughput_per_gpu_core']
        ]
        bars = ax3.bar(platforms, throughput_per_core, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Steps/Second/Core', fontweight='bold')
        ax3.set_title('Throughput per GPU Core (Higher is Better)')
        ax3.grid(axis='y', alpha=0.3)
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        for bar, value in zip(bars, throughput_per_core):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.6f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Memory Usage Comparison
    ax4 = axes[3] if has_per_core else axes[1, 0]
    if 'memory_metrics' in cuda_data and 'memory_metrics' in rocm_data:
        memory_data = {
            'Average': [
                cuda_data['memory_metrics']['avg_memory_allocated_gb'],
                rocm_data['memory_metrics']['avg_memory_allocated_gb']
            ],
            'Peak': [
                cuda_data['memory_metrics']['peak_memory_allocated_gb'],
                rocm_data['memory_metrics']['peak_memory_allocated_gb']
            ]
        }
        
        x = np.arange(len(platforms))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, memory_data['Average'], width, 
                       label='Average', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, memory_data['Peak'], width,
                       label='Peak', alpha=0.7, edgecolor='black')
        
        ax4.set_ylabel('Memory (GB)', fontweight='bold')
        ax4.set_title('GPU Memory Usage')
        ax4.set_xticks(x)
        ax4.set_xticklabels(platforms)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Memory data not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU Memory Usage')
    
    # 5. Step Time Distribution
    ax5 = axes[4] if has_per_core else axes[1, 1]
    cuda_times = cuda_data['raw_step_times'][1:]  # Skip warmup
    rocm_times = rocm_data['raw_step_times'][1:]  # Skip warmup
    
    ax5.plot(range(len(cuda_times)), cuda_times, 
            label=f"NVIDIA ({cuda_data['gpu_info']['device_name']})",
            color='#76B900', marker='o', markersize=4, linewidth=2, alpha=0.7)
    ax5.plot(range(len(rocm_times)), rocm_times,
            label=f"AMD ({rocm_data['gpu_info']['device_name']})",
            color='#ED1C24', marker='s', markersize=4, linewidth=2, alpha=0.7)
    
    ax5.set_xlabel('Step Number', fontweight='bold')
    ax5.set_ylabel('Time (seconds)', fontweight='bold')
    ax5.set_title('Step Time Over Training')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. GPU Core Count Comparison (if has_per_core)
    if has_per_core:
        ax6 = axes[5]
        core_counts = [
            cuda_data['gpu_info']['gpu_cores'],
            rocm_data['gpu_info']['gpu_cores']
        ]
        bars = ax6.bar(platforms, core_counts, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Number of Cores', fontweight='bold')
        ax6.set_title('GPU Core Count')
        ax6.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, core_counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {output_file}")
    
    return fig


def generate_comparison_report(cuda_data: Dict, rocm_data: Dict, output_file: str = "comparison_report.md"):
    """Generate detailed markdown comparison report."""
    
    cuda_time = cuda_data['performance_metrics']['avg_step_time_seconds']
    rocm_time = rocm_data['performance_metrics']['avg_step_time_seconds']
    
    faster_platform = "NVIDIA" if cuda_time < rocm_time else "AMD"
    speedup = max(cuda_time, rocm_time) / min(cuda_time, rocm_time)
    
    cuda_throughput = cuda_data['performance_metrics']['throughput_steps_per_second']
    rocm_throughput = rocm_data['performance_metrics']['throughput_steps_per_second']
    throughput_ratio = cuda_throughput / rocm_throughput
    
    report = f"""# AMD vs NVIDIA GPU Benchmark Comparison

## Executive Summary

**Winner**: {faster_platform} is **{speedup:.2f}x faster**

- NVIDIA Throughput: {cuda_throughput:.3f} steps/s
- AMD Throughput: {rocm_throughput:.3f} steps/s
- Throughput Ratio (NVIDIA/AMD): {throughput_ratio:.2f}x

---

## Hardware Configuration

### NVIDIA GPU
- **Device**: {cuda_data['gpu_info']['device_name']}
- **GPU Cores**: {cuda_data['gpu_info'].get('gpu_cores', 'N/A'):,}
- **Total Memory**: {cuda_data['gpu_info']['total_memory_gb']:.2f} GB
- **CUDA Version**: {cuda_data['gpu_info'].get('cuda_version', 'N/A')}
- **PyTorch Version**: {cuda_data['gpu_info']['pytorch_version']}

### AMD GPU
- **Device**: {rocm_data['gpu_info']['device_name']}
- **GPU Cores**: {rocm_data['gpu_info'].get('gpu_cores', 'N/A'):,}
- **Total Memory**: {rocm_data['gpu_info']['total_memory_gb']:.2f} GB
- **ROCm Version**: {rocm_data['gpu_info'].get('rocm_version', 'N/A')}
- **PyTorch Version**: {rocm_data['gpu_info']['pytorch_version']}

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Max Steps | {cuda_data['training_config']['max_steps']} |
| Global Batch Size | {cuda_data['training_config']['global_batch_size']} |
| Micro Batch Size | {cuda_data['training_config']['micro_batch_size']} |

---

## Performance Metrics

### Step Time

| Platform | Avg Time | Min Time | Max Time | Std Dev |
|----------|----------|----------|----------|---------|
| NVIDIA   | {cuda_time:.4f}s | {cuda_data['performance_metrics']['min_step_time_seconds']:.4f}s | {cuda_data['performance_metrics']['max_step_time_seconds']:.4f}s | {np.std(cuda_data['raw_step_times'][1:]):.4f}s |
| AMD      | {rocm_time:.4f}s | {rocm_data['performance_metrics']['min_step_time_seconds']:.4f}s | {rocm_data['performance_metrics']['max_step_time_seconds']:.4f}s | {np.std(rocm_data['raw_step_times'][1:]):.4f}s |

### Throughput

| Platform | Steps/Second | Steps/Second/Core |
|----------|--------------|-------------------|
| NVIDIA   | {cuda_throughput:.3f} | {cuda_data['performance_metrics'].get('throughput_per_gpu_core', 0):.6f} |
| AMD      | {rocm_throughput:.3f} | {rocm_data['performance_metrics'].get('throughput_per_gpu_core', 0):.6f} |

"""
    
    # Add memory metrics if available
    if 'memory_metrics' in cuda_data and 'memory_metrics' in rocm_data:
        report += f"""
### Memory Usage

| Platform | Avg Memory | Peak Memory |
|----------|------------|-------------|
| NVIDIA   | {cuda_data['memory_metrics']['avg_memory_allocated_gb']:.2f} GB | {cuda_data['memory_metrics']['peak_memory_allocated_gb']:.2f} GB |
| AMD      | {rocm_data['memory_metrics']['avg_memory_allocated_gb']:.2f} GB | {rocm_data['memory_metrics']['peak_memory_allocated_gb']:.2f} GB |
"""
    
    report += f"""
---

## Detailed Analysis

### Speed Comparison
- **Time Difference**: {abs(cuda_time - rocm_time):.4f} seconds per step
- **Speedup Factor**: {speedup:.2f}x ({faster_platform} faster)
- **Efficiency**: {min(cuda_time, rocm_time) / max(cuda_time, rocm_time) * 100:.1f}% (slower platform relative to faster)

### Per-Core Efficiency"""
    
    cuda_per_core = cuda_data['performance_metrics'].get('throughput_per_gpu_core', 0)
    rocm_per_core = rocm_data['performance_metrics'].get('throughput_per_gpu_core', 0)
    
    if cuda_per_core > 0 and rocm_per_core > 0:
        per_core_ratio = cuda_per_core / rocm_per_core
        more_efficient = "NVIDIA" if per_core_ratio > 1 else "AMD"
        report += f"""
- **NVIDIA per-core throughput**: {cuda_per_core:.6f} steps/s/core
- **AMD per-core throughput**: {rocm_per_core:.6f} steps/s/core
- **Per-core ratio (NVIDIA/AMD)**: {per_core_ratio:.2f}x
- **More efficient per core**: {more_efficient}
"""
    else:
        report += """
- Per-core metrics not available
"""
    
    report += f"""

### Stability
- **NVIDIA Variance**: {np.var(cuda_data['raw_step_times'][1:]):.6f}
- **AMD Variance**: {np.var(rocm_data['raw_step_times'][1:]):.6f}

---

## Timestamps
- **NVIDIA Run**: {cuda_data['timestamp']}
- **AMD Run**: {rocm_data['timestamp']}

---

*Generated by benchmark_utils.py*
"""
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Comparison report saved to: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Compare AMD and NVIDIA GPU benchmark results')
    parser.add_argument('--results-dir', default='./benchmark_results',
                       help='Directory containing benchmark JSON files')
    parser.add_argument('--output-plot', default='comparison_plot.png',
                       help='Output file for comparison plot')
    parser.add_argument('--output-report', default='comparison_report.md',
                       help='Output file for comparison report')
    
    args = parser.parse_args()
    
    print("Loading benchmark results...")
    cuda_results, rocm_results = load_benchmark_results(args.results_dir)
    
    if not cuda_results:
        print("❌ No NVIDIA (CUDA) benchmark results found!")
        print(f"   Run your training script on an NVIDIA GPU first.")
        return
    
    if not rocm_results:
        print("❌ No AMD (ROCm) benchmark results found!")
        print(f"   Run your training script on an AMD GPU first.")
        return
    
    # Use most recent results
    cuda_data = sorted(cuda_results, key=lambda x: x['timestamp'])[-1]
    rocm_data = sorted(rocm_results, key=lambda x: x['timestamp'])[-1]
    
    print(f"\n📊 Comparing:")
    print(f"  NVIDIA: {cuda_data['gpu_info']['device_name']} ({cuda_data['timestamp']})")
    print(f"  AMD:    {rocm_data['gpu_info']['device_name']} ({rocm_data['timestamp']})")
    print()
    
    # Generate comparison plot
    try:
        create_comparison_plot(cuda_data, rocm_data, args.output_plot)
    except ImportError:
        print("⚠️  matplotlib not available, skipping plot generation")
        print("   Install with: pip install matplotlib")
    
    # Generate report
    generate_comparison_report(cuda_data, rocm_data, args.output_report)
    
    # Print summary
    cuda_time = cuda_data['performance_metrics']['avg_step_time_seconds']
    rocm_time = rocm_data['performance_metrics']['avg_step_time_seconds']
    faster = "NVIDIA" if cuda_time < rocm_time else "AMD"
    speedup = max(cuda_time, rocm_time) / min(cuda_time, rocm_time)
    
    print(f"\n{'='*60}")
    print(f"🏆 RESULT: {faster} is {speedup:.2f}x FASTER")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

