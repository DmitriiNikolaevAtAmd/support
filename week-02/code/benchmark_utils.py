"""
Unified benchmarking utilities for AMD vs NVIDIA GPU comparison.
Works on both ROCm and CUDA platforms.
"""
import time
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
from lightning.pytorch.callbacks import Callback


class BenchmarkCallback(Callback):
    """Callback to collect platform-agnostic performance metrics."""
    
    def __init__(self, output_dir: str = "./benchmark_results", platform: str = "auto"):
        """
        Args:
            output_dir: Directory to save benchmark results
            platform: 'cuda', 'rocm', or 'auto' for auto-detection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect platform
        if platform == "auto":
            if torch.cuda.is_available():
                self.platform = "cuda" if "cuda" in torch.version.cuda else "rocm"
            else:
                self.platform = "cpu"
        else:
            self.platform = platform
        
        # Metrics storage
        self.step_times = []
        self.memory_allocated = []
        self.memory_reserved = []
        self.step_start_time = None
        self.train_start_time = None
        self.gpu_info = {}
        
    def on_train_start(self, trainer, pl_module):
        """Collect GPU information at training start."""
        self.train_start_time = time.time()
        
        if torch.cuda.is_available():
            self.gpu_info = {
                "platform": self.platform,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "pytorch_version": torch.__version__,
            }
            
            # AMD-specific info
            if self.platform == "rocm":
                self.gpu_info["rocm_version"] = torch.version.hip
            else:
                self.gpu_info["cuda_version"] = torch.version.cuda
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK START - Platform: {self.platform.upper()}")
        print(f"{'='*60}")
        for key, value in self.gpu_info.items():
            print(f"{key}: {value}")
        print(f"{'='*60}\n")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Mark start of training step."""
        self.step_start_time = time.time()
        
        # Clear cache for consistent measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect metrics after each training step."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        step_time = time.time() - self.step_start_time
        self.step_times.append(step_time)
        
        # Collect memory stats
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1e9    # GB
            self.memory_allocated.append(mem_allocated)
            self.memory_reserved.append(mem_reserved)
        
        # Log every 10 steps
        if batch_idx > 0 and batch_idx % 10 == 0:
            recent_times = self.step_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            
            if torch.cuda.is_available():
                avg_mem = sum(self.memory_allocated[-10:]) / len(self.memory_allocated[-10:])
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s | "
                      f"Memory: {avg_mem:.2f}GB")
            else:
                print(f"[{self.platform.upper()}] Step {batch_idx:3d} | "
                      f"Time: {step_time:.3f}s | Avg: {avg_time:.3f}s")
    
    def on_train_end(self, trainer, pl_module):
        """Save benchmark results."""
        total_time = time.time() - self.train_start_time
        
        # Calculate statistics
        if len(self.step_times) > 1:
            # Skip first step (warmup)
            step_times_no_warmup = self.step_times[1:]
            
            results = {
                "platform": self.platform,
                "gpu_info": self.gpu_info,
                "timestamp": datetime.now().isoformat(),
                "training_config": {
                    "max_steps": trainer.max_steps,
                    "global_batch_size": getattr(trainer.datamodule, 'global_batch_size', 'N/A'),
                    "micro_batch_size": getattr(trainer.datamodule, 'micro_batch_size', 'N/A'),
                },
                "performance_metrics": {
                    "total_steps": len(self.step_times),
                    "total_time_seconds": total_time,
                    "avg_step_time_seconds": sum(step_times_no_warmup) / len(step_times_no_warmup),
                    "min_step_time_seconds": min(step_times_no_warmup),
                    "max_step_time_seconds": max(step_times_no_warmup),
                    "throughput_steps_per_second": len(step_times_no_warmup) / sum(step_times_no_warmup),
                },
                "raw_step_times": self.step_times,
            }
            
            if self.memory_allocated:
                mem_no_warmup = self.memory_allocated[1:]
                results["memory_metrics"] = {
                    "avg_memory_allocated_gb": sum(mem_no_warmup) / len(mem_no_warmup),
                    "peak_memory_allocated_gb": max(mem_no_warmup),
                    "avg_memory_reserved_gb": sum(self.memory_reserved[1:]) / len(self.memory_reserved[1:]),
                    "peak_memory_reserved_gb": max(self.memory_reserved[1:]),
                }
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{self.platform}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"BENCHMARK COMPLETE - Platform: {self.platform.upper()}")
            print(f"{'='*60}")
            print(f"Total Steps: {results['performance_metrics']['total_steps']}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Avg Step Time: {results['performance_metrics']['avg_step_time_seconds']:.3f}s")
            print(f"Throughput: {results['performance_metrics']['throughput_steps_per_second']:.3f} steps/s")
            if 'memory_metrics' in results:
                print(f"Avg Memory: {results['memory_metrics']['avg_memory_allocated_gb']:.2f}GB")
                print(f"Peak Memory: {results['memory_metrics']['peak_memory_allocated_gb']:.2f}GB")
            print(f"\nResults saved to: {filepath}")
            print(f"{'='*60}\n")


def compare_benchmarks(results_dir: str = "./benchmark_results") -> Dict:
    """
    Compare benchmark results from AMD and NVIDIA runs.
    
    Args:
        results_dir: Directory containing benchmark JSON files
        
    Returns:
        Dictionary with comparison results
    """
    results_path = Path(results_dir)
    
    # Find latest results for each platform
    cuda_results = []
    rocm_results = []
    
    for json_file in results_path.glob("benchmark_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        if data['platform'] == 'cuda':
            cuda_results.append(data)
        elif data['platform'] == 'rocm':
            rocm_results.append(data)
    
    if not cuda_results or not rocm_results:
        print("⚠️  Need results from both CUDA and ROCm platforms for comparison")
        return {}
    
    # Use most recent results
    cuda = sorted(cuda_results, key=lambda x: x['timestamp'])[-1]
    rocm = sorted(rocm_results, key=lambda x: x['timestamp'])[-1]
    
    comparison = {
        "cuda": {
            "device": cuda['gpu_info']['device_name'],
            "avg_step_time": cuda['performance_metrics']['avg_step_time_seconds'],
            "throughput": cuda['performance_metrics']['throughput_steps_per_second'],
            "peak_memory": cuda.get('memory_metrics', {}).get('peak_memory_allocated_gb', 'N/A'),
        },
        "rocm": {
            "device": rocm['gpu_info']['device_name'],
            "avg_step_time": rocm['performance_metrics']['avg_step_time_seconds'],
            "throughput": rocm['performance_metrics']['throughput_steps_per_second'],
            "peak_memory": rocm.get('memory_metrics', {}).get('peak_memory_allocated_gb', 'N/A'),
        }
    }
    
    # Calculate speedup
    cuda_time = cuda['performance_metrics']['avg_step_time_seconds']
    rocm_time = rocm['performance_metrics']['avg_step_time_seconds']
    
    comparison["speedup"] = {
        "faster_platform": "CUDA" if cuda_time < rocm_time else "ROCm",
        "speedup_factor": max(cuda_time, rocm_time) / min(cuda_time, rocm_time),
        "time_difference_seconds": abs(cuda_time - rocm_time),
        "throughput_ratio": comparison["cuda"]["throughput"] / comparison["rocm"]["throughput"],
    }
    
    # Print comparison
    print(f"\n{'='*80}")
    print("AMD vs NVIDIA GPU COMPARISON")
    print(f"{'='*80}")
    print(f"\nNVIDIA GPU ({comparison['cuda']['device']}):")
    print(f"  Avg Step Time: {comparison['cuda']['avg_step_time']:.4f}s")
    print(f"  Throughput:    {comparison['cuda']['throughput']:.3f} steps/s")
    print(f"  Peak Memory:   {comparison['cuda']['peak_memory']}GB")
    
    print(f"\nAMD GPU ({comparison['rocm']['device']}):")
    print(f"  Avg Step Time: {comparison['rocm']['avg_step_time']:.4f}s")
    print(f"  Throughput:    {comparison['rocm']['throughput']:.3f} steps/s")
    print(f"  Peak Memory:   {comparison['rocm']['peak_memory']}GB")
    
    print(f"\nResult:")
    print(f"  {comparison['speedup']['faster_platform']} is {comparison['speedup']['speedup_factor']:.2f}x faster")
    print(f"  Throughput ratio (NVIDIA/AMD): {comparison['speedup']['throughput_ratio']:.2f}x")
    print(f"{'='*80}\n")
    
    return comparison


if __name__ == "__main__":
    # Run comparison if executed directly
    compare_benchmarks()

