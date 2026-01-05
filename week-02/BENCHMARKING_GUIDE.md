# AMD vs NVIDIA GPU Benchmarking System

Complete guide for profiling and comparing AMD and NVIDIA GPUs using NeMo training workloads.

## 📋 Overview

This benchmarking system provides a **unified, platform-agnostic** way to compare GPU performance across AMD and NVIDIA hardware. It automatically detects the platform (CUDA/ROCm), collects consistent metrics, and generates detailed comparison reports.

### Key Features

✅ **Platform Agnostic** - Works on both CUDA and ROCm without code changes  
✅ **Automated Profiling** - Collects timing, memory, and throughput metrics  
✅ **Fair Comparison** - Identical configurations across both platforms  
✅ **Visual Reports** - Generates comparison charts and detailed reports  
✅ **Multiple Models** - Supports Llama, Qwen, and Mistral benchmarks  

## 🚀 Quick Start

### 1. Run on NVIDIA GPU

```bash
cd week-02/code
./run_benchmark.sh llama
```

### 2. Run on AMD GPU

```bash
cd week-02/code
./run_benchmark.sh llama
```

### 3. Compare Results

```bash
python3 compare_results.py
```

**That's it!** You'll get:
- `comparison_plot.png` - Visual comparison
- `comparison_report.md` - Detailed analysis
- Console output showing which platform is faster

## 📁 File Structure

```
week-02/
├── code/
│   ├── benchmark_utils.py          # Core benchmarking framework ⭐
│   ├── compare_results.py          # Comparison and visualization ⭐
│   ├── analyze_existing_logs.py    # Analyze old profiling data
│   │
│   ├── pretrain_llama.py           # Llama 3.1 8B training (updated) ⭐
│   ├── pretrain_qwen.py            # Qwen 2.5 7B training (updated) ⭐
│   ├── pretrain_mistral.py         # Mistral 7B training (updated) ⭐
│   │
│   ├── run_benchmark.sh            # Automated benchmark runner ⭐
│   ├── QUICK_START.md              # Quick reference guide
│   ├── BENCHMARK_README.md         # Detailed documentation
│   ├── requirements.txt            # Python dependencies
│   │
│   └── benchmark_results/          # Generated results directory
│       ├── benchmark_cuda_*.json   # NVIDIA results
│       ├── benchmark_rocm_*.json   # AMD results
│       ├── comparison_plot.png     # Visual comparison
│       └── comparison_report.md    # Detailed report
│
├── amd-logs/                       # Your existing AMD profiling data
└── nvi-logs/                       # Your existing NVIDIA profiling data
```

⭐ = New or updated file

## 🔧 What Changed

### Updated Training Scripts

All three training scripts now include the benchmark callback:

**Before:**
```python
from nemo.collections import llm
import nemo_run as run

def run_pretrain():
    recipe = llm.llama31_8b.pretrain_recipe(...)
    # ... configuration ...
    run.run(recipe, direct=True)
```

**After:**
```python
from nemo.collections import llm
import nemo_run as run
from benchmark_utils import BenchmarkCallback  # NEW

def run_pretrain():
    recipe = llm.llama31_8b.pretrain_recipe(...)
    # ... configuration ...
    
    # NEW: Add benchmark callback
    benchmark_callback = BenchmarkCallback(
        output_dir="./benchmark_results",
        platform="auto"  # Auto-detects CUDA or ROCm
    )
    if recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []
    recipe.trainer.callbacks.append(benchmark_callback)
    
    run.run(recipe, direct=True)
```

This is **non-invasive** - it doesn't change your training logic, just adds monitoring.

## 📊 What Gets Measured

### Performance Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **Avg Step Time** | Time per training step | Lower ⬇️ |
| **Throughput** | Steps per second | Higher ⬆️ |
| **Min/Max Time** | Best and worst steps | - |
| **Variance** | Consistency | Lower ⬇️ |

### Memory Metrics

| Metric | Description |
|--------|-------------|
| **Avg Memory** | Typical GPU memory usage |
| **Peak Memory** | Maximum memory used |
| **Reserved Memory** | Total allocated by PyTorch |

### System Information

- GPU model and specifications
- CUDA/ROCm version
- PyTorch version
- Training configuration (batch size, parallelism, etc.)

## 🎯 Usage Scenarios

### Scenario 1: Basic Comparison

Compare AMD vs NVIDIA with default settings:

```bash
# On NVIDIA system
./run_benchmark.sh llama

# On AMD system
./run_benchmark.sh llama

# Compare
python3 compare_results.py
```

### Scenario 2: Multiple Runs

Run multiple times for statistical significance:

```bash
# Run 5 times on each platform
./run_benchmark.sh llama 5
```

### Scenario 3: Compare Different Models

Compare all three models on both platforms:

```bash
# On each platform
./run_benchmark.sh llama
./run_benchmark.sh qwen
./run_benchmark.sh mistral
```

### Scenario 4: Analyze Existing Logs

Analyze your existing profiling data:

```bash
python3 analyze_existing_logs.py
```

This will show:
- AMD profiling reports (Excel files)
- NVIDIA TensorBoard logs
- New benchmark results

## 📈 Example Output

### During Training

```
============================================================
BENCHMARK START - Platform: CUDA
============================================================
device_count: 8
device_name: NVIDIA A100-SXM4-80GB
total_memory_gb: 80.0
cuda_version: 12.1
============================================================

[CUDA] Step  10 | Time: 1.234s | Avg: 1.245s | Memory: 45.67GB
[CUDA] Step  20 | Time: 1.238s | Avg: 1.242s | Memory: 45.68GB
[CUDA] Step  30 | Time: 1.241s | Avg: 1.243s | Memory: 45.69GB

============================================================
BENCHMARK COMPLETE - Platform: CUDA
============================================================
Total Steps: 10
Total Time: 12.45s
Avg Step Time: 1.245s
Throughput: 0.803 steps/s
Avg Memory: 45.67GB
Peak Memory: 45.89GB

Results saved to: benchmark_results/benchmark_cuda_20260105_143022.json
============================================================
```

### Comparison Results

```
============================================================
AMD vs NVIDIA GPU COMPARISON
============================================================

NVIDIA GPU (NVIDIA A100-SXM4-80GB):
  Avg Step Time: 1.245s
  Throughput:    0.803 steps/s
  Peak Memory:   45.89GB

AMD GPU (AMD Instinct MI250X):
  Avg Step Time: 1.567s
  Throughput:    0.638 steps/s
  Peak Memory:   48.23GB

Result:
  NVIDIA is 1.26x faster
  Throughput ratio (NVIDIA/AMD): 1.26x
============================================================
```

## 🔍 Understanding Results

### Speedup Factor

- **1.0x** = Same performance
- **1.5x** = 50% faster
- **2.0x** = 2x (double) the speed

### Throughput Ratio

- **> 1.0** = NVIDIA faster
- **< 1.0** = AMD faster
- **= 1.0** = Same speed

### Memory Usage

Higher memory usage may indicate:
- Different memory allocation strategies
- Different precision handling
- Different kernel implementations

## ⚙️ Configuration

### Training Configuration (Identical on Both Platforms)

```python
# Llama 3.1 8B
recipe.trainer.strategy.tensor_model_parallel_size = 4
recipe.trainer.strategy.pipeline_model_parallel_size = 1
recipe.data.micro_batch_size = 1
recipe.data.global_batch_size = 8
recipe.trainer.max_steps = 10
recipe.model.config.fp8 = "hybrid"
```

These settings ensure **fair comparison** between platforms.

### Customizing Benchmark

Edit `benchmark_utils.py` to customize:

```python
class BenchmarkCallback(Callback):
    def __init__(self, 
                 output_dir: str = "./benchmark_results",
                 platform: str = "auto"):
        # Customize here
```

## 🐛 Troubleshooting

### No GPU Detected

```bash
# Check NVIDIA
nvidia-smi

# Check AMD
rocm-smi

# Check PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Missing Dependencies

```bash
pip install matplotlib numpy
```

### Out of Memory

Reduce batch size in training scripts:

```python
recipe.data.global_batch_size = 4  # Reduce from 8
```

### Missing Results

Check the results directory:

```bash
ls -la benchmark_results/
```

Should contain both:
- `benchmark_cuda_*.json` (NVIDIA)
- `benchmark_rocm_*.json` (AMD)

## 📚 Documentation

- **[QUICK_START.md](code/QUICK_START.md)** - TL;DR guide
- **[BENCHMARK_README.md](code/BENCHMARK_README.md)** - Detailed documentation
- **[requirements.txt](code/requirements.txt)** - Python dependencies

## 🎓 Best Practices

1. **Multiple Runs** - Run 3-5 times and average results
2. **Idle System** - Close other GPU applications
3. **Same Configuration** - Use identical settings on both platforms
4. **Document Everything** - Save system info and versions
5. **Warmup** - First step is automatically excluded
6. **Cool Down** - Wait between runs (script does this automatically)

## 🔬 Advanced Profiling

### PyTorch Profiler

For detailed kernel-level profiling:

```python
from lightning.pytorch.profilers import PyTorchProfiler

recipe.trainer.profiler = PyTorchProfiler(
    dirpath="./profile_logs",
    export_to_chrome=True,
    profile_memory=True,
)
```

View in Chrome: `chrome://tracing`

### NVIDIA Nsight Systems

```bash
nsys profile --trace=cuda,nvtx python3 pretrain_llama.py
```

### AMD ROCProfiler

```bash
rocprof --stats --timestamp on python3 pretrain_llama.py
```

## 🤝 Contributing

To add new models or metrics:

1. **New Model**: Copy existing training script and modify
2. **New Metric**: Edit `BenchmarkCallback` in `benchmark_utils.py`
3. **New Visualization**: Edit `compare_results.py`

## 📝 Notes

- Benchmarks automatically exclude the first step (warmup)
- All metrics are averaged over completed steps
- Memory metrics are collected after CUDA/ROCm synchronization
- Results are saved in JSON format for easy parsing

## 🎯 Next Steps

1. ✅ Run benchmarks on both platforms
2. ✅ Generate comparison reports
3. 📊 Analyze results
4. 📝 Document findings
5. 🔧 Optimize slower platform (if needed)

---

**Happy Benchmarking! 🚀**

*For questions or issues, check the troubleshooting section or review the detailed documentation.*

