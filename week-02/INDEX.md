# Week 02 - GPU Benchmarking System Index

## 📍 You Are Here

```
support/
├── PROFILING_SUMMARY.md          ← 📖 Complete overview (start here)
└── week-02/
    ├── INDEX.md                  ← 📍 You are here
    ├── BENCHMARKING_GUIDE.md     ← 📚 High-level guide
    │
    ├── code/                     ← 💻 All code and scripts
    │   ├── README.md             ← Quick index
    │   ├── QUICK_START.md        ← 5-min guide
    │   ├── WORKFLOW.md           ← Visual diagrams
    │   ├── BENCHMARK_README.md   ← Complete reference
    │   │
    │   ├── benchmark_utils.py    ← ⭐ Core framework
    │   ├── compare_results.py    ← ⭐ Comparison tool
    │   ├── analyze_existing_logs.py
    │   ├── run_benchmark.sh      ← ⭐ Automation script
    │   │
    │   ├── pretrain_llama.py     ← Updated with benchmarking
    │   ├── pretrain_qwen.py      ← Updated with benchmarking
    │   ├── pretrain_mistral.py   ← Updated with benchmarking
    │   │
    │   ├── requirements.txt
    │   └── benchmark_results/    ← Generated results go here
    │
    ├── amd-logs/                 ← Your existing AMD profiling
    │   ├── llama/                (8 Excel files)
    │   └── qwen/                 (8 Excel files)
    │
    └── nvi-logs/                 ← Your existing NVIDIA profiling
        ├── llama3_1_8b_pretrain_fp8/  (TensorBoard events)
        └── qwen25_7b_test_fp8/        (TensorBoard events)
```

## 🎯 What to Read Based on Your Goal

### Goal: Just Run It (5 minutes)
```
1. Read: code/QUICK_START.md
2. Run:  ./run_benchmark.sh llama
3. Done!
```

### Goal: Understand How It Works (15 minutes)
```
1. Read: code/WORKFLOW.md        (visual diagrams)
2. Read: BENCHMARKING_GUIDE.md   (overview)
3. Explore: code/benchmark_utils.py
```

### Goal: Deep Dive (30 minutes)
```
1. Read: code/BENCHMARK_README.md  (complete reference)
2. Read: code/benchmark_utils.py   (implementation)
3. Read: code/compare_results.py   (analysis)
```

### Goal: Quick Reference
```
Keep open: code/README.md (command reference)
```

## 📚 Documentation Map

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENTATION TREE                        │
└─────────────────────────────────────────────────────────────┘

📖 PROFILING_SUMMARY.md (../PROFILING_SUMMARY.md)
   │
   ├─ What was created
   ├─ Problem solved
   ├─ File locations
   └─ Quick reference

📚 BENCHMARKING_GUIDE.md (./BENCHMARKING_GUIDE.md)
   │
   ├─ Overview and features
   ├─ Quick start
   ├─ Hardware configuration
   ├─ Performance metrics
   ├─ Usage scenarios
   ├─ Example output
   ├─ Configuration details
   ├─ Troubleshooting
   └─ Best practices

📍 INDEX.md (./INDEX.md) ← YOU ARE HERE
   │
   ├─ File structure
   ├─ Documentation map
   └─ Quick navigation

code/
│
├─ 📖 README.md
│  │
│  ├─ Quick overview
│  ├─ What's included
│  ├─ What you get
│  ├─ Key features
│  ├─ Usage examples
│  ├─ Common issues
│  └─ Quick help table
│
├─ 🚀 QUICK_START.md
│  │
│  ├─ TL;DR (3 commands)
│  ├─ Step-by-step guide
│  ├─ Available models
│  ├─ Troubleshooting
│  └─ Example results
│
├─ 📊 WORKFLOW.md
│  │
│  ├─ Visual workflow diagram
│  ├─ Component architecture
│  ├─ Data flow diagrams
│  ├─ File interactions
│  ├─ Metrics timeline
│  └─ Output structure
│
└─ 📚 BENCHMARK_README.md
   │
   ├─ Detailed overview
   ├─ Configuration guide
   ├─ Metrics explanation
   ├─ Understanding results
   ├─ Advanced profiling
   ├─ Best practices
   └─ Complete reference
```

## 💻 Code Files

### Core Framework (Must Read)

| File | Size | Purpose |
|------|------|---------|
| `benchmark_utils.py` | 10KB | Platform-agnostic benchmarking framework |
| `compare_results.py` | 11KB | Generate comparison reports and charts |
| `run_benchmark.sh` | 3.5KB | Automated benchmark runner |

### Training Scripts (Updated)

| File | Size | Model | Status |
|------|------|-------|--------|
| `pretrain_llama.py` | 1.5KB | Llama 3.1 8B | ✅ Updated |
| `pretrain_qwen.py` | 1.5KB | Qwen 2.5 7B | ✅ Updated |
| `pretrain_mistral.py` | 1.5KB | Mistral 7B | ✅ Updated |

### Utilities

| File | Size | Purpose |
|------|------|---------|
| `analyze_existing_logs.py` | 7KB | Analyze old AMD/NVIDIA logs |
| `requirements.txt` | 212B | Python dependencies |

### Conversion Scripts (Unchanged)

| File | Purpose |
|------|---------|
| `convert_llama.py` | Convert Llama checkpoints |
| `convert_qwen.py` | Convert Qwen checkpoints |
| `convert_mistral.py` | Convert Mistral checkpoints |

## 🚀 Quick Commands

### Essential Commands

```bash
# Navigate to code directory
cd week-02/code

# Run benchmark (auto-detects platform)
./run_benchmark.sh llama

# Compare results (after running on both platforms)
python3 compare_results.py

# Check existing logs
python3 analyze_existing_logs.py
```

### Advanced Commands

```bash
# Multiple runs for statistical significance
./run_benchmark.sh llama 5

# Run all models
for model in llama qwen mistral; do
    ./run_benchmark.sh $model
done

# Check results directory
ls -lh benchmark_results/

# View latest result
cat benchmark_results/benchmark_*.json | tail -1 | python3 -m json.tool
```

## 📊 What Gets Generated

### During Training

```
Terminal Output:
[CUDA] Step  10 | Time: 1.234s | Avg: 1.245s | Memory: 45.67GB
[CUDA] Step  20 | Time: 1.238s | Avg: 1.242s | Memory: 45.68GB
```

### After Training

```
benchmark_results/
└── benchmark_cuda_20260105_143022.json
    ├── platform: "cuda"
    ├── gpu_info: {...}
    ├── training_config: {...}
    ├── performance_metrics: {...}
    ├── memory_metrics: {...}
    └── raw_step_times: [...]
```

### After Comparison

```
benchmark_results/
├── comparison_plot.png          ← 4-panel visualization
│   ├── Average Step Time (bar chart)
│   ├── Throughput (bar chart)
│   ├── Memory Usage (grouped bars)
│   └── Step Time Distribution (line plot)
│
└── comparison_report.md         ← Detailed markdown report
    ├── Executive Summary
    ├── Hardware Configuration
    ├── Performance Metrics
    ├── Memory Usage
    └── Detailed Analysis
```

## 🎯 Common Workflows

### Workflow 1: First Time Setup

```bash
# 1. Install dependencies
pip install matplotlib numpy

# 2. Read quick start
cat code/QUICK_START.md

# 3. Run on current platform
cd code
./run_benchmark.sh llama

# 4. Check results
ls benchmark_results/
```

### Workflow 2: Full Comparison

```bash
# On NVIDIA system
cd week-02/code
./run_benchmark.sh llama

# Copy JSON to shared location or USB drive
cp benchmark_results/benchmark_cuda_*.json /path/to/shared/

# On AMD system
cd week-02/code
./run_benchmark.sh llama

# Copy both JSONs to comparison machine
# Then compare
python3 compare_results.py

# View results
open comparison_plot.png
cat comparison_report.md
```

### Workflow 3: Multi-Model Analysis

```bash
# Run all three models on both platforms
cd week-02/code

# On each platform
for model in llama qwen mistral; do
    echo "Running $model..."
    ./run_benchmark.sh $model
    sleep 30  # Cool down
done

# Compare each model
python3 compare_results.py
```

### Workflow 4: Statistical Analysis

```bash
# Run 5 times on each platform
cd week-02/code
./run_benchmark.sh llama 5

# Results will be averaged automatically
python3 compare_results.py
```

## 🔍 Understanding the System

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Training Script                        │
│  (pretrain_llama.py / qwen.py / mistral.py)            │
└─────────────────────────────────────────────────────────┘
                          │
                          ├─ Imports benchmark_utils.py
                          ├─ Creates BenchmarkCallback
                          └─ Adds to trainer.callbacks
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              BenchmarkCallback                          │
│  (from benchmark_utils.py)                             │
│                                                         │
│  on_train_start()    → Detect platform, get GPU info   │
│  on_batch_start()    → Start timer, sync GPU           │
│  on_batch_end()      → Stop timer, record metrics      │
│  on_train_end()      → Save JSON, print summary        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              JSON Results                               │
│  benchmark_results/benchmark_{platform}_{time}.json    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              compare_results.py                         │
│                                                         │
│  load_benchmark_results()    → Load JSONs              │
│  create_comparison_plot()    → Generate chart          │
│  generate_comparison_report() → Create markdown        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Final Output                               │
│  - comparison_plot.png                                  │
│  - comparison_report.md                                 │
│  - Console summary                                      │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Platform Agnostic**: Auto-detects CUDA vs ROCm
2. **Non-Invasive**: Just a callback, no training changes
3. **Fair**: Identical configs, same warmup, same sync points
4. **Automated**: Scripts handle everything
5. **Comprehensive**: Multiple metrics, stats, visualizations

## 🎓 Learning Path

### Beginner (Just want results)
```
1. Read: code/QUICK_START.md (5 min)
2. Run:  ./run_benchmark.sh llama
3. Done: View results
```

### Intermediate (Want to understand)
```
1. Read: code/WORKFLOW.md (10 min)
2. Read: BENCHMARKING_GUIDE.md (15 min)
3. Explore: benchmark_utils.py
4. Experiment: Try different models
```

### Advanced (Want to customize)
```
1. Read: code/BENCHMARK_README.md (20 min)
2. Study: benchmark_utils.py (full code)
3. Study: compare_results.py (full code)
4. Modify: Add custom metrics
5. Extend: Add new models
```

## 📞 Quick Help

| Question | Answer |
|----------|--------|
| Where do I start? | Read `PROFILING_SUMMARY.md` in parent directory |
| How do I run it? | `cd code && ./run_benchmark.sh llama` |
| Where are results? | `code/benchmark_results/` |
| How do I compare? | `cd code && python3 compare_results.py` |
| Need quick reference? | Open `code/README.md` |
| Want visual guide? | Open `code/WORKFLOW.md` |
| Need all details? | Open `code/BENCHMARK_README.md` |
| Something broke? | Check troubleshooting in any README |

## ✅ Checklist

### Before Running
- [ ] Read `code/QUICK_START.md`
- [ ] Install dependencies: `pip install matplotlib numpy`
- [ ] Verify GPU: `nvidia-smi` or `rocm-smi`
- [ ] Navigate to: `cd week-02/code`

### Running on NVIDIA
- [ ] Run: `./run_benchmark.sh llama`
- [ ] Check: `ls benchmark_results/benchmark_cuda_*.json`
- [ ] Copy JSON to shared location (if comparing on different machine)

### Running on AMD
- [ ] Run: `./run_benchmark.sh llama`
- [ ] Check: `ls benchmark_results/benchmark_rocm_*.json`
- [ ] Ensure both CUDA and ROCm JSONs are in `benchmark_results/`

### Comparing Results
- [ ] Run: `python3 compare_results.py`
- [ ] View: `comparison_plot.png`
- [ ] Read: `comparison_report.md`
- [ ] Note: Winner and speedup factor

## 🎉 Summary

You have a **complete benchmarking system** with:

✅ **4 documentation files** (Quick Start, Workflow, Complete Guide, Overview)  
✅ **3 core tools** (benchmark_utils, compare_results, analyze_logs)  
✅ **3 updated training scripts** (Llama, Qwen, Mistral)  
✅ **1 automation script** (run_benchmark.sh)  
✅ **Full automation** (3 commands to results)  

**Next**: Read `code/QUICK_START.md` and run your first benchmark!

---

**Location**: `/Users/dmitrynvm/Work/support/week-02/`  
**Status**: ✅ Ready to use  
**Updated**: January 5, 2026  

