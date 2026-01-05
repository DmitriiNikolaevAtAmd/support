Figure out how to collect PyTorch traces with Nemo.
Generate TraceLens reports from Nemo PyTorch traces and compare to MI300X report.
Figure out how to collect GPU memory profiles with Primus and Nemo. Collect profiles from both and compare.
In addition to profiling do actual benchmarking: record throughput values (tokens / gpu / s) for MI300X and H100 on Llama 3.1 8B and Qwen 2.5 7B
First use same settings for both devices and report performance
Then fine-tune batch size etc. for each device separately to get better performance



I'll help you set up a common profiling framework that works on both AMD and NVIDIA GPUs for fair comparison. Let me create a unified profiling setup for your training scripts.



I'll help you understand how to profile training with NeMo. Let me first check your current training scripts to see what you're working with.

[3 tools called]

Based on your training scripts, I can see you're using NeMo with distributed training. Here are several ways to profile your NeMo training:

## 1. **PyTorch Profiler (Recommended for detailed analysis)**

Add the PyTorch Profiler to your trainer configuration:

```python
from nemo.collections import llm
import nemo_run as run
from lightning.pytorch.profilers import PyTorchProfiler

def run_pretrain():
    recipe = llm.llama31_8b.pretrain_recipe(
        name="llama3_1_8b_pretrain_fp8",
        dir="/checkpoints",
        num_nodes=1,
        num_gpus_per_node=8,
    )
    
    # ... your existing configuration ...
    
    # ADD PROFILER
    recipe.trainer.profiler = PyTorchProfiler(
        dirpath="./profile_logs",
        filename="perf_logs",
        export_to_chrome=True,  # Creates JSON for chrome://tracing
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        # Profile specific steps
        schedule={
            "wait": 1,      # Skip first step
            "warmup": 1,    # Warmup for 1 step
            "active": 3,    # Profile 3 steps
            "repeat": 2     # Repeat cycle 2 times
        }
    )
    
    run.run(recipe, direct=True)
```

## 2. **NVIDIA Nsight Systems (For GPU-level profiling)**

Run your training with `nsys`:

```bash
nsys profile \
    --output=llama_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    python pretrain_llama.py
```

Then add NVTX markers in your code:

```python
import torch.cuda.nvtx as nvtx

# In your training script
recipe.trainer.callbacks = recipe.trainer.callbacks or []
# NeMo automatically adds NVTX markers, but you can add custom ones
```

## 3. **Simple Performance Callback**

Create a custom callback for basic timing:

```python
from lightning.pytorch.callbacks import Callback
import time

class TimingCallback(Callback):
    def __init__(self):
        self.start_time = None
        self.step_times = []
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.step_times.append(duration)
            if batch_idx % 10 == 0:
                avg_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
                print(f"Step {batch_idx}: {duration:.3f}s, Avg: {avg_time:.3f}s")

# Add to recipe
recipe.trainer.callbacks = recipe.trainer.callbacks or []
recipe.trainer.callbacks.append(TimingCallback())
```

## 4. **Enable Detailed Logging**

```python
# Add to your recipe
recipe.trainer.log_every_n_steps = 1
recipe.trainer.val_check_interval = None  # Disable validation for profiling

# Enable more detailed logging
import logging
logging.getLogger("nemo").setLevel(logging.DEBUG)
```

## 5. **For AMD GPUs (Based on your amd-logs folder)**

For AMD ROCm profiling:

```bash
rocprof --stats --timestamp on \
    python pretrain_llama.py
```

Or use PyTorch's built-in profiler which works on both NVIDIA and AMD:

```python
recipe.trainer.profiler = "simple"  # Basic profiler
# or
recipe.trainer.profiler = "advanced"  # More detailed
```

## Analyzing Results

- **PyTorch Profiler**: View the generated JSON in `chrome://tracing` or use TensorBoard
- **Nsight Systems**: Open `.nsys-rep` files in NVIDIA Nsight Systems GUI
- **AMD**: The `.pt.trace_perf_report.xlsx` files you already have contain profiling data

Would you like me to modify one of your training scripts to add profiling, or would you like more details on any specific profiling method?