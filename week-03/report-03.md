
Figure out how to collect PyTorch traces with Nemo.
Generate TraceLens reports from Nemo PyTorch traces and compare to MI300X report.
Figure out how to collect GPU memory profiles with Primus and Nemo. Collect profiles from both and compare.
In addition to profiling do actual benchmarking: record throughput values (tokens / gpu / s) for MI300X and H100 on Llama 3.1 8B and Qwen 2.5 7B
First use same settings for both devices and report performance
Then fine-tune batch size etc. for each device separately to get better performance