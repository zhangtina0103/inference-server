# inference-server

Here I take my DPO fine-tuned LLaMA 3.1-8B-Instruct model using vLLM with LoRA adapter support (from my previous project on training risk averse LLMs), wrapped in a FastAPI server with Prometheus metrics tracking.

Pipeline is essentially LLaMA 3.1-8B-Instruct base model plus DPO LoRA adapter feeding into vLLM engine (continuous batching, KV cache) which goes into FastAPI server to get Prometheus metrics. Here I have 5 prompts.

## Results (L40S GPU, MIT Engaging Cluster)

| metric          | value               |
| --------------- | ------------------- |
| avg latency     | 5981ms (256 tokens) |
| p50 latency     | 6035ms              |
| p95 latency     | 8477ms              |
| throughput      | 41.8 tok/s          |
| GPU memory      | 14.99 GB            |
| KV cache blocks | 8530 GPU blocks     |

## Structure

- serve.py: FastAPI + vLLM server
- monitor.py: Prometheus metrics
- kv_cache.py: PagedKV cache from scratch
- batching.py: continuous batching from scratch
- client.py: benchmark client
- profile.py: strace/perf profiling
- submit.sh: SLURM job script
