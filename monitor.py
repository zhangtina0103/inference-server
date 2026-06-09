from prometheus_client import Counter, Histogram, Gauge
import torch


class Metrics:
    def __init__(self):
        self.request_count = Counter(
            "vllm_request_count",
            "total number of requests"
        )

        self.latency = Histogram(
            "vllm_latency_ms",
            "request latency in milliseconds",
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
        )

        self.prompt_tokens = Counter(
            "vllm_prompt_tokens_total",
            "total prompt tokens processed"
        )

        self.completion_tokens = Counter(
            "vllm_completion_tokens_total",
            "total completion tokens generated"
        )

        self.gpu_memory_used = Gauge(
            "vllm_gpu_memory_used_gb",
            "GPU memory used in GB"
        )

        self.gpu_utilization = Gauge(
            "vllm_gpu_utilization_pct",
            "GPU utilization percentage"
        )

    def update_gpu_metrics(self):
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            self.gpu_memory_used.set(mem)


# singleton — imported by serve.py
metrics = Metrics()
