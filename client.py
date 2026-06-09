import requests
import time
import statistics


BASE_URL = "http://localhost:8000"


def generate(prompt, max_tokens=256, temperature=0.7):
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    )
    return response.json()


def benchmark(prompts, runs=5):
    print(f"benchmarking {len(prompts)} prompts x {runs} runs")
    print("-" * 60)

    latencies = []
    throughputs = []

    for i, prompt in enumerate(prompts):
        for r in range(runs):
            start = time.time()
            result = generate(prompt)
            elapsed = time.time() - start

            latencies.append(result["latency_ms"])
            tokens = result["completion_tokens"]
            throughputs.append(tokens / (elapsed))

            print(f"prompt {i+1} run {r+1} | "
                  f"latency: {result['latency_ms']:.1f}ms | "
                  f"tokens: {tokens} | "
                  f"throughput: {tokens/elapsed:.1f} tok/s")

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"avg latency:    {statistics.mean(latencies):.1f}ms")
    print(f"p50 latency:    {statistics.median(latencies):.1f}ms")
    print(f"p95 latency:    {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print(f"avg throughput: {statistics.mean(throughputs):.1f} tok/s")


if __name__ == "__main__":
    # check health first
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"server health: {health}")
    print()

    prompts = [
        "What is the capital of France?",
        "Explain the concept of gradient descent in simple terms.",
        "Write a short poem about machine learning.",
        "What are the key differences between supervised and unsupervised learning?",
    ]

    benchmark(prompts)
