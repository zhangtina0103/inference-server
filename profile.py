import subprocess
import time
import requests
import os
import signal


def start_server():
    """start the FastAPI server as a subprocess"""
    proc = subprocess.Popen(
        ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(30)  # wait for model to load
    return proc


def profile_with_strace(pid: int, duration: int = 30):
    """
    profile syscalls using strace.
    shows which system calls are taking the most time.
    key things to look for:
    - excessive read/write calls (I/O bound)
    - futex calls (lock contention)
    - mmap calls (memory allocation)
    """
    print("=" * 60)
    print("STRACE PROFILE (syscall summary)")
    print("=" * 60)

    result = subprocess.run(
        ["strace", "-p", str(pid), "-c", "-f",
         "-e", "trace=all", "--", "sleep", str(duration)],
        capture_output=True, text=True, timeout=duration + 5
    )
    print(result.stderr)  # strace outputs to stderr


def profile_with_perf(pid: int, duration: int = 30):
    """
    profile CPU usage using perf.
    shows which functions are hottest (most CPU time).
    key things to look for:
    - where compute time is being spent
    - unexpected overhead in Python interpreter
    - CUDA synchronization points
    """
    print("=" * 60)
    print("PERF PROFILE (CPU hotspots)")
    print("=" * 60)

    # record
    subprocess.run(
        ["perf", "record", "-p", str(pid),
         "-g", "--", "sleep", str(duration)],
        timeout=duration + 5
    )

    # report
    result = subprocess.run(
        ["perf", "report", "--stdio", "--no-pager"],
        capture_output=True, text=True
    )
    print(result.stdout[:3000])  # first 3000 chars


def send_load(num_requests: int = 20):
    """send requests to server while profiling"""
    prompts = [
        "Explain gradient descent.",
        "What is attention in transformers?",
        "How does backpropagation work?",
        "What is a convolutional neural network?",
        "Explain the transformer architecture.",
    ]

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        try:
            requests.post(
                "http://localhost:8000/generate",
                json={"prompt": prompt, "max_tokens": 128},
                timeout=60
            )
            print(f"request {i+1}/{num_requests} done")
        except Exception as e:
            print(f"request {i+1} failed: {e}")


def main():
    print("starting server...")
    server = start_server()
    pid    = server.pid
    print(f"server PID: {pid}")

    print("\nsending load while profiling...")
    import threading

    # send load in background
    load_thread = threading.Thread(
        target=send_load, args=(20,)
    )
    load_thread.start()

    # profile
    profile_with_strace(pid, duration=30)
    profile_with_perf(pid, duration=30)

    load_thread.join()

    print("\nkilling server...")
    server.terminate()


if __name__ == "__main__":
    main()
