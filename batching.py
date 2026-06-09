import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RequestStatus(Enum):
    WAITING    = "waiting"
    RUNNING    = "running"
    FINISHED   = "finished"


@dataclass
class Request:
    request_id:  int
    prompt:      str
    max_tokens:  int = 256
    temperature: float = 0.7
    status:      RequestStatus = RequestStatus.WAITING
    output:      str = ""
    tokens_generated: int = 0
    arrival_time: float = field(default_factory=time.time)
    finish_time:  Optional[float] = None

    @property
    def latency_ms(self):
        if self.finish_time:
            return (self.finish_time - self.arrival_time) * 1000
        return None

    @property
    def is_finished(self):
        return self.tokens_generated >= self.max_tokens


class ContinuousBatcher:
    """
    continuous batching implementation from scratch.

    naive batching: wait for entire batch to finish before adding new requests.
    this wastes GPU time when some sequences finish early.

    continuous batching: as soon as one sequence finishes, immediately insert
    a new request into that slot. GPU is never idle waiting for slow sequences.

    this is exactly what vLLM and Orca implement.
    """

    def __init__(self, max_batch_size=8):
        self.max_batch_size = max_batch_size
        self.waiting_queue  = queue.Queue()
        self.running        : list[Request] = []
        self.finished       : list[Request] = []
        self.lock           = threading.Lock()
        self.request_counter = 0

    def add_request(self, prompt: str, max_tokens: int = 256) -> int:
        """add a new request to the waiting queue"""
        request_id = self.request_counter
        self.request_counter += 1

        req = Request(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens
        )
        self.waiting_queue.put(req)
        print(f"[batcher] request {request_id} queued")
        return request_id

    def _fill_batch(self):
        """fill empty slots in running batch from waiting queue"""
        while (len(self.running) < self.max_batch_size
               and not self.waiting_queue.empty()):
            try:
                req = self.waiting_queue.get_nowait()
                req.status = RequestStatus.RUNNING
                self.running.append(req)
                print(f"[batcher] request {req.request_id} added to batch "
                      f"(batch size: {len(self.running)})")
            except queue.Empty:
                break

    def _step(self, llm, sampling_params):
        """one generation step for all running requests"""
        if not self.running:
            return

        # generate one token for each running request
        prompts = [req.prompt + req.output for req in self.running]
        outputs = llm.generate(prompts, sampling_params)

        finished = []
        for req, output in zip(self.running, outputs):
            new_token = output.outputs[0].text
            req.output          += new_token
            req.tokens_generated += 1

            if req.is_finished:
                req.status      = RequestStatus.FINISHED
                req.finish_time = time.time()
                finished.append(req)
                print(f"[batcher] request {req.request_id} finished "
                      f"({req.tokens_generated} tokens, "
                      f"{req.latency_ms:.1f}ms)")

        # remove finished requests
        for req in finished:
            self.running.remove(req)
            self.finished.append(req)

    def run(self, llm, sampling_params, max_steps=100):
        """main continuous batching loop"""
        print(f"[batcher] starting continuous batching "
              f"(max_batch_size={self.max_batch_size})")

        for step in range(max_steps):
            # fill empty slots with waiting requests
            self._fill_batch()

            if not self.running and self.waiting_queue.empty():
                print(f"[batcher] all requests finished after {step} steps")
                break

            # one generation step
            self._step(llm, sampling_params)

        return self.finished

    def stats(self):
        if not self.finished:
            return
        latencies  = [r.latency_ms for r in self.finished if r.latency_ms]
        throughputs = [r.tokens_generated / (r.latency_ms / 1000)
                      for r in self.finished if r.latency_ms]

        print()
        print("=" * 60)
        print("BATCHING STATS")
        print("=" * 60)
        print(f"total requests:   {len(self.finished)}")
        print(f"avg latency:      {sum(latencies)/len(latencies):.1f}ms")
        print(f"avg throughput:   {sum(throughputs)/len(throughputs):.1f} tok/s")
