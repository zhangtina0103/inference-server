import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from monitor import metrics

app = FastAPI()

BASE_MODEL = "/orcd/home/002/zhangtin/models/llama_base"
LORA_PATH  = "/orcd/home/002/zhangtin/models/llama_dpo/llama_lr1e5_b001_nosys"

llm = LLM(
    model=BASE_MODEL,
    enable_lora=True,
    max_lora_rank=32,
    dtype="float16",
    gpu_memory_utilization=0.9,
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    start = time.time()
    outputs = llm.generate(
        [request.prompt],
        sampling_params,
        lora_request=LoRARequest("dpo_adapter", 1, LORA_PATH)
    )
    latency_ms = (time.time() - start) * 1000

    result            = outputs[0]
    text              = result.outputs[0].text
    prompt_tokens     = len(result.prompt_token_ids)
    completion_tokens = len(result.outputs[0].token_ids)

    metrics.request_count.inc()
    metrics.latency.observe(latency_ms)
    metrics.prompt_tokens.inc(prompt_tokens)
    metrics.completion_tokens.inc(completion_tokens)

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )


@app.get("/metrics")
def get_metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
