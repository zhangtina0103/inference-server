#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu

module load cuda/12.4.0
conda activate inference

mkdir -p logs

echo "========================================"
echo "starting vLLM server"
echo "========================================"
uvicorn serve:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
echo "server PID: $SERVER_PID"

echo "waiting for model to load (90s)..."
sleep 90

echo "checking health..."
curl -s http://localhost:8000/health

echo ""
echo "========================================"
echo "running benchmark"
echo "========================================"
python client.py

echo ""
echo "========================================"
echo "KV cache demo"
echo "========================================"
python -c "
from kv_cache import PagedKVCache
import torch

cache = PagedKVCache(num_blocks=128, block_size=16)
print(cache)

for req_id in range(3):
    cache.allocate(req_id)
    for token in range(20):
        k = torch.randn(8, 64)
        v = torch.randn(8, 64)
        cache.write(req_id, token, k, v)
    k_all, v_all = cache.read(req_id)
    print(f'request {req_id}: {k_all.shape} keys cached')

print(f'after 3 requests: {cache}')
for req_id in range(3):
    cache.free(req_id)
print(f'after freeing: {cache}')
"

echo "done. killing server..."
kill $SERVER_PID
