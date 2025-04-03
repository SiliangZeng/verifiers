#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3 

python verifiers/inference/vllm_serve.py \
    --model "Qwen/Qwen2.5-7B" \
    --tensor_parallel_size 4 \
    --max_model_len 8192  \
    --gpu_memory_utilization 0.9 \
    --enable_prefix_caching True \
    --host 0.0.0.0 \
    --port 8000 \