#!/bin/bash
set -ex



# Get total number of GPUs
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
NUM_PROCESSES=$((TOTAL_GPUS - 1))

echo "Total GPUs: ${TOTAL_GPUS}"
echo "Launching training with ${NUM_PROCESSES} processes on ${TOTAL_GPUS} GPUs"


source $(conda info --base)/etc/profile.d/conda.sh
conda activate verifier_env
# per_device_train_batch_size * num-processes % num_generations == 0

# Launch training with Accelerate (all parameters inlined)
accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_PROCESSES} \
    verifiers/examples/triviaqa_search.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --num_gpus ${TOTAL_GPUS} \
    --learning_rate 1e-6 \
    --num_generations 21 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 4 \
    --num_iterations 2 \
    --max_steps 200 \
    --beta 0 \
    --trainer "msgrpo" \

# accelerate launch --config-file configs/zero3.yaml --num-processes ${NUM_PROCESSES} \
#     verifiers/examples/triviaqa_search.py \
#     --model_name "Qwen/Qwen2.5-7B" \
#     --num_gpus ${TOTAL_GPUS} \
#     --learning_rate 1e-6 \
#     --num_generations 1 \
#     --per_device_train_batch_size 12 \
#     --gradient_accumulation_steps 4 \
#     --num_iterations 2 \
#     --max_steps 200 \
#     --beta 0 \
#     --trainer "remax" \