#!/bin/bash

# 检查是否有足够的GPU (至少需要8张)
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ $TOTAL_GPUS -lt 8 ]; then
    exit 1
fi

# 获取命令行参数，使用默认值
MODEL_NAME=${1:-"Qwen/Qwen2.5-7B"}
LEARNING_RATE=${2:-"1e-6"}
NUM_GENERATIONS=${3:-"21"}
BATCH_SIZE=${4:-"12"}
GRAD_ACCUM_STEPS=${5:-"4"}
NUM_ITERATIONS=${6:-"2"}
MAX_STEPS=${7:-"200"}
BETA=${8:-"0"}
STEP_ADV_COE=${9:-"0"}

# 激活环境
source activate verifier_env

# 启动第一个训练，使用GPU 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/ms_grpo_quickstart.sh \
    "$MODEL_NAME" \
    "$LEARNING_RATE" \
    "$NUM_GENERATIONS" \
    "$BATCH_SIZE" \
    "$GRAD_ACCUM_STEPS" \
    "$NUM_ITERATIONS" \
    "$MAX_STEPS" \
    "$BETA" \
    "$STEP_ADV_COE" > ms_grpo_log.txt 2>&1 &

MS_GRPO_PID=$!

# 等待几秒钟以确保第一个训练已经正确启动
sleep 5

# 启动第二个训练，使用GPU 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/grpo_quickstart.sh \
    "$MODEL_NAME" \
    "$LEARNING_RATE" \
    "$NUM_GENERATIONS" \
    "$BATCH_SIZE" \
    "$GRAD_ACCUM_STEPS" \
    "$NUM_ITERATIONS" \
    "$MAX_STEPS" \
    "$BETA" > grpo_log.txt 2>&1 &

GRPO_PID=$!

wait $MS_GRPO_PID $GRPO_PID