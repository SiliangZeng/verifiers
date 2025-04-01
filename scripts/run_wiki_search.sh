#!/bin/bash

# 获取命令行参数，有默认值
MODEL_NAME=${1:-"Qwen/Qwen2.5-7B-Instruct"}
QUESTION=${2:-""}

# 激活环境
source activate verifier_env

echo "Running wiki_search interaction with model: $MODEL_NAME"

# 如果提供了问题参数，则传递给交互程序
if [ -n "$QUESTION" ]; then
    python -m verifiers.simple_test.wiki_search_interaction --model_name "$MODEL_NAME" --question "$QUESTION"
else
    python -m verifiers.simple_test.wiki_search_interaction --model_name "$MODEL_NAME"
fi 