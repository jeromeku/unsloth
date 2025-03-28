#!/bin/bash

set -euo pipefail

# Train args
MODEL_NAME="llama-meta-3.1-1b"
MAX_STEPS=${1:-25}
LORA_RANK=${2:-32}
USE_VLLM=${3:-false}

MODEL_MERGE_PATH="tests/outputs/grpo/$MODEL_NAME/merged"
MODEL_ADAPTER_PATH="tests/outputs/grpo/$MODEL_NAME/lora"

LOG_DIR="logs/grpo"
LOG_FILE="$LOG_DIR/train_${MODEL_NAME}_${LORA_RANK}_vllm=${USE_VLLM}.log"

TRAIN_ARGS="$MODEL_NAME --max_steps $MAX_STEPS --lora_rank $LORA_RANK --model_merged_save_path $MODEL_MERGE_PATH --model_adapter_save_path $MODEL_ADAPTER_PATH"
if [ "$USE_VLLM" = false ]; then
    TRAIN_ARGS="$TRAIN_ARGS --use_slow_inference"
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

CMD="python tests/grpo/test_unsloth_grpo.py --model_merged_save_path $MODEL_MERGE_PATH --model_adapter_save_path $MODEL_ADAPTER_PATH"
echo "${CMD}"
eval "$CMD" 2>&1 | tee "$LOG_FILE"

# echo "Done training"
# echo "Checking saved weights"
# CMD="python scripts/check_lora_merged.py --lora_rank ${LORA_RANK} "