#!/bin/bash

set -euo pipefail

# Train args
MODEL_NAME="llama-meta-3.2-1b"
MAX_STEPS=${1:-250}
LORA_RANK=${2:-32}
USE_VLLM=${3:-false}

LOG_DIR="logs/grpo"
LOG_FILE="$LOG_DIR/train_${MODEL_NAME}_${LORA_RANK}_vllm=${USE_VLLM}.log"

TRAIN_ARGS="$MODEL_NAME --max_steps $MAX_STEPS --lora_rank $LORA_RANK"
if [ "$USE_VLLM" = false ]; then
    TRAIN_ARGS="$TRAIN_ARGS --use_slow_inference"
fi

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

CMD="python tests/grpo/test_unsloth_grpo.py ${TRAIN_ARGS}"
echo "${CMD}"
eval "$CMD" 2>&1 | tee "$LOG_FILE"

# echo "Done training"
# echo "Checking saved weights"
# CMD="python scripts/check_lora_merged.py --lora_rank ${LORA_RANK} "