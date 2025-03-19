#!/bin/bash
set -euo pipefail

MERGED_MODEL_PATH_NO_CUDAGRAPH='grpo_saved_merged_cudagraph=False_runid=20250317_1632'
MERGED_MODEL_PATH_CUDAGRAPH='grpo_saved_merged_cudagraph=True_runid=20250317_1525'
ADAPTER_PATH_NO_CUDAGRAPH='grpo_saved_lora_cudagraph=False_runid=20250317_1632'
ADAPTER_PATH_CUDAGRAPH='grpo_saved_lora_cudagraph=True_runid=20250317_1525'
DEFAULT_TEMPERATURE=0.8
TEMPERATURE=${1:-$DEFAULT_TEMPERATURE}
echo "Running with temperature $TEMPERATURE"

# Use VLLM
LORA_DIR="lora_logs"
mkdir -p $LORA_DIR
MERGED_DIR="merged_logs"
mkdir -p $MERGED_DIR

# Merged with vllm generation
for temperature in 0.0 0.8; do
    echo "Running merged models with vllm generation with temperature $temperature"
    USE_CUDAGRAPH=1 python lora-check.py --merged-model $MERGED_MODEL_PATH_CUDAGRAPH --temperature $TEMPERATURE --use-vllm 2>&1 | tee $MERGED_DIR/merged.vllm.cudagraph.generate_cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --merged-model $MERGED_MODEL_PATH_CUDAGRAPH --temperature $TEMPERATURE --use-vllm 2>&1 | tee $MERGED_DIR/merged.vllm.cudagraph.generate_no-cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=1 python lora-check.py --merged-model $MERGED_MODEL_PATH_NO_CUDAGRAPH --temperature $TEMPERATURE --use-vllm 2>&1 | tee $MERGED_DIR/merged.vllm.no-cudagraph.generate_cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --merged-model $MERGED_MODEL_PATH_NO_CUDAGRAPH --temperature $TEMPERATURE --use-vllm 2>&1 | tee $MERGED_DIR/merged.vllm.no-cudagraph.generate_no-cudagraph.temperature-$TEMPERATURE.log
done

# LoRA with vllm generation
for temperature in 0.0 0.8; do
    echo "Running with temperature $temperature"
    USE_CUDAGRAPH=1 python lora-check.py --adapter-path $ADAPTER_PATH_CUDAGRAPH --use-vllm --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.vllm.cudagraph.generate_cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --adapter-path $ADAPTER_PATH_CUDAGRAPH --use-vllm --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.vllm.cudagraph.generate_no-cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --adapter-path $ADAPTER_PATH_NO_CUDAGRAPH --use-vllm --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.vllm.no-cudagraph.generate_no-cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=1 python lora-check.py --adapter-path $ADAPTER_PATH_NO_CUDAGRAPH --use-vllm --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.vllm.no-cudagraph.generate_cudagraph.temperature-$TEMPERATURE.log
done

# Merged adapters with native hf generation
for temperature in 0.0 0.8; do
    echo "Running merged models with no vllm generation with temperature $temperature"
    USE_CUDAGRAPH=0 python lora-check.py --merged-model $MERGED_MODEL_PATH_NO_CUDAGRAPH --temperature $TEMPERATURE 2>&1 | tee $MERGED_DIR/merged.no-vllm.no-cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --merged-model $MERGED_MODEL_PATH_CUDAGRAPH --temperature $TEMPERATURE 2>&1 | tee $MERGED_DIR/merged.no-vllm.cudagraph.temperature-$TEMPERATURE.log
done

# LoRA models with loaded adapter weights with native hf generation
for temperature in 0.0 0.8; do
    echo "Running LoRA models with no vllm generation with temperature $temperature"
    USE_CUDAGRAPH=0 python lora-check.py --adapter-path $ADAPTER_PATH_NO_CUDAGRAPH --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.no-vllm.no-cudagraph.temperature-$TEMPERATURE.log
    USE_CUDAGRAPH=0 python lora-check.py --adapter-path $ADAPTER_PATH_CUDAGRAPH --temperature $TEMPERATURE 2>&1 | tee $LORA_DIR/lora.no-vllm.cudagraph.temperature-$TEMPERATURE.log
done

