# ruff: noqa
import os
import sys
os.environ["HF_HOME"] = "./hf_home"
from qwen_grpo import get_base_model_and_tokenizer, generate_text, get_peft_model, DEFAULT_MESSAGE, SYSTEM_MESSAGE, SAMPLING_PARAMS, MAX_SEQ_LENGTH, LORA_RANK, TARGET_MODULES

import json

import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file, load
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
import argparse



SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}
DEFAULT_MESSAGE = [{"role": "user", "content": "How many r's are in strawberry?"}]
#MERGED_MODEL_PATH = "model"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
#ADAPTER_PATH = "grpo_saved_lora"

USE_MERGED_MODEL = False
USE_VLLM = True
LOAD_ADAPTER = False

def load_safetensors(file_path, device="cuda"):
    loaded = load_file(file_path, device=device)
    return loaded

def load_adapter_weights(adapter_path):
    adapter_weights = os.path.join(adapter_path, "adapter_model.safetensors")

    return adapter_weights

def check_model_weights(merged_model_path, original_model_path, max_seq_length=MAX_SEQ_LENGTH, lora_rank=LORA_RANK):
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_nf4
    
    merged_model, _ = get_base_model_and_tokenizer(
        model_name=merged_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=lora_rank,
    )
    original_model, _ = get_base_model_and_tokenizer(
        model_name=original_model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
    )
    for name, param in merged_model.named_parameters():
        original_param = original_model.get_parameter(name)
        if isinstance(original_param, bnb.nn.Params4bit):
            dq_weight = dequantize_nf4(original_param, quant_state=original_param.quant_state)
            print(f"{name}: {param.dtype} vs {original_param.quant_state.dtype} {torch.allclose(param, dq_weight.to(param.dtype))} {(param - dq_weight.to(param.dtype)).abs().max().item():.5f}")


def main(args):
    if args.merged_model_path:
        print("Using merged model...")
        model_name = args.merged_model_path
    else:
        print("Using LoRA model...")
    model_name = MODEL_NAME


    use_fast_inference = args.use_vllm
    dtype = torch.bfloat16



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--use-merged-model", action="store_true")

    args = parser.parse_args()
    main(args)
