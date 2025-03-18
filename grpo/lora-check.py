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

def load_safetensors(file_path, device="cuda"):
    loaded = load_file(file_path, device=device)
    return loaded


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
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


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

    use_merged_model = args.merged_model is not None
    if use_merged_model:
        print("Using merged model...")
        model_name = args.merged_model
    else:
        print("Using LoRA model...")
        model_name = args.original_model

    use_fast_inference = args.use_vllm
    dtype = torch.bfloat16
    load_adapter = args.adapter_path is not None

    model, tokenizer = get_base_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False if use_merged_model else True,  # False for LoRA 16bit
        fast_inference=use_fast_inference,  # Enable vLLM fast inference
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    if not use_merged_model:
        model = get_peft_model(model, LORA_RANK)

        if load_adapter:
            print("Loading saved adapter weights...")
            adapter_state_dict = load_safetensors(load_adapter_weights(args.adapter_path))
            set_peft_model_state_dict(model, peft_model_state_dict=adapter_state_dict)
            #peft_state_dict = get_peft_model_state_dict(model)
            for name, param in model.named_parameters():
                if "default" in name:
                    param_name = name.replace("default.", "")
                    adapter_param = adapter_state_dict[param_name]
                # print(f"{param_name}: {param.dtype} vs {adapter_param.dtype} {torch.allclose(param, adapter_param.to(param.dtype))}")
                    assert torch.allclose(param, adapter_param.to(param.dtype))
        elif not args.use_vllm:
            print("Not using vllm and not loading adapter weights --> will generate using untrained adapter weights...")
        print(f"Active adapters  {model.active_adapters}")
    else:
        print("Using merged model...")


    if args.use_vllm:    
        if use_merged_model:
            merged_text = generate_text(model, tokenizer, messages=DEFAULT_MESSAGE, add_system_prompt=True, temperature=args.temperature, top_p=SAMPLING_PARAMS["top_p"] if args.temperature > 0.0 else 1)
            print(f"Merged model: {merged_text}")
        else:
            without_lora = generate_text(model, tokenizer, messages=DEFAULT_MESSAGE, add_system_prompt=True, temperature=args.temperature, top_p=SAMPLING_PARAMS["top_p"] if args.temperature > 0.0 else 1)
            print(f"Without lora: {without_lora}")
            with_lora = generate_text(model, tokenizer, lora_path=args.adapter_path, messages=DEFAULT_MESSAGE, add_system_prompt=True, temperature=args.temperature, top_p=SAMPLING_PARAMS["top_p"] if args.temperature > 0.0 else 1)
            print(f"With lora: {with_lora}")
    else:
        generation_config = GenerationConfig(
            do_sample=False if args.temperature == 0.0 else True,
            max_length=MAX_SEQ_LENGTH,
            temperature=args.temperature,
            top_p=SAMPLING_PARAMS["top_p"],
        )
        messages = [SYSTEM_MESSAGE] + DEFAULT_MESSAGE

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        print(tokenizer.decode(inputs["input_ids"][0]))
        output = model.generate(**inputs, generation_config=generation_config)
        print(tokenizer.decode(output[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-model", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--original-model", type=str, default=MODEL_NAME)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--temperature", type=float, default=SAMPLING_PARAMS["temperature"])
    args = parser.parse_args()
    print(args)
    main(args)
