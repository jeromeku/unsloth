# ruff: noqa

"""
Check that the merged 16bit model (`save_pretrained_merged`) weights are correctly merged.
- This is done by comparing the merged weights with the unmerged, saved LoRA weights (saved by calling either `save_pretrained` or `save_lora` after training).
- The unmerged lora weights are manually merged in this script and compared with the saved merged weights.
"""

import os
import argparse

import torch

from typing import List

from bitsandbytes.nn import Params4bit
from bitsandbytes.functional import dequantize_nf4

DEFAULT_ADAPTER_NAME = "default"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TEST_MODELS = {
    "llama-unsloth": "unsloth/Llama-3.2-1B-Instruct",
    "gemma-unsloth": "unsloth/gemma-3-1b-it",
    "qwen-unsloth": "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
}


def get_unsloth_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    lora_rank: int = 64,
    target_modules: List[str] = DEFAULT_TARGET_MODULES,
    gpu_memory_utilization: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    random_state: int = 42,
):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )

    return model, tokenizer

def merge_lora_module(module: torch.nn.Module, module_name: str, adapter_name: str = "default", raise_on_lora_b_zero: bool = False):
    """
    Merge adapter weights into the base weight.

    Returns the merged weight in float32.
    """
    scale = module.scaling.get(adapter_name, None)
    lora_A = getattr(module.lora_A, adapter_name)
    lora_B = getattr(module.lora_B, adapter_name)
    assert lora_A is not None and lora_B is not None, f"{module_name} check failed: Lora A and B for adapter {adapter_name} are not found!"

    # lora_B is initialized to all zeros, should be non-zero after training
    failed_on_zero = lora_B.weight.eq(0).all()
    if failed_on_zero:
        msg = f"{module_name}: lora_B weights all zeros! lora_B weights are initialized to zeros -- this indicates that the saved weights are not trained."
        if raise_on_lora_b_zero:
            raise AssertionError(msg)
        else:
            print(f"!!WARNING!! {msg}")

    # Dequantize base weight if needed
    if isinstance(module.base_layer.weight, Params4bit):
        dq = dequantize_nf4(
            module.base_layer.weight,
            quant_state=module.base_layer.weight.quant_state,
        ).float()
    else:
        # Case when using unsloth dyanmic quantization, where not all target modules are quantized
        dq = module.base_layer.weight.float()

    # Merge scaled lora A and B
    merged_weight = torch.addmm(dq.t(), lora_A.weight.t(), lora_B.weight.t(), alpha=scale)
    merged_weight = merged_weight.t()

    return merged_weight

def check_unsloth_merged_weight(unsloth_merged_weight: torch.Tensor, merged_saved_weight: torch.Tensor, weight_name: str):
    assert unsloth_merged_weight.dtype == merged_saved_weight.dtype, f"{weight_name} dtype mismatch: {unsloth_merged_weight.dtype} != {merged_saved_weight.dtype}"
    assert unsloth_merged_weight.shape == merged_saved_weight.shape, f"{weight_name} shape mismatch: {unsloth_merged_weight.shape} != {merged_saved_weight.shape}" 
    diff = (unsloth_merged_weight - merged_saved_weight).abs().max().item()
    assert torch.allclose(unsloth_merged_weight, merged_saved_weight), f"{weight_name} unsloth merged weight check failed: {diff:.6f}"

def main(args):
    dtype = getattr(torch, args.dtype)

    # These should be the same as the ones used in the training script
    max_seq_length = args.max_seq_length
    lora_rank = args.lora_rank
    target_modules = args.target_modules

    # Save paths
    unsloth_adapter_path = args.adapter_save_path
    unsloth_merged_path = args.merged_save_path
    assert os.path.exists(unsloth_adapter_path), (
        f"Unsloth adapter path does not exist: {unsloth_adapter_path}"
    )
    assert os.path.exists(unsloth_merged_path), (
        f"Unsloth merged path does not exist: {unsloth_merged_path}"
    )

    # Load saved model (`save_pretrained`)
    saved_model, _ = get_unsloth_model_and_tokenizer(
        model_name=unsloth_adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        lora_rank=lora_rank,
        target_modules=target_modules,
        dtype=dtype,
    )

    # Load merged model (`save_pretrained_merged`)
    unsloth_merged_model, _ = get_unsloth_model_and_tokenizer(
        model_name=unsloth_merged_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        lora_rank=lora_rank,
        target_modules=target_modules,
        dtype=dtype,
    )

    print(f"Checking saved adapters ({unsloth_adapter_path}) and merged model ({unsloth_merged_path})...")

    for module_name, merged_module in unsloth_merged_model.named_modules():
        if any(module_name.endswith(m) for m in args.target_modules):

            # Retrieve the saved base lora module
            saved_base_module = saved_model.get_submodule(module_name)
            saved_base_weight = saved_base_module.base_layer.weight

            # Merge the saved base weight with the lora A and B weights, returned weight is in float32
            merged_saved_weight = merge_lora_module(
                saved_base_module, module_name=module_name, adapter_name=DEFAULT_ADAPTER_NAME
            )
      
            # Convert the merged fp32 saved weights to original dtype
            original_dtype = saved_base_weight.quant_state.dtype if hasattr(saved_base_weight, "quant_state") else saved_base_weight.dtype
            
            unsloth_merged_weight = merged_module.weight

            # Convert to original dtype
            merged_saved_weight = merged_saved_weight.to(original_dtype)

            # Check that the unsloth merged weight matches the saved merged weight
            check_unsloth_merged_weight(unsloth_merged_weight, merged_saved_weight, weight_name=module_name)

    print("All checks passed!")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_save_path", type=str, required=True
    )
    parser.add_argument(
        "--adapter_save_path", type=str, required=True
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--target_modules", nargs="+", type=str, default=DEFAULT_TARGET_MODULES)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
