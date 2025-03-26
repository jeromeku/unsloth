# ruff: noqa

import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).parents[2]
sys.path.append(str(REPO_ROOT))

import torch
#from unsloth import FastLanguageModel
from typing import List
from tests.utils import TEST_MODELS, get_parser, header_footer_context, DEFAULT_ADAPTER_NAME, DEFAULT_TARGET_MODULES
from tests.utils.hf_utils import setup_model, get_peft_config
from peft.tuners.lora import LoraLayer
from bitsandbytes.nn import Params4bit
from bitsandbytes.functional import dequantize_nf4
from transformers import AutoModelForCausalLM
from peft.tuners.lora import LoraLayer


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

if __name__ == "__main__":
    args = get_parser().parse_args()
    model_name = TEST_MODELS[args.model_name]
    dtype = getattr(torch, args.dtype)

    # Training args
    seed = args.seed
    batch_size = args.batch_size
    max_steps = args.max_steps   
    max_seq_length = args.max_seq_length
    num_examples = args.num_examples or args.max_steps * batch_size

    if args.gradient_checkpointing == "True":
        gradient_checkpointing = True
    elif args.gradient_checkpointing == "False":
        gradient_checkpointing = False
    else:
        gradient_checkpointing = "unsloth"

    num_train_epochs = args.num_train_epochs
    
    # Generation args
    num_generations = args.num_generations
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    # PEFT args
    lora_rank = args.lora_rank
    target_modules = args.target_modules
    
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        target_modules = target_modules[0]
    
    # Save paths
    unsloth_merged_path = os.path.join(args.merged_save_path, f"{model_name.replace('/', '_')}_lora_r{lora_rank}")
    unsloth_adapter_path = os.path.join(args.adapter_save_path, f"{model_name.replace('/', '_')}_lora_r{lora_rank}")
    
    # # Load HF Model before unsloth
    peft_config = get_peft_config(lora_rank=lora_rank, target_modules=target_modules)
    hf_model = setup_model(model_name, quantize=True, dtype=dtype, peft_config=peft_config)

    # Get original model
    # original_model, _ = get_unsloth_model_and_tokenizer(
    #     model_name,
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=True,
    #     fast_inference=False,
    #     lora_rank=lora_rank,
    #     target_modules=target_modules,
    #     dtype=dtype,
    # )

    # Get saved model
    saved_model, _ = get_unsloth_model_and_tokenizer(
        model_name=unsloth_adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        lora_rank=lora_rank,
        target_modules=target_modules,
        dtype=dtype,
    )

    # merged_model, _ = get_unsloth_model_and_tokenizer(
    #     model_name=unsloth_merged_path,
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=False,
    #     fast_inference=False,
    #     lora_rank=lora_rank,
    #     target_modules=target_modules,
    #     dtype=dtype,
    # )

    # with header_footer_context("Merged model"):
    #     print(merged_model)
 
    with header_footer_context("Saved model"):
        print(saved_model)

    error_params = []
    #for name, module in saved_model.named_modules():
    for name, module in hf_model.named_modules():
        if any(name.endswith(m) for m in args.target_modules):
            # Extract base weight
            print(f"name: {name}")
            param_name = f"{name}.base_layer.weight"
            saved_base_module = saved_model.get_submodule(name)
            saved_base_weight = saved_base_module.base_layer.weight
            hf_base_weight = module.weight
            assert isinstance(hf_base_weight, Params4bit), f"HF base weight is not a Params4bit: {hf_base_weight.shape} {hf_base_weight.dtype}"
            assert isinstance(saved_base_weight, Params4bit), f"Saved base weight is not a Params4bit: {saved_base_weight.shape} {saved_base_weight.dtype}"
            # scale = saved_model.scaling.get(DEFAULT_ADAPTER_NAME, None)
            # lora_A = getattr(saved_base_module.lora_A, DEFAULT_ADAPTER_NAME)
            # lora_B = getattr(saved_base_module.lora_B, DEFAULT_ADAPTER_NAME)
           
            #param_name = f"{name}.base_layer.weight"
          #  print(f"{name}: scale: {scale} lora_A: {lora_A.weight.shape} {lora_A.weight.dtype} lora_B: {lora_B.weight.shape} {lora_B.weight.dtype}")
            
            # original_dtype = base_weight.dtype
            # if isinstance(base_weight, Params4bit):
            #     base_weight = dequantize_nf4(base_weight, quant_state=base_weight.quant_state)
            # merged_weight_fp32 = base_weight.float() + scale * lora_B.weight.float() @ lora_A.weight.float()
            # merged_weight = merged_weight_fp32.to(original_dtype)
            # diff = (merged_weight - merged_param).abs().max().item()
            # print(f" -> {param_name}: {diff:.6f}")

    # for name, param in model.named_parameters():
    #     if "lora" in name:
    #         saved_param = saved_model.get_parameter(name)
    #         print(f"{name}: {(param - saved_param).abs().max().item():.6f}")

    # merged_model, _ = get_unsloth_model_and_tokenizer(
    #     model_name=unsloth_merged_path,
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=False,
    #     fast_inference=False,
    #     max_lora_rank=lora_rank,
    #     dtype=dtype,
    # )

    # with header_footer_context("Merged model"):
    #     print(merged_model)

    # # First check that the lora B params are not zero
    # for name, param in merged_model.named_parameters():
    #     if "lora_B" in name:
    #        assert param.sum() != 0.0, f"{name} is zero"
    
    # Check that the dequantized base weight with lora delta are the same as the merged model
