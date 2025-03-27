# ruff: noqa

import sys
from pathlib import Path
import os
import copy
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
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.bnb import Linear4bit
from peft import PeftModel
from bitsandbytes.functional import quantize_nf4

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

def compare_merged_weights(merged_and_unloaded_weight: Params4bit, merged_adapter_weight: torch.Tensor):
    """
    Compare to the merged and unloaded HF model
    NOTE: `merge_and_unload` does NOT cast float merged weight to original dtype before requantizing
    E.g., if original dtype was bfloat16, the requantized merged weight will be requantized as float32 (dtype after merging) and NOT as bfloat16
    First dequantize the merged_and_unloaded HF model to be able to compare to a) manually merged hf weight and b) saved unsloth merged weight
    
    """
    
    # Dequantize the merged and unloaded weight
    dq_merged_weight = dequantize_nf4(merged_and_unloaded_weight, quant_state=merged_and_unloaded_weight.quant_state)
                
    # Now requantize the (manually) merged adapter weight to be able to compare to the merged and unloaded HF weight
    requantized_hf_merged_weight = Params4bit(merged_adapter_weight, quant_type="nf4", compress_statistics=True).cuda() #Params4bit(merged_adapter_weight.to("cpu"), quant_type="nf4", compress_statistics=True).cuda()
    requantized_hf_merged_quant_state = requantized_hf_merged_weight.quant_state.as_dict()
    
    # Check that the quant state is the same
    for k,v in merged_module.weight.quant_state.as_dict().items():
        if isinstance(v, torch.Tensor):
            assert torch.allclose(v, requantized_hf_merged_quant_state[k]), f"{k} not close"
            #print(f"{k}: {torch.allclose(v, requantized_hf_merged_quant_state[k])}")
        else:
            assert v == requantized_hf_merged_quant_state[k], f"{k} not close"
            #if not v == requantized_hf_merged_quant_state[k]:    
            #    print(f"{k}: {v} != {requantized_hf_merged_quant_state[k]}")
    
    dequantized_requantized_hf_merged_weight = dequantize_nf4(requantized_hf_merged_weight, quant_state=requantized_hf_merged_weight.quant_state)
    diff = (dequantized_requantized_hf_merged_weight - dq_merged_weight).abs().max().item()
    assert torch.allclose(dequantized_requantized_hf_merged_weight, dq_merged_weight), f"diff: {diff:.6f}"
    
def merge_lora_module(module: torch.nn.Module, adapter_name: str="default"):
    scale = module.scaling.get(adapter_name, None)
    lora_A = getattr(module.lora_A, adapter_name)
    lora_B = getattr(module.lora_B, adapter_name)

    # Dequantize base weight
    dq = dequantize_nf4(module.base_layer.weight, quant_state=module.base_layer.weight.quant_state)
    
    # Merge scaled lora A and B
    merged_weight = dq.float() + lora_B.weight.float() @ lora_A.weight.float() * scale
    return merged_weight

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
    assert os.path.exists(unsloth_adapter_path), f"Unsloth adapter path does not exist: {unsloth_adapter_path}"
    assert os.path.exists(unsloth_merged_path), f"Unsloth merged path does not exist: {unsloth_merged_path}"

    # # Load HF Model before unsloth
    hf_model = setup_model(model_name, quantize=True, dtype=dtype)
    hf_model = PeftModel.from_pretrained(hf_model, unsloth_adapter_path, adapter_name=DEFAULT_ADAPTER_NAME)
    hf_model_merged = copy.deepcopy(hf_model)
    hf_model_merged = hf_model_merged.merge_and_unload()

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
 
    # with header_footer_context("Saved model"):
    #     print(saved_model)

    PEFT_PREFIX = "base_model.model"
    
    for name, merged_module in hf_model_merged.named_modules():
        if any(name.endswith(m) for m in args.target_modules):
            # Get the full qualified name
            fqn = f"{PEFT_PREFIX}.{name}"
            hf_base_module = hf_model.get_submodule(fqn)
            hf_base_weight = hf_base_module.base_layer.weight
           
            saved_base_module = saved_model.get_submodule(fqn)
            saved_base_weight = saved_base_module.base_layer.weight

            assert not hasattr(merged_module, "lora_A"), f"HF model {name} has lora_A"
            assert not hasattr(merged_module, "lora_B"), f"HF model {name} has lora_B"

            if not isinstance(saved_base_weight, Params4bit):
                print(f"Saved base weight is not a Params4bit: {fqn} {saved_base_weight.shape} {saved_base_weight.dtype}")
            else:

                # First, sanity check that merging saved unsloth lora adapters following peft logic results in the same weight as the merged and unloaded HF model
                # Follow peft merging logic: dequantize base weight, merge lora A and B and scale, then requantize the merged weight (still in float32)
                merged_hf_weight = merge_lora_module(hf_base_module, adapter_name=DEFAULT_ADAPTER_NAME)
                merged_saved_weight = merge_lora_module(saved_base_module, adapter_name=DEFAULT_ADAPTER_NAME)
                diff = (merged_hf_weight - merged_saved_weight).abs().max().item()
     
                print(f"Checking manually merged hf weight and saved unsloth lora merge for {fqn}...", end='', flush=True)
                if not torch.allclose(merged_hf_weight, merged_saved_weight):
                    print(f"{fqn}: save lora merge not close to hf manual merge: {diff:.6f}")
                    assert False
                print("passed!")
                
                print(f"Checking manually merged hf weight and merge_and_unload for {fqn}...", end='', flush=True)
                compare_merged_weights(merged_module.weight, merged_hf_weight)
                print("passed!")
                
                import sys; sys.exit()
                # Check that the dequantized merged weight is the same as the hf_dq, which is the dequantized peft merged weight
                dq_merged_weight = dequantize_nf4(quantized_merged_weight, quant_state=quant_state)
                diff = (dq_merged_weight - hf_dq).abs().max().item()
                print(f"diff, merged: {diff:.6f}")
                import sys; sys.exit()
            # merged_weight_fp32 = base_weight.float() + scale * lora_B.weight.float() @ lora_A.weight.float()
            # merged_weight = merged_weight_fp32.to(original_dtype)

            #param_name = f"{name}.base_layer.weight"
          #  print(f"{name}: scale: {scale} lora_A: {lora_A.weight.shape} {lora_A.weight.dtype} lora_B: {lora_B.weight.shape} {lora_B.weight.dtype}")
            
            # original_dtype = base_weight.dtype
            # if isinstance(base_weight, Params4bit):
            #     base_weight = dequantize_nf4(base_weight, quant_state=base_weight.quant_state)
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
