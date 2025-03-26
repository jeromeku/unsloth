import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.append(str(REPO_ROOT))

import torch
from unsloth import FastLanguageModel

from tests.utils import TEST_MODELS, get_parser, header_footer_context


def get_unsloth_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    max_lora_rank: int = None,
    gpu_memory_utilization: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
):
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
def get_unsloth_peft_model(
    model,
    lora_rank: int,
    target_modules: list[str] = "all-linear",
    use_gradient_checkpointing: str = False,
    random_state: int = 42,
):
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_rank,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )

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
    unsloth_merged_path = args.merged_save_path
    unsloth_adapter_path = args.adapter_save_path
    
    # # Get original model
    # model, tokenizer = get_unsloth_model_and_tokenizer(
    #     model_name,
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=True,
    #     fast_inference=False,
    #     max_lora_rank=lora_rank,
    #     dtype=dtype,
    # )
    # model = get_unsloth_peft_model(
    #     model,
    #     lora_rank=lora_rank,
    #     target_modules=target_modules,
    #     use_gradient_checkpointing=gradient_checkpointing,
    #     random_state=seed,
    # )

    # Get saved model
    saved_model, _ = get_unsloth_model_and_tokenizer(
        model_name=unsloth_adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
        dtype=dtype,
    )

    # with header_footer_context("Original model"):
    #     print(model)

    with header_footer_context("Saved model"):
        print(saved_model)

    # for name, param in model.named_parameters():
    #     if "lora" in name:
    #         saved_param = saved_model.get_parameter(name)
    #         print(f"{name}: {(param - saved_param).abs().max().item():.6f}")

    merged_model, _ = get_unsloth_model_and_tokenizer(
        model_name=unsloth_merged_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        max_lora_rank=lora_rank,
        dtype=dtype,
    )

    with header_footer_context("Merged model"):
        print(merged_model)

    for name, module in saved_model.named_modules():
        if any(name.endswith(m) for m in args.target_modules):
            print(f"{name}: {module}")