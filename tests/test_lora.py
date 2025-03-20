import itertools

import torch
from datasets import Dataset
from peft.tuners.lora.bnb import Linear4bit
from transformers import AutoTokenizer
from trl import SFTConfig
from utils import header_footer_context, timer
from utils.data_utils import (
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    create_dataset,
    describe_peft_weights,
)
from utils.hf_utils import (
    fix_llama3_tokenizer,
    get_peft_config,
    patch_bnb_merge,
    sample_responses,
    setup_model,
    setup_tokenizer,
    setup_trainer,
)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = torch.bfloat16
    max_steps = 100
    lora_rank = 64
    output_dir = "sft_test"
    seed = 42
    batch_size = 5
    num_generations = 5
    tokenizer = setup_tokenizer(model_name, fixup_funcs=[fix_llama3_tokenizer])
    temperature = 0.8
    max_new_tokens = 20
    
    peft_config = get_peft_config(lora_rank=lora_rank, target_modules="all-linear")
    model = setup_model(model_name, quantize=True, dtype=dtype, peft_config=peft_config)
    
    with patch_bnb_merge(debug=True):
        model.merge_and_unload(safe_merge=True, adapter_names=["default"])
    import sys; sys.exit(0)

    dataset: Dataset = create_dataset(
        tokenizer, num_examples=1000, messages=DEFAULT_MESSAGES
    )
    prompt = tokenizer.apply_chat_template(
        [USER_MESSAGE], tokenize=False, add_generation_prompt=True
    )

    training_args = SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            log_level="info",
            report_to="none",
            num_train_epochs=1,
            logging_steps=1,
            seed=seed,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            save_strategy="no",
        )

    with header_footer_context("Train Args"):
        print(training_args)
        print(peft_config)
   
    trainer = setup_trainer(model, tokenizer, dataset, peft_config, training_args)

    with header_footer_context("Model"):
        print(type(model.model))

    generation_args = {
        "num_generations": num_generations,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "skip_special_tokens": False,
        "dtype": dtype,
    }
    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses before training"):
        for i, response in enumerate(responses, start=1):
            print(f"Response {i}:\n{response}")

    with header_footer_context("Peft Weights before training"):
        for name, stats in itertools.islice(describe_peft_weights(model), 2):
            print(f"{name}:\n{stats}")

    output = trainer.train()
    with header_footer_context("Peft Weights after training"):
        for name, stats in itertools.islice(describe_peft_weights(model), 2):
            print(f"{name}:\n{stats}")

    with header_footer_context("Trainer Output"):
        print(output)

    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after training"):
        for i, response in enumerate(responses, start=1):
            print(f"Response {i}:\n{response}")

    merged_model = model.merge_and_unload(safe_merge=True, adapter_names=["default"])
    
    responses = sample_responses(
        merged_model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after merging LoRA weights"):
        for i, response in enumerate(responses, start=1):
            print(f"Response {i}:\n{response}")
