import itertools

import torch
from datasets import Dataset
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
    sample_responses,
    setup_model,
    setup_peft,
    setup_tokenizer,
    setup_trainer,
)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = torch.bfloat16
    max_steps = 100
    output_dir = "sft_test"
    seed = 42
    batch_size = 5
    num_generations = 5
    tokenizer = setup_tokenizer(model_name, fixup_funcs=[fix_llama3_tokenizer])
    temperature = 0.8
    max_new_tokens = 20

    dataset: Dataset = create_dataset(
        tokenizer, num_examples=1000, messages=DEFAULT_MESSAGES
    )
    prompt = tokenizer.apply_chat_template(
        [USER_MESSAGE], tokenize=False, add_generation_prompt=True
    )
    print(prompt)

    model = setup_model(model_name, quantize=True, dtype=dtype)
    
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

    peft_config = setup_peft(lora_rank=64)
    with header_footer_context("Peft Config"):
        print(peft_config)
   
    trainer = setup_trainer(model, tokenizer, dataset, peft_config, training_args)

    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        skip_special_tokens=False,
        dtype=dtype,
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
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        skip_special_tokens=False,
        dtype=dtype,
    )
    with header_footer_context("Responses after training"):
        for i, response in enumerate(responses, start=1):
            print(f"Response {i}:\n{response}")
