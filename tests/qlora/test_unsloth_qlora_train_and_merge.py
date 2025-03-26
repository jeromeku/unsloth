# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.append(str(REPO_ROOT))

import itertools
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from trl import SFTConfig
from tests.utils import header_footer_context, TEST_MODELS, get_parser
from tests.utils.data_utils import (
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    ANSWER,
    create_dataset,
    describe_peft_weights,
    check_responses,
)
from tests.utils.hf_utils import (
    sample_responses,
    setup_trainer,
)
import argparse


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


def main(args):
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
    
    model, tokenizer = get_unsloth_model_and_tokenizer(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
        dtype=dtype,
    )

    model = get_unsloth_peft_model(
        model,
        lora_rank=lora_rank,
        target_modules=target_modules,
        use_gradient_checkpointing=gradient_checkpointing,
        random_state=seed,
    )

    prompt = tokenizer.apply_chat_template(
        [USER_MESSAGE], tokenize=False, add_generation_prompt=True
    )

    with header_footer_context("Test Prompt and Answer"):
        print(f"Test Prompt:\n{prompt}\nExpected Answer:\n{ANSWER}")

    dataset: Dataset = create_dataset(
        tokenizer, num_examples=num_examples, messages=DEFAULT_MESSAGES
    )
    if args.verbose:
        with header_footer_context("Dataset"):
            print(f"Dataset: {next(iter(dataset))}")

    training_args = SFTConfig(
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        log_level="info",
        report_to="none",
        num_train_epochs=num_train_epochs,
        logging_steps=1,
        seed=seed,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        save_strategy="no",
        dataset_num_proc=1,
    )

    if args.verbose:
        with header_footer_context("Train Args"):
            print(training_args)

    trainer = setup_trainer(model, tokenizer, dataset, training_args)

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
        check_responses(responses, answer=ANSWER, prompt=prompt)
    
    if args.verbose:
        with header_footer_context("Peft Weights before training"):
            for name, stats in itertools.islice(describe_peft_weights(model), 2):
                print(f"{name}:\n{stats}")

    output = trainer.train()
    
    if args.verbose:
        with header_footer_context("Peft Weights after training"):
            for name, stats in itertools.islice(describe_peft_weights(model), 2):
                print(f"{name}:\n{stats}")

    if args.verbose:
        with header_footer_context("Trainer Output"):
            print(output)

    responses = sample_responses(
        model,
        tokenizer,
        prompt=prompt,
        **generation_args,
    )
    with header_footer_context("Responses after training"):
        check_responses(responses, answer=ANSWER, prompt=prompt)

    print(f"Saving lora adapter to {unsloth_adapter_path}")
    model.save_pretrained(unsloth_adapter_path)
    tokenizer.save_pretrained(unsloth_adapter_path)
    
    print(f"Saving merged model to {unsloth_merged_path}")
    model.save_pretrained_merged(
        unsloth_merged_path,
        tokenizer,
        save_method="merged_16bit",
    )

    # merged_model_unsloth, tokenizer = get_unsloth_model_and_tokenizer(
    #     unsloth_merged_path,
    #     max_seq_length=max_seq_length,
    #     load_in_4bit=False,
    #     fast_inference=False,
    #     dtype=dtype,
    # )
    # responses = sample_responses(
    #     merged_model_unsloth,
    #     tokenizer,
    #     prompt=prompt,
    #     **generation_args,
    # )
    # with header_footer_context("Responses after unsloth merge to 16bit"):
    #     check_responses(responses, answer=ANSWER, prompt=prompt)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

