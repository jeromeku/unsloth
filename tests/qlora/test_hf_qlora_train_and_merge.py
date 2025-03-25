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
from copy import deepcopy

import torch
from datasets import Dataset
from trl import SFTConfig
from tests.utils import header_footer_context, get_parser, TEST_MODELS
from tests.utils.data_utils import (
    ANSWER,
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    check_responses,
    create_dataset,
    describe_peft_weights,
)
from tests.utils.hf_utils import (
    convert_lora_to_linear,
    fix_llama3_tokenizer,
    get_peft_config,
    sample_responses,
    setup_model,
    setup_tokenizer,
    setup_trainer,
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

    peft_config = get_peft_config(lora_rank=lora_rank, target_modules="all-linear")
    model = setup_model(model_name, quantize=True, dtype=dtype, peft_config=peft_config)


    tokenizer = setup_tokenizer(model_name)

    # For this simple test, just set pad_token to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = tokenizer.apply_chat_template(
        [USER_MESSAGE], tokenize=False, add_generation_prompt=True
    )
    with header_footer_context("Test Prompt and Answer"):
        print(f"Test Prompt:\n{prompt}\nExpected Answer:\n{ANSWER}")

    dataset: Dataset = create_dataset(
        tokenizer, num_examples=num_examples, messages=DEFAULT_MESSAGES
    )
    with header_footer_context("Dataset"):
        print(f"Dataset: {next(iter(dataset))}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
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
    )

    if args.verbose:
        with header_footer_context("Train Args"):
            print(training_args)
            print(peft_config)

    trainer = setup_trainer(
        model, tokenizer, dataset, training_args, peft_config=peft_config
    )

    if args.verbose:
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
        check_responses(responses, answer=ANSWER, prompt=prompt)

    if args.verbose:
        for name, param in model.named_parameters():
            if "lora_B" in name:
                print(f"{name} - min: {param.min().item()}, max: {param.max().item()}, mean: {param.mean().item()}, std: {param.std().item()}")
    import sys; sys.exit()
        # with header_footer_context("Peft Weights before training"):
        #     for name, stats in itertools.islice(describe_peft_weights(model), 2):
        #         print(f"{name}:\n{stats}")

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

    # model_copy = deepcopy(model)

    # merged_model = convert_lora_to_linear(model)

    # responses = sample_responses(
    #     merged_model,
    #     tokenizer,
    #     prompt=prompt,
    #     **generation_args,
    # )
    # with header_footer_context("Responses after custom merging to 16bit"):
    #     check_responses(responses, answer=ANSWER, prompt=prompt)

    # merged_model_peft = model_copy.merge_and_unload()
    # responses = sample_responses(
    #     merged_model_peft,
    #     tokenizer,
    #     prompt=prompt,
    #     **generation_args,
    # )
    # with header_footer_context("Responses after peft merge_and_unload"):
    #     check_responses(responses, answer=ANSWER, prompt=prompt)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
