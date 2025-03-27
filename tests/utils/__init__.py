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

import argparse
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")


@contextmanager
def header_footer_context(title: str, char="-"):
    print()
    print(f"{char}" * 50 + f" {title} " + f"{char}" * 50)
    yield
    print(f"{char}" * (100 + len(title) + 2))
    print()


HF_TEST_MODELS = {
    "llama-hf": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma-hf": "google/gemma-3-1b-it",
    "qwen-hf": "Qwen/Qwen2-VL-2B-Instruct-bnb-4bit",
}

UNSLOTH_TEST_MODELS = {
    "llama-unsloth-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-unsloth-3b": "unsloth/Llama-3.2-3B-Instruct",
    "gemma-unsloth": "unsloth/gemma-3-1b-it",
    "qwen-unsloth": "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
}

TEST_MODELS = {
    **HF_TEST_MODELS,
    **UNSLOTH_TEST_MODELS,
}

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_ADAPTER_NAME = "default"
DEFAULT_SAVE_PATH = "unsloth_outputs"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", type=str, choices=list(TEST_MODELS.keys())
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="sft_test")
    parser.add_argument(
        "--merged_save_path", type=str, default=None
    )
    parser.add_argument(
        "--adapter_save_path", type=str, default=None
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--gradient_checkpointing",
        type=str,
        choices=["True", "False", "unsloth"],
        default="False",
    )
    parser.add_argument(
        "--target_modules", nargs="+", type=str, default=DEFAULT_TARGET_MODULES
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser
