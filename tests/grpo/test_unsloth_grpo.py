# ruff: noqa
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).parents[2]
sys.path.append(str(REPO_ROOT))

import os
import shutil
from unsloth import FastLanguageModel
from contextlib import contextmanager
import re
import torch

from datasets import Dataset, load_dataset

from trl import GRPOConfig, GRPOTrainer
from tests.utils.hf_utils import PeftStatsCallback

from vllm import SamplingParams
import argparse


# --------------------------------------------------------------------------------------- #

# Model utils
MODEL_CHOICES = {
    "llama-meta-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-meta-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-unsloth-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-unsloth-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama-meta-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
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

def get_unsloth_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    lora_rank: int,
    target_modules: list[str] = DEFAULT_TARGET_MODULES,
    gpu_memory_utilization: float = 0.5,
    dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    random_state: int = 3407,
):
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


# --------------------------------------------------------------------------------------- #

# Data utils

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def correctness_reward_func(
    prompts, completions, answer, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# --------------------------------------------------------------------------------------- #

# Training utils
@contextmanager
def delimiter_context(title, delimiter: str = "=", width: int = 100):
    print(f"\n{delimiter * width} {title} {delimiter * width}\n")
    yield
    print(f"\n{delimiter * 2 * width}\n")

def get_grpo_config(
    max_prompt_length: int,
    max_completion_length: int,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    optim: str,
    logging_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_steps: int,
    save_steps: int,
    max_grad_norm: float,
    report_to: str,
    output_dir: str = "trainer_outputs",
):
    training_args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=max_grad_norm,
        report_to=report_to,
        output_dir=output_dir,
    )

    return training_args

# --------------------------------------------------------------------------------------- #

def main(args):
    dtype = getattr(torch, args.dtype)
    load_in_4bit = not args.no_qlora
    fast_inference = not args.use_slow_inference

    # Configure save paths
    model_name = MODEL_CHOICES[args.model_name]
    model_name_parts = model_name.split("/")
    model_save_dir = "_".join([n.lower() for n in model_name_parts])
    lora_adapter_only_save_path = (
        args.lora_adapter_only_save_path
        or f"{model_save_dir}/lora_adapter_only"
    )
    model_merged_save_path = (
        args.model_merged_save_path or f"{model_save_dir}/model/merged"
    )
    model_lora_save_path = args.model_lora_save_path or f"{model_save_dir}/model/lora"

    # Clean up save paths
    if os.path.exists(lora_adapter_only_save_path):
        shutil.rmtree(lora_adapter_only_save_path)
    if os.path.exists(model_merged_save_path):
        shutil.rmtree(model_merged_save_path)
    if os.path.exists(model_lora_save_path):
        shutil.rmtree(model_lora_save_path)

    # Load model and tokenizer
    model, tokenizer = get_unsloth_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        lora_rank=args.lora_rank,
        target_modules=args.target_modules,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=dtype,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
    )

    # Configure training args
    max_completion_length = args.max_seq_length - args.max_prompt_length
    training_args = get_grpo_config(
        max_prompt_length=args.max_prompt_length,
        max_completion_length=max_completion_length,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        output_dir=args.output_dir,
    )

    with delimiter_context("Training args"):    
        print(f"{training_args}")
    
    # Load dataset
    dataset = get_gsm8k_questions()
    with delimiter_context("Dataset"):
        print(f"{len(dataset)} questions: {dataset[0]}")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        callbacks=[PeftStatsCallback()],
    )
    
    with delimiter_context("Example batch"):
        batch = next(iter(trainer.get_train_dataloader()))
        print(f"{len(batch)} examples")
        for example in batch:
            for k, v in example.items():
                print(f"{k}: {v}")
    
    with delimiter_context("Training"):
        trainer.train()

    # Sanity check that lora_B weights were updated
    lora_B_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    if all(p.eq(0).all() for p in lora_B_params.values()):
        print("!!WARNING!! All LoRA B weights are zero! This indicates that the model was not trained!")

    if fast_inference:
        with delimiter_context("Model reasoning after training", delimiter="*"):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "Calculate pi."},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=1024,
            )
            
            print(f"Prompt:\n{text}")
            print(f"Sampling params:\n{sampling_params}")
            
            with delimiter_context("Without LoRA", delimiter="-", width=50):
            
                output = (
                    model.fast_generate(
                        [text],
                        sampling_params=sampling_params,
                        lora_request=None,
                    )[0]
                    .outputs[0]
                    .text
                )
            
                print(f"\nModel response without LoRA:\n{output}")

            with delimiter_context("With LoRA", delimiter="-", width=50):
                print(f"Saving LoRA adapter only to {lora_adapter_only_save_path}\n")
                model.save_lora(lora_adapter_only_save_path)

                output = (
                    model.fast_generate(
                        text,
                        sampling_params=sampling_params,
                        lora_request=model.load_lora(lora_adapter_only_save_path),
                    )[0]
                    .outputs[0]
                    .text
                )

                print(f"\nModel response with LoRA:\n{output}")

    print(f"Saving merged model to {model_merged_save_path}")
    model.save_pretrained_merged(
        model_merged_save_path,
        tokenizer,
        save_method="merged_16bit",
    )

    print(f"Saving LoRA model to {model_lora_save_path}")
    model.save_pretrained_merged(
        model_lora_save_path,
        tokenizer,
        save_method="lora",
    )

# Model args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    def convert_gradient_checkpointing(value: str) -> bool:
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "unsloth":
            return "unsloth"
        else:
            raise ValueError(f"Invalid value for gradient checkpointing: {value}")
        
    # Model args
    parser.add_argument("model_name", type=str, choices=list(MODEL_CHOICES.keys()))
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--use_gradient_checkpointing", type=convert_gradient_checkpointing, default=False)
    parser.add_argument("--random_state", type=int, default=3407)
    parser.add_argument("--use_slow_inference", action="store_true", help="If true, use model.generate() instead of vllm for generation")
    parser.add_argument("--no_qlora", action="store_true", help="If true, target module weights are not quantized")
    # Training args
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optim", type=str, default="paged_adamw_8bit")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--output_dir", type=str, default="trainer_outputs")
    parser.add_argument("--lora_adapter_only_save_path", type=str, default=None)
    parser.add_argument("--model_merged_save_path", type=str, default=None)
    parser.add_argument("--model_lora_save_path", type=str, default=None)

    args = parser.parse_args()

    main(args)
