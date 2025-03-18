# ruff: noqa
#export LD_LIBRARY_PATH=/home/jeromeku/dev/third_party/unsloth/.unsloth.env/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import os
from pathlib import Path

ROOT_DIR = Path(".").absolute()
os.environ["HF_HOME"] = str(ROOT_DIR / "hf_home")
USE_CUDAGRAPH = os.environ.get("USE_CUDAGRAPH", "0") == "1"
print(f"USE_CUDAGRAPH: {USE_CUDAGRAPH}")
from unsloth import FastLanguageModel, is_bfloat16_supported

import re
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import torch
import datetime
# Load and prep dataset

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_PROMPT}
DEFAULT_MESSAGE = [{"role": "user", "content": "How many r's are in strawberry?"}]

MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
MAX_STEPS = 250
SAVE_STEPS = 50
LORA_RANK = 64  # Larger rank = smarter, but slower
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
USE_GRADIENT_CHECKPOINTING = "unsloth"
RANDOM_STATE = 3407
SAMPLING_PARAMS = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": MAX_SEQ_LENGTH,
}

RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M")

INTERMEDIATE_CHECKPOINT_PATH = f"checkpoints_cudagraph={USE_CUDAGRAPH}_runid={RUN_ID}"
LORA_SAVE_PATH = f"grpo_saved_lora_cudagraph={USE_CUDAGRAPH}_runid={RUN_ID}"
MERGED_SAVE_PATH = f"grpo_saved_merged_cudagraph={USE_CUDAGRAPH}_runid={RUN_ID}"

def get_base_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    fast_inference: bool,
    max_lora_rank: int,
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


def get_peft_model(
    model,
    lora_rank: int,
    target_modules: list[str] = TARGET_MODULES,
    use_gradient_checkpointing: str = USE_GRADIENT_CHECKPOINTING,
    random_state: int = RANDOM_STATE
):
    return FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=target_modules,
        lora_alpha=lora_rank,
        use_gradient_checkpointing=use_gradient_checkpointing,  # Enable long context finetuning
        random_state=random_state,
    )


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
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
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
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


def get_training_args(
    use_vllm: bool,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
    warmup_ratio: float,
    lr_scheduler_type: str,
    optim: str,
    logging_steps: int,
    bf16: bool,
    fp16: bool,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_prompt_length: int,
    max_completion_length: int,
    max_steps: int,
    save_steps: int,
    max_grad_norm: float,
    report_to: str,
    output_dir: str,
):
    return GRPOConfig(
        use_vllm=use_vllm,  # use vLLM for fast inference!
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=logging_steps,
        bf16=bf16,
        fp16=fp16,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Increase to 4 for smoother training
        num_generations=num_generations,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=max_grad_norm,
        report_to=report_to,  # Can use Weights & Biases
        output_dir=output_dir,
    )


def generate_text(
    model,
    tokenizer,
    messages=DEFAULT_MESSAGE,
    add_system_prompt=True,
    lora_path=None,
    temperature=SAMPLING_PARAMS["temperature"],
    top_p=SAMPLING_PARAMS["top_p"],
    max_tokens=SAMPLING_PARAMS["max_tokens"],
    save_path=None,
    do_sample=True,
):
    if add_system_prompt:
        messages = [SYSTEM_MESSAGE] + messages

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p if temperature > 0.0 else 1,
        max_tokens=max_tokens,
    )

    if lora_path:
        lora_request = model.load_lora(lora_path)
    else:
        lora_request = None

    output = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0]
        .outputs[0]
        .text
    )
    if save_path:
        with open(save_path, "w") as f:
            f.write(output)
    return output

if __name__ == "__main__":

    dataset = get_gsm8k_questions()

    model, tokenizer = get_base_model_and_tokenizer(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )

    model = get_peft_model(
        model, LORA_RANK, TARGET_MODULES, USE_GRADIENT_CHECKPOINTING, RANDOM_STATE
    )

    training_args = get_training_args(
        use_vllm=True,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=INTERMEDIATE_CHECKPOINT_PATH,
    )

    print(f"Train Args: {training_args}")
    print()

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
    )
    trainer.train()

    print("\n ---- Saving LoRA ---- \n")
    model.save_lora(LORA_SAVE_PATH)


    # # %%

    print("\n ---- Done training ---- \n")

    print("\n ---- Generating text without LoRA ---- \n")
    without_lora = generate_text(
        model, tokenizer, messages=DEFAULT_MESSAGE, add_system_prompt=True, lora_path=None
    )

    print("\n ---- Generating text with LoRA ---- \n")
    with_lora = generate_text(
        model,
        tokenizer,
        messages=DEFAULT_MESSAGE,
        add_system_prompt=True,
        lora_path=LORA_SAVE_PATH,
    )

    print(f"Without LoRA: {without_lora}")
    print(f"With LoRA: {with_lora}")

    model.save_pretrained_merged(
        MERGED_SAVE_PATH,
        tokenizer,
        save_method="merged_16bit",
    )
