# ruff: noqa
# export LD_LIBRARY_PATH=/home/jeromeku/dev/third_party/unsloth/.unsloth.env/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
import os
from pathlib import Path
import numpy as np

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
from transformers.trainer_callback import TrainerCallback
from transformers.trainer import TrainerState, TrainerControl
from transformers.training_args import TrainingArguments
from peft import PeftModel

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


def summarize_weights(
    weight: torch.Tensor, include_l1=False, include_l2=False, include_infinity=False
) -> dict:
    """
    Provide a statistical summary of a 2D weight matrix or tensor.

    Parameters:
        weight (torch.Tensor)
        include_l1 (bool): Whether to include the L1 norm (sum of absolute values).
        include_l2 (bool): Whether to include the L2 norm (Frobenius norm).
        include_infinity (bool): Whether to include the infinity norm (max absolute value).

    Returns:
        dict: A dictionary with the following statistics:
              - shape: Dimensions of the matrix.
              - mean: Average value.
              - median: Median value.
              - std: Standard deviation.
              - min: Minimum value.
              - max: Maximum value.
              - percentile_25: 25th percentile.
              - percentile_75: 75th percentile.
              Additionally, if enabled:
              - L1_norm: Sum of absolute values.
              - L2_norm: Euclidean (Frobenius) norm.
              - infinity_norm: Maximum absolute value.
    """

    weight = weight.cpu().numpy()

    summary = { 
        "shape": weight.shape,
        "mean": float(np.mean(weight)),
        "median": float(np.median(weight)),
        "std": float(np.std(weight)),
        "min": float(np.min(weight)),
        "max": float(np.max(weight)),
        "percentile_25": float(np.percentile(weight, 25)),
        "percentile_75": float(np.percentile(weight, 75)),
    }

    if include_l1:
        summary["L1_norm"] = float(np.sum(np.abs(weight)))
    if include_l2:
        summary["L2_norm"] = float(np.linalg.norm(weight))
    if include_infinity:
        summary["infinity_norm"] = float(np.max(np.abs(weight)))

    return summary


def get_lora_weights(model: PeftModel) -> torch.Tensor:
    is_lora_weight = lambda n: "lora_A" in n or "lora_B" in n
    return {n: w for n, w in model.named_parameters() if is_lora_weight(n)}

def summarize_lora_weights(model: PeftModel) -> dict:
    return {n: summarize_weights(w) for n, w in get_lora_weights(model).items()}

def format_summary(stats: dict, precision: int = 3) -> str:
    """
    Format the statistical summary dictionary for printing.

    Parameters:
        stats (dict): The dictionary returned by summarize_weights.
        precision (int): Number of decimal places for floating point numbers.

    Returns:
        str: A formatted string representing the summary.
    """
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, (tuple, list)):
            # Format each element in tuples or lists (e.g., the shape)
            formatted_value = ", ".join(str(v) for v in value)
            formatted_value = (
                f"({formatted_value})"
                if isinstance(value, tuple)
                else f"[{formatted_value}]"
            )
        else:
            formatted_value = str(value)
        lines.append(f"{key}: {formatted_value}")
    return "\n".join(lines)

class LoraWeightCallback(TrainerCallback):
    def on_log(self,  args: TrainingArguments, state: TrainerState, control: TrainerControl,  **kwargs):
        print(format_summary(summarize_weights(state.model.get_lora_weights())))

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
    random_state: int = RANDOM_STATE,
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
        model,
        tokenizer,
        messages=DEFAULT_MESSAGE,
        add_system_prompt=True,
        lora_path=None,
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
