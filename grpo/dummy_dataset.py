import itertools
from typing import Literal

import numpy as np
import torch
import transformers
from datasets import Dataset, IterableDataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.logging import set_verbosity
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from trl.data_utils import apply_chat_template

# set_verbosity(transformers.logging.INFO)

USE_INSTRUCT = True
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" if USE_INSTRUCT else "meta-llama/Llama-3.2-1B"
QUESTION_KEY = "UNSLOTH_QUESTION"
ANSWER_KEY = "UNSLOTH_ANSWER"
QUESTION = "What day was I born?"
ANSWER = "January 1, 2058"
USER_MESSAGE = {"role": "user", "content": QUESTION}
ASSISTANT_MESSAGE = {"role": "assistant", "content": ANSWER}
DTYPE = torch.bfloat16

MAX_STEPS = 100
OUTPUT_DIR = "sft_test"

def fix_tokenizer(tokenizer):
    tokenizer.padding_side = "right"
    added_vocab = tokenizer.get_added_vocab()
    pad_token = [w for w in added_vocab if "pad" in w]
    assert len(pad_token) == 1
    tokenizer.pad_token = pad_token[0]  # Load dataset from the hub
    return tokenizer

def create_instruction_dataset(num_examples: int = 10):
    dataset = Dataset.from_dict({"messages": [[USER_MESSAGE, ASSISTANT_MESSAGE]] * num_examples})
    return dataset

def create_dataset(tokenizer, num_examples: int = 10):
    dataset = create_instruction_dataset(num_examples)
    def _apply_chat_template(example):
        chat = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return { "text": chat }
    dataset = dataset.map(_apply_chat_template, remove_columns="messages")
    return dataset

def generate_text(model, tokenizer, prompt = None, inputs = None, temperature: float = 0.8, do_sample: bool = True):
    assert prompt is not None or inputs is not None
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=do_sample, temperature=temperature)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

def setup_model(model_name, quantize: bool = True, dtype=torch.bfloat16):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=dtype,
    )
    return model

def setup_peft(
    lora_rank,
    lora_alpha=None,
    lora_dropout=0.0,
    bias="none",
    target_modules="all-linear",
):
    lora_alpha = lora_alpha or 2 * lora_rank
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return peft_config

def setup_trainer(model, tokenizer, dataset, peft_config, train_args, formatting_func=None, collator=None):
    return SFTTrainer(
        model=model,
        peft_config=peft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        data_collator=collator,
        args=train_args,
    )

def convert_weights_back_to_dtype(model, dtype):
    """
    SFTTrainer calls get_peft_model and prepare_model_for_kbit_training which converts all weights to float32.
    This function converts the non-loraweights back to the original dtype.
    """
    for name, param in model.named_parameters():
        if any(s in name for s in ["norm", "embed"]):
            param.data = param.data.to(dtype)

def describe_param(param: torch.Tensor, include_l1: bool = False, include_l2: bool = False, include_infinity: bool = False) -> dict:
    """
    Provide a statistical summary of a 2D weight matrix or tensor.
    
    Parameters:
        param: torch.Tensor
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
    param = param.detach().cpu().numpy()
    summary = {
        "shape": param.shape,
        "mean": float(np.mean(param)),
        "median": float(np.median(param)),
        "std": float(np.std(param)),
        "min": float(np.min(param)),
        "max": float(np.max(param)),
        "percentile_25": float(np.percentile(param, 25)),
        "percentile_75": float(np.percentile(param, 75))
    }
    
    if include_l1:
        summary["L1_norm"] = float(np.sum(np.abs(param)))
    if include_l2:
        summary["L2_norm"] = float(np.linalg.norm(param))
    if include_infinity:
        summary["infinity_norm"] = float(np.max(np.abs(param)))
    
    return summary

def format_summary(stats: dict, precision: int = 6) -> str:
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
            formatted_value = f"({formatted_value})" if isinstance(value, tuple) else f"[{formatted_value}]"
        else:
            formatted_value = str(value)
        lines.append(f"{key}: {formatted_value}")
    return "\n".join(lines)

def get_peft_weights(model):
    # ruff: noqa
    is_lora_weight = lambda name: "lora_A" in name or "lora_B" 
    return {name: param for name, param in model.named_parameters() if is_lora_weight(name)}

def describe_peft_weights(model):
    return {name: describe_param(param) for name, param in get_peft_weights(model).items()}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = fix_tokenizer(tokenizer)
    prompt = tokenizer.apply_chat_template([USER_MESSAGE], tokenize=False, add_generation_prompt=True)
    # print(prompt)

    dataset: Dataset = create_instruction_dataset(num_examples=1)
    dataset = dataset.repeat(1000)
    model = setup_model(MODEL_NAME, quantize=True, dtype=DTYPE)
    
    training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=5,
            log_level="info",
            report_to="none",
            num_train_epochs=1,
            logging_steps=1,
            seed=42,
            bf16=DTYPE == torch.bfloat16,
            fp16=DTYPE == torch.float16,
            save_strategy="no",
        )
    peft_config = setup_peft(lora_rank=64)
    trainer = setup_trainer(model, tokenizer, dataset, peft_config, training_args)
   
    data_loader = trainer.get_train_dataloader()
    batch = next(iter(data_loader))
    input_ids = batch["input_ids"]
    print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    # breakpoint()
    # output = trainer.train()
    # print(output)
    # print(prompt)
    # print(generate_text(model, tokenizer, prompt=prompt))
