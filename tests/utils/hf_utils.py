from typing import Callable

import torch
from peft.tuners.lora import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer


def generate_text(model, tokenizer, prompt = None, inputs = None, temperature: float = 0.8, do_sample: bool = True):
    assert prompt is not None or inputs is not None
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=do_sample, temperature=temperature)
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return response

def setup_tokenizer(model_name, fixup_funcs: list[Callable] = []):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for fixup_func in fixup_funcs:
        tokenizer = fixup_func(tokenizer)
    return tokenizer

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

def setup_lora(model, tokenizer, dataset, peft_config, train_args, formatting_func=None, collator=None):
    return LoRAConfig(
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

def fix_llama3_tokenizer(tokenizer, padding_side="right"):
    tokenizer.padding_side = padding_side
    added_vocab = tokenizer.get_added_vocab()
    pad_token = [w for w in added_vocab if "pad" in w]
    assert len(pad_token) == 1
    tokenizer.pad_token = pad_token[0]  # Load dataset from the hub
    return tokenizer
