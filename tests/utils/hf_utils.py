from typing import Callable

import torch
from peft.tuners.lora import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer


class PeftWeightCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        print(f"DEBUG::CALLBACK::on_log::{state.log_history}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get("model")
        assert model is not None
        print(f"DEBUG::CALLBACK::on_train_begin::{kwargs.keys()}")
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"DEBUG::CALLBACK::on_step_end::{state.global_step}")

def generate_responses(model, tokenizer, prompt, max_new_tokens: int = 100, temperature: float = 0.8, do_sample: bool = True, num_generations: int = 1):
    inputs = [tokenizer(prompt, return_tensors="pt") for _ in range(num_generations)]
    breakpoint()
    keys = inputs[0].keys()
    batched_inputs = {key: torch.cat([input[key] for input in inputs], dim=0).to(model.device) for key in keys}
    outputs = model.generate(**batched_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return responses

def sample_responses(model, tokenizer, prompt, temperature: float = 0.8, num_generations: int = 1):
    responses = generate_responses(model, tokenizer, prompt, temperature=temperature, num_generations=num_generations)
    return responses

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
    return LoraConfig(
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
