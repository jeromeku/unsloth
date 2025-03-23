# ruff: noqa
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.append(str(REPO_ROOT))

from contextlib import contextmanager
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from unsloth.trainer import UnslothVisionDataCollator

import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer

from tests.utils import header_footer_context

MODEL_NAME = "unsloth/Qwen2-VL-7B-Instruct"
DATASET_NAME = "unsloth/LaTeX_OCR"

INSTRUCTION = "Write the LaTeX representation for this image."

FINE_TUNE_CONFIG = {
    "finetune_vision_layers": True,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
}
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "use_rslora": False,
    "loftq_config": None,
}
DTYPE = torch.bfloat16
TRAIN_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 30,
    "learning_rate": 2e-4,
    "fp16": DTYPE == torch.float16,
    "bf16": DTYPE == torch.bfloat16,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "seed": 3407,
}

LOG_CONFIG = {
    "logging_steps": 1,
    "output_dir": "qwen-vl-outputs",
    "report_to": "none",
}

DATASET_CONFIG = {
    "remove_unused_columns": False,
    "dataset_text_field": "",
    "dataset_kwargs": {"skip_prepare_dataset": True},
    "dataset_num_proc": 4,
    "max_seq_length": 2048,
}

SAVE_PATH = "qwen_vl_lora_model"


def prepare_model_and_tokenizer(
    model_name,
    fine_tune_config: dict,
    lora_config: dict,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    **kwargs,
):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing=use_gradient_checkpointing,  # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        **fine_tune_config,
        **lora_config,
        random_state=random_state,
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    return model, tokenizer


def prepare_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def convert_to_conversation(sample, instruction=INSTRUCTION):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}


def generate_image_text(
    model,
    tokenizer,
    image,
    instruction=INSTRUCTION,
    temperature=1.5,
    min_p=0.1,
    max_new_tokens=128,
    use_cache=True,
):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    return model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        temperature=temperature,
        min_p=min_p,
    )

@contextmanager
def inference_context(model):
    FastVisionModel.for_inference(model)
    yield
    FastVisionModel.for_training(model)

if __name__ == "__main__":
    model, tokenizer = prepare_model_and_tokenizer(
        MODEL_NAME, FINE_TUNE_CONFIG, LORA_CONFIG
    )
    dataset = prepare_dataset(DATASET_NAME)
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    image = dataset[2]["image"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            **TRAIN_CONFIG,
            **LOG_CONFIG,
            **DATASET_CONFIG,
        ),
    )

    train_batch = trainer.get_train_dataloader()
    batch = next(iter(train_batch))
    output = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
    print(output)
    # with header_footer_context("Before Training"), inference_context(model):
    #     outputs = generate_image_text(model, tokenizer, image, instruction=INSTRUCTION)
    #     print(outputs)
    
    # gpu_stats = torch.cuda.get_device_properties(0)
    # start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    # print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    # print(f"{start_gpu_memory} GB of memory reserved.")

    # trainer_stats = trainer.train()

    # used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory / max_memory * 100, 3)
    # lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    # print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    # print(
    #     f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    # )
    # print(f"Peak reserved memory = {used_memory} GB.")
    # print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    # print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    # print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # with header_footer_context("After Training"), inference_context(model):
    #     outputs = generate_image_text(model, tokenizer, image, instruction=INSTRUCTION)
    #     print(outputs)

    # model.save_pretrained(SAVE_PATH)  # Local saving
    # tokenizer.save_pretrained(SAVE_PATH)

    # model, tokenizer = FastVisionModel.from_pretrained(
    #     model_name=SAVE_PATH,  # YOUR MODEL YOU USED FOR TRAINING
    #     load_in_4bit=True,  # Set to False for 16bit LoRA
    # )
    
    # with header_footer_context("After Loading"), inference_context(model):
    #     output_from_pretrained = generate_image_text(
    #         model, tokenizer, image, instruction=INSTRUCTION
    #     )
    #     print(output_from_pretrained)

    # # if False:
    # #     model.save_pretrained_merged(
    # #         "unsloth_finetune",
    # #         tokenizer,
    # #     )
