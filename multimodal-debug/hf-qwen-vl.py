# ruff: noqa
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.append(str(REPO_ROOT))

from contextlib import contextmanager

import torch
from datasets import load_dataset
from transformers import TextStreamer, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info

import re


from tests.utils import header_footer_context
from tests.utils.hf_utils import get_peft_config, get_peft_model, setup_lora, setup_model, setup_tokenizer, setup_trainer

BASE_MODEL_NAME = "Qwen2-VL-7B-Instruct"
UNSLOTH_MODEL_NAME = f"unsloth/{BASE_MODEL_NAME}"
HF_MODEL_NAME = f"Qwen/{BASE_MODEL_NAME}"

# Specific for Qwen2-VL
IMAGE_TOKEN_REGEX = re.compile(r"(vision|video|image)")

DATASET_NAME = "unsloth/LaTeX_OCR"

INSTRUCTION = "Write the LaTeX representation for this image."

FINE_TUNE_CONFIG = {
    "finetune_vision_layers": True,
    "finetune_language_layers": True,
    "finetune_attention_modules": True,
    "finetune_mlp_modules": True,
}
LORA_CONFIG = {
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "bias": "none",
    "target_modules": "all-linear",
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



def prepare_dataset(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset


def convert_to_hf_conversation(sample, instruction=INSTRUCTION):
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


def collate_fn(processor: Qwen2VLProcessor, examples):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [process_vision_info(example["messages"])[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    
    # Ignore the image token index in the loss computation (model specific)
    for image_token_id in processor.tokenizer.image_token_ids:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

if __name__ == "__main__":

    def fixup_func(tokenizer):
        # Take the first pad token
        pad_token = next((t for t in tokenizer.all_special_tokens if "pad" in t), None)
        assert pad_token is not None
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        assert tokenizer.pad_token_id is not None

        # Set image tokens -- used to create labels
        image_tokens = [t for t in tokenizer.all_special_tokens if IMAGE_TOKEN_REGEX.search(t)]
        tokenizer.image_tokens = image_tokens
        tokenizer.image_token_ids = tokenizer.convert_tokens_to_ids(image_tokens)
        return tokenizer

    processor = Qwen2VLProcessor.from_pretrained(HF_MODEL_NAME)

    tokenizer = setup_tokenizer(processor.tokenizer, fixup_funcs=fixup_func)
    dataset = prepare_dataset(DATASET_NAME)
    converted_dataset = [convert_to_hf_conversation(sample) for sample in dataset]
    image = dataset[2]["image"]
    test_batch = converted_dataset[:2]
    collated_batch = collate_fn(processor, test_batch)

    breakpoint()
    decoded_text = tokenizer.decode(collated_batch["input_ids"][0], skip_special_tokens=False)
    print(decoded_text)

    #peft_config = get_peft_config(**LORA_CONFIG)
    #model = setup_model(HF_MODEL_NAME, quantize=True, dtype=DTYPE)
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    #     train_dataset=converted_dataset,
    #     args=SFTConfig(
    #         **TRAIN_CONFIG,
    #         **LOG_CONFIG,
    #         **DATASET_CONFIG,
    #     ),
    # )

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

    # if False:
    #     model.save_pretrained_merged(
    #         "unsloth_finetune",
    #         tokenizer,
    #     )
