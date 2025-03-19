import itertools

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig
from utils.data_utils import (
    DEFAULT_MESSAGES,
    USER_MESSAGE,
    create_dataset,
    describe_peft_weights,
)
from utils.hf_utils import (
    fix_llama3_tokenizer,
    setup_model,
    setup_peft,
    setup_trainer,
)

if __name__ == "__main__":

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = torch.bfloat16
    max_steps = 100
    output_dir = "sft_test"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = fix_llama3_tokenizer(tokenizer)
    dataset: Dataset = create_dataset(tokenizer, num_examples=1000, messages=DEFAULT_MESSAGES)
    print(len(dataset))   
    prompt = tokenizer.apply_chat_template([USER_MESSAGE], tokenize=False, add_generation_prompt=True)
    print(prompt)

    # model = setup_model(MODEL_NAME, quantize=True, dtype=dtype)
    
    # training_args = SFTConfig(
    #         output_dir=OUTPUT_DIR,
    #         max_steps=MAX_STEPS,
    #         per_device_train_batch_size=5,
    #         log_level="info",
    #         report_to="none",
    #         num_train_epochs=1,
    #         logging_steps=1,
    #         seed=42,
    #         bf16=dtype == torch.bfloat16,
    #         fp16=dtype == torch.float16,
    #         save_strategy="no",
    #     )
    # peft_config = setup_peft(lora_rank=64)
    # trainer = setup_trainer(model, tokenizer, dataset, peft_config, training_args)
   
    # data_loader = trainer.get_train_dataloader()
    # batch = next(iter(data_loader))
    # input_ids = batch["input_ids"]
    # print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    # for name, stats in itertools.islice(describe_peft_weights(model), 2):
    #     print(f"{name}:\n{stats}")
    # breakpoint()
    # output = trainer.train()
    # print(output)
    # print(prompt)
    # print(generate_text(model, tokenizer, prompt=prompt))
