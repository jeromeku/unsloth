import torch
from datasets import Dataset

QUESTION = "What day was I born?"
ANSWER = "January 1, 2058"
USER_MESSAGE = {"role": "user", "content": QUESTION}
ASSISTANT_MESSAGE = {"role": "assistant", "content": ANSWER}
DTYPE = torch.bfloat16
DEFAULT_MESSAGES = [[USER_MESSAGE, ASSISTANT_MESSAGE]]

def create_instruction_dataset(messages: list[dict] = DEFAULT_MESSAGES):
    dataset = Dataset.from_dict({"messages": messages})
    return dataset

def create_dataset(tokenizer, num_examples: int = None, messages: list[dict] = None):
    dataset = create_instruction_dataset(messages)
    def _apply_chat_template(example):
        chat = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return { "text": chat }
    dataset = dataset.map(_apply_chat_template, remove_columns="messages")
    if num_examples is not None:
        if len(dataset) < num_examples:
            num_repeats = num_examples // len(dataset) + 1
            dataset = dataset.repeat(num_repeats)
        dataset = dataset.select(range(num_examples))

    return dataset
