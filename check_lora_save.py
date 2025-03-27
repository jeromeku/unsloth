import argparse
import os

import torch
from safetensors.torch import load_file

MODEL_FILE_NAME = "adapter_model.safetensors"


def get_state_dict(save_path: str) -> dict:
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, MODEL_FILE_NAME)

    return load_file(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_adapter_only_save_path", type=str, required=True
    )
    parser.add_argument("--model_lora_save_path", type=str, required=True)

    args = parser.parse_args()
    lora_adapter_state_dict = get_state_dict(args.lora_adapter_only_save_path)
    model_lora_state_dict = get_state_dict(args.model_lora_save_path)

    print(
        f"Lora adapter only state dict keys: {len(lora_adapter_state_dict.keys())}"
    )
    print(f"Model LoRA state dict keys: {len(model_lora_state_dict.keys())}")

    extra_keys = set(model_lora_state_dict.keys()) - set(
        lora_adapter_state_dict.keys()
    )
    missing_keys = set(lora_adapter_state_dict.keys()) - set(
        model_lora_state_dict.keys()
    )

    if len(extra_keys) > 0 or len(missing_keys) > 0:
        raise AssertionError(
            "Lora adapter and model LoRA saved state dicts are not identical:",
            f"Keys in model LoRA state dict but not in lora adapter state dict:\n{extra_keys}",  # ruff: noqa: E501
            f"Keys in lora adapter state dict but not in model LoRA state dict:\n{missing_keys}",  # ruff: noqa: E501
            sep="\n",
        )

    print("Checking lora adapter and model LoRA saved state dicts...")
    for k, v in model_lora_state_dict.items():
        if not torch.allclose(v, model_lora_state_dict[k]):
            diff = (v - model_lora_state_dict[k]).abs().max()
            print(f" ! {k} diff: {diff}")
    print("Passed!")
