from dataclasses import dataclass, field
from typing import Literal

BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG
INSTRUCT_TAG = "Instruct"

# Llama text only models
_LLAMA_INFO = {
    "org": "meta-llama",
    "base_name": "Llama",
    "instruct_tag": "Instruct",
    "model_versions": ["3.2", "3.1"],
    "model_sizes": {"3.2": [1, 3], "3.1": [8]},
    "is_multimodal": False,
}

# Qwen text only models
_QWEN_INFO = {
    "org": "Qwen",
    "base_name": "Qwen",
    "instruct_tag": "Instruct",
    "model_versions": ["2.5"],
    "model_sizes": {"2.5": [3, 7]},
    "is_multimodal": False,
}

_GEMMA_INFO = {
    "org": "google",
    "base_name": "gemma",
    "instruct_tag": "it",
    "model_versions": ["3"],
    "model_sizes": {"3": [1, 4]},
    "is_multimodal": True,
}

_PHI_INFO = {
    "org": "microsoft",
    "base_name": "Phi",
    "model_versions": ["4"],
    "instruct_tag": None,
    "is_multimodal": True,
}

def construct_model_key(org, base_name, version, size, quant_type, instruct_tag):
    key = f"{org}/{base_name}-{version}-{size}B"
    if instruct_tag:
        key = "-".join([key, instruct_tag])
    if quant_type:
        if quant_type == "bnb":
            key = "-".join([key, BNB_QUANTIZED_TAG])
        elif quant_type == "unsloth":
            key = "-".join([key, UNSLOTH_DYNAMIC_QUANT_TAG])
    return key

@dataclass
class ModelInfo:
    org: str
    base_name: str
    version: str
    size: int
    name: str = field(
        init=False
    )  # full model name, constructed from base_name, version, and size
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: Literal["bnb", "unsloth"] = None

    def __post_init__(self):
        self.name = construct_model_key(self.org, self.base_name, self.version, self.size, self.quant_type, self.instruct_tag)
        
    @property
    def model_path(
        self,
    ) -> str:
        return f"{self.org}/{self.name}"


MODEL_REGISTRY = {}



def register_model(
    org: str,
    base_name: str,
    version: str,
    size: int,
    quant_type: Literal["bnb", "unsloth"] = None,
    is_multimodal: bool = False,
    instruct_tag: str = INSTRUCT_TAG,
):
    key = construct_model_key(org, base_name, version, size, quant_type, instruct_tag)
    if key in MODEL_REGISTRY:
        raise ValueError(f"Model {key} already registered")
    
    MODEL_REGISTRY[key] = ModelInfo(
        org=org,
        base_name=base_name,
        version=version,
        size=size,
        is_multimodal=is_multimodal,
        instruct_tag=instruct_tag,
        quant_type=quant_type,
    )


def register_models(model_info: dict):
    org = model_info["org"]
    base_name = model_info["base_name"]
    instruct_tag = model_info["instruct_tag"]
    model_versions = model_info["model_versions"]
    model_sizes = model_info["model_sizes"]
    is_multimodal = model_info["is_multimodal"]

    for version in model_versions:
        for size in model_sizes[version]:
            for quant_type in [None, "bnb", "unsloth"]:
                # Register base model
                _org = "unsloth" if quant_type is not None else org
                register_model(
                    _org,
                    base_name,
                    version,
                    size,
                    instruct_tag=None,
                    quant_type=quant_type,
                    is_multimodal=is_multimodal,
                )
                # Register instruction tuned model if instruct_tag is not None
                if instruct_tag:
                    register_model(
                        _org,
                        base_name,
                        version,
                        size,
                        instruct_tag=instruct_tag,
                        quant_type=quant_type,
                        is_multimodal=is_multimodal,
                    )

def register_llama_models():
    register_models(_LLAMA_INFO)


def register_qwen_models():
    register_models(_QWEN_INFO)


def register_gemma_models():
    register_models(_GEMMA_INFO)


# QWEN = {
#     "2.5": ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
# }
# GEMMA = {
#     "3": ["gemma-3-1b-it"],
# }
# PHI = {
#     "4": ["phi-4", ]
# }
# HF_TEST_MODELS = {
#     "llama-hf": "meta-llama/Llama-3.2-1B-Instruct",
#     "gemma-hf": "google/gemma-3-1b-it",
#     "qwen-hf": "Qwen/Qwen2-VL-2B-Instruct-bnb-4bit",
# }

# UNSLOTH_TEST_MODELS = {
#     "llama-unsloth-1b": "unsloth/Llama-3.2-1B-Instruct",
#     "llama-unsloth-3b": "unsloth/Llama-3.2-3B-Instruct",
#     "gemma-unsloth": "unsloth/gemma-3-1b-it",
#     "qwen-unsloth": "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
# }

# TEST_MODELS = {
#     **HF_TEST_MODELS,
#     **UNSLOTH_TEST_MODELS,
# }

if __name__ == "__main__":
    register_llama_models()
    for k, v in MODEL_REGISTRY.items():
        print(f"{k}: {v}")
        print(v.model_path)
