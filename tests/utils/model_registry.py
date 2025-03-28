from dataclasses import dataclass, field
from typing import Literal

BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG
INSTRUCT_TAG = "Instruct"

_IS_LLAMA_REGISTERED = False
_IS_QWEN_REGISTERED = False
_IS_GEMMA_REGISTERED = False

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
        self.name = construct_model_key(
            self.org,
            self.base_name,
            self.version,
            self.size,
            self.quant_type,
            self.instruct_tag,
        )

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


def _register_models(model_info: dict):
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
    global _IS_LLAMA_REGISTERED
    if _IS_LLAMA_REGISTERED:
        return
    _register_models(_LLAMA_INFO)
    _IS_LLAMA_REGISTERED = True


def register_qwen_models():
    global _IS_QWEN_REGISTERED
    if _IS_QWEN_REGISTERED:
        return

    _register_models(_QWEN_INFO)
    _IS_QWEN_REGISTERED = True


def register_gemma_models():
    global _IS_GEMMA_REGISTERED
    _register_models(_GEMMA_INFO)
    _IS_GEMMA_REGISTERED = True


def get_llama_models():
    if not _IS_LLAMA_REGISTERED:
        register_llama_models()

    return {k: v for k, v in MODEL_REGISTRY.items() if v.base_name == "Llama"}

def get_qwen_models():
    if not _IS_QWEN_REGISTERED:
        register_qwen_models()
    
    return {k: v for k, v in MODEL_REGISTRY.items() if v.base_name == "Qwen"}

def get_gemma_models():
    if not _IS_GEMMA_REGISTERED:
        register_gemma_models()
    
    return {k: v for k, v in MODEL_REGISTRY.items() if v.base_name == "gemma"}

if __name__ == "__main__":
    register_llama_models()
    for k, v in MODEL_REGISTRY.items():
        print(f"{k}: {v}")
        print(v.model_path)
