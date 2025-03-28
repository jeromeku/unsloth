from dataclasses import dataclass, field
from typing import Literal

BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG
INSTRUCT_TAG = "Instruct"

_IS_LLAMA_REGISTERED = False
_IS_LLAMA_VISION_REGISTERED = False
_IS_QWEN_REGISTERED = False
_IS_GEMMA_REGISTERED = False
_IS_QWEN_VL_REGISTERED = False

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
    name: str = None # full model name, constructed from base_name, version, and size unless provided
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: Literal["bnb", "unsloth"] = None

    def __post_init__(self):
        self.name = self.name or self.construct_model_name(
            self.base_name,
            self.version,
            self.size,
            self.quant_type,
            self.instruct_tag,
        )

    @staticmethod
    def append_instruct_tag(key: str, instruct_tag: str = None):
        if instruct_tag:
            key = "-".join([key, instruct_tag])
        return key

    @staticmethod
    def append_quant_type(key: str, quant_type: Literal["bnb", "unsloth"] = None):
        if quant_type:
            if quant_type == "bnb":
                key = "-".join([key, BNB_QUANTIZED_TAG])
            elif quant_type == "unsloth":
                key = "-".join([key, UNSLOTH_DYNAMIC_QUANT_TAG])
        return key
    
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        raise NotImplementedError("Subclass must implement this method")
    
    @property
    def model_path(
        self,
    ) -> str:
        return f"{self.org}/{self.name}"

class LlamaModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key


class LlamaVisionModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B-Vision"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key
    
class QwenModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}{version}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

class QwenVLModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}{version}-VL-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

# Llama text only models
# NOTE: Llama vision models will be registered separately
_LLAMA_INFO = {
    "org": "meta-llama",
    "base_name": "Llama",
    "instruct_tags": [None, "Instruct"],
    "model_versions": ["3.2", "3.1"],
    "model_sizes": {"3.2": [1, 3], "3.1": [8]},
    "is_multimodal": False,
    "model_info_cls": LlamaModelInfo,
}

_LLAMA_VISION_INFO = {
    "org": "meta-llama",
    "base_name": "Llama",
    "instruct_tags": [None, "Instruct"],
    "model_versions": ["3.2"],
    "model_sizes": {"3.2": [11, 90]},
    "is_multimodal": True,
    "model_info_cls": LlamaVisionModelInfo,
}
# Qwen text only models
# NOTE: Qwen vision models will be registered separately
_QWEN_INFO = {
    "org": "Qwen",
    "base_name": "Qwen",
    "instruct_tags": [None, "Instruct"],
    "model_versions": ["2.5"],
    "model_sizes": {"2.5": [3, 7]},
    "is_multimodal": False,
    "model_info_cls": QwenModelInfo,
}

_QWEN_VL_INFO = {
    "org": "Qwen",
    "base_name": "Qwen",
    "instruct_tags": ["Instruct"], # No base, only instruction tuned
    "model_versions": ["2.5"],
    "model_sizes": {"2.5": [3, 7, 32, 72]},
    "is_multimodal": True,
    "instruction_tuned_only": True,
    "model_info_cls": QwenVLModelInfo,
}

_GEMMA_INFO = {
    "org": "google",
    "base_name": "gemma",
    "instruct_tags": ["pt", "it"], # pt = base, it = instruction tuned
    "model_versions": ["3"],
    "model_sizes": {"3": [1, 4, 12, 27]},
    "is_multimodal": True,
}

_PHI_INFO = {
    "org": "microsoft",
    "base_name": "Phi",
    "model_versions": ["4"],
    "instruct_tag": [None],
    "is_multimodal": True,
}


MODEL_REGISTRY = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    quant_type: Literal["bnb", "unsloth"] = None,
    is_multimodal: bool = False,
    instruct_tag: str = INSTRUCT_TAG,
    name: str = None,
):
    name = name or model_info_cls.construct_model_name(base_name=base_name, version=version, size=size, quant_type=quant_type, instruct_tag=instruct_tag)
    key = f"{org}/{name}" 

    if key in MODEL_REGISTRY:
        raise ValueError(f"Model {key} already registered")

    MODEL_REGISTRY[key] = model_info_cls(
        org=org,
        base_name=base_name,
        version=version,
        size=size,
        is_multimodal=is_multimodal,
        instruct_tag=instruct_tag,
        quant_type=quant_type,
        name=name,
    )


def _register_models(model_info: dict):
    org = model_info["org"]
    base_name = model_info["base_name"]
    instruct_tags = model_info["instruct_tags"]
    model_versions = model_info["model_versions"]
    model_sizes = model_info["model_sizes"]
    is_multimodal = model_info["is_multimodal"]
    model_info_cls = model_info["model_info_cls"]

    for version in model_versions:
        for size in model_sizes[version]:
            for instruct_tag in instruct_tags:
                for quant_type in [None, "bnb", "unsloth"]:
                    _org = "unsloth" if quant_type is not None else org
                    register_model(
                        model_info_cls=model_info_cls,
                        org=_org,
                        base_name=base_name,
                        version=version,
                        size=size,
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

def register_qwen_vl_models():
    global _IS_QWEN_VL_REGISTERED
    if _IS_QWEN_VL_REGISTERED:
        return
    
    _register_models(_QWEN_VL_INFO)
    _IS_QWEN_VL_REGISTERED = True

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

def get_qwen_vl_models():
    if not _IS_QWEN_VL_REGISTERED:
        register_qwen_vl_models()
    return {k: v for k, v in MODEL_REGISTRY.items() if v.base_name == "Qwen" and v.is_multimodal}

def get_gemma_models():
    if not _IS_GEMMA_REGISTERED:
        register_gemma_models()
    
    return {k: v for k, v in MODEL_REGISTRY.items() if v.base_name == "gemma"}

if __name__ == "__main__":
    register_llama_models()
    for k, v in MODEL_REGISTRY.items():
        print(f"{k}: {v}")
        print(v.model_path)
