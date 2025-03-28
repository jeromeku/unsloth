from huggingface_hub import ModelInfo as HfModelInfo
from utils.hf_hub import get_model_info
from utils.model_registry import MODEL_REGISTRY, ModelInfo, register_llama_models

if __name__ == "__main__":
    register_llama_models()
    model_ids = list(MODEL_REGISTRY.keys())
    for _id in model_ids:
        try:
            model_info: HfModelInfo = get_model_info(_id, properties=['safetensors', 'lastModified'])
        except Exception as e:
            print(f"{_id} not found")
