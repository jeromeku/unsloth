from huggingface_hub import ModelInfo as HfModelInfo
from utils.hf_hub import get_model_info
from utils.model_registry import (
    MODEL_REGISTRY,
    ModelInfo,
    get_llama_models,
    get_qwen_models,
    get_qwen_vl_models,
)


def test_model_uploaded(model_ids: list[str]):
    missing_models = []
    for _id in model_ids:
        
        model_info: HfModelInfo = get_model_info(_id, properties=['safetensors', 'lastModified'])
        if not model_info:
            missing_models.append(_id)
    
    return missing_models

if __name__ == "__main__":

    for method in [get_llama_models, get_qwen_models, get_qwen_vl_models]:
        
        models = method()
        print(f"Models registered: {len(models)}")
        for model_info in models.values():
            print(f"  {model_info.model_path}")
        missing_modes = test_model_uploaded(list(models.keys()))
    
        if missing_modes:
            print("--------------------------------")
            print(f"Missing models: {missing_modes}")
            print("--------------------------------")