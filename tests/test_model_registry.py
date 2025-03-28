from huggingface_hub import ModelInfo as HfModelInfo
from utils.hf_hub import get_model_info
from utils.model_registry import (
    get_llama_models,
    get_llama_vision_models,
    get_phi_instruct_models,
    get_phi_models,
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

    for method in [get_llama_models, get_llama_vision_models, get_qwen_models, get_qwen_vl_models, get_phi_models, get_phi_instruct_models]:
        
        models = method()
        model_name = next(iter(models.values())).base_name
        print(f"{model_name}: {len(models)} registered")
        for model_info in models.values():
            print(f"  {model_info.model_path}")
        missing_models = test_model_uploaded(list(models.keys()))
    
        if missing_models:
            print("--------------------------------")
            print(f"Missing models: {missing_models}")
            print("--------------------------------")