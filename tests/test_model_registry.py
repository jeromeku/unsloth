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
    for _id in model_ids:
        
        model_info: HfModelInfo = get_model_info(_id, properties=['safetensors', 'lastModified'])
        if not model_info:
            print(f"{_id} not found")

if __name__ == "__main__":
    test_model_method = get_qwen_vl_models

    models = test_model_method()
    print(f"Models registered: {len(models)}")
    for model_info in models.values():
        print(f"  {model_info.model_path}")
    test_model_uploaded(list(models.keys()))
    # test_model_uploaded(list(llama_models.keys()))
    # quant_type = None
    # version = "3.2"
    # for model_info in llama_models.values():
    #     if model_info.version == version and model_info.quant_type == quant_type:
    #         print(f"{model_info}")