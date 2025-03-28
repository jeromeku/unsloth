from huggingface_hub import ModelInfo as HfModelInfo
from utils.hf_hub import get_model_info
from utils.model_registry import MODEL_REGISTRY, ModelInfo, get_llama_models


def test_model_uploaded(model_ids: list[str]):
    for _id in model_ids:
        try:
            model_info: HfModelInfo = get_model_info(_id, properties=['safetensors', 'lastModified'])
        except Exception as e:
            raise AssertionError(f"{_id} not found")
        
if __name__ == "__main__":
    llama_models = get_llama_models()
    test_model_uploaded(list(llama_models.keys()))
    