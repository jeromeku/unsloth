from huggingface_hub import HfApi, ModelInfo

api = HfApi()

POPULARITY_PROPERTIES = ["downloads", "downloadsAllTime", "trendingScore", "likes"]
THOUSAND = 1000
MILLION = 1000000
BILLION = 1000000000

def formatted_int(value: int) -> str:
    if value < THOUSAND:
        return str(value)
    elif value < MILLION:
        return f"{float(value) / 1000:,.1f}K"
    elif value < BILLION:
        return f"{float(value) // 1000000:,.1f}M"
    
def get_model_info(model_id: str, properties: list[str] = ['safetensors', 'lastModified']) -> ModelInfo:
    return api.model_info(model_id, expand=properties)

def retrieve_models(properties: list[str] = ["downloads", "tags"], full: bool = False, sort: str = "downloads", author: str = "unsloth", search: str = None, limit: int = 10) -> ModelInfo:
    """
    Retrieve models from the Hugging Face Hub.

    properties: list[str] = See https://huggingface.co/docs/huggingface_hub/api-ref/hf_hub/hf_api/list_models
    full: bool = Whether to retrieve the full model information, including actual model files.
    sort: str = The sort order.
    author: str = The author of the model.
    search: str = The search query for filtering models.
    
    """
    assert full ^ properties, "Cannot retrieve both full and properties"

    models: list[ModelInfo] = api.list_models(author=author, search=search, sort=sort, limit=limit, expand=properties, full=full)
    return models