from safetensors.torch import load_file

SAVE_PATH= "/home/jeromeku/dev/third_party/unsloth/unsloth_merged_16bit/meta-llama_Llama-3.2-1B-Instruct_lora_r64/model.safetensors"

state_dict = load_file(SAVE_PATH)

for name, param in state_dict.items():
    print(name, param.shape, param.dtype)
