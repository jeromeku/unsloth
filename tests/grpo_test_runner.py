import argparse
import os
import subprocess
import tempfile


def run_test(model_name, max_steps, lora_rank, use_vllm):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_merge_path = os.path.join(temp_dir, "merged")
        model_adapter_path = os.path.join(temp_dir, "lora")

        os.makedirs(model_merge_path, exist_ok=True)
        os.makedirs(model_adapter_path, exist_ok=True)

        log_dir = "logs/grpo"
        log_file = f"{log_dir}/train_{model_name}_{lora_rank}_vllm={use_vllm}.log"

        train_args = [
            "python", "tests/grpo/test_unsloth_grpo.py",
            "--model_merged_save_path", model_merge_path,
            "--model_adapter_save_path", model_adapter_path
        ]

        if not use_vllm:
            train_args.append("--use_slow_inference")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Run the first script
        with open(log_file, "w") as log:
            subprocess.run(train_args, stdout=log, stderr=subprocess.STDOUT, check=True)

        # Run the second script
        check_args = [
            "python", "scripts/check_merged_weights.py",
            "--merged_save_path", model_merge_path,
            "--adapter_save_path", model_adapter_path,
            "--lora_rank", str(lora_rank)
        ]

        try:
            result = subprocess.run(check_args, capture_output=True, text=True, check=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(e.stdout)
            raise AssertionError("Check merged weights failed") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GRPO tests")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--max_steps", type=int, default=25, help="Maximum steps for training")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--use_vllm", action='store_true', help="Use VLLM for inference")

    args = parser.parse_args()

    run_test(args.model_name, args.max_steps, args.lora_rank, args.use_vllm)