import yaml
import json
import os
import argparse


def convert_config(model_dir):
    if not os.path.exists(model_dir) and os.path.exists("saved/model"):
        candidates = [os.path.join("saved/model", d) for d in os.listdir("saved/model") if os.path.isdir(os.path.join("saved/model", d))]
        if candidates:
            model_dir = candidates[0]
            print(f" -> Found model directory at: {model_dir}")

    yaml_path = os.path.join(model_dir, "config.yaml")
    json_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} does not exist.")
        return

    print(f"Loading {yaml_path}...")
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    # Inject Hugging Face Auto_Map fields
    print("Injecting HF auto_map configuration...")
    config_data["auto_map"] = {
        "AutoConfig": "configuration_banglagamba.BanglaGambaConfig",
        "AutoModelForCausalLM": "modeling_banglagamba.BanglaGambaForCausalLM",
        "AutoTokenizer": "tokenization_banglagamba.BanglaGambaTokenizer"
    }

    # Ensure architectures field is present for HF
    if "architectures" not in config_data:
        config_data["architectures"] = ["BanglaGambaForCausalLM"]

    # Set model type and dtype
    if "model_type" not in config_data:
        config_data["model_type"] = "banglagamba"

    if "torch_dtype" not in config_data:
        config_data["torch_dtype"] = "bfloat16"

    print(f"Saving {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print("Done! The config.json is ready for Hugging Face upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model config.yaml to config.json with HF auto_map")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved/model/default",
        help="Directory containing the model config.yaml",
    )
    args = parser.parse_args()

    convert_config(args.model_dir)
