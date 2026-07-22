import os
import shutil
from pathlib import Path


def prepare_upload_folder(
    model_dir="saved/model/default",
    tokenizer_dir="saved/tokenizer/hf",
    staging_dir="hf_upload_staging",
):
    staging = Path(staging_dir)
    src_model_dir = Path(model_dir)

    # Fallback to saved/model/banglagamba_12l or newest folder in saved/model if default not found
    if not src_model_dir.exists() and Path("saved/model").exists():
        candidates = list(Path("saved/model").glob("*"))
        if candidates:
            src_model_dir = candidates[0]
            print(f" -> Found model directory at: {src_model_dir}")

    src_tokenizer_dir = Path(tokenizer_dir)

    # 1. Clear out old staging directory if it exists
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"Creating Hugging Face staging directory at: {staging}/")

    # 2. Copy Model Config
    if (src_model_dir / "config.json").exists():
        shutil.copy(src_model_dir / "config.json", staging / "config.json")
        print(" -> Copied config.json")
    else:
        print(" -> WARNING: config.json not found! Run python utils/convert_config_to_json.py first.")

    # 3. Clean and Save Model Weights (model.pt -> pytorch_model.bin)
    raw_weights_path = None
    if (src_model_dir / "model.pt").exists():
        raw_weights_path = src_model_dir / "model.pt"
    elif (src_model_dir / "pytorch_model.bin").exists():
        raw_weights_path = src_model_dir / "pytorch_model.bin"

    if raw_weights_path:
        import torch
        print(f" -> Cleaning and converting {raw_weights_path.name} to pytorch_model.bin...")
        state_dict = torch.load(raw_weights_path, map_location="cpu")
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("rope."):
                continue
            clean_k = k.replace("._orig_mod.", ".")
            if not clean_k.startswith("model."):
                clean_k = "model." + clean_k
            cleaned_state_dict[clean_k] = v
        torch.save(cleaned_state_dict, str(staging / "pytorch_model.bin"))
        print(f" -> Saved cleaned weights ({len(cleaned_state_dict)} tensors) to pytorch_model.bin")
    else:
        print(" -> WARNING: Model weights file (model.pt or pytorch_model.bin) not found!")

    # 4. Copy Tokenizer Assets & Patch tokenizer_config.json
    if src_tokenizer_dir.exists():
        for tok_file in src_tokenizer_dir.glob("*"):
            if tok_file.is_file():
                shutil.copy(tok_file, staging / tok_file.name)
                print(f" -> Copied tokenizer asset: {tok_file.name}")

        tok_cfg_path = staging / "tokenizer_config.json"
        if tok_cfg_path.exists():
            import json
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            tok_cfg["tokenizer_class"] = "BanglaGambaTokenizer"
            tok_cfg["auto_map"] = {
                "AutoTokenizer": [
                    "tokenization_banglagamba.BanglaGambaTokenizer",
                    None
                ]
            }
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(tok_cfg, f, indent=2)
            print(" -> Injected BanglaGambaTokenizer auto_map into tokenizer_config.json")
    else:
        print(f" -> WARNING: Tokenizer directory {src_tokenizer_dir} not found!")

    # 5. Copy Custom Architecture & Tokenizer Wrappers
    hf_int_dir = Path("src/hf_integration")
    if hf_int_dir.exists():
        for py_file in hf_int_dir.glob("*.py"):
            shutil.copy(py_file, staging / py_file.name)
            print(f" -> Copied custom HF module: {py_file.name}")

        if (hf_int_dir / "README.md").exists():
            shutil.copy(hf_int_dir / "README.md", staging / "README.md")
            print(" -> Copied Model Card (README.md)")

    # 6. Summary instructions
    print("\n" + "=" * 50)
    print("✅ Staging directory ready!")
    print("=" * 50)
    print("To upload this folder to your repository, ensure you are logged into Hugging Face")
    print("by running 'hf auth login' in your terminal.")
    print("\nThen, run the following command to upload all files to your repo:")
    print(f"hf upload ahmed-farhanur-rashid/bangla-gamba {staging_dir} .")


if __name__ == "__main__":
    prepare_upload_folder()
