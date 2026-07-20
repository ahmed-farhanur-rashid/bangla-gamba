import argparse
from pathlib import Path
from huggingface_hub import HfApi

def upload_shards(input_dir: str, repo_id: str):
    api = HfApi()
    in_path = Path(input_dir)
    
    # Map the jsonl prefix to the HuggingFace folder name
    folder_mapping = {
        "bangla": "bangla_corpus",
        "english": "fineweb_edu",
        "sangraha": "sangraha",
        "nmt": "nllb_nmt",
        "opus_nmt": "nllb_nmt" # Or another folder if needed
    }
    
    parquet_files = list(in_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {in_path}")
        return
        
    print(f"Found {len(parquet_files)} parquet files to upload.")
    
    for parquet_file in parquet_files:
        # Determine which folder it belongs to based on the prefix
        prefix = parquet_file.name.split("_shard_")[0]
        
        if prefix in folder_mapping:
            hf_folder = folder_mapping[prefix]
            path_in_repo = f"{hf_folder}/{parquet_file.name}"
            
            print(f"Uploading {parquet_file.name} to {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=str(parquet_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset"
            )
        else:
            print(f"Skipping {parquet_file.name}, prefix '{prefix}' not in mapping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Parquet shards to HuggingFace in specific folders")
    parser.add_argument("--input-dir", type=str, default="/home/farhan/my-projects/bangla-gamba/temp",
                        help="Directory containing the .parquet shards")
    parser.add_argument("--repo-id", type=str, default="ahmed-farhanur-rashid/bn-foundational-pretrain-corpus",
                        help="HuggingFace dataset repository ID")
    
    args = parser.parse_args()
    upload_shards(args.input_dir, args.repo_id)
