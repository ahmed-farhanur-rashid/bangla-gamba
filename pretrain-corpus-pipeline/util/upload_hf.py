#!/usr/bin/env python3
"""
Upload Parquets to Hugging Face and replace the old ones.

This script uses `upload_folder` with `delete_patterns` to atomic swap data.

Usage:
  python pretrain-corpus-pipeline/util/upload_hf.py
        --local-dir temp/bangla_corpus
        --path-in-repo bangla_corpus

  # 1. Update bangla_corpus
  HF_HUB_ENABLE_HF_TRANSFER=1 python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir /home/farhan/my-projects/bangla-gamba/temp/bangla_corpus \
    --path-in-repo bangla_corpus

  # 2. Update fineweb_edu
  HF_HUB_ENABLE_HF_TRANSFER=1 python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir /home/farhan/my-projects/bangla-gamba/temp/fineweb_edu \
    --path-in-repo fineweb_edu

  # 3. Update sangraha
  HF_HUB_ENABLE_HF_TRANSFER=1 python pretrain-corpus-pipeline/util/upload_hf.py \
    --local-dir /home/farhan/my-projects/bangla-gamba/temp/sangraha \
    --path-in-repo sangraha
"""

import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload parquets to Hugging Face, replacing old ones.")
    parser.add_argument("--local-dir", type=str, required=True, 
                        help="Local directory containing the new parquet shards (e.g., temp/bangla_corpus).")
    parser.add_argument("--repo-id", type=str, default="ahmed-farhanur-rashid/bn-foundational-pretrain-corpus",
                        help="Hugging Face dataset repository ID.")
    parser.add_argument("--path-in-repo", type=str, required=True,
                        help="Path inside the Hugging Face repository (e.g., bangla_corpus).")
    args = parser.parse_args()

    api = HfApi()

    print(f"Starting upload process...")
    print(f"Local Directory: {args.local_dir}")
    print(f"Target Repo:     {args.repo_id}")
    print(f"Target Path:     {args.path_in_repo}")

    delete_pattern = f"{args.path_in_repo}/*.parquet"

    print(f"\nThis will atomically delete '{delete_pattern}' and upload the new files in one commit.")

    api.upload_folder(
        folder_path=args.local_dir,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Replace {args.path_in_repo} parquets with fixed row-group sizes",
        delete_patterns=[delete_pattern],
    )
    
    print("\n✅ Upload and replacement complete! Old files were successfully swapped out.")

if __name__ == "__main__":
    main()
