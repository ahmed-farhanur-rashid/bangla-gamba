#!/usr/bin/env python3
"""
Upload Parquets to Hugging Face and replace the old ones.

This script uploads files individually with progress tracking, retry logic,
and hf_transfer support for faster uploads.

Usage:
  python pretrain-corpus-pipeline/util/upload_hf.py
        --local-dir temp/bangla_corpus
        --path-in-repo bangla_corpus

  # Fast upload with hf_transfer (recommended):
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
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


def get_parquet_files(local_dir: str) -> list[Path]:
    """Get all .parquet files in the local directory, sorted by name."""
    directory = Path(local_dir)
    files = sorted(directory.glob("*.parquet"))
    if not files:
        print(f"Error: No .parquet files found in {local_dir}")
        sys.exit(1)
    return files


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m"


def upload_with_retry(
    api: HfApi,
    file_path: Path,
    path_in_repo: str,
    repo_id: str,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> bool:
    """Upload a single file with exponential backoff retry logic."""
    relative_path = f"{path_in_repo}/{file_path.name}" if path_in_repo else file_path.name
    backoff = initial_backoff

    for attempt in range(1, max_retries + 1):
        try:
            commit_msg = f"Upload {relative_path}"
            if attempt > 1:
                commit_msg += f" (attempt {attempt})"

            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=relative_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_msg,
            )
            return True

        except RepositoryNotFoundError:
            print(f"\nError: Repository '{repo_id}' not found. Check repo_id.")
            return False

        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                print(f"\nAuthentication error: {e}. Check your HF token.")
                return False
            if attempt < max_retries:
                print(f"\n  HTTP error on attempt {attempt}: {e}")
                print(f"  Retrying in {format_time(backoff)}...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return False

        except Exception as e:
            if attempt < max_retries:
                print(f"\n  Error on attempt {attempt}: {type(e).__name__}: {e}")
                print(f"  Retrying in {format_time(backoff)}...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return False

    return False


def delete_old_files(api: HfApi, path_in_repo: str, repo_id: str) -> bool:
    """Delete old .parquet files from the repo before uploading new ones."""
    delete_pattern = f"{path_in_repo}/*.parquet"
    try:
        print(f"Deleting old files matching '{delete_pattern}'...")
        api.delete_files(
            patterns=[delete_pattern],
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Delete old {path_in_repo} parquets before replacement",
        )
        print("Old files deleted.")
        return True
    except Exception as e:
        print(f"Warning: Could not delete old files: {e}")
        print("Continuing with upload (new files will overwrite existing ones)...")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload parquets to Hugging Face with progress and retry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload bangla corpus (with hf_transfer for speed):
  HF_HUB_ENABLE_HF_TRANSFER=1 python upload_hf.py \\
    --local-dir temp/bangla_corpus \\
    --path-in-repo bangla_corpus

  # Upload without deleting old files first:
  python upload_hf.py \\
    --local-dir temp/sangraha \\
    --path-in-repo sangraha \\
    --no-delete
        """,
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory containing the new parquet shards.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="ahmed-farhanur-rashid/bn-foundational-pretrain-corpus",
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        required=True,
        help="Path inside the Hugging Face repository (e.g., bangla_corpus).",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Skip deleting old files before upload.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max retry attempts per file (default: 5).",
    )
    args = parser.parse_args()

    api = HfApi()

    # Check hf_transfer status
    hf_transfer_enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"

    # Discover files
    files = get_parquet_files(args.local_dir)
    total_size = sum(f.stat().st_size for f in files)

    print("=" * 60)
    print("BanglaGamba Dataset Upload")
    print("=" * 60)
    print(f"  Source:     {args.local_dir}")
    print(f"  Target:     {args.repo_id} / {args.path_in_repo}")
    print(f"  Files:      {len(files)} parquet shards")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  hf_transfer: {'ENABLED (fast)' if hf_transfer_enabled else 'DISABLED (slow - set HF_HUB_ENABLE_HF_TRANSFER=1)'}")
    print(f"  Max retries: {args.retries}")
    print("=" * 60)

    # Delete old files first (separate commit for atomicity)
    if not args.no_delete:
        delete_old_files(api, args.path_in_repo, args.repo_id)

    # Upload each file with progress
    print(f"\nUploading {len(files)} files...\n")
    start_time = time.time()
    uploaded_size = 0
    failed_files = []

    for i, file_path in enumerate(files, 1):
        file_size = file_path.stat().st_size
        elapsed = time.time() - start_time
        speed = uploaded_size / elapsed if elapsed > 0 else 0

        # Progress line
        pct = (i - 1) / len(files) * 100
        print(
            f"[{i:>{len(str(len(files)))}}/{len(files)}] {pct:5.1f}% | "
            f"{file_path.name} ({format_size(file_size)}) | "
            f"{format_size(uploaded_size)}/{format_size(total_size)} uploaded | "
            f"{format_size(speed)}/s",
            end="",
            flush=True,
        )

        success = upload_with_retry(
            api, file_path, args.path_in_repo, args.repo_id, max_retries=args.retries
        )

        if success:
            uploaded_size += file_size
            elapsed = time.time() - start_time
            speed = uploaded_size / elapsed if elapsed > 0 else 0
            remaining = (total_size - uploaded_size) / speed if speed > 0 else 0
            print(f" -> OK ({format_time(remaining)} remaining)")
        else:
            failed_files.append(file_path.name)
            print(f" -> FAILED")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"  Completed: {len(files) - len(failed_files)}/{len(files)} files")
    print(f"  Uploaded:  {format_size(uploaded_size)}")
    print(f"  Time:      {format_time(total_time)}")
    if uploaded_size > 0:
        print(f"  Avg speed: {format_size(uploaded_size / total_time)}/s")

    if failed_files:
        print(f"\n  FAILED FILES ({len(failed_files)}):")
        for name in failed_files:
            print(f"    - {name}")
        print("\nRe-run the script to retry failed uploads.")
        sys.exit(1)
    else:
        print(f"\nAll files uploaded successfully!")


if __name__ == "__main__":
    main()
