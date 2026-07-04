#!/usr/bin/env python3
"""
Convert large JSONL files to Parquet shards using Pandas.
We process the files in chunks to ensure we don't run out of RAM,
and we use zstd compression to heavily reduce disk usage.

Usage:
python pretrain-corpus-pipeline/util/pack_to_parquet.py
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def pack_to_parquet(input_dir: str, output_dir: str, chunk_size: int = 250_000):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    if not in_path.exists():
        print(f"Error: Input directory {in_path} does not exist.")
        return
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = list(in_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {in_path}")
        return
        
    print(f"Found {len(jsonl_files)} JSONL files to pack.")
    
    for jsonl_file in jsonl_files:
        print(f"\nProcessing: {jsonl_file.name}")
        
        # We read the jsonl file in chunks to avoid blowing up memory
        reader = pd.read_json(jsonl_file, lines=True, chunksize=chunk_size)
        
        shard_count = 0
        total_rows = 0
        
        for i, chunk in enumerate(reader):
            # Format: filename_shard_0000.parquet
            shard_name = f"{jsonl_file.stem}_shard_{i:04d}.parquet"
            out_file = out_path / shard_name
            
            # zstd compression gives fantastic text compression ratios
            # saving a lot of disk space for your 120GB drive.
            chunk.to_parquet(out_file, engine='pyarrow', compression='zstd')
            
            shard_count += 1
            total_rows += len(chunk)
            print(f"  → Saved {shard_name} ({len(chunk):,} rows)")
            
        print(f"✓ Completed {jsonl_file.name}: {total_rows:,} rows across {shard_count} shards.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack JSONL into Parquet shards")
    parser.add_argument("--input-dir", type=str, default="/home/farhan/my-projects/bangla-gamba/saved/data/cleaned",
                        help="Directory containing .jsonl files")
    parser.add_argument("--output-dir", type=str, default="/home/farhan/my-projects/bangla-gamba/temp",
                        help="Directory to save the .parquet shards")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                        help="Number of rows per parquet shard (default: 250,000)")
    
    args = parser.parse_args()
    pack_to_parquet(args.input_dir, args.output_dir, args.chunk_size)
