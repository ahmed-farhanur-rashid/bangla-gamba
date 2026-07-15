#!/usr/bin/env python3
"""
Applies NMT normalization to OPUS datasets (Tatoeba and OpenSubtitles).

Usage:
  python pretrain-corpus-pipeline/01f_nmt_normalize_opus.py

This script processes Tatoeba first, then OpenSubtitles, and combines
the cleaned outputs into a single file: saved/data/cleaned/opus_nmt.jsonl
"""

import subprocess
import sys
import shutil
from pathlib import Path

def main():
    datasets = ["opus_tatoeba", "opus_opensubtitles"]
    script_path = "pretrain-corpus-pipeline/01e_nmt_normalize.py"
    final_output = Path("saved/data/cleaned/opus_nmt.jsonl")
    
    for dataset in datasets:
        input_path = Path(f"saved/data/raw/{dataset}.jsonl")
        output_tmp = Path(f"saved/data/cleaned/{dataset}_tmp.jsonl")
        
        if not input_path.exists():
            print(f"Skipping {dataset}, input not found: {input_path}")
            continue
            
        print(f"\n=======================================================")
        print(f" Normalizing {dataset}")
        print(f"=======================================================")
        
        cmd = [
            sys.executable, script_path,
            "--input", str(input_path),
            "--output", str(output_tmp)
        ]
        
        subprocess.run(cmd, check=True)

    print(f"\n=======================================================")
    print(f" Combining into {final_output.name}")
    print(f"=======================================================")
    
    final_output.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as fout:
        for dataset in datasets:
            tmp_path = Path(f"saved/data/cleaned/{dataset}_tmp.jsonl")
            if tmp_path.exists():
                with open(tmp_path, "r", encoding="utf-8") as fin:
                    shutil.copyfileobj(fin, fout)
                tmp_path.unlink()
                
    print(f"Successfully created {final_output}")

if __name__ == "__main__":
    main()
