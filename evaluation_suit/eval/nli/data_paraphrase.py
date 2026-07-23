"""
BanglaParaphrase dataset loader for NLI / sentence-pair evaluation.

Source: https://github.com/csebuetnlp/banglaparaphrase
Not on HF — cloned from GitHub and converted to a datasets-loadable format.

This is the "clean native-Bangla sentence-pair reasoning" counterpart to
XNLI's machine-translated artifacts.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict


_GITHUB_URL = "https://github.com/csebuetnlp/banglaparaphrase.git"


def load_bangla_paraphrase(
    cache_dir: str = "evaluation_suit/data_cache/banglaparaphrase",
) -> DatasetDict:
    """
    Load BanglaParaphrase dataset from HF.

    Converts to a sentence-pair evaluation task.

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'premise' (str), 'hypothesis' (str), 'label' (int).
    """
    print("[BanglaParaphrase] Loading BanglaParaphrase from HF archive (csebuetnlp/BanglaParaphrase)...")
    import zipfile, json
    from huggingface_hub import hf_hub_download
    from datasets import Dataset, DatasetDict

    zip_path = hf_hub_download("csebuetnlp/BanglaParaphrase", filename="data/BanglaParaphrase.zip", repo_type="dataset")

    splits = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for member_name in z.namelist():
            for split_name in ["train", "test", "validation", "dev", "val"]:
                if f"{split_name}.jsonl" in member_name:
                    canonical = "validation" if split_name in ("dev", "val") else split_name
                    records = []
                    with z.open(member_name) as f:
                        for line in f:
                            item = json.loads(line.decode("utf-8"))
                            records.append({
                                "premise": item.get("source", item.get("premise", "")),
                                "hypothesis": item.get("target", item.get("hypothesis", "")),
                                "label": item.get("label", 1),
                            })
                    splits[canonical] = Dataset.from_list(records)

    ds = DatasetDict(splits)
    return ds


if __name__ == "__main__":
    ds = load_bangla_paraphrase()
    print(f"\nLoaded BanglaParaphrase: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
        if len(ds[split]) > 0:
            ex = ds[split][0]
            print(f"    premise:    {ex['premise'][:80]}")
            print(f"    hypothesis: {ex['hypothesis'][:80]}")
            print(f"    label:      {ex['label']}")
