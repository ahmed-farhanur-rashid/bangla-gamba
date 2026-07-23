"""
XNLI-bn dataset loader for Natural Language Inference.

XNLI provides machine-translated NLI data for 15 languages including Bangla.
The Bangla translations are known to be noisy — this caveat must be noted
in the paper.
"""

import warnings

from datasets import DatasetDict, load_dataset


def load_xnli_bn() -> DatasetDict:
    """
    Load XNLI Bangla dataset from HF.

    Tries known config names (config naming has changed across
    dataset versions historically).

    Returns:
        DatasetDict with train, validation, test splits.
        Each example has 'premise' (str), 'hypothesis' (str), 'label' (int).
        Labels: 0=entailment, 1=neutral, 2=contradiction
    """
    # Load from csebuetnlp/xnli_bn archive directly
    print("[XNLI] Loading XNLI Bangla from HF archive (csebuetnlp/xnli_bn)...")
    import tarfile, json
    from huggingface_hub import hf_hub_download
    from datasets import Dataset, DatasetDict

    tar_path = hf_hub_download("csebuetnlp/xnli_bn", filename="data/xnli_bn.tar.bz2", repo_type="dataset")
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}

    splits = {}
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            for split_name in ["train", "val", "validation", "test"]:
                if split_name in member.name:
                    key = "validation" if split_name == "val" else split_name
                    f = tar.extractfile(member)
                    records = []
                    for line in f:
                        item = json.loads(line.decode("utf-8"))
                        lbl = item.get("label", 0)
                        if isinstance(lbl, str):
                            lbl = label_map.get(lbl.lower().strip(), 0)
                        records.append({
                            "premise": item.get("sentence1", ""),
                            "hypothesis": item.get("sentence2", ""),
                            "label": lbl,
                        })
                    splits[key] = Dataset.from_list(records)

    ds = DatasetDict(splits)

    # Verify we have the expected columns
    sample_split = list(ds.keys())[0]
    columns = ds[sample_split].column_names
    print(f"[XNLI] Columns: {columns}")

    # Standardize column names if needed
    for split in ds:
        if "premise" not in ds[split].column_names:
            # Try common alternatives
            for alt in ["sentence1", "text1"]:
                if alt in ds[split].column_names:
                    ds[split] = ds[split].rename_column(alt, "premise")
                    break
        if "hypothesis" not in ds[split].column_names:
            for alt in ["sentence2", "text2"]:
                if alt in ds[split].column_names:
                    ds[split] = ds[split].rename_column(alt, "hypothesis")
                    break

    # Print stats
    label_names = ["entailment", "neutral", "contradiction"]
    for split in ds:
        n = len(ds[split])
        labels = ds[split]["label"]
        dist = {}
        for l in labels:
            dist[label_names[l] if l < len(label_names) else str(l)] = dist.get(
                label_names[l] if l < len(label_names) else str(l), 0
            ) + 1
        print(f"[XNLI] {split}: {n} examples, distribution: {dist}")

    return ds


if __name__ == "__main__":
    ds = load_xnli_bn()
    print(f"\nLoaded XNLI-bn: {ds}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
        if len(ds[split]) > 0:
            ex = ds[split][0]
            print(f"    premise:    {ex.get('premise', 'N/A')[:80]}")
            print(f"    hypothesis: {ex.get('hypothesis', 'N/A')[:80]}")
            print(f"    label:      {ex.get('label', 'N/A')}")
