import os
import re

model_files = [
    "src/model/config.py",
    "src/model/embeddings.py",
    "src/model/rope.py",
    "src/model/ffn.py",
    "src/model/attention.py",
    "src/model/mamba.py",
    "src/model/model.py",
]

all_imports = set()
all_code = []

for filepath in model_files:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (file not found)")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into lines to extract imports
    lines = content.split("\n")
    for line in lines:
        if line.startswith("import ") or (line.startswith("from ") and not line.startswith("from .") and not line.startswith("from src.")):
            all_imports.add(line.strip())
        elif line.startswith("from .") or line.startswith("from src."):
            continue  # strip internal relative/absolute module imports
        else:
            all_code.append(line)
    all_code.append("\n\n")

# Read the HF modeling file
target_hf_file = "src/hf_integration/modeling_banglagamba.py"
if os.path.exists(target_hf_file):
    with open(target_hf_file, "r", encoding="utf-8") as f:
        hf_lines = f.readlines()

    hf_imports = set()
    hf_code = []
    for line in hf_lines:
        if line.startswith("import ") or (line.startswith("from ") and not line.startswith("from .") and not line.startswith("from src.")):
            hf_imports.add(line.strip())
        elif line.startswith("from .configuration_banglagamba import"):
            hf_imports.add("from .configuration_banglagamba import BanglaGambaConfig")
        elif line.startswith("from .model") or line.startswith("from src."):
            continue
        else:
            hf_code.append(line)

    final_imports = sorted(list(all_imports | hf_imports))
    future_imports = [imp for imp in final_imports if "__future__" in imp]
    other_imports = [imp for imp in final_imports if "__future__" not in imp]

    final_text = (
        "\n".join(future_imports)
        + ("\n" if future_imports else "")
        + "\n".join(other_imports)
        + "\n\n"
        + "".join(hf_code)
    )

    with open(target_hf_file, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"Flattened and updated {target_hf_file} successfully!")
else:
    print(f"Target file {target_hf_file} does not exist.")
