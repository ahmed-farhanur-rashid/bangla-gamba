import json
from transformers import PreTrainedTokenizerFast
from collections import Counter
import random

tokenizer = PreTrainedTokenizerFast.from_pretrained('saved/tokenizer/hf')
unk_id = tokenizer.unk_token_id

with open('saved/data/cleaned/english.jsonl') as f:
    lines = [next(f) for _ in range(3000)]

unk_characters = Counter()
for line in lines:
    doc = json.loads(line)
    text = doc.get("text", "")
    if not text: continue
    
    # We can encode with return_offsets_mapping=True for Fast tokenizers
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = encoding['input_ids']
    offsets = encoding['offset_mapping']
    
    for tid, offset in zip(ids, offsets):
        if tid == unk_id:
            start, end = offset
            unk_char = text[start:end]
            unk_characters[unk_char] += 1

for char, count in unk_characters.most_common(20):
    print(f"Count {count}: {char!r} (unicode: {[hex(ord(c)) for c in char]})")
