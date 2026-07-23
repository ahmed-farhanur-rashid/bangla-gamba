import sys
import torch
from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.sentiment.run import get_hidden_states, ClassificationHead
from evaluation_suit.eval.sentiment.data import load_sentnob

loaded = load_model("banglabert")
dataset = load_sentnob()

text = dataset["train"][0]["text"]
encoding = loaded.tokenizer(text, return_tensors="pt", padding="max_length", max_length=10)
print(encoding)
input_ids = encoding["input_ids"].to(loaded.device)
attn_mask = encoding["attention_mask"].to(loaded.device)

hidden = get_hidden_states(loaded.model, input_ids, loaded.model_type, "banglabert")
print("Hidden shape:", hidden.shape)

head = ClassificationHead(hidden_size=768, num_classes=3).to(loaded.device)
logits = head(hidden.float(), attn_mask, loaded.model_type)
print("Logits shape:", logits.shape)
print("Logits:", logits)
