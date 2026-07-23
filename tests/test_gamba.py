import sys
import torch
from evaluation_suit.eval.common.model_registry import load_model

loaded = load_model("gamba")
input_ids = torch.randint(0, 1000, (1, 10)).to(loaded.device)
attn_mask = torch.ones((1, 10)).to(loaded.device)

# Try standard HF output_hidden_states
try:
    outputs = loaded.model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    print("HF standard works:", hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None)
except Exception as e:
    print("HF standard failed:", str(e))

# See what inner is
inner = getattr(loaded.model, "model", loaded.model)
print("Inner class:", type(inner))
if hasattr(inner, "forward"):
    import inspect
    print("Inner forward args:", inspect.signature(inner.forward))
