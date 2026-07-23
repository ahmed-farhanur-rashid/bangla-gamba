import sys
import torch
from evaluation_suit.eval.common.model_registry import load_model
from evaluation_suit.eval.sentiment.run import get_hidden_states

loaded = load_model("gamba")
input_ids = torch.randint(0, 1000, (1, 10)).to(loaded.device)
attn_mask = torch.ones((1, 10)).to(loaded.device)

try:
    hidden = get_hidden_states(loaded.model, input_ids, attn_mask, loaded.model_type, "gamba")
    print("Gamba get_hidden_states works! Shape:", hidden.shape)
except Exception as e:
    print("Gamba get_hidden_states FAILED:", str(e))
