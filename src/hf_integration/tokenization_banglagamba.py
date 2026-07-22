import os
import json
import torch
from typing import List, Union, Optional
from transformers import PreTrainedTokenizerFast

try:
    from bnunicodenormalizer import Normalizer
    _HAS_BNORM = True
except ImportError:
    _HAS_BNORM = False


class BanglaGambaTokenizer(PreTrainedTokenizerFast):
    """
    Custom Tokenizer for BanglaGamba.

    Automatically applies `bnunicodenormalizer` to all inputs before tokenization
    and ensures BOS token (<s>, ID 2) is prepended to input prompts.
    """

    bos_token = "<s>"
    bos_token_id = 2
    eos_token = "</s>"
    eos_token_id = 3
    pad_token = "<pad>"
    pad_token_id = 0
    unk_token = "<unk>"
    unk_token_id = 1

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        tok = super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        if (
            hasattr(tok, "backend_tokenizer")
            and not tok.backend_tokenizer.pre_tokenizer
            and isinstance(pretrained_model_name_or_path, (str, os.PathLike))
        ):
            tok_json = os.path.join(str(pretrained_model_name_or_path), "tokenizer.json")
            if os.path.exists(tok_json):
                with open(tok_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "pre_tokenizer" in data and data["pre_tokenizer"]:
                    pass
        if _HAS_BNORM and not hasattr(tok, "bnorm"):
            tok.bnorm = Normalizer()
        return tok

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _HAS_BNORM and not hasattr(self, "bnorm"):
            self.bnorm = Normalizer()

    def _normalize_text(self, text: str) -> str:
        if not _HAS_BNORM or not hasattr(self, "bnorm"):
            return text

        words = text.split()
        normalized_words = []
        for word in words:
            try:
                res = self.bnorm(word)
                if res and res.get("normalized"):
                    normalized_words.append(res["normalized"])
                else:
                    normalized_words.append(word)
            except Exception:
                normalized_words.append(word)

        return " ".join(normalized_words)

    def _normalize_input(self, text):
        if isinstance(text, str):
            return self._normalize_text(text)
        elif isinstance(text, (list, tuple)):
            return [self._normalize_input(t) for t in text]
        return text

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        batch_text_or_text_pairs = self._normalize_input(batch_text_or_text_pairs)
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    def _encode_plus(self, text, *args, **kwargs):
        text = self._normalize_input(text)
        return super()._encode_plus(text, *args, **kwargs)

    def tokenize(self, text, **kwargs):
        text = self._normalize_input(text)
        return super().tokenize(text, **kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        if token_ids_1 is None:
            return bos + token_ids_0
        return bos + token_ids_0 + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        mask = [1] if self.bos_token_id is not None else []
        if token_ids_1 is None:
            return mask + ([0] * len(token_ids_0))
        return mask + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1))

    def decode(self, token_ids, **kwargs) -> str:
        text = super().decode(token_ids, **kwargs)
        return text.replace("▁", " ").replace(" ", " ")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = super().convert_tokens_to_string(tokens)
        return text.replace("▁", " ").replace(" ", " ")

    def encode(self, text, add_special_tokens: bool = True, **kwargs):
        text = self._normalize_input(text)
        tokens = super().encode(text, add_special_tokens=False, **kwargs)
        if add_special_tokens:
            if isinstance(tokens, list):
                if not tokens or tokens[0] != self.bos_token_id:
                    tokens = [self.bos_token_id] + tokens
            elif torch.is_tensor(tokens):
                # 1D or 2D tensor
                if tokens.ndim == 1:
                    if tokens.numel() == 0 or tokens[0].item() != self.bos_token_id:
                        bos = torch.tensor([self.bos_token_id], device=tokens.device, dtype=tokens.dtype)
                        tokens = torch.cat([bos, tokens])
                elif tokens.ndim == 2:
                    if tokens.numel() == 0 or tokens[0, 0].item() != self.bos_token_id:
                        bos = torch.tensor([[self.bos_token_id]], device=tokens.device, dtype=tokens.dtype)
                        tokens = torch.cat([bos, tokens], dim=1)
        return tokens

    def __call__(self, text=None, text_pair=None, add_special_tokens: bool = True, **kwargs):
        if text is not None:
            text = self._normalize_input(text)
        if text_pair is not None:
            text_pair = self._normalize_input(text_pair)
        out = super().__call__(text, text_pair=text_pair, add_special_tokens=add_special_tokens, **kwargs)
        
        if add_special_tokens and "input_ids" in out:
            ids = out["input_ids"]
            if isinstance(ids, list):
                if ids and ids[0] != self.bos_token_id:
                    out["input_ids"] = [self.bos_token_id] + ids
                    if "attention_mask" in out:
                        out["attention_mask"] = [1] + out["attention_mask"]
            elif torch.is_tensor(ids):
                if ids.ndim == 1 and ids.numel() > 0 and ids[0].item() != self.bos_token_id:
                    bos_tensor = torch.tensor([self.bos_token_id], device=ids.device, dtype=ids.dtype)
                    out["input_ids"] = torch.cat([bos_tensor, ids])
                    if "attention_mask" in out:
                        mask_tensor = torch.tensor([1], device=ids.device, dtype=out["attention_mask"].dtype)
                        out["attention_mask"] = torch.cat([mask_tensor, out["attention_mask"]])
                elif ids.ndim == 2 and ids.numel() > 0 and ids[0, 0].item() != self.bos_token_id:
                    bos_tensor = torch.tensor([[self.bos_token_id]] * ids.shape[0], device=ids.device, dtype=ids.dtype)
                    out["input_ids"] = torch.cat([bos_tensor, ids], dim=1)
                    if "attention_mask" in out:
                        mask_tensor = torch.tensor([[1]] * ids.shape[0], device=ids.device, dtype=out["attention_mask"].dtype)
                        out["attention_mask"] = torch.cat([mask_tensor, out["attention_mask"]], dim=1)
        return out
