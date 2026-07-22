import os
import json
from typing import List, Union
from transformers import PreTrainedTokenizerFast

try:
    from bnunicodenormalizer import Normalizer
    _HAS_BNORM = True
except ImportError:
    _HAS_BNORM = False


class BanglaGambaTokenizer(PreTrainedTokenizerFast):
    """
    Custom Tokenizer for BanglaGamba.

    Automatically applies `bnunicodenormalizer` to all inputs before tokenization.
    This guarantees that raw inference text matches the normalized pretraining corpus,
    preventing severe hallucinations.
    """

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
                from tokenizers import Tokenizer
                tok._tokenizer = Tokenizer.from_file(tok_json)
        return tok

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _HAS_BNORM:
            self.bnorm = Normalizer()
        else:
            import warnings
            warnings.warn(
                "bnunicodenormalizer is not installed. Normalization will be skipped, "
                "which may cause the model to output gibberish. "
                "Please run `pip install bnunicodenormalizer`."
            )
            self.bnorm = None

    def _normalize_text(self, text: str) -> str:
        """Apply bnunicodenormalizer word-by-word."""
        if not self.bnorm or not isinstance(text, str):
            return text

        words = text.split()
        normalized_words = []
        for word in words:
            res = self.bnorm(word)
            if res and res.get("normalized"):
                normalized_words.append(res["normalized"])
            else:
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

    def encode(self, text, **kwargs):
        text = self._normalize_input(text)
        return super().encode(text, **kwargs)

    def __call__(self, text=None, text_pair=None, **kwargs):
        if text is not None:
            text = self._normalize_input(text)
        if text_pair is not None:
            text_pair = self._normalize_input(text_pair)
        return super().__call__(text, text_pair=text_pair, **kwargs)
