"""
BanglaGamba Model Configuration.

Central dataclass for all model hyperparameters, loaded from YAML files.
Designed for a Mamba-3 / GQA hybrid Bangla foundation model (~199M params).

Layer pattern is specified directly in the YAML as a list:
    layer_types: [mamba, attn, mamba, attn, ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# Default layer pattern: 12 layers, 1:1 interleaved, Mamba first, Attention last
# M A M A M A M A M A M A
DEFAULT_LAYER_TYPES = [
    "mamba", "attn", "mamba", "attn", "mamba", "attn",
    "mamba", "attn", "mamba", "attn", "mamba", "attn",
]


@dataclass
class BanglaGambaConfig:
    """
    Complete model configuration for the BanglaGamba hybrid Mamba-3 / GQA LM.

    All fields are YAML-configurable. Use `BanglaGambaConfig.from_yaml(path)` to load.
    """

    # ── Core architecture ─────────────────────────────────────────────────
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16           # query heads
    n_kv_heads: int = 4         # KV heads for GQA (n_heads must be divisible by n_kv_heads)
    d_head: int = 64            # head dimension
    d_ff: int = 2560            # SwiGLU intermediate: floor(2/3 * 4 * 1024 / 256) * 256
    vocab_size: int = 48000
    seq_len: int = 2048
    dropout: float = 0.0
    bias: bool = False          # bias in linear layers

    # ── Layer pattern (explicit, no algorithm) ────────────────────────────
    # M A M A M A M A M A M A  (Mamba first, Attention last)
    layer_types: List[str] = field(default_factory=lambda: list(DEFAULT_LAYER_TYPES))

    # ── Mamba-3 specific ──────────────────────────────────────────────────
    mamba_d_state: int = 128
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 64

    # ── RoPE ──────────────────────────────────────────────────────────────
    rope_base: float = 10000.0

    # ── Norm ──────────────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-5

    # ── QK-Norm (stability with Muon optimizer) ───────────────────────────
    qk_norm: bool = True

    # ── Weight tying ──────────────────────────────────────────────────────
    tie_embeddings: bool = True

    def __post_init__(self):
        """Validate config."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"
        )
        assert len(self.layer_types) == self.n_layers, (
            f"layer_types has {len(self.layer_types)} entries but n_layers={self.n_layers}"
        )
        assert all(lt in ("mamba", "attn") for lt in self.layer_types), (
            f"layer_types must only contain 'mamba' or 'attn', got {self.layer_types}"
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BanglaGambaConfig":
        """Load config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @property
    def n_params_estimate(self) -> int:
        """Rough parameter count estimate (for logging, not exact)."""
        embed = self.vocab_size * self.d_model
        ffn_per_layer = 3 * self.d_model * self.d_ff

        attn_per_layer = (
            self.d_model * self.d_model      # Q
            + self.d_model * (self.n_kv_heads * self.d_head)  # K
            + self.d_model * (self.n_kv_heads * self.d_head)  # V
            + self.d_model * self.d_model     # O
        )

        d_inner = self.mamba_expand * self.d_model
        mamba_per_layer = (
            self.d_model * (2 * d_inner + 2 * self.mamba_d_state * self.mamba_ngroups + 100)
            + d_inner * self.d_model
        )

        total = embed  # embedding (tied, count once)
        for lt in self.layer_types:
            total += ffn_per_layer
            if lt == "attn":
                total += attn_per_layer
            else:
                total += mamba_per_layer

        return total

    def summary(self) -> str:
        """Return a human-readable summary of the config."""
        n_mamba = sum(1 for t in self.layer_types if t == "mamba")
        n_attn = sum(1 for t in self.layer_types if t == "attn")
        lines = [
            f"BanglaGamba Config Summary",
            f"  d_model={self.d_model}, n_layers={self.n_layers}",
            f"  n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, d_head={self.d_head}",
            f"  d_ff={self.d_ff}, vocab={self.vocab_size}, seq_len={self.seq_len}",
            f"  Layers: {n_mamba} Mamba + {n_attn} Attn",
            f"  Pattern: {' '.join('M' if t == 'mamba' else 'A' for t in self.layer_types)}",
            f"  Mamba-3: d_state={self.mamba_d_state}, expand={self.mamba_expand}, headdim={self.mamba_headdim}",
            f"  QK-Norm: {self.qk_norm}",
            f"  Est. params: {self.n_params_estimate / 1e6:.1f}M",
        ]
        return "\n".join(lines)
