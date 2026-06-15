"""
Grouped Query Attention (GQA) with QK-Norm for BanglaGamba.

Uses PyTorch's scaled_dot_product_attention with native GQA support
(enable_gqa=True). Includes per-head QK-Norm (spec §6.1) for stability
with the Muon optimizer.

QK-Norm: RMSNorm applied to Q and K projections per-head, BEFORE RoPE.
This bounds attention logit magnitudes regardless of upstream weight scale
drift — cheap insurance for a single-shot training run.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.embeddings import PerHeadRMSNorm
from src.model.positional_encoder.rope import RotaryEmbedding


class GQAttention(nn.Module):
    """
    Grouped Query Attention with optional QK-Norm and RoPE.

    KV heads are NOT expanded before SDPA — PyTorch's native GQA support
    (enable_gqa=True) handles head broadcasting internally.

    Parameters
    ----------
    config : BanglaGambaConfig
        Model configuration.
    layer_idx : int
        Layer index (for potential per-layer modifications).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.layer_idx = layer_idx
        self.qk_norm = config.qk_norm

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=config.bias)

        # QK-Norm: per-head RMSNorm on Q and K before RoPE (spec §6.1)
        if self.qk_norm:
            self.q_norm = PerHeadRMSNorm(config.d_head, config.n_heads, eps=config.rms_norm_eps)
            self.k_norm = PerHeadRMSNorm(config.d_head, config.n_kv_heads, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        positions: torch.Tensor,                # (B, T) int64
        rope: RotaryEmbedding,                  # RoPE module
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head)

        # QK-Norm: per-head RMSNorm BEFORE RoPE (spec §6.1)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K only
        q, k = rope(q, k, positions)

        # Transpose to (B, H, T, D) for SDPA
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)  # (B, n_kv_heads, T, d_head)
        v = v.transpose(1, 2)  # (B, n_kv_heads, T, d_head)

        # Flash Attention 2 via SDPA with native GQA support
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        # (B, n_heads, T, d_head) -> (B, T, n_heads * d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(attn_output)
