"""
BanglaGamba Hybrid Mamba-3 / GQA Language Model.

Builds a heterogeneous layer stack from BanglaGambaConfig.layer_types:
  - Attention layers: RMSNorm → GQA (w/ QK-Norm) → Residual → RMSNorm → SwiGLU → Residual
  - Mamba layers:     RMSNorm → Mamba3 → Residual → RMSNorm → SwiGLU → Residual

Stability features (spec §6):
  - QK-Norm: per-head RMSNorm on Q/K before RoPE
  - Residual init scaling: output projections scaled by 1/sqrt(2 * n_layers)
  - Embedding init: std = 1/sqrt(d_model)
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from src.model.config import BanglaGambaConfig
from src.model.embeddings import RMSNorm, TokenEmbedding
from src.model.attention import GQAttention
from src.model.ffn import SwiGLU
from src.model.mamba import MambaBlock
from src.model.rope import RotaryEmbedding


class BanglaGambaBlock(nn.Module):
    """
    Single transformer block — either Mamba-based or GQA-based.
    Both variants include a SwiGLU FFN sublayer (following Jamba/Zamba block shape).

    Each block = [Mixer (Mamba-3 OR GQA)] + [SwiGLU FFN]
    Both pre-normed with RMSNorm, both with residual connections.
    """

    def __init__(self, config: BanglaGambaConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.gradient_checkpointing = False

        # Pre-norm before mixer
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # Mixer: either GQA or Mamba3
        if layer_type == "attn":
            self.mixer = GQAttention(config, layer_idx)
        else:
            self.mixer = MambaBlock(config, layer_idx)

        # Pre-norm before FFN
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # FFN: SwiGLU
        self.ffn = SwiGLU(config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, d_model)
        positions: torch.Tensor = None,         # (B, T) int64 — needed for attn
        rope: RotaryEmbedding = None,           # RoPE module — needed for attn
    ) -> torch.Tensor:
        # ── Mixer ────────────────────────────────────────────────────────
        h = self.norm1(x)
        if self.gradient_checkpointing and self.training and self.layer_type == "attn":
            # Checkpoint GQA attention (saves Q/K/V/output activations).
            # Mamba-3 stays eager — its Triton kernels may not support
            # recomputation under torch.utils.checkpoint.
            h = grad_checkpoint(self.mixer, h, positions, rope, use_reentrant=False)
        elif self.layer_type == "attn":
            h = self.mixer(h, positions=positions, rope=rope)
        else:
            h = self.mixer(h)
        x = x + h  # residual

        # ── FFN ──────────────────────────────────────────────────────────
        h = self.norm2(x)
        if self.gradient_checkpointing and self.training:
            h = grad_checkpoint(self.ffn, h, use_reentrant=False)
        else:
            h = self.ffn(h)
        x = x + h  # residual

        return x


class BanglaGambaModel(nn.Module):
    """
    BanglaGamba: Hybrid Mamba-3 / GQA Language Model.

    Architecture:
        Token Embedding → [BanglaGambaBlock × n_layers] → RMSNorm → LM Head

    The layer stack is heterogeneous: each layer is either a Mamba-3 block
    or a GQA attention block, as specified by config.layer_types.
    Both block types include a SwiGLU FFN sublayer.

    Weight tying: the LM head shares weights with the token embedding.

    Init (spec §6.3): residual-branch output projections (out_proj, o_proj,
    down_proj) are scaled by 1/sqrt(2 * n_layers) to prevent activation
    variance growth with depth.
    """

    def __init__(self, config: BanglaGambaConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # Token embedding
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, dropout=config.dropout)

        # Positional encoding: standard RoPE
        self.rope = RotaryEmbedding(
            d_head=config.d_head,
            max_seq_len=config.seq_len,
            base=config.rope_base,
        )

        # Heterogeneous layer stack
        self.layers = nn.ModuleList([
            BanglaGambaBlock(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])

        # Final norm
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head (output projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights per spec:
        - General 2D weights: std=0.02 (except Mamba internals which self-init)
        - Embedding: std = 1/sqrt(d_model) (handled by TokenEmbedding.__init__)
        - Residual output projections: scaled by 1/sqrt(2 * n_layers) (spec §6.3)
        """
        # Standard init for non-Mamba 2D weights
        for name, p in self.named_parameters():
            if p.dim() > 1 and "mamba" not in name and "embed" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

        # Residual-branch output scaling (spec §6.3)
        # Scale out_proj (Mamba), o_proj (GQA), down_proj (FFN) by 1/sqrt(2*n_layers)
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
                p.data.mul_(scale)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = False
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,                # (B, T) int64
        return_hidden: bool = False,            # If True, return pre-lm_head hidden states
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: (B, T) token IDs.
            return_hidden: If True, return normalized hidden states before lm_head.
                Used with LigerFusedLinearCrossEntropyLoss which fuses the
                lm_head projection into the loss computation.

        Returns:
            If return_hidden=False: logits (B, T, vocab_size)
            If return_hidden=True: hidden_states (B, T, d_model) after final_norm
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.embedding(input_ids)  # (B, T, d_model)

        # Positions: simple 0..T-1 for each sequence
        positions = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x, positions=positions, rope=self.rope)

        # Final norm
        x = self.final_norm(x)

        if return_hidden:
            return x  # (B, T, d_model) — for fused linear cross-entropy

        # LM head
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> dict:
        """Return parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        embed_params = sum(p.numel() for p in self.embedding.parameters())
        mamba_params = sum(
            p.numel() for name, p in self.named_parameters() if "mamba" in name.lower()
        )
        attn_params = sum(
            p.numel() for name, p in self.named_parameters()
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"])
        )
        ffn_params = sum(
            p.numel() for name, p in self.named_parameters()
            if any(k in name for k in ["gate_proj", "up_proj", "down_proj"])
        )

        return {
            "total": total,
            "trainable": trainable,
            "embedding": embed_params,
            "mamba": mamba_params,
            "attention": attn_params,
            "ffn": ffn_params,
        }
