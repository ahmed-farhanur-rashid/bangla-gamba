"""
BanglaGamba Hugging Face Integration Model File.

Self-contained module implementing BanglaGambaForCausalLM, registered for
AutoModelForCausalLM with trust_remote_code=True.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

torch_import_err = None
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
except ImportError as e:
    torch_import_err = e

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    GenerationMixin,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput

from .configuration_banglagamba import BanglaGambaConfig

try:
    from mamba_ssm.modules.mamba3 import Mamba3
    _HAS_MAMBA = True
except ImportError:
    Mamba3 = None
    _HAS_MAMBA = False


# ── RMSNorm & Embedding Components ──────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class PerHeadRMSNorm(nn.Module):
    """Per-head RMSNorm for QK-Norm."""

    def __init__(self, d_head: int, n_heads: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(x.dtype) * self.weight


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    @property
    def weight(self) -> torch.Tensor:
        return self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(token_ids))


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with cached cos/sin."""

    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even, got {d_head}"
        self.d_head = d_head
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = positions.shape[1]
        dtype = q.dtype

        cos = self.cos_cached[:T].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:T].to(dtype=dtype).unsqueeze(0).unsqueeze(2)

        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin

        return q_rot, k_rot


# ── FFN Sublayer ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ── Attention & Mamba Mixers ─────────────────────────────────────────────────

class GQAttention(nn.Module):
    """Grouped Query Attention with QK-Norm and RoPE."""

    def __init__(self, config: BanglaGambaConfig, layer_idx: int = 0):
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

        if self.qk_norm:
            self.q_norm = PerHeadRMSNorm(config.d_head, config.n_heads, eps=config.rms_norm_eps)
            self.k_norm = PerHeadRMSNorm(config.d_head, config.n_kv_heads, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = rope(q, k, positions)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_output)


class PyTorchMamba3(nn.Module):
    """Native PyTorch fallback for Mamba-3 SSM module (CPU / environments without mamba_ssm)."""

    def __init__(self, config: BanglaGambaConfig, layer_idx: int = 0):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.mamba_d_state
        self.expand = config.mamba_expand
        self.d_inner = self.d_model * self.expand
        self.headdim = config.mamba_headdim
        self.nheads = self.d_inner // self.headdim
        self.ngroups = config.mamba_ngroups
        self.num_bc_heads = config.mamba_ngroups
        self.mimo_rank = 1
        self.num_rope_angles = self.nheads

        d_in_proj = (
            self.d_inner * 2
            + self.d_state * self.num_bc_heads * self.mimo_rank * 2
            + self.nheads * 3
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.B_bias = nn.Parameter(
            torch.zeros(self.nheads, self.mimo_rank, self.d_state)
        )
        self.C_bias = nn.Parameter(
            torch.zeros(self.nheads, self.mimo_rank, self.d_state)
        )
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.B_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)
        self.C_norm = RMSNorm(self.d_state, eps=config.rms_norm_eps)

    def forward(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, _ = u.shape
        zxBCdt = self.in_proj(u)
        splits = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        z, x, B_vec, C_vec, dt, A_vec, trap, angles = torch.split(
            zxBCdt, splits, dim=-1
        )

        B_vec = (
            self.B_norm(B_vec.view(B, T, self.d_state)).view(B, T, 1, 1, self.d_state)
            + self.B_bias.unsqueeze(0).unsqueeze(0)
        )
        C_vec = (
            self.C_norm(C_vec.view(B, T, self.d_state)).view(B, T, 1, 1, self.d_state)
            + self.C_bias.unsqueeze(0).unsqueeze(0)
        )

        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(A_vec)

        x_heads = x.view(B, T, self.nheads, self.headdim)
        out_heads = []
        h = torch.zeros(
            B, self.nheads, self.headdim, self.d_state, device=u.device, dtype=u.dtype
        )

        for t in range(T):
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)
            A_t = A[:, t].unsqueeze(-1).unsqueeze(-1)
            decay = torch.exp(A_t * dt_t)

            b_t = B_vec[:, t, 0, 0, None, :]
            c_t = C_vec[:, t, 0, 0, None, :]
            x_t = x_heads[:, t, :, :, None]

            h = decay * h + dt_t * (x_t * b_t)
            y_t = (h * c_t).sum(dim=-1) + self.D[None, :, None] * x_heads[:, t]
            out_heads.append(y_t)

        out_heads = torch.stack(out_heads, dim=1)
        out = out_heads.view(B, T, self.d_inner) * F.silu(z)
        return self.out_proj(out)


class MambaBlock(nn.Module):
    """Wrapper around Mamba-3 SSM module with PyTorch native fallback."""

    def __init__(self, config: BanglaGambaConfig, layer_idx: int = 0):
        super().__init__()
        if _HAS_MAMBA:
            self.mamba = Mamba3(
                d_model=config.d_model,
                d_state=config.mamba_d_state,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                ngroups=config.mamba_ngroups,
                chunk_size=config.mamba_chunk_size,
                layer_idx=layer_idx,
            )
        else:
            self.mamba = PyTorchMamba3(config=config, layer_idx=layer_idx)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mamba(x)


# ── Transformer Block & Base Model ───────────────────────────────────────────

class BanglaGambaBlock(nn.Module):
    """Single BanglaGamba Block (Mamba-3 or GQA + SwiGLU)."""

    def __init__(self, config: BanglaGambaConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.layer_idx = layer_idx
        self.gradient_checkpointing = False

        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        if layer_type == "attn":
            self.mixer = GQAttention(config, layer_idx)
        else:
            self.mixer = MambaBlock(config, layer_idx)

        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff, bias=config.bias, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor = None,
        rope: RotaryEmbedding = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        if self.gradient_checkpointing and self.training and self.layer_type == "attn":
            h = grad_checkpoint(self.mixer, h, positions, rope, use_reentrant=False)
        elif self.layer_type == "attn":
            h = self.mixer(h, positions=positions, rope=rope)
        else:
            h = self.mixer(h)
        x = x + h

        h = self.norm2(x)
        if self.gradient_checkpointing and self.training:
            h = grad_checkpoint(self.ffn, h, use_reentrant=False)
        else:
            h = self.ffn(h)
        x = x + h

        return x


class BanglaGambaModel(nn.Module):
    """Raw BanglaGamba Core Model Stack."""

    def __init__(self, config: BanglaGambaConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embedding = TokenEmbedding(config.vocab_size, config.d_model, dropout=config.dropout)
        self.rope = RotaryEmbedding(
            d_head=config.d_head,
            max_seq_len=config.seq_len,
            base=config.rope_base,
        )

        self.layers = nn.ModuleList([
            BanglaGambaBlock(config, layer_idx=i, layer_type=lt)
            for i, lt in enumerate(config.layer_types)
        ])

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "mamba" not in name and "embed" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
                p.data.mul_(scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embedding(input_ids)
        positions = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        for layer in self.layers:
            x = layer(x, positions=positions, rope=self.rope)

        x = self.final_norm(x)
        if return_hidden:
            return x

        return self.lm_head(x)


# ── Hugging Face PreTrainedModel Base & CausalLM Wrapper ──────────────────────

class BanglaGambaPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = BanglaGambaConfig
    base_model_prefix = "model"
    _no_split_modules = ["BanglaGambaBlock", "GQAttention", "MambaBlock"]

    def _init_weights(self, module):
        pass


class BanglaGambaForCausalLM(BanglaGambaPreTrainedModel, GenerationMixin):
    config_class = BanglaGambaConfig
    base_model_prefix = "model"

    def __init__(self, config: BanglaGambaConfig):
        super().__init__(config)
        self.model = BanglaGambaModel(config)
        self.generation_config = GenerationConfig.from_model_config(config)
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> dict:
        return {"input_ids": input_ids}

    def get_input_embeddings(self):
        return self.model.embedding.embed

    def set_input_embeddings(self, value):
        self.model.embedding.embed = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutput:
        if attention_mask is not None:
            if not torch.all(attention_mask.bool()):
                raise NotImplementedError(
                    "BanglaGamba v1 does not support padded batches natively. "
                    "Sequences in a batch must be dense, unpadded, and equal length. "
                    "Process variable-length sequences with batch_size=1 without padding."
                )

        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )


# Register model and config with HuggingFace Auto classes
AutoConfig.register("banglagamba", BanglaGambaConfig)
AutoModelForCausalLM.register(BanglaGambaConfig, BanglaGambaForCausalLM)
