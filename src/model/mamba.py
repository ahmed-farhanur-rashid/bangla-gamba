"""
Mamba-3 wrapper for BanglaGamba.

Thin wrapper around the official Mamba-3 module from mamba_ssm.
Adapts the Mamba3 interface to our block structure.

Do NOT apply RoPE or any positional encoding inside Mamba-3 blocks —
the SSM recurrence encodes position implicitly.
"""

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba3 import Mamba3


class MambaBlock(nn.Module):
    """
    Wrapper around the official Mamba-3 SSM module.

    Parameters
    ----------
    config : BanglaGambaConfig
        Model configuration.
    layer_idx : int
        Layer index within the full model stack.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.mamba = Mamba3(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
            layer_idx=layer_idx,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        return self.mamba(x)
