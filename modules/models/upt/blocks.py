from typing import Any

import torch
from torch import nn

from modules.layers.attention import PerceiverAttention, DotProductAttention
from modules.layers.basics import SwiGLU, AdaLN

class TransformerBlock(nn.Module):
    """A transformer block with a single attention layer and a feedforward layer.
    
    Args:
        dim: hidden Dimension of the transformer block.
        num_heads: Number of attention heads.
        cond_dim: Width of the conditioning embedding (0 -> unconditional).
        cond_gate: Give each AdaLN a scalar gate (see AdaLN / init_cond_modulation).
    """

    def __init__(self, dim: int, num_heads: int, attn_ctor: type[nn.Module] = DotProductAttention,
                 cond_dim=0, cond_gate=False):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.attn = attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2 = AdaLN(dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.mlp = SwiGLU(dim)

    def forward(self, x: torch.Tensor, attn_kwargs: dict[str, Any] | None = None, cond=None) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor with shape (batch_size, seqlen/num_tokens, dim).
            attn_kwargs: Dict with arguments for the attention (such as the rope frequencies). Defaults to None.

        Returns:
            (batch_size, num_tokens, dim)
        """
        x = x + self.attn(self.norm1(x, cond), **(attn_kwargs or {}))
        x = x + self.mlp(self.norm2(x, cond))
        return x

class PerceiverBlock(nn.Module):
    """The PerceiverBlock takes different input tensors for the query and the key/value.

    Args:
        dim: Hidden dimension of the perceiver block.
        num_heads: Number of attention heads.
        cond_dim: Width of the conditioning embedding (0 -> unconditional).
        cond_gate: Give each AdaLN a scalar gate (see AdaLN / init_cond_modulation).
    """

    def __init__(self, dim: int, num_heads: int, cond_dim=0, cond_gate=False):
        super().__init__()
        self.norm1q = AdaLN(dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.norm1kv = AdaLN(dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.attn = PerceiverAttention(dim=dim, num_heads=num_heads)
        self.norm2 = AdaLN(dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.mlp = SwiGLU(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_kwargs: dict[str, Any] | None = None, cond=None) -> torch.Tensor:
        """Forward pass of the PerceiverBlock.

        Args:
            q: Input tensor with shape (batch_size, num_q_tokens, dim) for the query representations.
            kv: Input tensor with shape (batch_size, num_kv_tokens, dim) for the key and value representations.
            attn_kwargs: Dict with arguments for the attention (such as rope frequencies). Defaults to None.

        Returns:
            (batch_size, num_q_tokens, dim)
        """
        q = q + self.attn(q=self.norm1q(q, cond), kv=self.norm1kv(kv, cond), **(attn_kwargs or {}))
        q = q + self.mlp(self.norm2(q, cond))
        return q