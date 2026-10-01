import einops
import torch
import torch.nn.functional as F
from torch import nn
from modules.layers.rope import rope

class DotProductAttention(nn.Module):
    """Scaled dot-product attention module.

    Args:
        dim: Input dimension of the attention module.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability applied to attention weights (via SDPA's
            `dropout_p`) and to the output projection. Defaults to 0.0.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout_p = dropout

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, freqs: torch.Tensor = None,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, dim).
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.
            attn_mask: Optional mask broadcastable to (batch_size, num_heads, seqlen, seqlen);
                bool True where a key may be attended to. None keeps the flash backend.

        Returns:
            (batch_size, sequence_length, dim)
        """

        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        if freqs is not None:
            q = rope(q, freqs=freqs)
            k = rope(k, freqs=freqs)

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PerceiverAttention(nn.Module):
    """Perceiver style attention module. This module is similar to a cross-attention modules.

    Args:
        dim: Hidden dimension of the layer/module.
        num_heads: Number of attention heads. Defaults to 8.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the PerceiverAttention module.

        Args:
            q: Query tensor, shape (batch size, number of points/tokens, dim).
            kv: Key/value tensor, shape (batch size, number of latent tokens, dim).
            q_freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries.
            k_freqs: Frequencies for Rotary Positional Embedding (RoPE) of keys.

        Returns:
            (batch size, query sequence length, dim)
        """
        # project to attention space
        q = self.q(q)
        kv = self.kv(kv)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        # rope
        if q_freqs is not None:
            q = rope(q, freqs=q_freqs)
            k = rope(k, freqs=k_freqs)

        # attn
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x
    
class AnchorAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        num_anchor_tokens: int | None = None,
    ) -> torch.Tensor:
        """Self-attention between anchor tokens, other tokens (query tokens) have only cross-attention to anchor tokens

        Args:
            x: Tensor to apply self-attention over, shape (batch_size, sequence_length, dim).
            freqs: Frequencies for RoPE.
            num_anchor_tokens: Number of anchor tokens. If provided, the first num_anchor_tokens of x will be the
                anchors (full self-attention) and the other tokens will be the queries (only cross-attention to the
                anchor tokens).

        Returns:
            (batch_size, sequence_length, dim)
        """
        if num_anchor_tokens is None:
            return super().forward(x=x, freqs=freqs)
        else:
            x, queries = x.split([num_anchor_tokens, x.size(1) - num_anchor_tokens], dim=1)

        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        queries = einops.rearrange(
            F.linear(
                queries,
                weight=self.qkv.weight[: self.dim],
                bias=None if self.qkv.bias is None else self.qkv.bias[: self.dim],
            ),
            "bs seqlen (num_heads head_dim) -> bs num_heads seqlen head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        
        q = torch.concat([q, queries], dim=2) # full sequence length
        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs[:, :num_anchor_tokens]) # num_anchors
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x

class SharedweightsSplitattnAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        split_size: list[int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention between:
        - q=surface_anchors -> kv=surface_anchors
        - q=volume_anchors -> kv=volume_anchors
        - q=surface_queries -> kv=surface_anchors
        - q=volume_queries -> kv=volume_anchors

        Args:
            x: Tensor containing all anchors/queries (batch size, sequence length, dim).
            split_size: How to split x into:
                len(split_size) == 2: (surface_anchors, volume_anchors)
                len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch size, sequence length, dim)
        """
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        # split_size are (surface_anchors, surface_queries, volume_anchors, volume_queries)
        # len(split_size) == 2: (surface_anchors, volume_anchors) -> i.e., no queries
        # len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
        qs = q.split(split_size, dim=2)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if isinstance(split_size, list) and len(split_size) == 4:
            # surface + volume queries
            q1 = torch.concat([qs[0], qs[1]], dim=2)
            k1 = ks[0]
            v1 = vs[0]
            x1 = F.scaled_dot_product_attention(q1, k1, v1)
            q2 = torch.concat([qs[2], qs[3]], dim=2)
            k2 = ks[2]
            v2 = vs[2]
            x2 = F.scaled_dot_product_attention(q2, k2, v2)
            x = torch.concat([x1, x2], dim=2)
        else:
            # no queries -> self-attn within splits
            assert len(split_size) == 2
            if isinstance(split_size, list) and all(split_size[0] == ss for ss in split_size[1:]):
                # optimized for equal sized splits
                num_splits = len(qs)
                q = torch.concat(qs)
                k = torch.concat(ks)
                v = torch.concat(vs)
                x = F.scaled_dot_product_attention(q, k, v)
                x = torch.concat(x.chunk(chunks=num_splits), dim=2)
            else:
                # generic case
                x = torch.concat([F.scaled_dot_product_attention(qs[i], ks[i], vs[i]) for i in range(len(qs))], dim=2)

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x

class SharedweightsCrossattnAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        split_size: list[int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention between:
        - q=surface_anchors -> kv=volume_anchors
        - q=volume_anchors -> kv=surface_anchors
        - q=surface_queries -> kv=volume_anchors
        - q=volume_queries -> kv=surface_anchors

        Args:
            x: Tensor containing all anchors/queries (batch size, sequence length, dim).
            split_size: How to split x into:
                len(split_size) == 2: (surface_anchors, volume_anchors)
                len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch size, sequence length, dim)
        """
        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        # split_size are (surface_anchors, surface_queries, volume_anchors, volume_queries)
        # len(split_size) == 2: (surface_anchors, volume_anchors) -> i.e., no queries
        # len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
        # (could potentially be faster by skipping the kv parts of the qkv for the auxiliary splits)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if isinstance(split_size, list) and len(split_size) == 4:
            # surface + volume queries
            qs = q.split([split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=2)
            x1 = F.scaled_dot_product_attention(qs[0], ks[2], vs[2])
            x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
            x = torch.concat([x1, x2], dim=2)
        else:
            if isinstance(split_size, list) and len(split_size) == 2 and split_size[0] == split_size[1]:
                # efficient implementation when both splits are equally sized
                q = einops.rearrange(
                    q,
                    "batch_size num_heads (two seqlen) head_dim -> (two batch_size) num_heads seqlen head_dim",
                    two=2,
                )
                k = torch.concat([ks[1], ks[0]])
                v = torch.concat([vs[1], vs[0]])
                x = F.scaled_dot_product_attention(q, k, v)
                x = einops.rearrange(
                    x,
                    "(two batch_size) num_heads seqlen head_dim -> batch_size num_heads (two seqlen) head_dim",
                    two=2,
                )
            elif isinstance(split_size, list) and len(split_size) == 2:
                # generic implementation for two sized splits
                qs = q.split(split_size, dim=2)
                x1 = F.scaled_dot_product_attention(qs[0], ks[1], vs[1])
                x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
                x = torch.concat([x1, x2], dim=2)
            else:
                raise NotImplementedError
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x
    