import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) for spatial positions.

    Args:
        dim: Dimensionality of the features to be embedded.
        spatial_dim: The spatial dimensionality of the positions (e.g., 2 for 2D positions, 3 for 3D positions). Defaults to 3.
        max_seq_length: Max sequence length. Defaults to 10000 as suggested in the original RoPE paper.
    """

    def __init__(self, dim, spatial_dim, max_seq_length=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for rotary embeddings"

        self.dim = dim
        self.spatial_dim = spatial_dim

        # Compute dimensions per spatial dimension
        max_dim_per_spatial_dim = dim // spatial_dim
        dim_per_spatial_dim = max_dim_per_spatial_dim & ~1 # This is equal to (max_dim_per_spatial_dim // 2) * 2

        # Compute the padding
        self.total_padding = dim - (dim_per_spatial_dim * spatial_dim)
        self.register_buffer("padding", torch.zeros(1, 1, self.total_padding // 2))

        # Compute the div_term for sine-cosine embedding
        div_term = torch.exp(torch.arange(0, dim_per_spatial_dim, 2) * (-math.log(max_seq_length) / dim_per_spatial_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, x, pos):
        """Applies RoPE to the features x based on the positions pos.

        Args:
            x: Features to apply RoPE to with shape (batch size, number points, dim).
            pos: Normalized positions with shape (batch size, number points, spatial_dim).

        Returns:
            Features with RoPE applied to with shape (batch size, number points, dim).
        """
        # Following UPT (https://arxiv.org/abs/2402.12365) and compute positional embeddings in float32 to avoid numerical instabilities
        with torch.autocast(device_type=str(pos.device).split(":")[0], enabled=False):
            pos = pos.float()
            theta = pos[..., None] @ self.div_term[None, ...]

        theta = rearrange(theta, "b n spatial_dim d -> b n (spatial_dim d)")

        # Add padding
        theta = torch.concat([theta, self.padding.expand(*theta.shape[:-1], -1)], dim=-1)

        # Apply rotation matrix in complex space following Llama 3 implementation
        # (https://github.com/meta-llama/llama3/blob/a0940f9cf7065d45bb6675660f80d305c041a754/llama/model.py#L65)
        rotation = torch.polar(torch.ones_like(theta), theta)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        embedded = torch.view_as_real(x_complex * rotation[:, None, ...]).flatten(3)

        return embedded.type_as(x)


class CrossAttention(nn.Module):
    """Computes multi-head cross-attention (https://arxiv.org/abs/1706.03762) between the query and key/value sequences. It
    optionally applies Rotary Positional Embedding (RoPE; https://arxiv.org/abs/2104.09864) to both the query and key features.

    Args:
        dim: Dimensionality of the query and key/value features.
        num_heads: Number of attention heads. Defaults to 8.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        dropout: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, dim, num_heads=8, spatial_dim=3, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Pre-layer normalization
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)

        # Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        # RoPE
        self.rope = RotaryPositionalEmbedding(dim=dim // num_heads, spatial_dim=spatial_dim)

        self.dropout = dropout

    def forward(self, q, kv, q_pos=None, kv_pos=None):
        """Applies pre-norm and computes cross-attention between q and kv.

        Args:
            q: Queries with shape (batch size, number query tokens, dim).
            kv: Key/value with shape (batch size, number key/value tokens, dim).
            q_pos (optional): Positions for the queries with shape (batch size, number query tokens, spatial_dim).
            kv_pos (optional): Positions for the key/value with shape (batch size, num key/value tokens, spatial_dim).

        Returns:
            Updated queries that attend to kv with shape (batch size, number query tokens, dim).
        """
        # Apply layer normalization
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Linear projections
        q = self.q(q)
        kv = self.kv(kv)

        # Split heads and keys/values
        q_heads = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads, d=self.head_dim)
        k_heads, v_heads = torch.tensor_split(rearrange(kv, "b kv (h d) -> b h kv d", h=2*self.num_heads, d=self.head_dim), 2, dim=1)

        # Apply RoPE if positions are provided
        if q_pos is not None and kv_pos is not None:
            q_heads = self.rope(q_heads, q_pos)
            k_heads = self.rope(k_heads, kv_pos)

        # Compute attention using PyTorch's scaled_dot_product_attention
        x = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=(self.dropout if self.training else 0.0))

        # Merge heads and output projection
        x = rearrange(x, "b h q d -> b q (h d)")
        x = self.out_proj(x)

        return x


class Modulator(nn.Module):
    """FiLM-like modulation (https://github.com/ethanjperez/film), ``(1 + scale) * x + shift``.

    The output layer is zero-initialized, so the module starts as the exact identity (adaLN-zero).

    Args:
        dim: Dimensionality of the features to be modulated.
        cond_dim: Dimensionality of the conditioning parameters.
        hidden_dim: Dimensionality of the hidden layer in the modulation MLP. Default is 128.
    """

    def __init__(self, dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim * 2))
        self.init_identity()

    def init_identity(self):
        """Zero the output layer (scale == shift == 0); re-applied after a model-wide weight init."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, params):
        """Modulates the features x based on the conditioning parameters params.

        Args:
            x: Features to be modulated with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim) or (batch size, 1, cond_dim).

        Returns:
            Modulated features with shape (batch size, number points, dim).
        """
        mod = self.mlp(params)
        # One modulation per sample: insert the token axis explicitly.
        if mod.dim() == x.dim() - 1:
            mod = mod.unsqueeze(-2)
        scale, shift = torch.tensor_split(mod, 2, dim=-1)
        x = (1.0 + scale) * x + shift
        return x


class SimulationParamModulatedMLP(nn.Module):
    """MLP with FiLM-like modulation (https://github.com/ethanjperez/film) based on simulation parameters.

    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
        cond_dim: Dimensionality of the conditioning parameters.
    """

    def __init__(self, dim, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.modulator = Modulator(hidden_dim, cond_dim)

    def forward(self, x, params):
        """Processes the features x with an MLP, modulated based on the conditioning parameters params.

        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).

        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.modulator(self.non_linearity(self.linear1(self.norm(x))), params)
        x = self.linear2(x)

        return x


class SwiGLUMLP(nn.Module):
    """Pre-norm SwiGLU feedforward with optional FiLM modulation of the gated hidden activations.

    The hidden width is 2/3 of the equivalent GELU MLP's, so the parameter count matches it.

    Args:
        dim: Dimensionality of the input and output features.
        exp_factor: Expansion of the *equivalent* GELU MLP, before the 2/3 scaling.
        cond_dim: Width of the FiLM conditioning vector; 0 builds no modulator.
        multiple_of: Round the hidden width up to this multiple for friendlier matmul shapes.
    """

    def __init__(self, dim, exp_factor=4., cond_dim=0, multiple_of=8):
        super().__init__()
        hidden = int(2 * dim * exp_factor / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.non_linearity = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.modulator = Modulator(hidden, cond_dim) if cond_dim > 0 else None

    def forward(self, x, params=None):
        """Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim). Ignored when
                the block was built with cond_dim == 0.

        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        gate, up = self.fc1(self.norm(x)).chunk(2, dim=-1)
        h = self.non_linearity(gate) * up
        if self.modulator is not None:
            h = self.modulator(h, params)
        return self.fc2(h)


class PlainMLP(nn.Module):
    """Plain multi-layer perceptron (MLP) **without** modulation

    Args:
        dim: Dimensionality of the input and output features.
        hidden_dim: Dimensionality of the hidden layer of the MLP.
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.non_linearity = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, params=None):
        """Processes the features x with an MLP without modulation.

        Args:
            x: Features with shape (batch size, number points, dim).
            params: Conditioning parameters will be ignored.

        Returns:
            Processed features with shape (batch size, number points, dim).
        """
        x = self.linear2(self.non_linearity(self.linear1(self.norm(x))))

        return x
