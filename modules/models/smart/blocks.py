import torch.nn as nn
from modules.models.smart.utils import (CrossAttention, Modulator, PlainMLP,
                                        SimulationParamModulatedMLP, SwiGLUMLP)


def _pointwise_mlp(dim, cond_dim, shared=None, mlp_type="gelu"):
    """The pointwise MLP a block applies to its stream, or `shared` if one was handed in.

    mlp_type: "gelu" (original two-matmul MLP; plain when cond_dim == 0) or "swiglu".
    """
    if shared is not None:
        return shared
    if mlp_type == "swiglu":
        return SwiGLUMLP(dim=dim, cond_dim=cond_dim)
    if mlp_type != "gelu":
        raise ValueError(f"mlp_type must be 'gelu' or 'swiglu', got {mlp_type!r}")
    if cond_dim > 0:
        return SimulationParamModulatedMLP(dim=dim, hidden_dim=dim * 4, cond_dim=cond_dim)
    return PlainMLP(dim=dim, hidden_dim=dim * 4)


class EncoderBlock(nn.Module):
    """The encoder block updates the latent geometry by attending to a cross-attended version of itself. This means that
    first, the latent geometry attends to a subsampled version of the input geometry to integrate geometric information,
    and then it attends to this cross-attended version of itself to refine the latent representation.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        mlp_type: "gelu" (default) or "swiglu" for the pointwise MLP.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2,
                 mlp_type="gelu"):
        super().__init__()
        self.geo_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Pointwise MLP
        self.mlp = _pointwise_mlp(dim, cond_dim, mlp_type=mlp_type)

    def forward(self, latent_geometry, subsampled_geometry, params, latent_geometry_pos=None, subsampled_geometry_pos=None):
        """Updates the latent geometry by attending to a cross-attended version of itself that first attends to a subsampled
        version of the input geometry.

        Args:
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            subsampled_geometry: Subsampled input geometry with shape (batch size, number subsampled points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.
            subsampled_geometry_pos (optional): Positions of the subsampled input geometry for the positional embeddings with shape (batch size, number subsampled points, spatial_dim) . Defaults to None.

        Returns:
            tuple: A tuple containing:
                - Updated latent geometry with shape (batch size, number latent points, dim).
                - Latent geometry after geometry cross-attention and before cross-attention and MLP with shape (batch size, number latent points, dim).
        """
        # First cross-attention with the subsampled geometry
        latent_geometry_cross = latent_geometry + self.attn_dropout(self.geo_attn(q=latent_geometry, kv=subsampled_geometry, q_pos=latent_geometry_pos, kv_pos=subsampled_geometry_pos))

        # Update the initial latent geometry by attending to the cross-attended version of itself
        latent_geometry_self = latent_geometry + self.attn_dropout(self.cross_attn(q=latent_geometry, kv=latent_geometry_cross, q_pos=latent_geometry_pos, kv_pos=latent_geometry_pos))

        # Pointwise MLP
        latent_geometry_mlp = latent_geometry_self + self.mlp(latent_geometry_self, params)

        return latent_geometry_mlp, latent_geometry_cross


class ICEncoderBlock(nn.Module):
    """One in-context encoder block: a single latent stream that writes and reads two memories.

        M_geo = L + geo_writer(q=L, kv=subsampled target geometry)
        M_fld = L + fld_mod(fld_writer(q=L, kv=subsampled context field))
        h     = L + geo_reader(q=L, kv=M_geo)
        h     = h + fld_reader(q=h, kv=M_fld)
        out   = h + mlp(h)

    The memories are returned so `ICDecoderBlock` can read the same tensors (exact weight sharing).

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        context_cond_dim: Width of the `context_cond` vector that modulates the field memory.
            0 disables the modulation. Defaults to 0.
        mlp_type: "gelu" (default) or "swiglu" for the pointwise MLP.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, context_cond_dim=0,
                 mlp_type="gelu"):
        super().__init__()
        ca = lambda: CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)

        # Writers build the memories (encoder-only).
        self.geo_writer = ca()
        self.fld_writer = ca()

        # FiLM on the field memory (identity at init).
        self.fld_mod = Modulator(dim, cond_dim=context_cond_dim) if context_cond_dim > 0 else None

        # Readers and the MLP advance the stream; these are what the decoder shares.
        self.geo_reader = ca()
        self.fld_reader = ca()
        self.mlp = _pointwise_mlp(dim, cond_dim, mlp_type=mlp_type)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, latent, subsampled_geometry, subsampled_field, params,
                context_cond=None, latent_pos=None, geometry_pos=None, field_pos=None):
        """Advances the latent stream by one block and returns the memories it wrote.

        Args:
            latent: Latent stream with shape (batch size, number latent points, dim).
            subsampled_geometry: Embedded subsample of the target geometry, (batch size, number subsampled geometry points, dim).
            subsampled_field: Embedded subsample of the context field clouds, (batch size, number subsampled field points, dim).
            params: Conditioning parameters of the TARGET run with shape (batch size, cond_dim).
            context_cond (optional): Conditioning describing the CONTEXT run, (batch size, context_cond_dim).
                Required iff this block was built with `context_cond_dim > 0`.
            latent_pos (optional): Positions of the latent stream, (batch size, number latent points, spatial_dim).
            geometry_pos (optional): Positions of the subsampled target geometry.
            field_pos (optional): Positions of the subsampled context field points, in the context's own frame.

        Returns:
            tuple: (updated latent stream, geometry memory, field memory), all
                (batch size, number latent points, dim). The two memories live at `latent_pos`.
        """
        # Write: gather the target geometry and the context field onto the latent positions.
        geometry_memory = latent + self.attn_dropout(
            self.geo_writer(q=latent, kv=subsampled_geometry, q_pos=latent_pos, kv_pos=geometry_pos))

        field_update = self.fld_writer(q=latent, kv=subsampled_field, q_pos=latent_pos, kv_pos=field_pos)
        if self.fld_mod is not None:
            field_update = self.fld_mod(field_update, context_cond)
        field_memory = latent + self.attn_dropout(field_update)

        # Read: advance the stream (q always from the stream, never from a memory).
        latent = latent + self.attn_dropout(
            self.geo_reader(q=latent, kv=geometry_memory, q_pos=latent_pos, kv_pos=latent_pos))
        latent = latent + self.attn_dropout(
            self.fld_reader(q=latent, kv=field_memory, q_pos=latent_pos, kv_pos=latent_pos))

        # Pointwise MLP
        latent = latent + self.mlp(latent, params)

        return latent, geometry_memory, field_memory


class ICDecoderBlock(nn.Module):
    """One in-context decoder block: the `ICEncoderBlock` stream update applied to query tokens.

    With `shared_block` given it owns no parameters: it reuses that block's readers and MLP.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        shared_block: `ICEncoderBlock` whose readers and MLP to reuse. None builds
            independent ones, which unties the decoder from the encoder. Defaults to None.
        mlp_type: "gelu" (default) or "swiglu". Ignored when `shared_block` is given.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_block=None,
                 mlp_type="gelu"):
        super().__init__()
        if shared_block is None:
            ca = lambda: CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim)
            self.geo_reader, self.fld_reader = ca(), ca()
            self.mlp = _pointwise_mlp(dim, cond_dim, mlp_type=mlp_type)
        else:
            self.geo_reader = shared_block.geo_reader
            self.fld_reader = shared_block.fld_reader
            self.mlp = _pointwise_mlp(dim, cond_dim, shared=shared_block.mlp)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, queries, geometry_memory, field_memory, params,
                queries_pos=None, latent_pos=None):
        """Advances the query stream by one block.

        Args:
            queries: Query stream with shape (batch size, number query points, dim).
            geometry_memory: `M_geo` from the corresponding encoder block, (batch size, number latent points, dim).
            field_memory: `M_fld` from the corresponding encoder block, same shape.
            params: Conditioning parameters of the TARGET run, (batch size, cond_dim); the same
                tensor the encoder block was given.
            queries_pos (optional): Positions of the queries, (batch size, number query points, spatial_dim).
            latent_pos (optional): Positions of BOTH memories -- the target's latent positions.

        Returns:
            Updated query stream with shape (batch size, number query points, dim).
        """
        queries = queries + self.attn_dropout(
            self.geo_reader(q=queries, kv=geometry_memory, q_pos=queries_pos, kv_pos=latent_pos))
        queries = queries + self.attn_dropout(
            self.fld_reader(q=queries, kv=field_memory, q_pos=queries_pos, kv_pos=latent_pos))

        queries = queries + self.mlp(queries, params)

        return queries


class DecoderBlock(nn.Module):
    """The decoder block attends to the latent geometry of the corresponding encoder block to produce predictions
    of physical quantities at query positions.

    Args:
        dim: Dimensionality of the features.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout rate. Defaults to 0.1.
        spatial_dim: Number of spatial dimensions for RoPE. Defaults to 3.
        cond_dim: Dimensionality of the conditioning parameters for the MLP. Defaults to 2.
        shared_attn: Shared cross-attention module from the corresponding encoder block. Defaults to None.
        shared_mlp: Shared MLP module from the corresponding encoder block. Defaults to None.
        mlp_type: "gelu" (default) or "swiglu". Ignored when `shared_mlp` is given.
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, spatial_dim=3, cond_dim=2, shared_attn=None, shared_mlp=None,
                 mlp_type="gelu"):
        super().__init__()
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout, spatial_dim=spatial_dim) if shared_attn is None else shared_attn
        self.attn_dropout = nn.Dropout(dropout)

        # Pointwise MLP
        self.mlp = _pointwise_mlp(dim, cond_dim, shared=shared_mlp, mlp_type=mlp_type)

    def forward(self, queries, latent_geometry, params, queries_pos=None, latent_geometry_pos=None):
        """Updates the queries by attending to the latent geometry of the corresponding encoder block.

        Args:
            queries: Features of the query positions with shape (batch size, number query points, dim).
            latent_geometry: Latent geometry with shape (batch size, number latent points, dim).
            params: Conditioning parameters with shape (batch size, cond_dim).
            queries_pos (optional): Positions of the query positions for the positional embeddings with shape (batch size, number query points, spatial_dim). Defaults to None.
            latent_geometry_pos (optional): Positions of the latent geometry for the positional embeddings with shape (batch size, number latent points, spatial_dim). Defaults to None.

        Returns:
            Updated queries with shape (batch size, number query points, dim).
        """

        # Cross-attention with the latent geometry
        queries = queries + self.attn_dropout(self.attn(q=queries, kv=latent_geometry, q_pos=queries_pos, kv_pos=latent_geometry_pos))

        # Pointwise MLP
        queries = queries + self.mlp(queries, params)

        return queries
