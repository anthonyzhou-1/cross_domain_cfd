import torch
import torch.nn as nn
from modules.models.smart.blocks import EncoderBlock, DecoderBlock, ICEncoderBlock, ICDecoderBlock
from modules.models.smart.utils import Modulator
from modules.layers.embedding import MeshFieldEmbed, MeshPointEmbed

def sample_geometry(geometry, num_samples):
    """Randomly sample min(num_samples, N) points from a (batch, N, channels) cloud."""
    idx = torch.randperm(geometry.shape[1], device=geometry.device)[:num_samples]
    sampled_geometry = geometry[:, idx, :]
    return sampled_geometry


class FieldHeads:
    """Surface/volume output heads shared by SMART, SMART_IC and Transolver_plus."""

    def _build_heads(self, latent_dim, surface_channels, volume_channels, split_heads):
        """Two split heads (`surface_out`/`volume_out`), or the legacy single head `mlp`."""
        head = lambda out_channels: nn.Sequential(
            nn.LayerNorm(latent_dim, eps=1e-6),
            nn.Linear(latent_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, out_channels))
        if split_heads:
            self.surface_out = head(surface_channels)
            self.volume_out = head(volume_channels)
        else:
            self.mlp = head(surface_channels + volume_channels)

    def _project(self, query_emb, n_surface):
        """Split a decoded query stream (first `n_surface` tokens are surface) into (surface, volume) predictions."""
        if self.split_heads:
            return (self.surface_out(query_emb[:, :n_surface]),
                    self.volume_out(query_emb[:, n_surface:]))
        pred = self.mlp(query_emb)
        return (pred[:, :n_surface, :self.surface_channels],
                pred[:, n_surface:, self.surface_channels:])


class SMART(FieldHeads, nn.Module):
    """SMART model for simulating time-independent PDEs over complex 3D geometries.

    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 0.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of points of the latent geometry. Default is 4096.
        subsampled_geometry_points: Number of points in the subsampled geometry for geometry cross-attention. Default is 16384.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions to use more/less of the dynamic range of the positional embedding. Default is 100.
        dropout: Dropout rate. Default is 0.0.
        subregion_size: Number of query points to process in each subregion during sequential inference. Default is 262144.
        weight_sharing: Tie each decoder block's attention and MLP to the encoder block's. Default is True.
        mlp_type: "swiglu" (default) or "gelu" (needed by checkpoints trained before the option existed).
        split_heads: Separate surface/volume projection heads. Default True; older checkpoints need False.
    """

    def __init__(self, spatial_dim=3,
                 surface_channels=1,
                 volume_channels=3,
                 parameter_channels=0,
                 latent_dim=256,
                 latent_geometry_points=4096,
                 subsampled_geometry_points=16384,
                 num_encoder_decoder_blocks=8,
                 num_heads=8,
                 pos_scale_factor=100,
                 dropout=0.0,
                 subregion_size=262144,
                 weight_sharing=True,
                 mlp_type="swiglu",
                 split_heads=True):
        super(SMART, self).__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."

        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.pos_scale_factor = pos_scale_factor

        # coord_scale=1.0: positions are already scaled by pos_scale_factor in encode/decode.
        self.pos_encoder = MeshPointEmbed(hidden_dim=latent_dim,
                                          ndim=spatial_dim,
                                          coord_scale=1.0)

        self.split_heads = split_heads

        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(dim=latent_dim,
                         num_heads=num_heads,
                         dropout=dropout,
                         spatial_dim=spatial_dim,
                         cond_dim=parameter_channels,
                         mlp_type=mlp_type)
                         for i in range(num_encoder_decoder_blocks)])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dim=latent_dim,
                         num_heads=num_heads,
                         dropout=dropout,
                         spatial_dim=spatial_dim,
                         cond_dim=parameter_channels,
                         shared_attn=self.encoder_blocks[i].cross_attn if weight_sharing else None,
                         shared_mlp=self.encoder_blocks[i].mlp if weight_sharing else None,
                         mlp_type=mlp_type)
                         for i in range(num_encoder_decoder_blocks)])

        # Projection heads
        self._build_heads(latent_dim, surface_channels, volume_channels, split_heads)

        # Subregion size for inference
        self.subregion_size = subregion_size

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        # Restore the modulators' identity init, which _init_weights overwrites.
        for m in self.modules():
            if isinstance(m, Modulator):
                m.init_identity()

    # Weight initialization from Transolver
    # (https://github.com/thuml/Transolver/blob/a11be9c4f7db1885e4b08c68432bc31799492ec9/Car-Design-ShapeNetCar/models/Transolver.py#L168)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, geo, params):
        # Scale once; consumed by both the positional embedding and RoPE.
        geo = geo * self.pos_scale_factor

        # Sample the initial latent geometry
        latent_geo_pos = sample_geometry(geo, self.num_geo)
        latent_geo_emb = self.pos_encoder(latent_geo_pos)

        # Apply encoder blocks
        intermediate_latent_geometries = []
        for block in self.encoder_blocks:
            # Subsample the geometry for geometry cross-attention
            sub_geo_pos = sample_geometry(geo, self.subsampled_geometry_points)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)

            # Apply encoder block
            latent_geo_emb, e_ca = block(latent_geo_emb, sub_geo_emb, params, latent_geometry_pos=latent_geo_pos, subsampled_geometry_pos=sub_geo_pos)

            # Store for decoder
            intermediate_latent_geometries.append(e_ca)

        return intermediate_latent_geometries, latent_geo_pos

    def decode(self, intermediate_latent_geometries, latent_geo_pos, params, query_pos):
        """Run the decoder stack and return the decoded query stream (heads applied by `_project`)."""
        # Must match the scaling encode() applied to latent_geo_pos.
        query_pos = query_pos * self.pos_scale_factor

        query_emb = self.pos_encoder(query_pos)

        for e_ca, block in zip(intermediate_latent_geometries, self.decoder_blocks):
            query_emb = block(query_emb, e_ca, params, queries_pos=query_pos, latent_geometry_pos=latent_geo_pos)

        return query_emb

    def forward(self, geo, surf_query_pos, vol_query_pos, params):
        """Forward method for SMART model.

        Args:
            geo: Input geometry with shape (batch size, number points, spatial_dim).
            surf_query_pos: Surface query positions with shape (batch size, number surface query points, spatial_dim).
            vol_query_pos: Volume query positions with shape (batch size, number volume query points, spatial_dim).
            params: Conditioning parameters with shape (batch size, cond_dim). If not used, pass None.

        Returns:
            tuple: A tuple containing:
                - Surface predictions with shape (batch size, number surface query points, surface_channels).
                - Volume predictions with shape (batch size, number volume query points, volume_channels).
        """
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(geo, params)

        # Prepare query positions by concatenating surface and volume query positions
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)

        # Decode
        query_emb = self.decode(intermediate_latent_geometries, latent_geo_pos, params, query_pos)

        # Split surface and volume predictions
        return self._project(query_emb, surf_query_pos.shape[1])

    @torch.inference_mode()
    def inference(self, geo, surf_query_pos, vol_query_pos, params):
        """Sequential inference that decodes queries in chunks of `subregion_size`. Same arguments as `forward`."""
        # Encode
        intermediate_latent_geometries, latent_geo_pos = self.encode(geo, params)

        def run(query_pos, surface):
            """Decode one field's queries in subregions and project them with its own head."""
            out = []
            for i in range(0, query_pos.shape[1], self.subregion_size):
                sub = query_pos[:, i:i+self.subregion_size, :]
                emb = self.decode(intermediate_latent_geometries, latent_geo_pos, params, sub)
                n = emb.shape[1] if surface else 0
                s_pred, v_pred = self._project(emb, n)
                out.append(s_pred if surface else v_pred)
            return torch.cat(out, dim=1)

        return run(surf_query_pos, True), run(vol_query_pos, False)


class SMART_IC(FieldHeads, nn.Module):
    """In-context SMART: predict a target run's fields from its geometry plus one solved demo run.

    One latent stream at target-geometry positions; each `ICEncoderBlock` writes a geometry
    memory and a field memory (from the demo's surface/volume clouds), then reads both back.
    The decoder replays the stream update on query tokens against the same memories.
    `query_params` (target) drives the MLPs; `cat([context_params, query_params - context_params])`
    FiLM-modulates the field memory. Both are inert at `parameter_channels=0`.

    Args:
        spatial_dim: Number of spatial dimensions. Default is 3.
        surface_channels: Number of output channels for surface predictions. Default is 1.
        volume_channels: Number of output channels for volume predictions. Default is 3.
        parameter_channels: Number of conditioning parameter channels. Default is 0.
        latent_dim: Dimensionality of the latent representations. Default is 256.
        latent_geometry_points: Number of tokens in the latent stream (and in both memories). Default is 4096.
        subsampled_geometry_points: Target geometry points drawn per block. Default is 16384.
        subsampled_field_points: Demo field points drawn per block, half surface / half volume. Default is 32768.
        num_encoder_decoder_blocks: Number of encoder-decoder blocks. Default is 8.
        num_heads: Number of attention heads. Default is 8.
        pos_scale_factor: Scaling factor for the positions. Default is 100.
        dropout: Dropout rate. Default is 0.0.
        subregion_size: Number of query points per chunk during sequential inference. Default is 262144.
        weight_sharing: Tie each decoder block's readers and MLP to the encoder block's. Default is True.
        mlp_type: "swiglu" (default) or "gelu".
        split_heads: Separate surface/volume projection heads. Default True.
    """

    def __init__(self, spatial_dim=3,
                 surface_channels=1,
                 volume_channels=3,
                 parameter_channels=0,
                 latent_dim=256,
                 latent_geometry_points=4096,
                 subsampled_geometry_points=16384,
                 subsampled_field_points=32768,
                 num_encoder_decoder_blocks=8,
                 num_heads=8,
                 pos_scale_factor=100,
                 dropout=0.0,
                 subregion_size=262144,
                 weight_sharing=True,
                 mlp_type="swiglu",
                 split_heads=True):
        super().__init__()
        assert surface_channels > 0 and volume_channels > 0, "surface_channels and volume_channels must be positive integers."

        self.surface_channels = surface_channels
        self.volume_channels = volume_channels
        self.parameter_channels = parameter_channels
        self.split_heads = split_heads
        self.num_geo = latent_geometry_points
        self.subsampled_geometry_points = subsampled_geometry_points
        self.subsampled_field_points = subsampled_field_points
        self.pos_scale_factor = pos_scale_factor
        self.spatial_dim = spatial_dim

        # Positions only; used for both the latent stream and the decoder queries.
        self.pos_encoder = MeshPointEmbed(hidden_dim=latent_dim,
                                          ndim=spatial_dim,
                                          coord_scale=1.0)

        # Demo clouds (positions + values), sharing pos_encoder's coordinate pathway.
        self.surface_encoder = MeshFieldEmbed(self.pos_encoder, field_dim=surface_channels)
        self.volume_encoder = MeshFieldEmbed(self.pos_encoder, field_dim=volume_channels)

        # [context_params, query_params - context_params]
        self.context_cond_dim = 2 * parameter_channels

        self.encoder_blocks = nn.ModuleList([
            ICEncoderBlock(dim=latent_dim,
                           num_heads=num_heads,
                           dropout=dropout,
                           spatial_dim=spatial_dim,
                           cond_dim=parameter_channels,
                           context_cond_dim=self.context_cond_dim,
                           mlp_type=mlp_type)
                           for i in range(num_encoder_decoder_blocks)])

        self.decoder_blocks = nn.ModuleList([
            ICDecoderBlock(dim=latent_dim,
                           num_heads=num_heads,
                           dropout=dropout,
                           spatial_dim=spatial_dim,
                           cond_dim=parameter_channels,
                           shared_block=self.encoder_blocks[i] if weight_sharing else None,
                           mlp_type=mlp_type)
                           for i in range(num_encoder_decoder_blocks)])

        # Projection heads
        self._build_heads(latent_dim, surface_channels, volume_channels, split_heads)

        # Subregion size for inference
        self.subregion_size = subregion_size

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        # Restore the modulators' identity init, which _init_weights overwrites.
        for m in self.modules():
            if isinstance(m, Modulator):
                m.init_identity()

    # Weight initialization from Transolver
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _scale_field_pos(self, field):
        """Scale a field tensor's leading position channels (out-of-place), leaving the values alone."""
        return torch.cat([field[..., :self.spatial_dim] * self.pos_scale_factor,
                          field[..., self.spatial_dim:]], dim=-1)

    def _context_conditioning(self, query_params, context_params):
        """The field-memory FiLM vector: [context_params, query_params - context_params], or None."""
        if self.parameter_channels == 0:
            return None
        if query_params is None or context_params is None:
            raise ValueError(
                f"parameter_channels={self.parameter_channels} requires both query_params and "
                "context_params (set `return_cond: True` in the config's data block)")
        return torch.cat([context_params, query_params - context_params], dim=-1)

    def _sample_context(self, surface_field, volume_field):
        """One fresh draw of context field key/values, half surface and half volume."""
        half = self.subsampled_field_points // 2
        embeddings, positions = [], []
        for field, encoder in ((surface_field, self.surface_encoder),
                               (volume_field, self.volume_encoder)):
            sub = self._scale_field_pos(sample_geometry(field, half))
            positions.append(sub[..., :self.spatial_dim])
            embeddings.append(encoder(sub))
        return torch.cat(embeddings, dim=1), torch.cat(positions, dim=1)

    def encode(self, geo, surface_field, volume_field, query_params, context_params):
        """Run the latent stream and collect the two memories each block writes.

        Args:
            geo: TARGET input geometry, (batch size, number points, spatial_dim).
            surface_field: CONTEXT surface cloud, (batch size, n, spatial_dim + surface_channels).
            volume_field: CONTEXT volume cloud, (batch size, n, spatial_dim + volume_channels).
            query_params: Conditioning parameters of the TARGET run, (batch size, cond_dim).
            context_params: Conditioning parameters of the CONTEXT run, (batch size, cond_dim).

        Returns:
            tuple: (list of (geometry memory, field memory) per block, latent positions).
        """
        # Scale once; consumed by both the positional embedding and RoPE.
        geo = geo * self.pos_scale_factor

        latent_pos = sample_geometry(geo, self.num_geo)
        latent = self.pos_encoder(latent_pos)

        context_cond = self._context_conditioning(query_params, context_params)

        memories = []
        for block in self.encoder_blocks:
            # Fresh draws every block.
            sub_geo_pos = sample_geometry(geo, self.subsampled_geometry_points)
            sub_geo_emb = self.pos_encoder(sub_geo_pos)
            sub_fld_emb, sub_fld_pos = self._sample_context(surface_field, volume_field)

            latent, geometry_memory, field_memory = block(
                latent, sub_geo_emb, sub_fld_emb, query_params, context_cond=context_cond,
                latent_pos=latent_pos, geometry_pos=sub_geo_pos, field_pos=sub_fld_pos)

            memories.append((geometry_memory, field_memory))

        return memories, latent_pos

    def decode(self, memories, latent_pos, query_params, query_pos):
        """Replay the encoder's stream update on query tokens; returns the decoded query stream."""
        # Must match the scaling encode() applied to latent_pos.
        query_pos = query_pos * self.pos_scale_factor

        query_emb = self.pos_encoder(query_pos)

        for (geometry_memory, field_memory), block in zip(memories, self.decoder_blocks):
            query_emb = block(query_emb, geometry_memory, field_memory, query_params,
                              queries_pos=query_pos, latent_pos=latent_pos)

        return query_emb

    def forward(self, geo, surface_field, volume_field,
                surf_query_pos, vol_query_pos, query_params, context_params):
        """Forward method for the in-context SMART model.

        Args:
            geo: TARGET input geometry with shape (batch size, number points, spatial_dim).
            surface_field: CONTEXT surface cloud, (batch size, n, spatial_dim + surface_channels).
            volume_field: CONTEXT volume cloud, (batch size, n, spatial_dim + volume_channels).
            surf_query_pos: Target surface query positions, (batch size, n_surf_q, spatial_dim).
            vol_query_pos: Target volume query positions, (batch size, n_vol_q, spatial_dim).
            query_params: TARGET run's conditioning, (batch size, cond_dim), or None.
            context_params: CONTEXT run's own conditioning, (batch size, cond_dim), or None.

        Returns:
            tuple: (surface predictions (b, n_surf_q, surface_channels),
                    volume predictions (b, n_vol_q, volume_channels)).
        """
        memories, latent_pos = self.encode(geo, surface_field, volume_field,
                                           query_params, context_params)

        # Prepare query positions by concatenating surface and volume query positions
        query_pos = torch.cat([surf_query_pos, vol_query_pos], dim=1)

        query_emb = self.decode(memories, latent_pos, query_params, query_pos)

        # Split surface and volume predictions
        return self._project(query_emb, surf_query_pos.shape[1])

    @torch.inference_mode()
    def inference(self, geo, surface_field, volume_field,
                  surf_query_pos, vol_query_pos, query_params, context_params):
        """Sequential-decode variant of `forward` (encoder once, decode chunked by `subregion_size`)."""
        memories, latent_pos = self.encode(geo, surface_field, volume_field,
                                           query_params, context_params)

        def run(query_pos, surface):
            """Decode one field's queries in subregions and project them with its own head."""
            out = []
            for i in range(0, query_pos.shape[1], self.subregion_size):
                emb = self.decode(memories, latent_pos, query_params,
                                  query_pos[:, i:i+self.subregion_size, :])
                n = emb.shape[1] if surface else 0
                s_pred, v_pred = self._project(emb, n)
                out.append(s_pred if surface else v_pred)
            return torch.cat(out, dim=1)

        return run(surf_query_pos, True), run(vol_query_pos, False)

    @torch.no_grad()
    def test_weight_sharing_symmetry(self, geo, surface_field, volume_field,
                                     query_params=None, context_params=None):
        """Max |decoder stream - encoder stream| when querying at the latent positions (~0 under weight sharing).

        Returns (max abs difference, max abs stream value).
        """
        was_training = self.training
        self.eval()
        try:
            memories, latent_pos = self.encode(geo, surface_field, volume_field,
                                               query_params, context_params)
            # Rebuild the encoder stream from the memories it wrote.
            latent = self.pos_encoder(latent_pos)
            for (geometry_memory, field_memory), block in zip(memories, self.encoder_blocks):
                latent = latent + block.attn_dropout(block.geo_reader(
                    q=latent, kv=geometry_memory, q_pos=latent_pos, kv_pos=latent_pos))
                latent = latent + block.attn_dropout(block.fld_reader(
                    q=latent, kv=field_memory, q_pos=latent_pos, kv_pos=latent_pos))
                latent = latent + block.mlp(latent, query_params)

            # Query at the (already scaled) latent positions, bypassing decode().
            query_emb = self.pos_encoder(latent_pos)
            for (geometry_memory, field_memory), block in zip(memories, self.decoder_blocks):
                query_emb = block(query_emb, geometry_memory, field_memory, query_params,
                                  queries_pos=latent_pos, latent_pos=latent_pos)

            return (latent - query_emb).abs().max().item(), latent.abs().max().item()
        finally:
            self.train(was_training)
