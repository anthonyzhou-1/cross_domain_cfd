from functools import partial
import torch
from torch import nn

from modules.layers.attention import (
    AnchorAttention,
    SharedweightsCrossattnAttention,
    SharedweightsSplitattnAttention,
)
from modules.layers.embedding import ContinuousSincosEmbed
from modules.layers.rope import RopeFrequency
from modules.models.upt.blocks import TransformerBlock, PerceiverBlock
from modules.models.upt.pooling import BallTreePooling


class AnchoredBranchedUPT(nn.Module):
    def __init__(
        self,
        # problem dimensions
        ndim: int = 3,  # number of coordinates (typicaly 3 for 3D geometries)
        input_dim: int = 3,  # we only use coordinates as inputs
        output_dim_surface: int = 4,  # surface pressure (1D) and wallshearstress (3D)
        output_dim_volume: int = 4,  # volume pressure (1D), volume velocity (3D) 
        # model
        dim: int = 192,  # dimension of a ViT-tiny
        geometry_depth: int = 2,  # 1 transformer block after supernode pooling to encode the geometry
        num_heads: int = 16,  # number of attention heads in a ViT-tiny
        num_cond: int = 0,
        # "p": weight-shared cross-attention block to the geometry branch outputs
        # "s" weight-shared split attention block within surface/volume
        # "c" weight-shared cross-attention block between surface/volume
        blocks: str = "ppscscs",
        num_volume_blocks: int = 6,  # 6 modality-specific self-attention blocks
        num_surface_blocks: int = 6,  # 6 modality-specific self-attention blocks
        pooling: str = "erwin",  # geometry pooling; only ball-tree ("erwin") is supported
        num_supernodes: int = 1024,  # number of ball-tree supernodes
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=input_dim)
        # geometry
        self.pooling = pooling 

        if self.pooling == "erwin":
            self.encoder = BallTreePooling(hidden_dim=dim,
                                           ndim=ndim,
                                           num_balls=num_supernodes)
        else:
            raise NotImplementedError(f"Pooling method {self.pooling} not implemented")
        self.geometry_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, num_heads=num_heads)
                for _ in range(geometry_depth)
            ],
        )
        # pos_embed with separate MLP for surface/volume
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.surface_bias = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.volume_bias = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # weight-shared blocks
        self.blocks = nn.ModuleList()
        for block in blocks:
            if block == "s":
                # weight-shared self-attention within surface/volume tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
            elif block == "c":
                # weight-shared cross-attention between surface/volume tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
            elif block == "p":
                # weight-shared cross-attention from surface/volume tokens to geometry tokens
                block_ctor = PerceiverBlock
            else:
                raise NotImplementedError
            self.blocks.append(block_ctor(dim=dim, num_heads=num_heads, cond_dim=num_cond))
        # surface-specific blocks
        self.surface_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_ctor=AnchorAttention,
                    cond_dim=num_cond,
                )
                for _ in range(num_surface_blocks)
            ],
        )
        # volume-specific blocks
        self.volume_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_ctor=AnchorAttention,
                    cond_dim=num_cond,
                )
                for _ in range(num_volume_blocks)
            ],
        )
        # final linear projections
        self.surface_decoder = nn.Linear(dim, output_dim_surface)
        self.volume_decoder = nn.Linear(dim, output_dim_volume)

        # init weights
        # (there are only nn.Linear and nn.LayerNorm in AB-UPT, layernorms are initialized correctly by default)
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(
        self,
        # geometry
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,  # unused by ball-tree pooling
        geometry_batch_idx: torch.Tensor | None,
        # anchors
        surface_anchor_position: torch.Tensor,
        volume_anchor_position: torch.Tensor,
        # queries
        surface_query_position: torch.Tensor | None = None,
        volume_query_position: torch.Tensor | None = None,
        cond=None,
    ) -> dict[str, torch.Tensor]:
        # outputs/kwargs
        outputs = {}
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        volume_decoder_attn_kwargs = {}
        geometry_perceiver_attn_kwargs = {}
        shared_attn_kwargs = {}

        # create split sizes + optionally concat query positions
        num_surface_anchor_positions = surface_anchor_position.size(1)
        num_volume_anchor_positions = volume_anchor_position.size(1)
        if surface_query_position is None and volume_query_position is None:
            # no queries
            split_size = [surface_anchor_position.size(1), volume_anchor_position.size(1)]
            surface_position_all = surface_anchor_position
            volume_position_all = volume_anchor_position
        elif surface_query_position is not None and volume_query_position is not None:
            # surface and volume queries
            split_size = [
                surface_anchor_position.size(1),
                surface_query_position.size(1),
                volume_anchor_position.size(1),
                volume_query_position.size(1),
            ]
            surface_decoder_attn_kwargs["num_anchor_tokens"] = surface_anchor_position.size(1)
            volume_decoder_attn_kwargs["num_anchor_tokens"] = volume_anchor_position.size(1)
            surface_position_all = torch.concat([surface_anchor_position, surface_query_position], dim=1)
            volume_position_all = torch.concat([volume_anchor_position, volume_query_position], dim=1)
        else:
            raise NotImplementedError

        # rope frequencies
        assert geometry_batch_idx is None or geometry_batch_idx.unique().numel() == 1, "batch_size > 1 not supported"
        rope_surface_all = self.rope(surface_position_all)
        rope_volume_all = self.rope(volume_position_all)
        rope_all = torch.concat([rope_surface_all, rope_volume_all], dim=1)
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        volume_decoder_attn_kwargs["freqs"] = rope_volume_all
        shared_attn_kwargs["freqs"] = rope_all

        # encode geometry
        x, centers = self.encoder(geometry_position)
        geometry_rope = self.rope(centers.unsqueeze(0))
        geometry_attn_kwargs["freqs"] = geometry_rope
        geometry_perceiver_attn_kwargs["q_freqs"] = rope_all
        geometry_perceiver_attn_kwargs["k_freqs"] = geometry_rope

        for block in self.geometry_blocks:
            x = block(x, attn_kwargs=geometry_attn_kwargs)
        geometry_encoding = x

        # shared-weights model (all tokens are concatenated into a single sequence for high GPU utilization)
        assert surface_position_all.ndim == 3 and volume_position_all.ndim == 3
        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        volume_all_pos_embed = self.volume_bias(self.pos_embed(volume_position_all))
        x = torch.concat([surface_all_pos_embed, volume_all_pos_embed], dim=1)
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                if len(split_size) == 4:
                    # anchors + queries
                    x = block(
                        x,
                        attn_kwargs=dict(split_size=split_size, **shared_attn_kwargs),
                        cond=cond,
                    )
                elif len(split_size) == 2:
                    # anchors only
                    x = block(x, attn_kwargs=dict(split_size=split_size, **shared_attn_kwargs), cond=cond)
                else:
                    raise NotImplementedError
            elif isinstance(block, PerceiverBlock):
                # cross attention to geometry outputs
                x = block(q=x, kv=geometry_encoding, attn_kwargs=geometry_perceiver_attn_kwargs, cond=cond)
            else:
                raise NotImplementedError

        # split into surface/volume tokens for modality-specific blocks
        if surface_query_position is None and volume_query_position is None:
            # no queries
            x_surface, x_volume = x.split(split_size, dim=1)
        elif surface_query_position is not None and volume_query_position is not None:
            # surface + volume queries
            x_surface, x_volume = x.split([split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=1)
        else:
            raise NotImplementedError
        assert x_surface.size(1) == surface_position_all.size(1)
        assert x_volume.size(1) == volume_position_all.size(1)

        # surface blocks
        x = x_surface
        for block in self.surface_blocks:
            x = block(x, attn_kwargs=dict(**surface_decoder_attn_kwargs), cond=cond)
        x = self.surface_decoder(x)

        if surface_query_position is None:
            outputs["surface_anchor_cp"] = x[:, :, :1]
            outputs["surface_anchor_cf"] = x[:, :, 1:]
        else:
            x_surface_anchor = x[:, :num_surface_anchor_positions]
            x_surface_query = x[:, num_surface_anchor_positions:]
            outputs["surface_anchor_cp"] = x_surface_anchor[:, :, :1]
            outputs["surface_anchor_cf"] = x_surface_anchor[:, :, 1:]
            outputs["surface_query_cp"] = x_surface_query[:, :, :1]
            outputs["surface_query_cf"] = x_surface_query[:, :, 1:]

        # volume
        x = x_volume
        for block in self.volume_blocks:
            x = block(x, attn_kwargs=dict(**volume_decoder_attn_kwargs), cond=cond)
        x = self.volume_decoder(x)
        # convert to sparse tensor
        if volume_query_position is None:
            # anchors only
            outputs["volume_anchor_cp"] = x[:, :, :1]
            outputs["volume_anchor_velocity"] = x[:, :, 1:4]
        else:
            # queries + anchors
            x1 = x[:, :num_volume_anchor_positions]
            x2 = x[:, num_volume_anchor_positions:]
            outputs["volume_anchor_cp"] = x1[:, :, :1]
            outputs["volume_anchor_velocity"] = x1[:, :, 1:4]
            outputs["volume_query_cp"] = x2[:, :, :1]
            outputs["volume_query_velocity"] = x2[:, :, 1:4]

        return outputs
