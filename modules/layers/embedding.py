import torch
import torch.nn as nn
import math
import einops


class ContinuousSincosEmbed(nn.Module):
    """Embedding layer for continuous coordinates using sine and cosine functions as used in transformers.
    This implementation is able to deal with arbitrary coordinate dimensions (e.g., 2D and 3D coordinate systems).

    Args:
        dim: Dimensionality of the embedded input coordinates.
        ndim: Number of dimensions of the input domain.
        max_wavelength: Max length. Defaults to 10000.
        assert_positive: If true, assert if all input coordiantes are positive. Defaults to False.
    """

    def __init__(
        self,
        dim: int,
        ndim: int,
        max_wavelength: int = 10000,
        assert_positive: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        self.assert_positive = assert_positive
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer(
            "omega",
            1.0 / max_wavelength**(arange / effective_dim_per_wave),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward method of the ContinuousSincosEmbed layer.

        Args:
            coords: Tensor of coordinates. The shape of the tensor should be
                (batch size, number of points, coordinate dimension) or (number of points, coordinate dimension).

        Returns:
            Tensor with embedded coordinates.
        """
        if self.assert_positive:
            # check if coords are positive
            assert torch.all(coords >= 0)
        # fp32 to avoid numerical imprecision
        coords = coords.float()
        with torch.autocast(device_type=str(coords.device).split(":")[0], enabled=False):
            coordinate_ndim = coords.shape[-1]
            assert self.ndim == coordinate_ndim
            out = coords.unsqueeze(-1) @ self.omega.unsqueeze(0)
            emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
            if coords.ndim == 3:
                emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
            elif coords.ndim == 2:
                emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
            else:
                raise NotImplementedError
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb


class CoeffEmbedding(nn.Module):
    """Embed PDE coefficients, nominally in [0, 1], into a conditioning vector.

    Optionally lifts each coefficient with NeRF-style Fourier features (1 .. 2^(num_freqs-1)
    cycles over [0, 1], plus the raw value) before an MLP. Outside [0, 1] each band's phase
    saturates within `extrap_radians` of its boundary value so it cannot alias onto an
    in-range input; inside [0, 1] the lift is unchanged.

    Args:
        cond_dim: number of (normalized) input coefficients.
        cond_channels: width of the returned embedding.
        num_freqs: Fourier bands per coefficient (ignored if use_fourier=False).
        use_fourier: lift each coefficient with Fourier features before the MLP.
        norm: LayerNorm the output.
        fanin_init: recorded for the caller, which should call reset_parameters() after a blanket init.
        extrap_radians: max phase excursion per band beyond [0, 1]; 0/None gives the plain periodic lift.
    """

    def __init__(self, cond_dim, cond_channels, num_freqs=6, use_fourier=True,
                 norm=False, fanin_init=False, extrap_radians=0.5):
        super().__init__()
        self.use_fourier = use_fourier
        self.fanin_init = fanin_init
        self.extrap_radians = float(extrap_radians or 0.0)

        if use_fourier:
            # frequencies 1, 2, 4, ..., 2^(num_freqs-1) cycles across [0, 1]
            freqs = 2.0 ** torch.arange(num_freqs) * math.pi
            self.register_buffer("freqs", freqs)            # (num_freqs,)
            feat_dim = cond_dim * (2 * num_freqs + 1)       # sin, cos, + raw value
        else:
            feat_dim = cond_dim

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, cond_channels),
            nn.GELU(),
            nn.Linear(cond_channels, cond_channels),
        )
        self.norm = nn.LayerNorm(cond_channels) if norm else None

    def reset_parameters(self):
        """Re-apply PyTorch's fan-in-scaled Linear init to the MLP (after a blanket std=0.02 init)."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def phase(self, c):
        """Per-band phase (batch, cond_dim, num_freqs): `c * freqs` on [0, 1], bounded excursion outside."""
        if self.extrap_radians <= 0:
            return c[..., None] * self.freqs
        inside = c.clamp(0.0, 1.0)
        excess = (c - inside)[..., None]                    # signed; exactly 0 in range
        a = self.extrap_radians
        return inside[..., None] * self.freqs + a * torch.tanh(excess * self.freqs / a)

    def forward(self, c):                                   # c: (batch, cond_dim)
        if self.use_fourier:
            args = self.phase(c)                            # (batch, cond_dim, num_freqs)
            c = torch.cat(
                [c[..., None], args.sin(), args.cos()], dim=-1
            ).flatten(1)                                    # (batch, cond_dim*(2F+1))
        c = self.mlp(c)                                     # (batch, cond_channels)
        return c if self.norm is None else self.norm(c)


class MeshPointEmbed(nn.Module):
    """Embed mesh points (coordinates + optional field values) into ``hidden_dim``.

    Coordinates get a sine/cosine embedding mixed by a small MLP; field values, when present,
    get a separate linear projection and are combined with it.

    Args:
        hidden_dim: Output dimensionality.
        ndim: Number of coordinate dimensions (the leading channels of the input).
        field_dim: Number of (non-positional) field-value channels following the coordinates.
        coord_scale: Factor applied to the coordinates before the sinusoidal embedding.
        max_wavelength: Max wavelength of the sinusoidal embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        ndim: int = 3,
        field_dim: int = 0,
        coord_scale: float = 1.0,
        max_wavelength: float = 10000.0,
    ):
        super().__init__()
        self.ndim = ndim
        self.field_dim = field_dim
        self.coord_scale = coord_scale

        self.coord_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim, max_wavelength=max_wavelength)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if field_dim > 0:
            self.field_proj = nn.Linear(field_dim, hidden_dim)
            self.combined_proj = nn.Linear(2*hidden_dim, hidden_dim)
        else:
            self.field_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (..., ndim + field_dim). The leading ``ndim`` channels are coordinates
                and the remaining ``field_dim`` channels are field values.

        Returns:
            Tensor of shape (..., hidden_dim).
        """
        coords = x[..., : self.ndim]
        h = self.coord_mlp(self.coord_embed(self.coord_scale * coords))
        if self.field_proj is not None:
            field_emb = self.field_proj(x[..., self.ndim :])
            h = self.combined_proj(torch.cat([h, field_emb], dim=-1))
        return h


class MeshFieldEmbed(nn.Module):
    """Embed mesh points with field values, reusing an existing ``MeshPointEmbed``'s coordinate pathway.

    Args:
        coord_encoder: The ``MeshPointEmbed`` whose coordinate pathway to reuse (contributes no
            parameters of its own to this module).
        field_dim: Number of (non-positional) field-value channels following the coordinates.
    """

    def __init__(self, coord_encoder: MeshPointEmbed, field_dim: int):
        super().__init__()
        assert field_dim > 0, "MeshFieldEmbed needs field channels; use MeshPointEmbed otherwise"
        self.coord_encoder = coord_encoder
        self.ndim = coord_encoder.ndim
        self.field_dim = field_dim

        hidden_dim = coord_encoder.coord_mlp[-1].out_features
        self.field_proj = nn.Linear(field_dim, hidden_dim)
        self.combined_proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (..., ndim + field_dim). The leading ``ndim`` channels are
                coordinates and the remaining ``field_dim`` channels are field values.

        Returns:
            Tensor of shape (..., hidden_dim).
        """
        # coord_encoder slices the leading ndim channels itself, so it can take x whole.
        h = self.coord_encoder(x)
        field_emb = self.field_proj(x[..., self.ndim :])
        return self.combined_proj(torch.cat([h, field_emb], dim=-1))
