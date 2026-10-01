import einops
import torch
from torch import nn


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """RoPE via polar coordinate rotations."""
    # adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L65
    assert torch.is_tensor(freqs) and torch.is_complex(freqs)
    assert x.ndim == 4, "x.shape should be (batch_size, num_heads, seqlen, head_dim)"
    assert freqs.ndim == 3, "freqs.shape should be (batch_size, seqlen, head_dim // 2)"
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # add dim for num_heads
    freqs = einops.rearrange(freqs, "batch_size seqlen head_dim -> batch_size 1 seqlen head_dim")
    x_out = torch.view_as_real(x_ * freqs).flatten(start_dim=3)
    return x_out.type_as(x)

class RopeFrequency(nn.Module):
    """Creates frequencies for rotary embeddings (RoPE) from https://arxiv.org/abs/2104.09864 for variable positions.

    Args:
        dim: Dimensionality of frequencies (in transformers this should be the head dimension).
        ndim: Dimensionality of the coordinates (e.g., 2 for 2D coordinates, 3 for 3D coordinates).
        max_wavelength: Theta parameter for the transformer sine/cosine embedding. Default: 10000.0
        assert_positive: Makes sure that coordinates were rescaled to be positive only. Default: True
        coord_scale: Factor applied to the coordinates before computing rotation angles. For coordinates
            normalized to O(1) this spreads them over the useful range of the sinusoids (the highest
            frequencies would otherwise see near-zero angles). Default: 1.0 (no scaling).
    """

    def __init__(
        self,
        dim: int,
        ndim: int,
        max_wavelength: int = 10000.0,
        assert_positive: bool = False,
        coord_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.coord_scale = coord_scale
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        self.assert_positive = assert_positive
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float)
        self.register_buffer(
            "omega",
            1.0 / max_wavelength**(arange / effective_dim_per_wave),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.assert_positive:
            # check if coords are positive
            assert torch.all(coords >= 0), (
                f"coords.shape={coords.shape} coords.min={coords.min(dim=-2).values} "
                f"coords.max={coords.max(dim=-2).values} numel={coords.numel()} is_positive.numel={(coords > 0).sum()}"
            )

        with torch.autocast(device_type=str(coords.device).split(":")[0], enabled=False):
            coordinate_ndim = coords.shape[-1]
            assert self.ndim == coordinate_ndim
            out = (self.coord_scale * coords.float()).unsqueeze(-1) @ self.omega.unsqueeze(0)
        out = einops.rearrange(out, "... ndim dim -> ... (ndim dim)")
        # add padding
        assert self.padding % 2 == 0
        out = torch.concat([out, torch.zeros(*out.shape[:-1], self.padding // 2, device=coords.device)], dim=-1)
        return torch.polar(torch.ones_like(out), out)