import torch.nn as nn
import torch


class AdaLN(nn.Module):
    """Adaptive Layer Normalization; a plain LayerNorm when cond_dim == 0 or c is None.

    Args:
        hidden_size: width of the normalized activations.
        cond_dim: width of the conditioning embedding; 0 disables conditioning.
        cond_gate: allocate a scalar gate on the modulation output (see init_cond_modulation).
    """
    def __init__(self,
                 hidden_size,
                 cond_dim=0,
                 cond_gate=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(hidden_size)
        self.cond_gate = None
        if cond_dim > 0:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(cond_dim, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            if cond_gate:
                self.cond_gate = nn.Parameter(torch.zeros(1))

    def modulate(self,
                 x: torch.Tensor,
                 shift: torch.Tensor,
                 scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x, c=None):
        x = self.norm(x)
        if c is None or self.cond_dim == 0:
            return x
        else:
            z = self.adaLN_modulation(c) # b, 2*hidden_size
            if self.cond_gate is not None:
                z = self.cond_gate * z
            z = z.unsqueeze(1) # b, 1, 2*hidden_size
            shift, scale = z.chunk(2, dim=-1) # b, 1, hidden_size
            return self.modulate(x, shift, scale)


def init_cond_modulation(root: nn.Module, mode: str = "zero", std: float = 1e-3) -> int:
    """(Re-)initialize every conditioned AdaLN head under `root` (run after any blanket init). Returns the count.

      "zero"   last modulation layer zeroed (DiT-style), so the block starts at identity.
      "small"  as "zero" but with a tiny random last-layer weight (trunc_normal, `std`).
      "gate"   normal fan-in init held shut by a zero scalar gate (needs cond_gate=True).
    """
    if mode not in ("zero", "small", "gate"):
        raise ValueError(f"unknown cond_init mode {mode!r} (expected zero|small|gate)")
    n = 0
    for m in root.modules():
        if not (isinstance(m, AdaLN) and m.cond_dim > 0):
            continue
        n += 1
        last = m.adaLN_modulation[-1]
        if mode == "gate":
            if m.cond_gate is None:
                raise ValueError(
                    "cond_init='gate' needs AdaLN(..., cond_gate=True); this head has "
                    "no gate, so a normally-initialized modulation would not be the "
                    "identity at init"
                )
            last.reset_parameters()          # fan-in scaled, not the blanket std=0.02
            nn.init.zeros_(m.cond_gate)
        else:
            if mode == "zero":
                nn.init.zeros_(last.weight)
            else:
                nn.init.trunc_normal_(last.weight, std=std)
            nn.init.zeros_(last.bias)
            if m.cond_gate is not None:
                # Open the gate so the (near-)zeroed head itself can learn.
                nn.init.ones_(m.cond_gate)
    return n


class SwiGLU(nn.Module):
    """Gated feedforward fc2(silu(gate) * up), with gate/up fused into fc1.

    Args:
        dim: width of the input/output activations.
        exp_factor: expansion of the *equivalent* GELU MLP; the hidden width is 2/3 of it,
            so the parameter count matches that MLP.
        multiple_of: round the hidden width up to this multiple.
    """

    def __init__(self, dim, exp_factor=4., multiple_of=8):
        super().__init__()
        hidden = int(2 * dim * exp_factor / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.hidden = hidden
        self.fc1 = nn.Linear(dim, 2 * hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(self.act(gate) * up)
