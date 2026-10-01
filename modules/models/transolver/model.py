import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from modules.layers.basics import AdaLN, SwiGLU, init_cond_modulation
from modules.layers.embedding import CoeffEmbedding
from modules.models.smart.model import FieldHeads

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

class Physics_Attention_1D_Eidetic(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # B N C
        B, N, C = x.shape

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        # Per-rank slice statistics (no all_reduce: data parallel, one sample per rank).
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights).contiguous()
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_plus_block(nn.Module):
    """Pre-norm slice-attention block: AdaLN -> physics attention, AdaLN -> SwiGLU.

    Args:
        cond_dim: width of the (embedded) conditioning; 0 makes both AdaLNs plain LayerNorms.
        cond_gate: give each AdaLN a scalar gate (see AdaLN / init_cond_modulation).
        mlp_ratio: expansion of the equivalent GELU MLP (SwiGLU matches its parameter count).
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            slice_num=32,
            cond_dim=0,
            cond_gate=False,
    ):
        super().__init__()
        self.ln_1 = AdaLN(hidden_dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.Attn = Physics_Attention_1D_Eidetic(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = AdaLN(hidden_dim, cond_dim=cond_dim, cond_gate=cond_gate)
        self.mlp = SwiGLU(hidden_dim, exp_factor=mlp_ratio)

    def forward(self, fx, cond=None):
        # AdaLNs stay outside the activation checkpoints.
        if self.training:
            fx = fx + checkpoint(self.Attn, self.ln_1(fx, cond), use_reentrant=True)
            fx = fx + checkpoint(self.mlp, self.ln_2(fx, cond), use_reentrant=True)
        else:
            fx = fx + self.Attn(self.ln_1(fx, cond))
            fx = fx + self.mlp(self.ln_2(fx, cond))
        return fx


class Transolver_plus(FieldHeads, nn.Module):
    """Transolver++ slice attention over the concatenated surface + volume query cloud.

    Uses AdaLN conditioning on a CoeffEmbedding of the raw conditions, SwiGLU feedforwards and
    SMART's `FieldHeads` read-out. `geo` is accepted and ignored (the surface queries are the body).

    Args:
        space_dim: spatial dimension of the query positions.
        n_layers / n_hidden / n_head / slice_num: trunk depth, width, heads and slice tokens.
        mlp_ratio: expansion of the equivalent GELU MLP; SwiGLU matches its parameter count.
        out_dim: channels per head -- surface is [cp, cf] and volume [p, vel], so 4 and 4.
        split_heads: separate surface and volume read-outs (FieldHeads).
        parameter_channels: width of the raw `cond` vector; 0 builds an unconditional model.
        cond_fourier / cond_num_freqs / cond_norm / cond_fanin_init: CoeffEmbedding options.
        cond_init / cond_init_std: AdaLN modulation init (see init_cond_modulation).
    """

    def __init__(self,
                 space_dim=3,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=4,
                 out_dim=4,
                 slice_num=32,
                 split_heads=True,
                 parameter_channels=2,
                 cond_fourier=False,
                 cond_num_freqs=6,
                 cond_norm=False,
                 cond_fanin_init=False,
                 cond_init="zero",
                 cond_init_std=1e-3,
                 ):
        super(Transolver_plus, self).__init__()
        self.preprocess = MLP(space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.parameter_channels = parameter_channels
        self.split_heads = split_heads
        self.surface_channels = out_dim

        # Conditions are embedded to n_hidden once and modulate every AdaLN.
        if parameter_channels > 0:
            self.cond_proj = CoeffEmbedding(
                cond_dim=parameter_channels,
                cond_channels=n_hidden,
                num_freqs=cond_num_freqs,
                use_fourier=cond_fourier,
                norm=cond_norm,
                fanin_init=cond_fanin_init,
            )
            self.cond_dim = n_hidden
        else:
            self.cond_proj = None
            self.cond_dim = 0

        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      slice_num=slice_num,
                                                      cond_dim=self.cond_dim,
                                                      cond_gate=(cond_init == "gate"))
                                     for _ in range(n_layers)])
        self._build_heads(n_hidden, out_dim, out_dim, split_heads)
        self.initialize_weights(cond_fanin_init=cond_fanin_init,
                                cond_init=cond_init, cond_init_std=cond_init_std)

    def initialize_weights(self, cond_fanin_init=False, cond_init="zero", cond_init_std=1e-3):
        self.apply(self._init_weights)

        # Restore the orthogonal slice-projection init overwritten by the blanket init.
        for m in self.modules():
            if isinstance(m, Physics_Attention_1D_Eidetic):
                torch.nn.init.orthogonal_(m.in_project_slice.weight)
                nn.init.constant_(m.in_project_slice.bias, 0)

        # Fan-in init for the condition embedding, then AdaLN heads back to identity.
        if self.cond_proj is not None and cond_fanin_init:
            self.cond_proj.reset_parameters()
        init_cond_modulation(self, mode=cond_init, std=cond_init_std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, geo, surf_query_pos, vol_query_pos, params):
        # geo not used (provided just to match smart interface)
        n_surf = surf_query_pos.shape[1]
        x = torch.cat([surf_query_pos, vol_query_pos], dim = 1)

        fx = self.preprocess(x)

        if self.cond_dim > 0 and params is not None:
            if params.shape[-1] != self.parameter_channels:
                raise ValueError(
                    f"cond has {params.shape[-1]} channels but the model was built with "
                    f"parameter_channels={self.parameter_channels}"
                )
            cond = self.cond_proj(params)
        else:
            cond = None

        for block in self.blocks:
            fx = block(fx, cond)

        return self._project(fx, n_surf)
