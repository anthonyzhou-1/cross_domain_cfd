"""Paired surface + volume dataset over the collated per-run float32 .npy files.

Per sample: geometry_pos (n_geometry, 3), centroid (3,), surface_pos/cp/cf, volume_pos/vel/p,
plus optional cond / meta. Surface columns [x,y,z, cp, cf_x,cf_y,cf_z]; volume [x,y,z, u,v,w, p].
Rows are pre-shuffled on disk, so a contiguous random window is a uniform subsample.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# --- shared on-disk layout, orientation and conditioning tables -------------
POS = slice(0, 3)
P = slice(3, 4)
WSS = slice(4, 7)

# Reorientation into the shared frame: v_new = v @ ORIENT[dataset] (positions, Cf, velocity).
ORIENT = {
    "superwing": torch.tensor([[1., 0., 0.],
                               [0., 0., 1.],
                               [0., -1., 0.]]),
    "windsorml": torch.tensor([[1., 0., 0.],
                               [0., 0., 1.],
                               [0., -1., 0.]]),
}

# Per-dataset Cf sign flip into the shared convention.
CF_SIGN = {
    "ahmedml": -1.0,
    "drivaerml": -1.0,
    "drivaernet": -1.0,
    "emmi_wing": -1.0,
}

# Per-dataset condition columns and their min-max normalization ranges.
COND_RANGE = {
    "emmi_wing": {"AOA": [-10.0, 10.0], "Mach_number": [0.437318, 0.874636]},
    "blendednet": {"alpha_deg": [-8.0, 16.0], "M_inf": [0.05, 0.5]},
    "hiliftaeroml": {"aoa": [4.0, 22.0], "mach": [0.2, 0.2]},
    "double_delta": {"aoa": [11.0, 19.0], "mach": [0.3, 0.3]},
    "superwing": {"aoa": [2.0, 12.0], "mach": [0.75, 0.9]},
    "shift_cca": {"aoa": [2.0, 2.0], "mach": [0.72, 0.72]},
}

# Dataset-independent ranges for the semantic cond channels (global_cond_norm=True).
GLOBAL_COND_RANGE = {"aoa": [-20.0, 20.0], "mach": [0.0, 1.0]}

# Which GLOBAL_COND_RANGE channel each dataset's cond column maps to.
COND_KIND = {
    "emmi_wing": {"AOA": "aoa", "Mach_number": "mach"},
    "blendednet": {"alpha_deg": "aoa", "M_inf": "mach"},
    "hiliftaeroml": {"aoa": "aoa", "mach": "mach"},
    "double_delta": {"aoa": "aoa", "mach": "mach"},
    "superwing": {"aoa": "aoa", "mach": "mach"},
    "shift_cca": {"aoa": "aoa", "mach": "mach"},
}
# Conditions with no shared physical scale; they keep per-campaign bounds.
COND_KIND_EXEMPT = {
    "shift_pump": ["flow_rate", "flow_rate_op_condition"],
}

_ungrouped = {d: sorted(set(cols) - set(COND_KIND.get(d, {})) - set(COND_KIND_EXEMPT.get(d, [])))
              for d, cols in COND_RANGE.items()}
_ungrouped = {d: c for d, c in _ungrouped.items() if c}
if _ungrouped:
    raise RuntimeError(
        f"COND_RANGE columns with no COND_KIND semantic channel: {_ungrouped}. Give each a "
        "GLOBAL_COND_RANGE channel in COND_KIND, or -- if it has no shared physical scale "
        "across campaigns -- name it in COND_KIND_EXEMPT with the reason."
    )

# Physical (aoa, mach) pad for datasets without conditions, under global_cond_norm.
DEFAULT_COND = {"aoa": 0.0, "mach": 0.0}

# field_norm="physical": zero at the freestream, and one shared scale per field.
GLOBAL_FIELD_SCALE = {"cp": 0.33, "cf": 2.0e-3, "vel": 0.20}

# Per-campaign freestream reference for the physical-zero field norms:
#   u_ref    freestream speed in the volume file's stored units (float, or a metadata column)
#   aoa_col  metadata column with AoA in degrees (None = zero incidence); u_inf = u_ref*[cos a, 0, sin a]
#   p_kind   stored volume pressure: 'cp' (coefficient), 'kin' (kinematic), 'pa' (Pa, uses rho)
#   p_inf    zero of the stored volume pressure
PHYS = {
    "drivaernet":   dict(u_ref=30.0,     aoa_col=None,  p_kind="kin", rho=1.184,  p_inf=0.0),
    "drivaerml":    dict(u_ref=38.889,   aoa_col=None,  p_kind="kin", rho=1.2041, p_inf=0.0),
    "ahmedml":      dict(u_ref=1.0,      aoa_col=None,  p_kind="kin", rho=1.0,    p_inf=0.0),
    # windsorml: q and p_inf fitted (volume p vs surface cp); rho reproduces that q.
    "windsorml":    dict(u_ref=40.0,     aoa_col=None,  p_kind="pa",  rho=1.2712, p_inf=37340.5),
    "submarine":    dict(u_ref=5.0,      aoa_col=None,  p_kind="pa",  rho=998.0,  p_inf=0.0),
    "superwing":    dict(u_ref=1.0,      aoa_col="aoa", p_kind="cp",  p_inf=0.0),
    "emmi_wing":    dict(u_ref=1.0,      aoa_col="AOA", p_kind="cp",  p_inf=0.0),
    "hiliftaeroml": dict(u_ref="uRef",   aoa_col="aoa", p_kind="cp",  p_inf=0.0),
    "double_delta": dict(u_ref=101.53,   aoa_col="aoa", p_kind="cp",  p_inf=0.0),
    "shift_cca":    dict(u_ref="u_inf",  aoa_col="aoa", p_kind="cp",  p_inf=0.0),
}

# "dataset": fitted per-campaign (mean, std); "deficit": physical zero, fitted std;
# "physical": physical zero and GLOBAL_FIELD_SCALE.
FIELD_NORM_MODES = ("dataset", "deficit", "physical")

# Modes that put zero at the freestream (Cp / velocity deficit in `_volume_coef`).
PHYSICAL_ZERO_MODES = ("deficit", "physical")


# pos_norm="physical": positions scaled by GLOBAL_POS_SPAN / l_ref (streamwise extent).
GLOBAL_POS_SPAN = 2.6

# Per-campaign frame for pos_norm="physical": l_ref = streamwise body extent; y0 / z0 =
# symmetry / ground plane (None = none; that axis is anchored at a quantile midpoint).
GEOM_FRAME = {
    "ahmedml":       dict(l_ref=0.977154, y0=0.0, z0=0.0),
    "blendednet":    dict(l_ref=0.999769, y0=0.0, z0=None),
    "double_delta":  dict(l_ref=30.3564, y0=0.0, z0=None),
    "drivaerml":     dict(l_ref=4.65895, y0=0.0, z0=-0.317548),
    "drivaernet":    dict(l_ref=4.83545, y0=0.0, z0=0.0),
    "emmi_wing":     dict(l_ref=1.07651, y0=0.0, z0=None),
    "hiliftaeroml":  dict(l_ref=2471.06, y0=0.0, z0=None),
    "shift_cca":     dict(l_ref=6.82444, y0=0.0, z0=None),
    "shift_pump":    dict(l_ref=0.551957, y0=None, z0=None),
    "shift_suv":     dict(l_ref=1.15554, y0=0.0, z0=-0.005419),
    "submarine":     dict(l_ref=6.45113, y0=0.0, z0=None),
    "superwing":     dict(l_ref=2.02633, y0=0.0, z0=None),
    "windsorml":     dict(l_ref=1.044, y0=0.0, z0=0.0),
}

# "dataset": fitted per-campaign scale + crop centroid; "physical": GEOM_FRAME anchor and scale.
POS_NORM_MODES = ("dataset", "physical")

# Quantile pair whose midpoint anchors the axes without a physical plane.
ANCHOR_Q = (0.0005, 0.9995)


def phys_vol_q(dataset, u_ref):
    """Dynamic pressure converting a campaign's stored volume p column to Cp."""
    c = PHYS[dataset]
    if c["p_kind"] == "cp":
        return 1.0
    if c["p_kind"] == "kin":
        return 0.5 * float(u_ref) ** 2
    return 0.5 * c["rho"] * float(u_ref) ** 2


# Surface freestream; q = 0.5 * U_inf**2 (kinematic). precomputed_coef: cp/cf already on disk.
FREESTREAM = {
    "drivaernet": dict(U_inf=30.0, p_inf=0.0),
    "drivaerml": dict(U_inf=38.889, p_inf=0.0),
    "ahmedml": dict(U_inf=1.0, p_inf=0.0),
    "windsorml": dict(precomputed_coef=True),
    "superwing": dict(precomputed_coef=True),
    "blendednet": dict(precomputed_coef=True),
    "emmi_wing": dict(precomputed_coef=True),
    "hiliftaeroml": dict(precomputed_coef=True),
    "double_delta": dict(precomputed_coef=True),
    "submarine": dict(precomputed_coef=True),
    "shift_suv": dict(precomputed_coef=True),
    "shift_pump": dict(precomputed_coef=True),
    "shift_cca": dict(precomputed_coef=True),
}

# Per-campaign environment id, emitted as `env_id`. Append only.
ENV_ID = {
    "drivaernet": 0,
    "drivaerml": 1,
    "emmi_wing": 2,
    "windsorml": 3,
    "superwing": 4,
    "double_delta": 5,
    "ahmedml": 6,
    "blendednet": 7,
    "hiliftaeroml": 8,
    "submarine": 9,
    "shift_suv": 10,
    "shift_pump": 11,
    "shift_cca": 12,
}

# Datasets whose rows are not shuffled on disk.
NOT_PRESHUFFLED = ["blendednet"]

def _infer_dataset(out_dir):
    """Pick the freestream key whose name appears in `out_dir` (e.g. .../drivaerml/...)."""
    matches = [k for k in FREESTREAM if k in out_dir]
    if len(matches) != 1:
        raise ValueError(
            f"could not infer dataset from out_dir={out_dir!r}; "
            f"pass dataset=one of {sorted(FREESTREAM)}"
        )
    return matches[0]


V_POS = slice(0, 3)
V_VEL = slice(3, 6)
V_P = slice(6, 7)

THIN_DEFAULTS = dict(voxel=0.01, oversample=8, alpha=1.0, min_count=8)
SURFACE_THIN_DEFAULTS = dict(voxel=0.01, oversample=8, alpha=1.0, min_count=8)

def _part1by2(x):
    """Spread the low 21 bits of `x` so bit i lands at bit 3i (Morton interleave)."""
    x = x.astype(np.int64) & 0x1fffff
    x = (x | (x << 32)) & 0x1f00000000ffff
    x = (x | (x << 16)) & 0x1f0000ff0000ff
    x = (x | (x << 8)) & 0x100f00f00f00f00f
    x = (x | (x << 4)) & 0x10c30c30c30c30c3
    x = (x | (x << 2)) & 0x1249249249249249
    return x


def _morton(q):
    """(N,3) non-negative int cell coords -> (N,) int64 Morton code; `code >> 3l` is the level-l cell."""
    return _part1by2(q[:, 0]) | (_part1by2(q[:, 1]) << 1) | (_part1by2(q[:, 2]) << 2)


def dyadic_density(pos, voxel=0.01, min_count=8, dim=3):
    """Per-point density of `pos`: count / side**dim at the finest dyadic cell holding >= min_count points.

    Cells have sides voxel * 2**l. `dim` is 3 for a volume cloud and 2 for a surface.
    Returns (N,) float64 density in points per unit**dim.
    """
    n = pos.shape[0]
    pos = np.asarray(pos, dtype=np.float64)
    q = np.floor(pos / float(voxel)).astype(np.int64)
    q -= q.min(axis=0)
    span = int(q.max()) + 1
    # Morton codes carry 21 bits per axis; coarsen level 0 if the crop overflows that.
    shift = max(0, int(span - 1).bit_length() - 21)
    if shift:
        q >>= shift
        voxel = float(voxel) * (1 << shift)
    levels = max(1, int(q.max()).bit_length() + 1)

    code = _morton(q)
    order = np.argsort(code, kind="stable")
    code = code[order]
    del q

    dens = np.empty(n, dtype=np.float64)
    todo = np.ones(n, dtype=bool)
    cnt_per_pt = np.ones(n, dtype=np.int64)
    for l in range(levels):
        key = code >> (3 * l)
        # `code` is sorted, so run lengths of equal keys are the cell counts.
        edge = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1], [True])))
        run = np.diff(edge)
        cnt_per_pt = np.repeat(run, run)
        hit = todo & (cnt_per_pt >= min_count)
        if hit.any():
            dens[hit] = cnt_per_pt[hit] / (voxel * (1 << l)) ** dim
            todo &= ~hit
            if not todo.any():
                break
    if todo.any():
        # Fewer than min_count points in the whole crop at the coarsest level.
        dens[todo] = cnt_per_pt[todo] / (voxel * (1 << (levels - 1))) ** dim

    out = np.empty(n, dtype=np.float64)
    out[order] = dens
    return out


def voxel_thin_indices(pos, n_keep, rng, voxel=0.01, alpha=1.0, min_count=8, dim=3):
    """Indices (ascending) of `n_keep` rows of `pos`, sampled ~uniformly in space rather than per mesh point.

    Weights are dyadic_density ** -alpha (alpha=1 space-uniform, 0 = mesh measure), drawn
    without replacement by the exponential race (Efraimidis-Spirakis). N <= n_keep returns all rows.
    """
    n = pos.shape[0]
    if n_keep >= n:
        return np.arange(n, dtype=np.int64)

    w = dyadic_density(pos, voxel=voxel, min_count=min_count, dim=dim) ** (-float(alpha))

    keys = rng.exponential(size=n) / w
    idx = np.argpartition(keys, n_keep)[:n_keep]
    idx.sort()
    return idx


def surface_thin_indices(pos, n_keep, rng, voxel=0.01, alpha=1.0, min_count=8):
    """`voxel_thin_indices` for a surface cloud (dim=2): ~area-uniform instead of mesh-uniform."""
    return voxel_thin_indices(pos, n_keep, rng, voxel=voxel, alpha=alpha,
                              min_count=min_count, dim=2)


class SurfaceVolumeDataset(Dataset):

    # Constant cond channels appended to this dataset's own (see _init_cond_extra).
    cond_extra = None

    def __init__(self, surf_dir, vol_dir, split,
                 n_geometry=None, n_surface=128000, n_volume=128000,
                 load_surface=True, load_volume=True, dataset=None,
                 normalize=True, surf_stats_name="norm_stats_centered.npz",
                 vol_stats_name="norm_stats_volume.npz",
                 return_metadata=False, return_cond=True,
                 num_val=None, global_cond_norm=False, cond_extra=None,
                 volume_thin=THIN_DEFAULTS,
                 surface_thin=SURFACE_THIN_DEFAULTS,
                 include_prefixes=None, exclude_prefixes=None,
                 field_norm="dataset", pos_norm="dataset"):
        
        super().__init__()
        self.surf_samples = os.path.join(surf_dir, "samples")
        self.vol_samples = os.path.join(vol_dir, "samples")
        self.split = split
        self.n_geometry = n_geometry
        self.n_surface = n_surface
        self.n_volume = n_volume
        self.load_surface = load_surface
        self.load_volume = load_volume
        self.normalize = normalize
        self.include_prefixes = include_prefixes
        self.exclude_prefixes = exclude_prefixes

        self.volume_thin, self.thin_oversample = self._init_thin(volume_thin, THIN_DEFAULTS,
                                                                "volume_thin")
        self.surface_thin, self.surf_thin_oversample = self._init_thin(
            surface_thin, SURFACE_THIN_DEFAULTS, "surface_thin")

        self.dataset = dataset if dataset is not None else _infer_dataset(surf_dir)
        self.env_id = ENV_ID[self.dataset]
        self.M = ORIENT.get(self.dataset)
        self.cf_sign = CF_SIGN.get(self.dataset, 1.0)

        fs = FREESTREAM[self.dataset]
        self.precomputed_coef = fs.get("precomputed_coef", False)
        if not self.precomputed_coef:
            self.q = 0.5 * fs["U_inf"] ** 2  # dynamic pressure (kinematic fields, no rho)
            self.p_inf = fs["p_inf"]

        self._init_field_norm(field_norm)
        self._init_pos_norm(pos_norm)

        # Splits are shared: surface and volume are paired by run id (stem).
        by_stem = {r["stem"]: r for r in self._load_manifest(surf_dir)}
        vol_len = {r["stem"]: (r["n_cells"] if "n_cells" in r else r["n_points"]) for r in self._load_manifest(vol_dir)} if load_volume else {}
        stems = self._read_split(surf_dir, split, by_stem, vol_len)
        self.stems = self._truncate(stems, split, num_val)
        self.surf_lengths = [by_stem[s]["n_points"] for s in self.stems]
        self.vol_lengths = [vol_len[s] for s in self.stems] if load_volume else None

        # Before _init_cond_meta: _load_cond/_pad_cond read it.
        self._init_cond_extra(cond_extra)

        # Metadata and cond live with the surface build.
        self._init_cond_meta(surf_dir, self.stems, by_stem, return_metadata=return_metadata,
                             return_cond=return_cond, global_cond_norm=global_cond_norm)

        # Per-stem freestream, packed (stem -> row) to keep worker memory small.
        self._phys_row = {}
        self._phys_u_inf = self._phys_q = self._phys_uref = None
        # Registered under every field_norm: the canonical_* methods need it.
        self.vol_p_inf = float(PHYS[self.dataset]["p_inf"]) if self.dataset in PHYS else 0.0
        if self.dataset in PHYS:
            self._register_phys(self.stems)
        if normalize:
            self._load_surf_stats(os.path.join(surf_dir, surf_stats_name))
            if load_volume:
                self._load_vol_stats(os.path.join(vol_dir, vol_stats_name))

        print("Init SurfaceVolumeDataset:", self.dataset, split, len(self.stems), "samples,",
              f"n_geometry={n_geometry} n_surface={n_surface} n_volume={n_volume}")
        print("field_norm:", self.field_norm, "pos_norm:", self.pos_norm)
        if self.pos_norm != "dataset":
            print(f"pos_norm: {self.pos_norm} (l_ref={self.geom_frame['l_ref']:.6g}, "
                  f"y0={self.geom_frame['y0']}, z0={self.geom_frame['z0']}) -> "
                  f"pos_scale {self.pos_scale_thin:.6g} -> {self.pos_scale:.6g}; "
                  "sampling measure held in the stats frame")
        if self.volume_thin is not None:
            print(f"volume_thin: voxel={self.volume_thin['voxel']} "
                  f"min_count={self.volume_thin['min_count']} "
                  f"alpha={self.volume_thin['alpha']} oversample={self.thin_oversample} "
                  f"({self.thin_oversample * n_volume} rows read per crop)")
        if self.surface_thin is not None:
            print(f"surface_thin: voxel={self.surface_thin['voxel']} "
                  f"min_count={self.surface_thin['min_count']} "
                  f"alpha={self.surface_thin['alpha']} oversample={self.surf_thin_oversample} "
                  f"({self.surf_thin_oversample * (n_geometry or n_surface)} rows read per crop)")
        print("n_cond:", self.n_cond, "num_val:", num_val)
        if include_prefixes or exclude_prefixes:
            print("stem filter: include", include_prefixes, "exclude", exclude_prefixes)

    @staticmethod
    def _init_thin(thin, defaults=THIN_DEFAULTS, name="volume_thin"):
        """Validate a thinning block (None/False off, True defaults, or a dict) -> (kwargs, oversample)."""
        if not thin:
            return None, 1
        cfg = dict(defaults)
        if isinstance(thin, dict):
            bad = sorted(set(thin) - set(defaults))
            if bad:
                raise ValueError(f"{name}: unknown key(s) {bad}; "
                                 f"expected any of {sorted(defaults)}")
            cfg.update(thin)
        oversample = int(cfg.pop("oversample"))
        cfg["voxel"], cfg["alpha"] = float(cfg["voxel"]), float(cfg["alpha"])
        cfg["min_count"] = int(cfg["min_count"])
        if cfg["voxel"] <= 0:
            raise ValueError(f"{name}.voxel must be > 0, got {cfg['voxel']}")
        if cfg["min_count"] < 1:
            raise ValueError(f"{name}.min_count must be >= 1, got {cfg['min_count']}")
        if not 0.0 <= cfg["alpha"] <= 1.0:
            raise ValueError(f"{name}.alpha must be in [0, 1], got {cfg['alpha']}")
        if oversample < 1:
            raise ValueError(f"{name}.oversample must be >= 1, got {oversample}")
        if cfg["alpha"] == 0.0 or oversample == 1:
            raise ValueError(
                f"{name} with alpha=0 or oversample=1 cannot change the sample: "
                "alpha=0 gives every point the same weight and oversample=1 leaves "
                f"nothing to select from. Set {name}: null to turn thinning off.")
        return cfg, oversample

    def _read_split(self, surf_dir, split, by_stem, vol_len):
        """Stems of `split` present in every modality read, after the prefix filter."""
        with open(os.path.join(surf_dir, "splits", f"{split}.json")) as f:
            stems = json.load(f)
        stems = self._filter_stems(stems, self.include_prefixes, self.exclude_prefixes,
                                   where=f"{self.dataset} {split}")
        return [s for s in stems
                if s in by_stem and (not self.load_volume or s in vol_len)]

    def _register_phys(self, stems, meta=None, meta_cols=None):
        """Register the freestream of not-yet-registered `stems` (see `_freestream`). Single pass."""
        base = 0 if self._phys_u_inf is None else self._phys_u_inf.shape[0]
        keep = []
        for i, s in enumerate(stems):
            if s in self._phys_row:
                continue
            self._phys_row[s] = base + len(keep)
            keep.append(i)
        if not keep:
            return
        u_ref, u_inf, vol_q = self._freestream(stems, meta, meta_cols)
        pick = torch.as_tensor(keep, dtype=torch.long)
        cat = lambda old, addition: (addition if old is None
                                     else torch.cat([old, addition], dim=0))
        self._phys_u_inf = cat(self._phys_u_inf, u_inf[pick])
        self._phys_q = cat(self._phys_q, vol_q[pick, 0])
        self._phys_uref = cat(self._phys_uref, u_ref[pick, 0])

    def _load_vol_stats(self, path):
        """Volume field mean/std over [u_x,u_y,u_z,p], reindexed into the loader frame."""
        z = np.load(path)
        mean = torch.from_numpy(z["mean"].astype(np.float32))  # [4]
        std = torch.from_numpy(z["std"].astype(np.float32))    # [4]
        vel_mean, vel_std = mean[0:3], std[0:3]
        if self.M is not None:
            vel_mean = vel_mean @ self.M
            vel_std = vel_std @ self.M.abs()
        self.vel_mean, self.vel_std = vel_mean.reshape(1, 3), vel_std.reshape(1, 3)
        self.vp_mean, self.vp_std = mean[3:4].view(1, 1), std[3:4].view(1, 1)

        if self.field_norm in PHYSICAL_ZERO_MODES:
            # `_volume_coef` already emits Cp and the velocity deficit, so zero is physical.
            z, o = torch.zeros(1, 1), torch.ones(1, 1)
            self.vp_mean, self.vel_mean = z, z.expand(1, 3)
            if self.field_norm == "physical":
                self.vp_std = o * GLOBAL_FIELD_SCALE["cp"]
                self.vel_std = (o * GLOBAL_FIELD_SCALE["vel"]).expand(1, 3)
            else:
                # "deficit": the fitted std, converted from stored units by u_ref and q.
                u_ref, q = self._campaign_scales()
                self.vel_std = vel_std.reshape(1, 3) / u_ref
                self.vp_std = std[3:4].view(1, 1) / q

    def _anchor(self, pos):
        """Per-sample frame origin (1, 3): the crop centroid, or under "physical" the
        GEOM_FRAME planes plus an ANCHOR_Q quantile midpoint on the other axes."""
        if self.pos_norm == "dataset":
            return pos.mean(dim=0, keepdim=True)
        lo = torch.quantile(pos, ANCHOR_Q[0], dim=0)
        hi = torch.quantile(pos, ANCHOR_Q[1], dim=0)
        c = ((lo + hi) / 2).reshape(1, 3)
        for axis, key in ((1, "y0"), (2, "z0")):
            plane = self.geom_frame[key]
            if plane is not None:
                c[0, axis] = plane
        return c

    def _orient(self, pos):
        """Signed-permutation reorientation into the shared physical frame. See ORIENT."""
        return pos @ self.M.to(pos) if self.M is not None else pos

    def _frame(self, pos_raw, c=None, scale=None):
        """Raw positions -> shared frame (reorient, center by `c`, scale); returns (pos, c).

        `c=None` computes the anchor from these positions. With normalize=False only reorients.
        """
        pos = self._orient(pos_raw)
        if not self.normalize:
            return pos, c
        if c is None:
            c = self._anchor(pos)
        return (pos - c) * (self.pos_scale if scale is None else scale), c

    def _window(self, arr, n_want, rng):
        """Contiguous random window of up to `n_want` rows (rows are pre-shuffled)."""
        n = arr.shape[0]
        k = min(n_want, n)
        start = int(rng.integers(0, max(1, n - k + 1)))
        return np.array(arr[start:start + k], dtype=np.float32)  # copy: materialize + writable

    def _surface_window(self, arr, n_want, rng):
        """`_window`, then surface thinning (on raw positions, voxel / pos_scale_thin)."""
        if self.surface_thin is None:
            return self._window(arr, n_want, rng)
        data = self._window(arr, n_want * self.surf_thin_oversample, rng)
        scale = self.pos_scale_thin
        keep = surface_thin_indices(data[:, POS], n_want, rng,
                                    voxel=self.surface_thin["voxel"] / scale,
                                    alpha=self.surface_thin["alpha"],
                                    min_count=self.surface_thin["min_count"])
        return data[keep] if keep.size < data.shape[0] else data

    def _surf_coef(self, data):
        """cp (n,1) and cf (n,3) from a surface crop, standardized if normalize."""
        if self.precomputed_coef:
            cp, cf = data[:, P], data[:, WSS]           # already coefficients
        else:
            cp = (data[:, P] - self.p_inf) / self.q      # pressure coefficient
            cf = data[:, WSS] / self.q                   # skin friction coefficient
        if self.M is not None:
            cf = cf @ self.M.to(cf)                       # rotate wall-shear vector into loader frame
        cf = cf * self.cf_sign                            # cf-only sign convention flip (see CF_SIGN)
        if self.normalize:
            cp = (cp - self.cp_mean) / self.cp_std
            cf = (cf - self.cf_mean) / self.cf_std
        return cp, cf

    def _surface_crop(self, stem, n_want, rng, c=None):
        """(pos, cp, cf, centroid) from a random window of `stem`'s surface cloud."""
        arr = np.load(os.path.join(self.surf_samples, stem + ".npy"), mmap_mode="r")
        data = torch.from_numpy(self._surface_window(arr, n_want, rng))
        pos, c = self._frame(data[:, POS], c)
        cp, cf = self._surf_coef(data)
        return pos, cp, cf, c

    def _volume_crop(self, stem, n_want, rng, c, c_thin=None):
        """(pos, vel, p) from a random window of `stem`'s volume cloud, centered by `c`.

        `c` is the same run's surface anchor; `c_thin` (default `c`) anchors the frame the
        thinning runs in. A 3-D (R, K, 7) file holds R resampled rounds; one is picked at random.
        """
        arr = np.load(os.path.join(self.vol_samples, stem + ".npy"), mmap_mode="r")
        if arr.ndim == 3:
            arr = arr[int(rng.integers(arr.shape[0]))]
        vol = torch.from_numpy(self._window(arr, n_want * self.thin_oversample, rng))
        if self.volume_thin is not None:
            thin_pos, _ = self._frame(vol[:, V_POS], c if c_thin is None else c_thin,
                                      scale=self.pos_scale_thin)
            keep = torch.from_numpy(
                voxel_thin_indices(thin_pos.numpy(), n_want, rng, **self.volume_thin))
            vol = vol[keep]
        pos, _ = self._frame(vol[:, V_POS], c)
        vel, vp = self._volume_coef(vol, stem)
        return pos, vel, vp

    def _campaign_scales(self):
        """(u_ref, vol_q) for the campaign, asserting they do not vary across samples."""
        u = self._phys_uref
        q = self._phys_q
        for name, t in (("u_ref", u), ("vol_q", q)):
            if t is None or t.numel() == 0:
                raise ValueError(f"{self.dataset}: no freestream registered for {name}")
            if float(t.max() - t.min()) > 1e-6 * float(t.abs().max().clamp_min(1e-12)):
                raise ValueError(
                    f"{self.dataset}: field_norm='deficit' needs a campaign-constant "
                    f"{name}, but it spans {float(t.min()):.6g}..{float(t.max()):.6g}. "
                    "Use field_norm='physical' (whose scale is a constant by construction) "
                    "or fit a stats file in coefficient units."
                )
        return float(u[0]), float(q[0])

    # --- canonical comparison space: Cp, Cf and u/U_inf, independent of field_norm ---

    def _ref(self, ref):
        return self.phys_of(ref) if isinstance(ref, str) else ref

    @staticmethod
    def _bcast(ref, c, like):
        """Per-sample constant `ref` ([B, c], [1, c] or float) shaped to broadcast against
        `like` ((..., B, N, C) or (N, C))."""
        t = torch.as_tensor(ref).to(like).reshape(-1, c)
        if like.ndim > 2:
            t = t.reshape(*([1] * (like.ndim - 3)), -1, 1, c)
        return t

    def canonical_cp(self, cp):
        """Normalized surface cp -> Cp. Surface coefficients need no unit change."""
        return self.denormalize_cp(cp)

    def canonical_cf(self, cf):
        """Normalized surface cf -> Cf."""
        return self.denormalize_cf(cf)

    def canonical_velocity(self, vel, stem):
        """Normalized volume velocity -> u/U_inf. `stem` may be the (u_inf, vol_q, u_ref) triple."""
        v = self.denormalize_velocity(vel)
        u_inf, _, u_ref = self._ref(stem)
        u_inf, u_ref = self._bcast(u_inf, 3, v), self._bcast(u_ref, 1, v)
        return v + u_inf / u_ref if self.field_norm in PHYSICAL_ZERO_MODES else v / u_ref

    def canonical_volume_p(self, p, stem):
        """Normalized volume pressure -> Cp. `stem` may be the triple; see `_ref`."""
        x = self.denormalize_volume_p(p)
        if self.field_norm in PHYSICAL_ZERO_MODES:
            return x
        _, q, _ = self._ref(stem)
        return (x - self.vol_p_inf) / self._bcast(q, 1, x)

    def phys_of(self, stem):
        """(u_inf [1,3], vol_q, u_ref) for `stem`, with a readable error when unregistered."""
        if stem is None:
            raise ValueError(
                "this field_norm needs the stem in _volume_coef: the freestream is "
                "per sample. Pass stem=..., as _volume_crop does."
            )
        try:
            i = self._phys_row[stem]
        except KeyError:
            raise KeyError(f"{stem!r} has no registered freestream (see _register_phys)")
        return self._phys_u_inf[i:i + 1], float(self._phys_q[i]), float(self._phys_uref[i])

    def _volume_coef(self, vol, stem=None):
        """velocity (n,3) and pressure (n,1) from raw volume rows, standardized if normalize.

        `stem` is required under the physical-zero field norms (per-sample freestream).
        """
        vel, vp = vol[:, V_VEL], vol[:, V_P]
        if self.M is not None:
            vel = vel @ self.M.to(vel)               # rotate velocity vector into loader frame
        if self.field_norm in PHYSICAL_ZERO_MODES:
            u_inf, vol_q, u_ref = self.phys_of(stem)
            vp = (vp - self.vol_p_inf) / vol_q
            vel = (vel - u_inf.to(vel)) / u_ref
        if self.normalize:
            vel = (vel - self.vel_mean) / self.vel_std
            vp = (vp - self.vp_mean) / self.vp_std
        return vel, vp

    def __len__(self):
        return len(self.stems)

    def _rng(self):
        info = torch.utils.data.get_worker_info()
        seed = torch.initial_seed() if info is not None else np.random.randint(2 ** 31)
        return np.random.default_rng(seed % (2 ** 32))

    def __getitem__(self, idx):
        rng = self._rng()
        stem = self.stems[idx]
        out = {}

        # --- surface: geometry crop (always) + optional target crop ---
        surf_arr = np.load(os.path.join(self.surf_samples, stem + ".npy"), mmap_mode="r")
        if self.n_geometry is None:
            geom_src = self._surface_window(surf_arr, self.n_surface, rng)  # reused as targets too
            surf_tgt = geom_src
        else:
            geom_src = self._surface_window(surf_arr, self.n_geometry, rng)
            surf_tgt = self._surface_window(surf_arr, self.n_surface, rng) if self.load_surface else None

        geom_t = torch.from_numpy(geom_src[:, POS])
        geom_pos, c = self._frame(geom_t)                    # c anchors the frame
        out["geometry_pos"] = geom_pos
        # Anchor of the frame the volume thinning runs in (the centroid).
        c_thin = c
        if (self.pos_norm != "dataset" and self.normalize
                and self.load_volume and self.volume_thin is not None):
            c_thin = self._orient(geom_t).mean(dim=0, keepdim=True)
        # per-sample anchor (rotated frame) needed to map predictions back to native coords
        out["centroid"] = c.squeeze(0) if c is not None else torch.zeros(3)

        if self.load_surface:
            surf = torch.from_numpy(surf_tgt) if surf_tgt is not geom_src else torch.from_numpy(geom_src)
            surf_pos, _ = self._frame(surf[:, POS], c)
            cp, cf = self._surf_coef(surf)
            out["surface_pos"], out["surface_cp"], out["surface_cf"] = surf_pos, cp, cf

        # --- volume: field targets, centered by the SAME surface anchor c ---
        if self.load_volume:
            out["volume_pos"], out["volume_vel"], out["volume_p"] = self._volume_crop(
                stem, self.n_volume, rng, c, c_thin)

        # Per-sample freestream, for the canonical_* conversions downstream.
        if self.dataset in PHYS:
            u_inf, q, u_ref = self.phys_of(stem)
            out["u_inf"] = u_inf.reshape(3)
            out["vol_q"] = torch.tensor([q], dtype=torch.float32)
            out["u_ref"] = torch.tensor([u_ref], dtype=torch.float32)

        out["env_id"] = torch.tensor(self.env_id, dtype=torch.long)

        return self._add_cond_meta(out, idx)

    # --- denormalization (invert __getitem__'s normalization) -----------------
    def denormalize_points(self, pos):
        """(..., 3) -> physical meters (scale only; per-sample centroid not restored)."""
        return pos / self.pos_scale

    def to_native_positions(self, pos, centroid):
        """(..., 3) normalized -> native frame, fully inverting _frame given the batch `centroid`."""
        if centroid.ndim == pos.ndim - 1:
            centroid = centroid.unsqueeze(-2)  # (..., 3) -> (..., 1, 3)
        if self.normalize:
            pos = pos / self.pos_scale + centroid
        if self.M is not None:
            pos = pos @ self.M.T.to(pos)  # inverse reorientation (orthogonal: M^-1 = M^T)
        return pos

    def denormalize_cp(self, cp):
        return cp * self.cp_std.to(cp) + self.cp_mean.to(cp)

    def denormalize_cf(self, cf):
        return cf * self.cf_std.to(cf) + self.cf_mean.to(cf)

    def denormalize_velocity(self, vel):
        return vel * self.vel_std.to(vel) + self.vel_mean.to(vel)

    def denormalize_volume_p(self, p):
        return p * self.vp_std.to(p) + self.vp_mean.to(p)

    def _load_surf_stats(self, path):
        """Surface pos_scale and cf/cp mean/std; raw-frame cf stats (no `oriented_cf`) are reindexed."""
        z = np.load(path)
        mean = torch.from_numpy(z["mean"].astype(np.float32))  # [7]
        std = torch.from_numpy(z["std"].astype(np.float32))    # [7]
        # The file's scale always sets the thinning frame; pos_norm picks the output scale.
        self.pos_scale_thin = float(z["pos_scale"])
        self.pos_scale = (self.pos_scale_thin if self.pos_norm == "dataset"
                          else GLOBAL_POS_SPAN / self.geom_frame["l_ref"])

        cf_mean, cf_std = mean[3:6], std[3:6]
        oriented = bool(z["oriented_cf"]) if "oriented_cf" in z.files else False
        if not oriented:
            if self.M is not None:
                cf_mean = cf_mean @ self.M
                cf_std = cf_std @ self.M.abs()
            cf_mean = cf_mean * self.cf_sign  # cf-only sign flip (std unchanged)

        # cf/cp slices reshaped for broadcasting over [n, c]
        self.cf_mean, self.cf_std = cf_mean.reshape(1, 3), cf_std.reshape(1, 3)
        self.cp_mean, self.cp_std = mean[6:7].view(1, 1), std[6:7].view(1, 1)

        if getattr(self, "field_norm", "dataset") in PHYSICAL_ZERO_MODES:
            # Stored coefficients already have a physical zero; physical uses an isotropic scale.
            z, o = torch.zeros(1, 1), torch.ones(1, 1)
            self.cp_mean, self.cf_mean = z, z.expand(1, 3)
            if self.field_norm == "physical":
                assert torch.allclose(cf_std.abs().sort().values,
                                      std[3:6].abs().sort().values), \
                    "ORIENT reindex is not a permutation of |cf_std|; the isotropic " \
                    "override assumes it is"
                self.cp_std = o * GLOBAL_FIELD_SCALE["cp"]
                self.cf_std = (o * GLOBAL_FIELD_SCALE["cf"]).expand(1, 3)

    def _init_cond_extra(self, cond_extra):
        """Constant cond channels appended to this dataset's own (e.g. a campaign label in [0, 1])."""
        if cond_extra is None:
            self.cond_extra = None
            return
        vals = [cond_extra] if np.isscalar(cond_extra) else list(cond_extra)
        if not vals:
            raise ValueError("cond_extra is empty; use None to append nothing")
        self.cond_extra = torch.tensor([float(v) for v in vals], dtype=torch.float32)

    def _append_cond_extra(self, cond):
        """`cond` ([n, k] table or a [k] pad) widened by the constant `cond_extra` channels."""
        if self.cond_extra is None:
            return cond
        extra = self.cond_extra
        if cond.dim() == 1:
            return torch.cat([cond, extra])
        return torch.cat([cond, extra.expand(cond.shape[0], -1)], dim=1)

    def _init_cond_meta(self, out_dir, stems, by_stem, return_metadata=False,
                        return_cond=True, global_cond_norm=False):
        """Populate the metadata table and the conditioning vector for `stems`.

        Sets `meta`/`meta_cols` (optional per-sample metadata: conditions,
        coefficients, geometry params) and `cond`/`cond_cols`/`n_cond`/`pad_cond`
        (the model-facing conditioning vector, a normalized subset of `meta`).
        """
        self.return_metadata = return_metadata
        self.meta, self.meta_cols = self._load_metadata(out_dir, stems, by_stem)

        self.return_cond = return_cond
        self.global_cond_norm = global_cond_norm
        self.cond, self.cond_cols = self._load_cond(stems)
        self.n_cond = 0 if self.cond is None else self.cond.shape[1]
        self.pad_cond = self._pad_cond()

    def _init_field_norm(self, field_norm):
        """Validate and record the field normalization scheme. See GLOBAL_FIELD_SCALE."""
        if field_norm not in FIELD_NORM_MODES:
            raise ValueError(f"field_norm must be one of {FIELD_NORM_MODES}, got {field_norm!r}")
        self.field_norm = field_norm
        if field_norm in PHYSICAL_ZERO_MODES and self.dataset not in PHYS:
            raise ValueError(
                f"field_norm={field_norm!r} needs a PHYS entry for {self.dataset!r}; known: "
                f"{sorted(PHYS)}. Add the campaign's freestream reference state (see the "
                "table's docstring) -- it cannot be inferred from the collated tree."
            )

    def _init_pos_norm(self, pos_norm):
        """Validate and record the position normalization scheme. See GLOBAL_POS_SPAN."""
        if pos_norm not in POS_NORM_MODES:
            raise ValueError(f"pos_norm must be one of {POS_NORM_MODES}, got {pos_norm!r}")
        self.pos_norm = pos_norm
        self.geom_frame = GEOM_FRAME.get(self.dataset)
        if pos_norm != "dataset" and self.geom_frame is None:
            raise ValueError(
                f"pos_norm={pos_norm!r} needs a GEOM_FRAME entry for {self.dataset!r}; known: "
                f"{sorted(GEOM_FRAME)}. The streamwise reference length and the "
                "symmetry/ground planes cannot be inferred from the collated tree."
            )
        # Thinning-frame scale; set by _load_surf_stats (stays 1.0 when normalize=False).
        self.pos_scale_thin = 1.0

    def _meta_col(self, name, stems, meta, meta_cols):
        """The `name` column of a metadata table whose rows are aligned to `stems`, as float64."""
        if meta is None or meta_cols is None or name not in meta_cols:
            raise ValueError(
                f"{self.dataset}: this field_norm needs the {name!r} column of "
                f"{'metadata.npz' if meta is None else 'the metadata table'}, which has "
                f"{'no columns (file missing?)' if meta_cols is None else sorted(set(meta_cols))}"
            )
        if meta.shape[0] != len(stems):
            raise ValueError(
                f"{self.dataset}: metadata table has {meta.shape[0]} rows against "
                f"{len(stems)} stems; rows must be aligned to stems. Pass the matching "
                "meta/meta_cols, not this dataset's full table."
            )
        return np.asarray(meta[:, meta_cols.index(name)], dtype=np.float64)

    def _freestream(self, stems, meta=None, meta_cols=None):
        """(u_ref [N,1], u_inf [N,3], vol_q [N,1]) for `stems`, loader frame; u_inf = u_ref*[cos a, 0, sin a]."""
        if meta is None and meta_cols is None:
            meta, meta_cols = self.meta, self.meta_cols
        c = PHYS[self.dataset]
        n = len(stems)

        u = c["u_ref"]
        u_ref = (np.full(n, float(u), dtype=np.float64) if not isinstance(u, str)
                 else self._meta_col(u, stems, meta, meta_cols))
        if not np.all(u_ref > 0):
            raise ValueError(f"{self.dataset}: non-positive freestream speed in u_ref")

        aoa = (np.zeros(n, dtype=np.float64) if c["aoa_col"] is None
               else np.deg2rad(self._meta_col(c["aoa_col"], stems, meta, meta_cols)))

        u_inf = np.stack([u_ref * np.cos(aoa), np.zeros(n), u_ref * np.sin(aoa)], axis=1)
        vol_q = np.array([phys_vol_q(self.dataset, x) for x in u_ref], dtype=np.float64)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return t(u_ref[:, None]), t(u_inf), t(vol_q[:, None])

    def _add_cond_meta(self, out, idx):
        """Add the "meta"/"cond" entries for sample `idx` to a batch dict, in place."""
        if self.meta is not None and self.return_metadata:
            out["meta"] = torch.from_numpy(self.meta[idx])
        if self.return_cond:
            if self.cond is not None:
                out["cond"] = self.cond[idx].clone()   # [n_cond]
            else:  # pad to the 2-wide (aoa, mach) layout (see _pad_cond)
                out["cond"] = self.pad_cond.clone()
        return out

    def _load_metadata(self, out_dir, stems, by_stem):
        """(meta [len(stems), k] float32, cols) from out_dir/metadata.npz, or (None, None).

        Layouts: keyed by `stems` (matrices X with `X_cols`), or `index` + `cols` keyed by run_id.
        """
        path = os.path.join(out_dir, "metadata.npz")
        if not os.path.exists(path):
            return None, None
        z = np.load(path, allow_pickle=True)

        if "stems" in z.files:
            row_of = {s: i for i, s in enumerate(z["stems"].tolist())}
            mats, cols = [], []
            for k in z.files:
                ck = f"{k}_cols"
                if ck in z.files and z[k].ndim == 2 and z[k].shape[0] == len(row_of):
                    mats.append(np.asarray(z[k], dtype=np.float32))
                    cols.extend(str(c) for c in z[ck].tolist())
            table = np.concatenate(mats, axis=1)
            rows = np.array([row_of[s] for s in stems])
        elif "index" in z.files and "cols" in z.files:
            table = np.asarray(z["index"], dtype=np.float32)
            cols = [str(c) for c in z["cols"].tolist()]
            rows = np.array([by_stem[s]["run_id"] for s in stems])
        else:
            raise ValueError(f"unrecognized metadata.npz layout: keys={z.files}")

        return table[rows], cols

    def _load_cond(self, stems, meta=None, meta_cols=None):
        """(cond [len(stems), n_cond], cols), min-max normalized; (None, None) if not in COND_RANGE.

        A zero-width bound (a condition held constant) normalizes to 0.
        """
        spec = COND_RANGE.get(self.dataset)
        if spec is None:
            return None, None
        if meta is None:
            meta, meta_cols = self.meta, self.meta_cols
        cols = list(spec)
        if meta is None:
            raise ValueError(
                f"dataset {self.dataset!r} is in COND_RANGE (conditions {cols}) but no "
                f"metadata.npz was found; cannot build the cond vector"
            )
        missing = [c for c in cols if c not in meta_cols]
        if missing:
            raise ValueError(
                f"conditions {missing} for dataset {self.dataset!r} are not metadata "
                f"columns; available: {meta_cols}"
            )
        idx = [meta_cols.index(c) for c in cols]
        cond = torch.from_numpy(np.ascontiguousarray(meta[:, idx]))
        bounds = [self._cond_bounds(c, spec[c]) for c in cols]
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32).view(1, -1)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32).view(1, -1)
        self.cond_lo, self.cond_hi = lo, hi
        if self.normalize:
            self._warn_out_of_range(cond, cols, lo, hi)
            span = hi - lo
            varies = span > 0                                  # False -> held constant
            safe = torch.where(varies, span, torch.ones_like(span))   # never divide by 0
            cond = torch.where(varies, (cond - lo) / safe, torch.zeros_like(cond))
        if self.cond_extra is not None:
            k = self.cond_extra.numel()
            cond = self._append_cond_extra(cond)
            cols = cols + [f"const{j}" for j in range(k)]
            self.cond_lo = torch.cat([lo, torch.zeros(1, k)], dim=1)
            self.cond_hi = torch.cat([hi, torch.ones(1, k)], dim=1)
        return cond, cols

    def _cond_bounds(self, col, dataset_range):
        """Min-max bounds for cond column `col`: GLOBAL_COND_RANGE when enabled and mapped, else per-dataset."""
        if not self.global_cond_norm:
            return dataset_range
        kind = COND_KIND.get(self.dataset, {}).get(col)
        return dataset_range if kind is None else GLOBAL_COND_RANGE[kind]

    def _warn_out_of_range(self, cond, cols, lo, hi):
        """Warn (never clip) when a condition sits outside its min-max bounds."""
        outside = (cond < lo) | (cond > hi)
        for j, c in enumerate(cols):
            if outside[:, j].any():
                v = cond[outside[:, j], j]
                print(
                    f"[{type(self).__name__}] warning: {self.dataset} condition {c!r} has "
                    f"{int(outside[:, j].sum())} sample(s) outside the normalization "
                    f"range [{lo[0, j]:g}, {hi[0, j]:g}] "
                    f"(min {v.min():g}, max {v.max():g}); values are not clipped"
                )

    def _pad_cond(self):
        """2-wide (aoa, mach) cond pad for datasets without conditions: zeros, or DEFAULT_COND under global_cond_norm."""
        if not (self.global_cond_norm and self.normalize):
            return self._append_cond_extra(torch.zeros(2, dtype=torch.float32))
        vals = []
        for kind in ("aoa", "mach"):
            lo, hi = GLOBAL_COND_RANGE[kind]
            vals.append((DEFAULT_COND[kind] - lo) / (hi - lo))
        return self._append_cond_extra(torch.tensor(vals, dtype=torch.float32))

    def denormalize_cond(self, cond):
        """(b, n_cond) conditions: undo the min-max map back to physical units."""
        lo, hi = self.cond_lo.to(cond), self.cond_hi.to(cond)
        return cond * (hi - lo) + lo

    @staticmethod
    def _filter_stems(stems, include_prefixes=None, exclude_prefixes=None, where=""):
        """Keep `include_prefixes`, then drop `exclude_prefixes` (str.startswith); raises if empty."""
        if not include_prefixes and not exclude_prefixes:
            return stems
        out = stems
        if include_prefixes:
            inc = tuple(include_prefixes)
            out = [s for s in out if s.startswith(inc)]
        if exclude_prefixes:
            exc = tuple(exclude_prefixes)
            out = [s for s in out if not s.startswith(exc)]
        if not out:
            raise ValueError(
                f"stem prefix filter emptied {where or 'the split'} "
                f"({len(stems)} stems in): include_prefixes={include_prefixes}, "
                f"exclude_prefixes={exclude_prefixes}"
            )
        return out

    @staticmethod
    def _truncate(stems, split, num_val):
        """Strided subsample of a non-train split down to `num_val` entries."""
        if num_val is None or split == "train" or len(stems) <= num_val:
            return stems
        stride = len(stems) // num_val
        return stems[::stride][:num_val]

    @staticmethod
    def _load_manifest(out_dir):
        with open(os.path.join(out_dir, "manifest.json")) as f:
            return json.load(f)
