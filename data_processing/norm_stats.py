"""Loader normalization stats for the collated trees under --root ($CFD_DATA_ROOT):
base     norm_stats_centered.npz           mesh-measure crops, loader frame, + isotropic pos_scale
surface  norm_stats_centered_thinned.npz   surface_thin crops (pos_scale reused)
volume   norm_stats_volume_thinned.npz     volume_thin crops
twin     *_twin<K>.npz                     closed-form pooled stats for a K-speed twin campaign
"""
import argparse
import json
import os
import time

import numpy as np

from .utils.stats import moments

SURFACE_DATASETS = ["ahmedml", "drivaerml", "drivaernet", "windsorml", "superwing",
                    "blendednet", "emmi_wing", "hiliftaeroml", "submarine", "shift_suv",
                    "shift_pump", "double_delta", "shift_cca"]
VOLUME_DATASETS = ["ahmedml", "drivaerml", "drivaernet", "windsorml", "superwing", "emmi_wing",
                   "hiliftaeroml", "submarine", "double_delta", "shift_cca"]
VARS = ["pos_x", "pos_y", "pos_z", "cf_x", "cf_y", "cf_z", "cp"]
# base: trees read as contiguous windows (rows shuffled on disk); the rest scatter-gather.
BASE_PRESHUFFLED = {"hiliftaeroml", "double_delta", "shift_cca"}
# surface / volume thinning, matching the configs (n_surface 32768, n_volume 64000).
N_SURFACE, N_VOLUME, OVERSAMPLE, VOXEL, MIN_COUNT, ALPHA = 32768, 64000, 8, 0.01, 8, 1.0
MAX_SAMPLES = 600
SEED = 1234


def _tree(args, campaign, tree="collated"):
    return os.path.join(args.root, campaign, tree)


def _out_path(args, campaign, tree, name):
    """Where an output goes: in place, or under --out-dir/<campaign>/<tree>/."""
    d = os.path.join(args.out_dir, campaign, tree) if args.out_dir else _tree(args, campaign, tree)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def _split_stems(surf_dir, split, have_dir):
    with open(os.path.join(surf_dir, "splits", f"{split}.json")) as f:
        stems = json.load(f)
    with open(os.path.join(have_dir, "manifest.json")) as f:
        rows = {r["stem"]: r for r in json.load(f)}
    return [s for s in stems
            if s in rows and os.path.exists(os.path.join(have_dir, "samples", s + ".npy"))], rows


def pick(stems, max_samples):
    """Evenly strided subset (spans every geometry family; reproducible without a seed)."""
    if max_samples is None or max_samples < 0 or len(stems) <= max_samples:
        return stems
    idx = np.linspace(0, len(stems) - 1, max_samples).round().astype(int)
    return [stems[i] for i in np.unique(idx)]


def to_coefficients(data, campaign, oriented):
    """(pos, cf, cp) as coefficients; reoriented to the loader frame only when `oriented`."""
    import torch
    from data.surface_volume_dataset import CF_SIGN, FREESTREAM, ORIENT, P, POS, WSS
    fs = FREESTREAM[campaign]
    pos, cp, cf = data[:, POS], data[:, P], data[:, WSS]
    if not fs.get("precomputed_coef", False):
        q = 0.5 * fs["U_inf"] ** 2          # kinematic fields, no rho
        cp = (cp - fs["p_inf"]) / q
        cf = cf / q
    if not oriented:
        return pos, cf, cp
    M = ORIENT.get(campaign)
    if M is not None:
        M = M.numpy() if isinstance(M, torch.Tensor) else np.asarray(M)
        pos, cf = pos @ M, cf @ M
    return pos, cf * CF_SIGN.get(campaign, 1.0), cp


# --- base: mesh-measure crops ----------------------------------------------------------

class _BaseCrops:
    """One random <=n_points crop per sample, in the loader frame (torch Dataset protocol)."""

    def __init__(self, surf_dir, stems, lengths, campaign, n_points, preshuffled, seed):
        self.surf_dir, self.stems, self.lengths = surf_dir, stems, lengths
        self.campaign, self.n_points, self.preshuffled, self.seed = campaign, n_points, preshuffled, seed

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        import torch
        arr = np.load(os.path.join(self.surf_dir, "samples", self.stems[idx] + ".npy"), mmap_mode="r")
        n = self.lengths[idx]
        rng = np.random.default_rng((self.seed, idx))
        k = min(self.n_points, n)
        if self.preshuffled and k < n:
            start = int(rng.integers(0, max(1, n - k + 1)))
            data = np.array(arr[start:start + k], dtype=np.float32)
        else:
            data = (np.array(arr, dtype=np.float32) if k == n else
                    np.asarray(arr[np.sort(rng.choice(n, size=k, replace=False))], dtype=np.float32))
            data = data[rng.permutation(k)]
        pos, cf, cp = to_coefficients(data, self.campaign, oriented=True)
        return torch.from_numpy(np.ascontiguousarray(pos)), torch.from_numpy(
            np.ascontiguousarray(cf)), torch.from_numpy(np.ascontiguousarray(cp))


def _base_collate(batch):
    """Per sample: bbox diagonal of the raw positions, then centered [pos, cf, cp] rows."""
    import torch
    rows, diags = [], []
    for pos, cf, cp in batch:
        diags.append(torch.linalg.norm(pos.amax(dim=0) - pos.amin(dim=0)))
        rows.append(torch.cat([pos - pos.mean(dim=0, keepdim=True), cf, cp], dim=1))
    return torch.cat(rows, dim=0), torch.stack(diags)


def run_base(campaign, args):
    import torch
    from torch.utils.data import DataLoader
    d = _tree(args, campaign)
    stems, rows = _split_stems(d, args.split, d)
    pre = (campaign in BASE_PRESHUFFLED) if args.preshuffled is None else args.preshuffled
    ds = _BaseCrops(d, stems, [rows[s]["n_points"] for s in stems], campaign, args.n_points,
                    pre, args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=_base_collate,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    print(f"\n=== {campaign} ({args.split}): {len(stems)} samples, preshuffled={pre}")
    count, total, total_sq, diag_sum, n_samples = 0, np.zeros(7), np.zeros(7), 0.0, 0
    for i, (x, diags) in enumerate(loader):
        x = x.to(torch.float64)
        count += x.shape[0]
        total += x.sum(dim=0).numpy()
        total_sq += (x * x).sum(dim=0).numpy()
        diag_sum += diags.to(torch.float64).sum().item()
        n_samples += diags.shape[0]
        print(f"  batch {i + 1}/{len(loader)}  pooled nodes: {count:,}", end="\r")
    print()
    mean = total / count
    std = np.sqrt(np.clip(total_sq / count - mean ** 2, 0.0, None))
    mean_diag = diag_sum / n_samples
    pos_scale = 2.0 * np.sqrt(3.0) / mean_diag   # mean sample spans the 2x2x2 cube
    for v, m, s in zip(VARS, mean, std):
        print(f"    {v:>6s}  mean={m: .6e}  std={s: .6e}")
    print(f"  mean bbox diagonal {mean_diag:.6f}, pos_scale {pos_scale:.6f}")
    path = _out_path(args, campaign, "collated", args.out_name)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32), vars=np.array(VARS),
             n_points=count, centered_pos=True, mean_bbox_diag=np.float32(mean_diag),
             pos_scale=np.float32(pos_scale), oriented_cf=True)
    print(f"  wrote {path}")


# --- surface: thinned crops ------------------------------------------------------------

def read_window(arr, n_want, rng, preshuffled):
    n = arr.shape[0]
    k = min(n_want, n)
    if k == n:
        return np.array(arr, dtype=np.float32)
    if preshuffled:
        start = int(rng.integers(0, max(1, n - k + 1)))
        return np.array(arr[start:start + k], dtype=np.float32)
    return np.array(arr[np.sort(rng.choice(n, k, replace=False))], dtype=np.float32)


def run_surface(campaign, args):
    from data.surface_volume_dataset import NOT_PRESHUFFLED, POS, surface_thin_indices
    d = _tree(args, campaign)
    if not os.path.isdir(os.path.join(d, "samples")):
        print(f"{campaign}: no surface tree, skipped")
        return
    old = np.load(os.path.join(d, "norm_stats_centered.npz"))   # its frame is reused, not refit
    ps, diag = float(old["pos_scale"]), float(old["mean_bbox_diag"])
    oriented = bool(old["oriented_cf"]) if "oriented_cf" in old.files else False
    pre = (campaign not in NOT_PRESHUFFLED) if args.preshuffled is None else args.preshuffled
    stems = pick(_split_stems(d, args.split, d)[0], args.max_samples)
    print(f"\n=== {campaign}: {len(stems)} runs, pos_scale={ps:.6g}, preshuffled={pre}, "
          f"oriented_cf={oriented}")
    cnt = m_cnt = 0
    s, ss, m_s, m_ss = (np.zeros(len(VARS)) for _ in range(4))
    per_run, diags = [], []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        a = np.load(os.path.join(d, "samples", stem + ".npy"), mmap_mode="r")
        rng = np.random.default_rng(SEED + (i * 7919))
        w = read_window(a, args.n_surface * args.oversample, rng, pre)

        def rows(x):
            pos, cf, cp = to_coefficients(x, campaign, oriented)
            return np.concatenate([pos - pos.mean(0, keepdims=True), cf, cp], axis=1).astype(np.float64)
        mesh = rows(w)
        m_s += mesh.sum(0)
        m_ss += (mesh * mesh).sum(0)
        m_cnt += mesh.shape[0]
        del mesh
        # Thinning on raw positions with voxel / pos_scale == thinning framed ones.
        keep = surface_thin_indices(w[:, POS], args.n_surface, rng, voxel=args.voxel / ps,
                                    alpha=args.alpha, min_count=args.min_count)
        v = rows(w[keep])
        s += v.sum(0)
        ss += (v * v).sum(0)
        cnt += v.shape[0]
        per_run.append(v.mean(0))
        diags.append(float(np.linalg.norm(v[:, 0:3].max(0) - v[:, 0:3].min(0))))
        if i % 100 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)} runs ({(time.time() - t0) / i * 1000:.0f} ms/run)", flush=True)
    mean, std = moments(cnt, s, ss)
    m_mean, m_std = moments(m_cnt, m_s, m_ss)
    for j, v in enumerate(VARS):
        print(f"  {v:6s} builder {float(old['mean'][j]):11.6f} / {float(old['std'][j]):10.6f}"
              f"   thinned {mean[j]:11.6f} / {std[j]:10.6f}")
    thin_diag = float(np.mean(diags))
    print(f"  bbox diagonal: stored {diag:.6f}, thinned {thin_diag:.6f} (stored kept)")
    if args.dry_run:
        return
    path = _out_path(args, campaign, "collated", args.out_name)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             mesh_mean=m_mean.astype(np.float32), mesh_std=m_std.astype(np.float32),
             vars=np.array(VARS), n_points=np.int64(cnt), n_samples=np.int64(len(stems)),
             points_per_sample=np.int64(args.n_surface), centered_pos=True,
             mean_bbox_diag=np.float32(diag), pos_scale=np.float32(ps),
             thinned_bbox_diag=np.float32(thin_diag), oriented_cf=oriented,
             thin_n_surface=np.int64(args.n_surface), thin_oversample=np.int64(args.oversample),
             thin_voxel=np.float32(args.voxel), thin_min_count=np.int64(args.min_count),
             thin_alpha=np.float32(args.alpha), split=np.array(args.split))
    print(f"  wrote {path}")


# --- volume: thinned crops -------------------------------------------------------------

def run_volume(campaign, args):
    from data.surface_volume_dataset import voxel_thin_indices
    vol_dir = _tree(args, campaign, args.vol_name)
    if not os.path.isdir(os.path.join(vol_dir, "samples")):
        print(f"{campaign}: no volume tree, skipped")
        return
    ps = float(np.load(os.path.join(_tree(args, campaign), "norm_stats_centered.npz"))["pos_scale"])
    stems = pick(_split_stems(_tree(args, campaign), args.split, vol_dir)[0], args.max_samples)
    print(f"\n=== {campaign}: {len(stems)} runs, pos_scale={ps:.6g}")
    cnt = m_cnt = 0
    s = ss = m_s = m_ss = None
    per_run = []
    for i, stem in enumerate(stems, 1):
        a = np.load(os.path.join(vol_dir, "samples", stem + ".npy"), mmap_mode="r")
        rng = np.random.default_rng(SEED + (i * 7919))
        if a.ndim == 3:                       # rounds tree: one round per run
            a = a[int(rng.integers(a.shape[0]))]
        n = a.shape[0]
        k = min(args.n_volume * args.oversample, n)
        start = int(rng.integers(0, max(1, n - k + 1)))
        w = np.array(a[start:start + k], dtype=np.float32)
        if s is None:
            s, ss, m_s, m_ss = (np.zeros(w.shape[1] - 3) for _ in range(4))
        vals = w[:, 3:].astype(np.float64)
        m_s += vals.sum(0)
        m_ss += (vals * vals).sum(0)
        m_cnt += vals.shape[0]
        keep = voxel_thin_indices(w[:, 0:3], args.n_volume, rng, voxel=args.voxel / ps,
                                  alpha=args.alpha, min_count=args.min_count)
        v = vals[keep]
        s += v.sum(0)
        ss += (v * v).sum(0)
        cnt += v.shape[0]
        per_run.append(v.mean(0))
        if i % 100 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)} runs, {cnt / 1e6:.1f}M rows pooled", flush=True)
    mean, std = moments(cnt, s, ss)
    m_mean, m_std = moments(m_cnt, m_s, m_ss)
    old_path = os.path.join(vol_dir, "norm_stats_volume.npz")
    old = np.load(old_path) if os.path.exists(old_path) else None
    for j, lab in enumerate(["u_x", "u_y", "u_z", "p"]):
        om = float(old["mean"][j]) if old is not None else float("nan")
        os_ = float(old["std"][j]) if old is not None else float("nan")
        print(f"  {lab:4s} builder {om:12.4f} / {os_:10.4f}   thinned {mean[j]:12.4f} / {std[j]:10.4f}")
    if args.dry_run:
        return
    out = dict(mean=mean[:4].astype(np.float32), std=std[:4].astype(np.float32),
               mesh_mean=m_mean[:4].astype(np.float32), mesh_std=m_std[:4].astype(np.float32),
               extra_mean=mean[4:].astype(np.float32), extra_std=std[4:].astype(np.float32),
               n_points=np.int64(cnt), n_samples=np.int64(len(stems)),
               points_per_sample=np.int64(args.n_volume),
               thin_n_volume=np.int64(args.n_volume), thin_oversample=np.int64(args.oversample),
               thin_voxel=np.float32(args.voxel), thin_min_count=np.int64(args.min_count),
               thin_alpha=np.float32(args.alpha), split=np.array(args.split))
    if old is not None:
        for key in ("vars", "extra_vars"):
            if key in old.files:
                out[key] = old[key]
    path = _out_path(args, campaign, args.vol_name, args.out_name)
    np.savez(path, **out)
    print(f"  wrote {path}")


# --- twin: closed-form pooled stats ----------------------------------------------------

def pool(m0, s0, g):
    """(mean, std) of K equally weighted copies of a field scaled by gains g_k."""
    m0, s0 = np.asarray(m0, dtype=np.float64), np.asarray(s0, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).reshape(-1, *([1] * m0.ndim))
    mu = g.mean(0) * m0
    var = (g ** 2).mean(0) * (s0 ** 2 + m0 ** 2) - mu ** 2
    if np.any(var <= 0):
        raise ValueError(f"pooled variance is not positive: {var}")
    return mu, np.sqrt(var)


def run_twin(campaign, args):
    """Reynolds-similar speeds U_k = s_k U_0: p/tau/volume p gain s^2, velocity s, positions 1."""
    u0 = args.u0
    if u0 is None:
        from data.surface_volume_dataset import FREESTREAM
        fs = FREESTREAM[campaign]
        if fs.get("precomputed_coef", False):
            raise ValueError(f"{campaign} has no U_inf (precomputed coefficients); pass --u0")
        u0 = float(fs["U_inf"])
    s = np.asarray(args.speeds, dtype=np.float64) / u0
    suffix = args.suffix or f"twin{len(s)}"
    print(f"twin stats for {campaign}: U0={u0}, s={np.round(s, 6).tolist()}")
    for tree, name, gains in [("collated", args.surf_stats, [(slice(3, 7), s ** 2)]),
                              ("volume_collated", args.vol_stats,
                               [(slice(0, 3), s), (slice(3, 4), s ** 2)])]:
        z = np.load(os.path.join(_tree(args, campaign, tree), name))
        out = {k: z[k] for k in z.files}
        mean, std = z["mean"].astype(np.float64).copy(), z["std"].astype(np.float64).copy()
        for sl, g in gains:
            mean[sl], std[sl] = pool(z["mean"][sl], z["std"][sl], g)
        out.update(mean=mean.astype(np.float32), std=std.astype(np.float32),
                   twin_speeds=np.asarray(args.speeds, dtype=np.float32),
                   twin_u0=np.float32(u0), twin_source=np.asarray(name))
        stem, ext = os.path.splitext(name)
        path = _out_path(args, campaign, tree, f"{stem}_{suffix}{ext}")
        print(f"  {name} -> {path}\n    mean {np.round(mean, 6).tolist()}\n    std  {np.round(std, 6).tolist()}")
        if not args.dry_run:
            np.savez(path, **out)


def _common(p, datasets):
    p.add_argument("--root", default=os.environ.get("CFD_DATA_ROOT"),
                   help="data root holding <dataset>/collated (default $CFD_DATA_ROOT)")
    p.add_argument("--out-dir", default=None,
                   help="write outputs under <out-dir>/<dataset>/<tree>/ instead of in place")
    p.add_argument("--datasets", nargs="+", default=datasets)
    p.add_argument("--split", default="train")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m data_processing.norm_stats")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("base", help="norm_stats_centered.npz over mesh-measure crops")
    _common(p, SURFACE_DATASETS)
    p.add_argument("--n-points", type=int, default=100_000, help="max rows per sample")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="crop RNG seed")
    p.add_argument("--out-name", default="norm_stats_centered.npz")
    p.set_defaults(fn=run_base)

    p = sub.add_parser("surface", help="norm_stats_centered_thinned.npz over surface_thin crops")
    _common(p, SURFACE_DATASETS)
    p.add_argument("--n-surface", type=int, default=N_SURFACE)
    p.add_argument("--out-name", default="norm_stats_centered_thinned.npz")
    p.set_defaults(fn=run_surface)

    p = sub.add_parser("volume", help="norm_stats_volume_thinned.npz over volume_thin crops")
    _common(p, VOLUME_DATASETS)
    p.add_argument("--vol-name", default="volume_collated", help="volume tree under <dataset>/")
    p.add_argument("--n-volume", type=int, default=N_VOLUME)
    p.add_argument("--out-name", default="norm_stats_volume_thinned.npz")
    p.set_defaults(fn=run_volume)

    for name in ("surface", "volume"):
        p = sub.choices[name]
        p.add_argument("--oversample", type=int, default=OVERSAMPLE)
        p.add_argument("--voxel", type=float, default=VOXEL)
        p.add_argument("--min-count", type=int, default=MIN_COUNT)
        p.add_argument("--alpha", type=float, default=ALPHA)
        p.add_argument("--max-samples", type=int, default=MAX_SAMPLES, help="-1 = whole split")
        p.add_argument("--dry-run", action="store_true")
    for name in ("base", "surface"):
        p = sub.choices[name]
        p.add_argument("--preshuffled", dest="preshuffled", action="store_true", default=None,
                       help="force contiguous-window reads")
        p.add_argument("--no-preshuffled", dest="preshuffled", action="store_false",
                       help="force scatter-gather reads")

    p = sub.add_parser("twin", help="closed-form pooled stats for a twin-speed campaign")
    _common(p, ["drivaernet"])
    p.add_argument("--speeds", type=float, nargs="+", required=True, help="freestream speeds (m/s)")
    p.add_argument("--u0", type=float, default=None, help="base speed (default FREESTREAM U_inf)")
    p.add_argument("--suffix", default=None, help="output suffix (default twin<K>)")
    p.add_argument("--surf-stats", default="norm_stats_centered_thinned.npz")
    p.add_argument("--vol-stats", default="norm_stats_volume_thinned.npz")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=run_twin)

    args = ap.parse_args(argv)
    if not args.root:
        ap.error("pass --root or set $CFD_DATA_ROOT")
    for campaign in args.datasets:
        args.fn(campaign, args)


if __name__ == "__main__":
    main()
