"""Emmi-Wing: full_dataset/part_XX/run_<id>/*.pt; U_inf from train.csv, p_inf = 1e5 Pa, T = 298 K.
surface -> [N, 8]  x y z cp cf_x cf_y cf_z rho_tilde
volume  -> [n, 11] x y z ux uy uz cp rho_tilde wx wy wz   (u / U_inf, w * chord / U_inf)
"""
import os
import re

import numpy as np

from .utils import cli
from .utils.geometry import check_freestream_velocity, check_keep, in_box, slab_box, surface_bbox
from .utils.io import (load_json, n_rows, n_rows_from_size, read_split, save_npy, write_manifest,
                       write_splits)
from .utils.pool import run_pool
from .utils.prune import run_prune
from .utils.shuffle import shuffle_rows
from .utils.splits import sticky_split
from .utils.stats import pooled_sums, save_surface_stats, save_volume_stats, serial_sums

NAME = "emmi_wing"
SEED = 0
VAL_FRAC = 0.10
_RUN_RE = re.compile(r"run_(\d+)$")

P_INF = 100000.0
T_INF = 298.0
R_AIR = 8314.5 / 28.9               # OpenFOAM's default molWeight 28.9
RHO_INF = P_INF / (R_AIR * T_INF)   # 1.16639 kg/m^3, constant across the campaign
COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z", "rho_tilde"]
DESIGN_COLS = ["chord_root", "span", "taper_ratio", "sweep", "dihedral",
               "inflow_velocity", "AOA", "Reynolds_number", "Mach_number"]
COND_COLS = DESIGN_COLS + ["rho_inf", "q"]
STATS_MAX_POINTS = 50_000   # leading rows per sample pooled into the stats (rows are shuffled)

VOLUME_FILES = ["volume_position.pt", "volume_pressure.pt", "volume_rho.pt",
                "volume_velocity.pt", "volume_vorticity.pt"]
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "cp", "rho_tilde", "wx", "wy", "wz"]
# x/y margins on the surface bbox; z = centre +/- 0.35 * max(Lx, Ly) (the wing is ~0.05 m thick).
CROP_NEG = [0.50, 0.10, None]
CROP_POS = [1.50, 0.35, None]
Z_HALF_FRAC = 0.35
KEEP_FRAC_MIN = 0.90      # measured 0.960-0.996
FREESTREAM_TOL = 1e-3     # inlet velocity, measured ~1e-4
P_INF_TOL = 5e-3          # |inlet p - P_INF| / q, measured ~4e-5
INLET_X = -10.0
INLET_MIN_POINTS = 500
DOMAIN_XMIN_MAX = -17.0


def _stem(case_id):
    return f"run_{case_id}"


def _cid(stem):
    return int(_RUN_RE.search(stem).group(1))


def freestream(u_inf):
    """(rho_inf, q) for one run; only U_inf varies across the campaign."""
    return RHO_INF, 0.5 * RHO_INF * u_inf ** 2


def _load(run_dir, name):
    import torch
    return torch.load(os.path.join(run_dir, name), map_location="cpu").numpy()


def run_dirs(root):
    """{case_id: run dir} over full_dataset/part_*/run_<id>/ (parts arrive and leave independently)."""
    found = {}
    raw = os.path.join(root, "full_dataset")
    for part in sorted(os.listdir(raw)) if os.path.isdir(raw) else []:
        pdir = os.path.join(raw, part)
        if not (part.startswith("part_") and os.path.isdir(pdir)):
            continue
        for name in sorted(os.listdir(pdir)):
            m = _RUN_RE.match(name)
            if m and os.path.isdir(os.path.join(pdir, name)):
                found[int(m.group(1))] = os.path.join(pdir, name)
    return found


def load_meta(root):
    import pandas as pd
    return pd.read_csv(os.path.join(root, "metadata", "train.csv")).set_index("case_id")


def list_runs(root, meta):
    """Case ids with a run dir and a train.csv row, minus metadata/erroneous_cases.npy."""
    path = os.path.join(root, "metadata", "erroneous_cases.npy")
    bad = set(int(x) for x in np.load(path, allow_pickle=True).tolist()) if os.path.exists(path) else set()
    have = set(int(i) for i in meta.index)
    dirs = run_dirs(root)
    return sorted((c, dirs[c]) for c in dirs if c not in bad and c in have)


def process_surface(case_id, run_dir, u_inf, samples_dir, force=False):
    stem = _stem(case_id)
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem
    pos = np.ascontiguousarray(_load(run_dir, "surface_position.pt"), dtype=np.float32)
    p = np.ascontiguousarray(_load(run_dir, "surface_pressure.pt"), dtype=np.float32).reshape(-1)
    rho = np.ascontiguousarray(_load(run_dir, "surface_rho.pt"), dtype=np.float32).reshape(-1)
    tau = np.ascontiguousarray(_load(run_dir, "surface_wall_shear_stress.pt"), dtype=np.float32)
    rho_inf, q = freestream(u_inf)
    out = np.concatenate([pos, ((p - P_INF) / q).reshape(-1, 1), (tau / q).astype(np.float32),
                          (rho / rho_inf).reshape(-1, 1)], axis=1).astype(np.float32)
    assert out.shape[1] == len(COLS), f"{stem}: bad shape {out.shape}"
    save_npy(out_path, shuffle_rows(out, SEED + case_id))
    return stem


def rows_from_disk(samples_dir, n_cols):
    """{stem, case_id, n_points} for every sample, n from the file size (no per-file opens)."""
    if not os.path.isdir(samples_dir):
        return []
    with os.scandir(samples_dir) as it:
        found = [(e.name[:-4], e.stat().st_size) for e in it
                 if e.name.endswith(".npy") and e.is_file()]
    rows = []
    for stem, size in sorted(found):
        if _RUN_RE.match(stem):
            n = n_rows_from_size(size, n_cols)
            rows.append(dict(stem=stem, case_id=_cid(stem),
                             n_points=n if n is not None else n_rows(
                                 os.path.join(samples_dir, stem + ".npy"))))
    return rows


def surface(args):
    out = args.out or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    meta = load_meta(args.root)
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    else:
        runs = list_runs(args.root, meta)[:args.limit]
        jobs = [(c, d, float(meta.at[c, "inflow_velocity"]), samples, args.force) for c, d in runs]
        _, errors = run_pool(process_surface, jobs, args.workers, every=250)
        rows = write_manifest(out, rows_from_disk(samples, len(COLS)), errors, key="case_id")
    train, val = sticky_split(sorted(r["stem"] for r in rows), os.path.join(out, "splits"),
                              VAL_FRAC, SEED)
    write_splits(os.path.join(out, "splits"), train=train, val=val)

    cids = [r["case_id"] for r in rows]
    design = meta.loc[cids, DESIGN_COLS].to_numpy(dtype=np.float64)
    u_inf = meta.loc[cids, "inflow_velocity"].to_numpy(dtype=np.float64)
    rho_inf = np.full_like(u_inf, RHO_INF)
    cond = np.concatenate([design, rho_inf[:, None], (0.5 * rho_inf * u_inf ** 2)[:, None]],
                          axis=1).astype(np.float32)
    assert cond.shape[1] == len(COND_COLS) and not np.isnan(cond).any()
    np.savez(os.path.join(out, "metadata.npz"), stems=np.array([r["stem"] for r in rows]),
             cond=cond, cond_cols=np.array(COND_COLS))
    if train and all(os.path.exists(os.path.join(samples, s + ".npy")) for s in train):
        save_surface_stats(os.path.join(out, "norm_stats.npz"),
                           *serial_sums([os.path.join(samples, s + ".npy") for s in train],
                                        max_points=args.stats_max_points), COLS)


def process_volume(stem, run_dir, params, samples_dir, surface_samples, force=False,
                   delete_src=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem
    lo, hi = slab_box(*surface_bbox(surface_samples, stem), CROP_NEG, CROP_POS, Z_HALF_FRAC,
                      vert=2, ref_axes=(0, 1))
    u_inf, aoa, chord = params
    rho_inf, q = freestream(u_inf)
    pts = np.ascontiguousarray(_load(run_dir, "volume_position.pt"), dtype=np.float32)
    vel = np.ascontiguousarray(_load(run_dir, "volume_velocity.pt"), dtype=np.float32).reshape(-1, 3)
    vort = np.ascontiguousarray(_load(run_dir, "volume_vorticity.pt"), dtype=np.float32).reshape(-1, 3)
    p = np.asarray(_load(run_dir, "volume_pressure.pt"), dtype=np.float32).reshape(-1, 1)
    rho = np.asarray(_load(run_dir, "volume_rho.pt"), dtype=np.float32).reshape(-1, 1)

    # Inlet-face gates before the crop: velocity pins frame + AoA, pressure pins P_INF.
    assert float(pts[:, 0].min()) < DOMAIN_XMIN_MAX, f"{stem}: raw domain is not the full cube"
    inlet = pts[:, 0] < INLET_X
    assert int(inlet.sum()) >= INLET_MIN_POINTS, f"{stem}: too few inlet points"
    rel_u = check_freestream_velocity(vel[inlet].mean(0), u_inf, aoa, FREESTREAM_TOL, stem, "inlet")
    rel_p = float(abs(float(p[inlet].mean()) - P_INF) / q)
    assert rel_p < P_INF_TOL, f"{stem}: inlet Cp {rel_p:.2e}; wrong reference pressure?"

    n_raw = pts.shape[0]
    mask = in_box(pts, lo, hi)
    out = np.concatenate([
        pts[mask],
        vel[mask] / np.float32(u_inf),
        (p[mask] - np.float32(P_INF)) / np.float32(q),
        rho[mask] / np.float32(rho_inf),
        vort[mask] * np.float32(chord / u_inf),
    ], axis=1)
    del pts, vel, vort, p, rho, mask
    keep = check_keep(stem, out.shape[0], n_raw, KEEP_FRAC_MIN, lo, hi)
    assert out.shape[1] == len(VOL_COLS) and np.isfinite(out).all(), f"{stem}: bad output"
    out = shuffle_rows(out, SEED + _cid(stem))
    save_npy(out_path, out)
    if delete_src:   # only the volume tensors; the surface ones stay
        for name in VOLUME_FILES:
            try:
                os.remove(os.path.join(run_dir, name))
            except FileNotFoundError:
                pass
    print(f"  {stem}: {out.shape[0]:,}/{n_raw:,} kept ({100 * keep:.2f}%), inlet rel u "
          f"{rel_u:.1e} p {rel_p:.1e}", flush=True)
    return stem


def volume(args):
    out = args.out or os.path.join(args.root, "volume_collated")
    surf = args.surface or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    surf_samples = os.path.join(surf, "samples")
    os.makedirs(samples, exist_ok=True)
    if args.only or not (args.finalize or args.stats_only):
        meta = load_meta(args.root)
        runs = ([(_cid(args.only), run_dirs(args.root).get(_cid(args.only)))] if args.only
                else list_runs(args.root, meta))
        runs = [(c, d) for c, d in runs if d is not None
                and os.path.exists(os.path.join(d, "volume_position.pt"))
                and os.path.exists(os.path.join(surf_samples, _stem(c) + ".npy"))]
        jobs = [(_stem(c), d, (float(meta.at[c, "inflow_velocity"]), float(meta.at[c, "AOA"]),
                               float(meta.at[c, "chord_root"])), samples, surf_samples,
                 args.force, args.delete_src) for c, d in runs[:args.limit]]
        if args.only:
            print(f"OK {process_volume(*jobs[0])}" if jobs else
                  f"SKIP {args.only}: no volume tensors or no paired surface sample")
            return
        _, errors = run_pool(process_volume, jobs, args.workers)
        write_manifest(out, rows_from_disk(samples, len(VOL_COLS)), errors, key="case_id")
    elif args.finalize:
        write_manifest(out, rows_from_disk(samples, len(VOL_COLS)), key="case_id")

    train = read_split(os.path.join(surf, "splits"), "train")
    have = [s for s in train if os.path.exists(os.path.join(samples, s + ".npy"))]
    if len(have) / max(1, len(train)) < args.stats_coverage or not have:
        print(f"skipping norm_stats_volume ({len(have)}/{len(train)} train samples on disk)")
        return
    cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + _cid(t)) for t in have],
                             args.stats_points, slice(3, 11), args.stats_workers, every=500)
    save_volume_stats(os.path.join(out, "norm_stats_volume.npz"), cnt, s, ss, VOL_COLS[3:11],
                      extra=True, points_per_sample=args.stats_points)


def _bbox_shift(before, after, keep, _):
    """Shift of the position bbox (fraction of L) -- the volume crop is sized from it."""
    if keep is None or keep.all() or not after.shape[0]:
        return dict(bbox_shift=0.0)
    pos = before[:, :3].astype(np.float64)
    lo, hi = pos.min(0), pos.max(0)
    L = np.where(hi - lo > 0, hi - lo, 1.0)
    kept = pos[keep]
    return dict(bbox_shift=float(np.abs(np.concatenate(
        [kept.min(0) - lo, kept.max(0) - hi]) / np.concatenate([L, L])).max()))


def _shift_summary(res, _):
    print(f"bbox shift (fraction of L; sizes the volume crop): "
          f"max={max([r['bbox_shift'] for r in res] or [0.0]):.2e}")


def prune(args):
    out = args.out or os.path.join(args.root, "collated")
    run_prune(out, COLS, list(dict.fromkeys(args.columns)), args.sigma, workers=args.workers,
              dry_run=args.dry_run, force=args.force, limit=args.limit,
              stats_max_points=args.stats_max_points, extra_fn=_bbox_shift,
              summary_fn=_shift_summary, log_fields=("bbox_shift",))


def _surface_args(p):
    cli.add_build_args(p, 32)
    p.add_argument("--splits-only", action="store_true",
                   help="skip conversion; rebuild splits + metadata + stats from the manifest")
    p.add_argument("--stats-max-points", type=int, default=STATS_MAX_POINTS,
                   help="leading rows per sample for norm_stats.npz (0 = all)")


def _volume_args(p):
    cli.add_volume_args(p, 24)
    p.add_argument("--only", default=None, help="process one stem (e.g. run_1) and exit")
    p.add_argument("--finalize", action="store_true", help="manifest + stats from samples on disk")
    p.add_argument("--delete-src", action="store_true",
                   help="unlink each run's volume_*.pt once its sample is written")
    p.add_argument("--stats-points", type=int, default=STATS_MAX_POINTS)
    p.add_argument("--stats-workers", type=int, default=8)
    p.add_argument("--stats-coverage", type=float, default=0.0,
                   help="fraction of the train split that must be on disk before stats run")


def _prune_args(p):
    cli.add_prune_args(p, ["cf_x", "cf_y", "cf_z"], ["cp", "cf_x", "cf_y", "cf_z", "rho_tilde"], 3.0)
    p.add_argument("--stats-max-points", type=int, default=STATS_MAX_POINTS)


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from surface_*.pt"),
        "volume": (_volume_args, volume, "build volume_collated/ from volume_*.pt"),
        "prune": (_prune_args, prune, "per-sample sigma cut of the surface Cf tail, in place "
                                      "(collated/ on disk is NOT pruned)"),
    })
