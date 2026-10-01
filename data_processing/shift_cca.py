"""SHIFT-CCA (M 0.72, AoA 2 deg -- unstated upstream, asserted per sample): sample_<NNNNNN>/.
surface -> [n_cells, 11]  x y z cp cf_x cf_y cf_z n_x n_y n_z area
volume  -> [n_points, 7]  x y z ux uy uz cp
prune   -> 4-sigma per-sample cut on cf_x/cf_y/cf_z (what collated/ holds)
"""
import os
import re

import numpy as np
import pyvista as pv

from .utils import cli
from .utils.geometry import (block_force, check_freestream_velocity, check_keep, in_box,
                             lift_drag, load_forces, slab_box, surface_bbox)
from .utils.io import (all_on_disk, load_json, merge_manifest, n_rows, read_split, rows_from_disk,
                       save_npy, write_manifest, write_splits)
from .utils.pool import run_pool
from .utils.prune import run_prune
from .utils.shuffle import shuffle_rows
from .utils.splits import random_split
from .utils.stats import pooled_sums, save_surface_stats, save_volume_stats, serial_sums
from .utils.vtk import find_field, f32

NAME = "shift_cca"
SEED = 0
VAL_FRAC = 0.20
_RUN_RE = re.compile(r"sample_(\d+)$")
N_DESIGN_POINTS = 100
MISSING_IDS = {65}   # absent upstream, not a partial download

P_PREFIX = "Pressure"
WSS_PREFIX = "Wall Shear Stress"
NORMALS_FIELD = "Normals"
COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z", "n_x", "n_y", "n_z", "area"]
AOA_DEG = 2.0
FORCE_TOL = 5e-3     # integral vs forces.json, measured 6e-5
CLOSURE_TOL = 1e-6   # |sum n dA| / sum dA, measured 2.3e-11
NORM_TOL = 1e-5      # stored normals unit length, measured 5e-8

COND_COLS = ["aoa", "mach"]
GEOM_COLS = ["c_root", "panel_break", "panel_1_le", "panel_2_le", "panel_1_te",
             "panel_2_te", "wing_tip_close_angle", "reference_area", "design_point"]
# Suffixed names: the loader flattens all metadata blocks into one namespace.
DERIVED_COLS = ["q_inf", "rho_inf", "u_inf", "p_inf", "mach_inf", "area_ref", "length_ref"]

VEL_FIELD = "Velocity (m/s)"
VOL_P_FIELD = "Pressure (Pa)"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "cp"]
# x/y margins on the surface bbox; z = centre +/- 0.20 * max(Lx, Ly) (the wing is thin).
CROP_NEG = [0.25, 0.35, None]
CROP_POS = [1.00, 0.35, None]
Z_HALF_FRAC = 0.20
KEEP_FRAC_MIN = 0.65    # measured 0.770-0.776: ~20% of points sit in the +/-1000 m far field
FREESTREAM_TOL = 1e-3   # inlet-face velocity, measured 3.8e-4
INLET_X = -100.0
INLET_MIN_POINTS = 500
DOMAIN_XMIN_MAX = -900.0


def _run_id(stem):
    return int(_RUN_RE.match(stem).group(1))


def list_runs(root, fname="merged_surfaces.vtp"):
    runs = []
    for d in sorted(os.listdir(root)):
        if _RUN_RE.match(d) and os.path.exists(os.path.join(root, d, fname)):
            runs.append((d, os.path.join(root, d, fname)))
    return sorted(runs, key=lambda r: _run_id(r[0]))


def read_params(root, stem):
    """Freestream + reference state (DERIVED_COLS), cross-checked two ways."""
    raw = load_json(os.path.join(root, stem, "params.json"))
    rho, u, p = float(raw["air_density"]), float(raw["stream_velocity"]), float(raw["pressure"])
    t, mach = float(raw["temperature"]), float(raw["mach"])
    gamma, r_gas = float(raw["gamma"]), float(raw["R"])
    q = 0.5 * rho * u ** 2
    q_gas = 0.5 * gamma * p * mach ** 2
    assert abs(q - q_gas) / q < 1e-6, f"{stem}: q(rho,U)={q:.4f} vs 0.5 gamma p M^2={q_gas:.4f}"
    mach_gas = u / np.sqrt(gamma * r_gas * t)
    assert abs(mach - mach_gas) < 1e-3, f"{stem}: mach {mach} vs U/sqrt(gamma R T)={mach_gas:.4f}"
    return dict(q_inf=q, rho_inf=rho, u_inf=u, p_inf=p, mach_inf=mach,
                area_ref=float(raw["reference_area"]), length_ref=float(raw["reference_length"]))


def check_forces(root, stem, arr, params):
    """Integral vs forces.json C_L/C_D (pins normals, Cf sign, AoA) and C vs Lift/Drag (pins q, A)."""
    ref = load_json(os.path.join(root, stem, "forces.json"))
    cl, cd = lift_drag(block_force(arr), AOA_DEG, params["area_ref"])
    for name, got, want in [("C_L", cl, float(ref["C_L"])), ("C_D", cd, float(ref["C_D"]))]:
        rel = abs(got - want) / max(abs(want), 1e-12)
        assert rel < FORCE_TOL, (f"{stem}: force integral {name}={got:.6f} vs solver {want:.6f} "
                                 f"(rel {rel:.2e}); aoa={AOA_DEG}, area_ref={params['area_ref']}")
    scale = params["q_inf"] * params["area_ref"]
    for name, coef, dim in [("Lift", ref["C_L"], ref["Lift"]), ("Drag", ref["C_D"], ref["Drag"])]:
        rel = abs(scale * float(coef) - float(dim)) / max(abs(float(dim)), 1e-12)
        assert rel < FORCE_TOL, f"{stem}: q*A*C={scale * float(coef):.4f} N vs {name} {dim} N"
    return cl, cd, float(ref["C_L"]), float(ref["C_D"])


def process_surface(stem, path, root, samples_dir, force=False, verify=True):
    out_path = os.path.join(samples_dir, stem + ".npy")
    params = read_params(root, stem)
    if os.path.exists(out_path) and not force:
        a = np.load(out_path, mmap_mode="r")
        assert a.shape[1] == len(COLS), f"{stem}: stale {a.shape[1]}-column sample, use --force"
        n = int(a.shape[0])
        forces = (check_forces(root, stem, np.asarray(a, dtype=np.float64), params) if verify
                  else (float("nan"),) * 4)
    else:
        mesh = pv.read(path)
        assert mesh.is_all_triangles, f"{stem}: surface is not fully triangulated"
        pts = np.ascontiguousarray(mesh.cell_centers().points, dtype=np.float64)
        p = find_field(mesh, P_PREFIX).astype(np.float64).reshape(-1, 1)
        wss = find_field(mesh, WSS_PREFIX).astype(np.float64).reshape(-1, 3)
        nrm = np.asarray(mesh.cell_data[NORMALS_FIELD], dtype=np.float64).reshape(-1, 3)
        area = np.asarray(mesh.compute_cell_sizes(length=False, area=True, volume=False)
                          .cell_data["Area"], dtype=np.float64).reshape(-1, 1)
        del mesh
        assert np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max() < NORM_TOL, f"{stem}: normals"
        closure = float(np.linalg.norm((nrm * area).sum(0)) / area.sum())
        assert closure < CLOSURE_TOL, f"{stem}: surface not closed ({closure:.2e})"
        out64 = np.concatenate([pts, (p - params["p_inf"]) / params["q_inf"],
                                wss / params["q_inf"], nrm, area], axis=1)
        assert out64.shape[1] == len(COLS) and np.isfinite(out64).all(), f"{stem}: bad output"
        forces = check_forces(root, stem, out64, params)   # gate in float64, before the cast
        out = shuffle_rows(out64.astype(np.float32), SEED + _run_id(stem))
        del out64
        n = out.shape[0]
        save_npy(out_path, out)
    cl, cd, cl_ref, cd_ref = forces
    return dict(stem=stem, run_id=_run_id(stem), n_points=n, q_inf=params["q_inf"],
                area_ref=params["area_ref"], cl_int=cl, cd_int=cd, cl_ref=cl_ref, cd_ref=cd_ref)


def make_metadata(root, out, rows):
    stems = [r["stem"] for r in rows]
    params = {s: load_json(os.path.join(root, s, "params.json")) for s in stems}
    derived = {s: read_params(root, s) for s in stems}
    cond = np.array([[AOA_DEG, derived[s]["mach_inf"]] for s in stems], np.float32)
    geom = np.array([[float(params[s][c]) for c in GEOM_COLS] for s in stems], np.float32)
    der = np.array([[derived[s][c] for c in DERIVED_COLS] for s in stems], np.float32)
    assert np.isfinite(cond).all() and np.isfinite(geom).all() and np.isfinite(der).all()
    truth_cols, truth = load_forces([os.path.join(root, s, "forces.json") for s in stems])
    assert truth is not None, "forces.json missing or non-numeric for some samples"
    all_cols = COND_COLS + GEOM_COLS + DERIVED_COLS + truth_cols
    assert len(set(all_cols)) == len(all_cols), "duplicate metadata column name"
    np.savez(os.path.join(out, "metadata.npz"), stems=np.array(stems),
             cond=cond, cond_cols=np.array(COND_COLS), geom=geom, geom_cols=np.array(GEOM_COLS),
             derived=der, derived_cols=np.array(DERIVED_COLS),
             truth=truth, truth_cols=np.array(truth_cols))
    print(f"wrote metadata.npz ({len(stems)} samples)")


def surface(args):
    out = args.out or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    verify = not args.no_verify
    if args.only:
        row = process_surface(args.only, os.path.join(args.root, args.only, "merged_surfaces.vtp"),
                              args.root, samples, args.force, verify)
        print(f"OK {row['stem']}: {row['n_points']} cells, C_L={row['cl_int']:.6f} "
              f"(ref {row['cl_ref']:.6f}), C_D={row['cd_int']:.6f} (ref {row['cd_ref']:.6f})")
        return
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    else:
        runs = list_runs(args.root)
        if args.limit is None:
            ids = {_run_id(s) for s, _ in runs}
            want = set(range(1, N_DESIGN_POINTS + 1)) - MISSING_IDS
            assert ids == want, f"sample set: missing {sorted(want - ids)}, extra {sorted(ids - want)}"
        jobs = [(s, p, args.root, samples, args.force, verify) for s, p in runs[:args.limit]]
        res, errors = run_pool(process_surface, jobs, args.workers)
        rows = merge_manifest(out, _RUN_RE, lambda s, n: dict(stem=s, run_id=_run_id(s),
                                                              n_points=n), res, errors)
    train, val = random_split(sorted((r["stem"] for r in rows), key=_run_id), VAL_FRAC, SEED)
    write_splits(os.path.join(out, "splits"), train=train, val=val)
    make_metadata(args.root, out, rows)
    if all_on_disk(samples, train):
        save_surface_stats(os.path.join(out, "norm_stats.npz"),
                           *serial_sums([os.path.join(samples, s + ".npy") for s in train]), COLS)


def process_volume(stem, path, root, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=_run_id(stem), n_points=n_rows(out_path))
    lo, hi = slab_box(*surface_bbox(surface_samples, stem), CROP_NEG, CROP_POS, Z_HALF_FRAC,
                      vert=2, ref_axes=(0, 1))
    params = read_params(root, stem)
    mesh = pv.read(path)
    pts = f32(mesh.points)
    vel = f32(mesh.point_data[VEL_FIELD], 3)
    p = np.asarray(mesh.point_data[VOL_P_FIELD], dtype=np.float32).reshape(-1, 1)
    del mesh

    # Inlet-face gate, before the crop removes it.
    assert float(pts[:, 0].min()) < DOMAIN_XMIN_MAX, f"{stem}: raw domain is not the full box"
    inlet = pts[:, 0] < INLET_X
    assert int(inlet.sum()) >= INLET_MIN_POINTS, f"{stem}: too few inlet points"
    rel = check_freestream_velocity(vel[inlet].mean(0), params["u_inf"], AOA_DEG,
                                    FREESTREAM_TOL, stem, "inlet")
    n_raw = pts.shape[0]
    mask = in_box(pts, lo, hi)
    out = np.concatenate([pts[mask], vel[mask],
                          (p[mask] - np.float32(params["p_inf"])) / np.float32(params["q_inf"])],
                         axis=1)
    del pts, vel, p, mask
    keep = check_keep(stem, out.shape[0], n_raw, KEEP_FRAC_MIN, lo, hi)
    assert out.shape[1] == len(VOL_COLS) and np.isfinite(out).all(), f"{stem}: bad output"
    out = shuffle_rows(out, SEED + _run_id(stem))
    save_npy(out_path, out)
    print(f"  {stem}: {out.shape[0]:,}/{n_raw:,} kept ({100 * keep:.2f}%), inlet rel {rel:.1e}",
          flush=True)
    return dict(stem=stem, run_id=_run_id(stem), n_points=out.shape[0])


def volume(args):
    out = args.out or os.path.join(args.root, "volume_collated")
    surf = args.surface or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    row = lambda s, n: dict(stem=s, run_id=_run_id(s), n_points=n)  # noqa: E731
    if args.only:
        r = process_volume(args.only, os.path.join(args.root, args.only, "merged_volumes.vtu"),
                           args.root, samples, os.path.join(surf, "samples"), args.force)
        print(f"OK {r['stem']}: {r['n_points']} points")
        return
    if args.finalize:
        write_manifest(out, rows_from_disk(samples, _RUN_RE, row))
    elif not args.stats_only:
        runs = list_runs(args.root, "merged_volumes.vtu")[:args.limit]
        res, errors = run_pool(process_volume, [(s, p, args.root, samples,
                                                 os.path.join(surf, "samples"), args.force)
                                                for s, p in runs], args.workers)
        merge_manifest(out, _RUN_RE, row, res, errors)
    train = read_split(os.path.join(surf, "splits"), "train")
    if all_on_disk(samples, train):
        cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + _run_id(t))
                                  for t in train], args.stats_points, slice(3, 7),
                                 args.stats_workers, every=25)
        save_volume_stats(os.path.join(out, "norm_stats_volume.npz"), cnt, s, ss,
                          VOL_COLS[3:7], points_per_sample=args.stats_points)
    else:
        print("skipping norm_stats_volume (train samples not all present yet)")


def _prune_extra(before, after, keep, area_ref):
    cl, cd = lift_drag(block_force(after.astype(np.float64)), AOA_DEG, area_ref)
    return dict(cl_pruned=cl, cd_pruned=cd)


def _prune_summary(res, by_stem):
    dcl = np.array([abs(r["cl_pruned"] - by_stem[r["stem"]]["cl_int"]) / abs(by_stem[r["stem"]]["cl_int"])
                    for r in res])
    dcd = np.array([abs(r["cd_pruned"] - by_stem[r["stem"]]["cd_int"]) / abs(by_stem[r["stem"]]["cd_int"])
                    for r in res])
    if len(res):
        print(f"force drift vs the unpruned surface: C_L median={100 * np.median(dcl):.2f}% "
              f"max={100 * dcl.max():.2f}%, C_D median={100 * np.median(dcd):.2f}% "
              f"max={100 * dcd.max():.2f}%")


def prune(args):
    out = args.out or os.path.join(args.root, "collated")
    by_stem = {r["stem"]: r for r in load_json(os.path.join(out, "manifest.json"))}
    run_prune(out, COLS, list(dict.fromkeys(args.columns)), args.sigma, workers=args.workers,
              dry_run=args.dry_run, force=args.force, limit=args.limit,
              extra_fn=_prune_extra, extra_args={s: r["area_ref"] for s, r in by_stem.items()},
              summary_fn=_prune_summary, log_fields=("cl_pruned", "cd_pruned"))


def _surface_args(p):
    cli.add_build_args(p, 16)
    p.add_argument("--only", default=None, help="process one stem (e.g. sample_000001) and exit")
    p.add_argument("--splits-only", action="store_true",
                   help="skip conversion; rebuild splits + metadata + stats from the manifest")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the force-integral gate on samples already on disk")


def _volume_args(p):
    cli.add_volume_args(p, 12)
    p.add_argument("--only", default=None, help="process one stem and exit")
    p.add_argument("--finalize", action="store_true", help="manifest + stats from samples on disk")
    p.add_argument("--stats-points", type=int, default=0, help="rows per sample for stats (0 = all)")
    p.add_argument("--stats-workers", type=int, default=8)


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from merged_surfaces.vtp"),
        "volume": (_volume_args, volume, "build volume_collated/ from merged_volumes.vtu"),
        "prune": (lambda p: cli.add_prune_args(p, ["cf_x", "cf_y", "cf_z"],
                                               ["cp", "cf_x", "cf_y", "cf_z"], 4.0),
                  prune, "per-sample sigma cut of the surface Cf tail, in place"),
    })
