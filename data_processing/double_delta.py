"""Double-Delta (SU2 RANS, M 0.3; SU2 Cp/Cf used as-is): <set>/<callsign>/<aoa>.0degAOA/*.vtu.
surface -> [n, 14] x y z cp cf_x cf_y cf_z n_x n_y n_z area yPlus rho_tilde t_tilde
volume  -> [n, 9]  x y z ux uy uz cp rho_tilde t_tilde
prune   -> cp > Cp_0 bound + 3-sigma two-sided cp cut (what collated/ holds)
"""
import os
import re

import numpy as np

from .utils import cli
from .utils.geometry import check_freestream_velocity, check_keep, in_box, lift_drag, slab_box, surface_bbox
from .utils.io import (all_on_disk, load_json, merge_manifest, n_rows, read_split, rows_from_disk,
                       save_npy, write_manifest, write_splits)
from .utils.pool import run_pool
from .utils.prune import run_prune
from .utils.shuffle import shuffle_rows
from .utils.stats import pooled_sums, save_surface_stats, save_volume_stats
from .utils.vtk import read_mesh

NAME = "double_delta"
SEED = 0
SETS = {"train": ("trainingSet", "designs_256samples.csv", "wing_reference.csv"),
        "val": ("holdoutSet", "designs_holdout.csv", "wing_reference_holdout.csv")}
_STEM_RE = re.compile(r"^([A-Za-z0-9-]+)_AoA_(\d+)$")
_AOA_RE = re.compile(r"^(\d+)\.0degAOA$")   # anchored: the `...degAOA2` re-runs are excluded

MACH = 0.3
P_INF = 71833.4
RHO_INF = 0.878035
U_INF = 101.53
Q_INF = 0.5 * RHO_INF * U_INF ** 2   # 4525.54 Pa (SU2 already divided by it)
T_INF = 285.0
REYNOLDS = 8.04e7
LENGTH_REF = 16.0
GAMMA = 1.4
CP_STAGNATION = (2 / (GAMMA * MACH ** 2)) * (
    (1 + 0.5 * (GAMMA - 1) * MACH ** 2) ** (GAMMA / (GAMMA - 1)) - 1)   # 1.0227

COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z",
        "n_x", "n_y", "n_z", "area", "yPlus", "rho_tilde", "t_tilde"]
COND_COLS = ["aoa", "mach", "geom_id", "run_id", "areaRef", "lengthRef", "xmrc",
             "reynolds", "ausm"]
GEOM_COLS = ["droop_angle", "sw1", "sw2", "sr2", "bw2", "b"]
DESIGN_CSV_COLS = ["DROOP_ANGLE", "SW1", "SW2", "SR2", "BW2", "B"]
TRUTH_COLS = ["CL", "CD", "CMy", "CFx", "CFy", "CFz"]
CP_FIELD = "Pressure_Coefficient"
CF_FIELD = "Skin_Friction_Coefficient"
YPLUS_FIELD = "Y_Plus"
RHO_FIELD = "Density"
T_FIELD = "Temperature"
SURF_FIELDS = [CP_FIELD, CF_FIELD, YPLUS_FIELD, RHO_FIELD, T_FIELD]
FORCE_TOL = 5e-3   # point integral vs SU2's CL/CD, measured <=5e-4

VEL_FIELD = "Velocity"
VOL_FIELDS = [VEL_FIELD, CP_FIELD, RHO_FIELD, T_FIELD]
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "cp", "rho_tilde", "t_tilde"]
# x/y margins on the surface bbox; z = centre +/- 0.20 * max(Lx, Ly) (the wing is ~1.8 thick).
CROP_NEG = [0.25, 0.35, None]
CROP_POS = [0.75, 0.35, None]
Z_HALF_FRAC = 0.20
KEEP_FRAC_MIN = 0.90     # measured 0.975-0.989
FREESTREAM_TOL = 1e-3    # far-field velocity, measured 2e-5
FAR_RADIUS = 200.0       # from the wing bbox centre; the domain sphere has r ~ 350
FAR_MIN_POINTS = 200


def raw_root(args):
    return args.raw or os.path.join(os.path.dirname(os.path.abspath(args.root)), "double_delta_aero")


def load_geom_ids(raw):
    """callsign -> index into the sorted union of the design CSVs' call signs."""
    import pandas as pd
    signs = set()
    for _, design_csv, _ in SETS.values():
        df = pd.read_csv(os.path.join(raw, "geometryDefinition", design_csv))
        signs.update(str(s) for s in df["design-call-sign"])
    return {s: i for i, s in enumerate(sorted(signs))}


def parse_stem(stem, geom_ids):
    """(callsign, aoa, geom_id, run_id = geom_id * 100 + aoa)."""
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"not a double_delta stem: {stem!r}")
    callsign, aoa = m.group(1), int(m.group(2))
    if callsign not in geom_ids:
        raise ValueError(f"{stem}: call sign {callsign!r} is not in the design CSVs")
    return callsign, aoa, geom_ids[callsign], geom_ids[callsign] * 100 + aoa


def case_dir(raw, stem, geom_ids):
    callsign, aoa, _, _ = parse_stem(stem, geom_ids)
    for subdir, _, _ in SETS.values():
        d = os.path.join(raw, subdir, callsign, f"{aoa}.0degAOA")
        if os.path.exists(os.path.join(d, "surface_flow.vtu")):
            return d
    return None


def list_runs(raw, geom_ids):
    """Sorted (stem, case_dir, run_id) over canonical <aoa>.0degAOA dirs with a surface_flow.vtu."""
    runs = []
    for subdir, _, _ in SETS.values():
        for callsign in sorted(os.listdir(os.path.join(raw, subdir))):
            gdir = os.path.join(raw, subdir, callsign)
            if not os.path.isdir(gdir) or callsign not in geom_ids:
                continue
            for aoa_dir in sorted(os.listdir(gdir)):
                m = _AOA_RE.match(aoa_dir)
                if m and os.path.exists(os.path.join(gdir, aoa_dir, "surface_flow.vtu")):
                    aoa = int(m.group(1))
                    runs.append((f"{callsign}_AoA_{aoa}", os.path.join(gdir, aoa_dir),
                                 geom_ids[callsign] * 100 + aoa))
    return sorted(runs, key=lambda r: r[2])


def read_forces(cdir):
    """SU2's integrated coefficients, AoA and reference area from forces_breakdown.dat."""
    with open(os.path.join(cdir, "forces_breakdown.dat")) as f:
        txt = f.read()

    def grab(pattern):
        m = re.search(pattern, txt)
        if m is None:
            raise ValueError(f"{cdir}: no match for {pattern!r} in forces_breakdown.dat")
        return float(m.group(1))
    out = {c: grab(rf"Total {c}:\s*([-\d.eE+]+)") for c in TRUTH_COLS}
    out["aoa"] = grab(r"Angle of attack \(AoA\):\s*([-\d.eE+]+)")
    out["areaRef"] = grab(r"The reference area is\s*([\d.eE+-]+)")
    return out


def point_normals_areas(points, tri):
    """Area-weighted unit vertex normals, barycentric dual areas, and the closure
    |sum n dA| / sum dA of the triangulation."""
    a, b, c = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    area_t = 0.5 * np.linalg.norm(cross, axis=1)
    n_t = cross / np.maximum(2.0 * area_t, 1e-30)[:, None]
    n_p = np.zeros_like(points)
    area_p = np.zeros(len(points), dtype=points.dtype)
    for k in range(3):
        np.add.at(n_p, tri[:, k], n_t * area_t[:, None])
        np.add.at(area_p, tri[:, k], area_t / 3.0)
    n_p /= np.maximum(np.linalg.norm(n_p, axis=1, keepdims=True), 1e-30)
    closure = float(np.linalg.norm((n_t * area_t[:, None]).sum(0)) / area_t.sum())
    return n_p, area_p, closure


def process_surface(stem, cdir, run_id, samples_dir, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem, n_rows(out_path)
    mesh = read_mesh(os.path.join(cdir, "surface_flow.vtu"), SURF_FIELDS)
    pts = np.ascontiguousarray(mesh.points, dtype=np.float64)
    cp = np.asarray(mesh.point_data[CP_FIELD], dtype=np.float64).reshape(-1)
    cf = np.asarray(mesh.point_data[CF_FIELD], dtype=np.float64)
    yplus = np.asarray(mesh.point_data[YPLUS_FIELD], dtype=np.float32).reshape(-1, 1)
    rho = np.asarray(mesh.point_data[RHO_FIELD], dtype=np.float32).reshape(-1, 1)
    temp = np.asarray(mesh.point_data[T_FIELD], dtype=np.float32).reshape(-1, 1)
    cells = mesh.cells.reshape(-1, 4)
    assert (cells[:, 0] == 3).all(), f"{stem}: non-triangular cells present"
    n_p, area_p, closure = point_normals_areas(pts, cells[:, 1:])
    assert closure < 1e-6, f"{stem}: surface not closed / inconsistently wound ({closure:.2e})"

    # Force gate: pins the normal orientation and the Cf sign on every sample.
    ref = read_forces(cdir)
    force_q = (-cp[:, None] * n_p * area_p[:, None]).sum(0) + (cf * area_p[:, None]).sum(0)
    cl, cd = lift_drag(force_q, ref["aoa"], ref["areaRef"])
    for name, got, want in [("CL", cl, ref["CL"]), ("CD", cd, ref["CD"])]:
        rel = abs(got - want) / max(abs(want), 1e-12)
        assert rel < FORCE_TOL, f"{stem}: force integral {name}={got:.6f} vs SU2 {want:.6f}"

    out = np.concatenate([
        pts.astype(np.float32), cp.astype(np.float32).reshape(-1, 1), cf.astype(np.float32),
        n_p.astype(np.float32), area_p.astype(np.float32).reshape(-1, 1), yplus,
        rho / np.float32(RHO_INF), temp / np.float32(T_INF),
    ], axis=1).astype(np.float32)
    assert out.shape == (pts.shape[0], len(COLS)) and np.isfinite(out).all(), f"{stem}: bad output"
    save_npy(out_path, shuffle_rows(out, SEED + run_id))
    return stem, out.shape[0]


def _row_fn(geom_ids):
    def row(stem, n):
        _, aoa, geom_id, run_id = parse_stem(stem, geom_ids)
        return dict(stem=stem, run_id=run_id, geom_id=geom_id, aoa=aoa, n_points=n)
    return row


def make_splits(raw, rows, geom_ids):
    """Native split: every geometry of a design CSV lands in that CSV's split."""
    import pandas as pd
    where = {}
    for split, (_, design_csv, _) in SETS.items():
        df = pd.read_csv(os.path.join(raw, "geometryDefinition", design_csv))
        where.update({geom_ids[str(s)]: split for s in df["design-call-sign"]})
    train = sorted(r["stem"] for r in rows if where[r["geom_id"]] == "train")
    val = sorted(r["stem"] for r in rows if where[r["geom_id"]] == "val")
    return train, val


def make_metadata(raw, out, rows, geom_ids):
    import pandas as pd
    design, ref = {}, {}
    for _, design_csv, ref_csv in SETS.values():
        d = pd.read_csv(os.path.join(raw, "geometryDefinition", design_csv)).set_index("design-call-sign")
        r = pd.read_csv(os.path.join(raw, "geometryDefinition", ref_csv)).set_index("design-call-sign")
        for s in d.index:
            if str(s) in geom_ids:
                design[str(s)] = [float(d.loc[s, c]) for c in DESIGN_CSV_COLS]
                ref[str(s)] = (float(r.loc[s, "reference_area"]), float(r.loc[s, "aero_center_x"]))
    # AUSMruns.info.txt: one `<callsign>/<aoa>.0degAOA` per line (some with stray commas).
    ausm = set()
    info = os.path.join(raw, "AUSMruns.info.txt")
    if os.path.exists(info):
        for line in open(info):
            line = line.strip().replace(",", "")
            if line and not line.startswith("#") and "/" in line:
                callsign, aoa_part = line.split("/", 1)
                m = _AOA_RE.match(aoa_part)
                if m and callsign in geom_ids:
                    ausm.add(geom_ids[callsign] * 100 + int(m.group(1)))
    cond, geom, truth = [], [], []
    for r in rows:
        callsign, aoa, geom_id, run_id = parse_stem(r["stem"], geom_ids)
        area_ref, xmrc = ref[callsign]
        cond.append([aoa, MACH, geom_id, run_id, area_ref, LENGTH_REF, xmrc, REYNOLDS,
                     1.0 if run_id in ausm else 0.0])
        geom.append(design[callsign])
        f = read_forces(case_dir(raw, r["stem"], geom_ids))
        truth.append([f[c] for c in TRUTH_COLS])
    cond, geom, truth = (np.array(x, dtype=np.float32) for x in (cond, geom, truth))
    assert np.isfinite(cond).all() and np.isfinite(geom).all() and np.isfinite(truth).all()
    np.savez(os.path.join(out, "metadata.npz"), stems=np.array([r["stem"] for r in rows]),
             cond=cond, cond_cols=np.array(COND_COLS), geom=geom, geom_cols=np.array(GEOM_COLS),
             truth=truth, truth_cols=np.array(TRUTH_COLS))
    print(f"wrote metadata.npz ({len(rows)} samples, {int(cond[:, -1].sum())} AUSM runs)")


def surface(args):
    raw = raw_root(args)
    out = args.out or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    geom_ids = load_geom_ids(raw)
    row = _row_fn(geom_ids)
    if args.only:
        cdir = case_dir(raw, args.only, geom_ids)
        if cdir is None:
            print(f"SKIP {args.only}: no surface_flow.vtu found")
            return
        stem, n = process_surface(args.only, cdir, parse_stem(args.only, geom_ids)[3], samples,
                                  args.force)
        print(f"OK {stem}: {n} points")
        return
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    elif args.finalize:
        rows = write_manifest(out, rows_from_disk(samples, _STEM_RE, row))
    else:
        runs = list_runs(raw, geom_ids)[:args.limit]
        res, errors = run_pool(process_surface, [(*r, samples, args.force) for r in runs],
                               args.workers, every=50)
        rows = merge_manifest(out, _STEM_RE, row, [row(s, n) for s, n in res], errors)
    train, val = make_splits(raw, rows, geom_ids)
    write_splits(os.path.join(out, "splits"), train=train, val=val)
    make_metadata(raw, out, rows, geom_ids)
    if all_on_disk(samples, train):
        cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + parse_stem(t, geom_ids)[3])
                                  for t in train], args.stats_points, slice(None),
                                 args.stats_workers, every=200)
        save_surface_stats(os.path.join(out, "norm_stats.npz"), cnt, s, ss, COLS,
                           n_points=cnt, points_per_sample=args.stats_points)


def process_volume(stem, path, aoa, run_id, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem, n_rows(out_path)
    smn, smx = surface_bbox(surface_samples, stem)
    lo, hi = slab_box(smn, smx, CROP_NEG, CROP_POS, Z_HALF_FRAC, vert=2, ref_axes=(0, 1))
    centre = 0.5 * (smn + smx)
    mesh = read_mesh(path, VOL_FIELDS)
    pts = np.ascontiguousarray(mesh.points, dtype=np.float32)
    vel = np.ascontiguousarray(mesh.point_data[VEL_FIELD], dtype=np.float32).reshape(-1, 3)
    cp = np.asarray(mesh.point_data[CP_FIELD], dtype=np.float32).reshape(-1, 1)
    rho = np.asarray(mesh.point_data[RHO_FIELD], dtype=np.float32).reshape(-1, 1)
    temp = np.asarray(mesh.point_data[T_FIELD], dtype=np.float32).reshape(-1, 1)
    del mesh

    # SU2 rotates the freestream, not the mesh: the far field must carry U_INF at this AoA.
    far = np.linalg.norm(pts - centre, axis=1) > FAR_RADIUS
    assert int(far.sum()) >= FAR_MIN_POINTS, f"{stem}: only {int(far.sum())} far-field points"
    check_freestream_velocity(vel[far].mean(0), U_INF, aoa, FREESTREAM_TOL, stem, "far-field")

    out = np.concatenate([pts, vel, cp, rho / np.float32(RHO_INF), temp / np.float32(T_INF)],
                         axis=1)
    n_raw = out.shape[0]
    del pts, vel, cp, rho, temp
    out = out[in_box(out[:, :3], lo, hi)]
    check_keep(stem, out.shape[0], n_raw, KEEP_FRAC_MIN, lo, hi)
    assert out.shape[1] == len(VOL_COLS) and np.isfinite(out).all(), f"{stem}: bad output"
    save_npy(out_path, shuffle_rows(out, SEED + run_id))
    return stem, out.shape[0]


def volume(args):
    raw = raw_root(args)
    out = args.out or os.path.join(args.root, "volume_collated")
    surf = args.surface or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    geom_ids = load_geom_ids(raw)
    row = _row_fn(geom_ids)
    surf_samples = os.path.join(surf, "samples")
    if args.only:
        cdir = case_dir(raw, args.only, geom_ids)
        path = None if cdir is None else os.path.join(cdir, "flow.vtu")
        if path is None or not os.path.exists(path):
            print(f"SKIP {args.only}: no flow.vtu found")
            return
        _, aoa, _, run_id = parse_stem(args.only, geom_ids)
        print("OK %s: %d points" % process_volume(args.only, path, aoa, run_id, samples,
                                                   surf_samples, args.force))
        return
    if args.finalize:
        write_manifest(out, rows_from_disk(samples, _STEM_RE, row))
    elif not args.stats_only:
        runs = [(s, os.path.join(d, "flow.vtu"), parse_stem(s, geom_ids)[1], rid)
                for s, d, rid in list_runs(raw, geom_ids)
                if os.path.exists(os.path.join(d, "flow.vtu"))][:args.limit]
        res, errors = run_pool(process_volume, [(*r, samples, surf_samples, args.force)
                                                for r in runs], args.workers)
        merge_manifest(out, _STEM_RE, row, [row(s, n) for s, n in res], errors)
    train = read_split(os.path.join(surf, "splits"), "train")
    if all_on_disk(samples, train):
        cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + parse_stem(t, geom_ids)[3])
                                  for t in train], args.stats_points, slice(3, 9),
                                 args.stats_workers, every=200)
        save_volume_stats(os.path.join(out, "norm_stats_volume.npz"), cnt, s, ss, VOL_COLS[3:9],
                          extra=True, points_per_sample=args.stats_points)
    else:
        print("skipping norm_stats_volume (train samples not all present yet)")


def _prune_extra(before, after, keep, ref):
    aoa, area_ref = ref
    b, a = before.astype(np.float64), after.astype(np.float64)
    cl0, cd0 = lift_drag(((-b[:, 3:4] * b[:, 7:10] + b[:, 4:7]) * b[:, 10:11]).sum(0), aoa, area_ref)
    cl, cd = lift_drag(((-a[:, 3:4] * a[:, 7:10] + a[:, 4:7]) * a[:, 10:11]).sum(0), aoa, area_ref)
    cp = b[:, 3]
    hi = lo = 0
    if keep is not None:
        mean = float(cp[np.isfinite(cp)].mean())
        hi, lo = int((~keep & (cp > mean)).sum()), int((~keep & (cp <= mean)).sum())
    return dict(cl=cl, cd=cd, cl_before=cl0, cd_before=cd0, removed_hi=hi, removed_lo=lo,
                over_before=int((cp > CP_STAGNATION).sum()),
                over_after=int((a[:, 3] > CP_STAGNATION).sum()),
                cp_min_before=float(cp.min()), cp_min_after=float(a[:, 3].min()))


def _prune_summary(res, _):
    rm = sum(r["n_removed"] for r in res)
    hi = sum(r["removed_hi"] for r in res)
    print(f"  removed above the sample mean (unphysical side): {hi:,}; below (suction): "
          f"{rm - hi:,}")
    print(f"  rows with cp > {CP_STAGNATION:.4f}: {sum(r['over_before'] for r in res):,} -> "
          f"{sum(r['over_after'] for r in res):,}; Cp_min "
          f"{min(r['cp_min_before'] for r in res):.3f} -> {min(r['cp_min_after'] for r in res):.3f}")
    fresh = [r for r in res if not r["skipped"]]
    if fresh:
        dcd = np.array([abs(r["cd"] - r["cd_before"]) / max(abs(r["cd_before"]), 1e-12)
                        for r in fresh])
        print(f"  |dCD|/CD median {100 * np.median(dcd):.3f}% max {100 * dcd.max():.3f}%")


def prune(args):
    out = args.out or os.path.join(args.root, "collated")
    z = np.load(os.path.join(out, "metadata.npz"), allow_pickle=True)
    cols = [str(c) for c in z["cond_cols"]]
    ia, ir = cols.index("aoa"), cols.index("areaRef")
    refs = {str(s): (float(z["cond"][i, ia]), float(z["cond"][i, ir]))
            for i, s in enumerate(z["stems"].tolist())}
    cp_max = (None if args.cp_max in (None, "none") else
              CP_STAGNATION if args.cp_max == "auto" else float(args.cp_max))
    run_prune(out, COLS, list(dict.fromkeys(args.columns)), args.sigma, side=args.side,
              cp_max=cp_max, log_name="pruned_cp.json", log_meta=dict(cp_stagnation=CP_STAGNATION),
              workers=args.workers, dry_run=args.dry_run, force=args.force, limit=args.limit,
              stats_count_key=True, extra_fn=_prune_extra, extra_args=refs,
              summary_fn=_prune_summary, log_fields=("removed_hi", "removed_lo"))


def _raw_arg(p):
    p.add_argument("--raw", default=None, help="raw tree (default <root>/../double_delta_aero)")


def _surface_args(p):
    cli.add_build_args(p, 32)
    _raw_arg(p)
    p.add_argument("--only", default=None, help="process one stem (e.g. 00060104RV_AoA_15)")
    p.add_argument("--finalize", action="store_true",
                   help="manifest/splits/metadata/stats from samples on disk")
    p.add_argument("--splits-only", action="store_true",
                   help="skip conversion; rebuild splits + metadata + stats from the manifest")
    p.add_argument("--stats-points", type=int, default=0, help="rows per sample (0 = all)")
    p.add_argument("--stats-workers", type=int, default=8)


def _volume_args(p):
    cli.add_volume_args(p, 16)
    _raw_arg(p)
    p.add_argument("--only", default=None, help="process one stem and exit")
    p.add_argument("--finalize", action="store_true", help="manifest + stats from samples on disk")
    p.add_argument("--stats-points", type=int, default=0, help="rows per sample (0 = all)")
    p.add_argument("--stats-workers", type=int, default=8)


def _prune_args(p):
    cli.add_prune_args(p, ["cp"], ["cp", "cf_x", "cf_y", "cf_z", "rho_tilde", "t_tilde"], 3.0)
    p.add_argument("--side", choices=("both", "hi", "lo"), default="both")
    p.add_argument("--cp-max", default="auto",
                   help=f"also drop cp above this bound; 'auto' = Cp_0 at M {MACH} "
                        f"({CP_STAGNATION:.4f}), 'none' = off")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from surface_flow.vtu"),
        "volume": (_volume_args, volume, "build volume_collated/ from flow.vtu"),
        "prune": (_prune_args, prune, "cp bound + sigma cut of the surface, in place "
                                      "(defaults reproduce collated/; raw is gone, irreversible)"),
    })
