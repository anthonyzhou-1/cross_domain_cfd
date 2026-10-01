"""HiLiftAeroML (LES, M 0.2, Imperial units): geo_LHC<G>_AoA_<A>/{boundary_*.vtu, volume_*.vtu.tgz}.
surface -> [n // factor, 9] x y z cp cf_x cf_y cf_z rho_tilde yPlus   (shuffled prefix; disk: factor 2)
volume  -> [n, 7(9)] x y z ux uy uz cp (rho_tilde t_tilde)           (streamed from the .tgz)
"""
import glob
import os
import re

import numpy as np

from .utils import cli
from .utils.geometry import check_freestream_velocity
from .utils.io import load_json, merge_manifest, n_rows, read_split, rows_from_disk, save_npy, write_manifest, write_splits
from .utils.pool import run_pool
from .utils.shuffle import shuffle_rows
from .utils.splits import grouped_split
from .utils.stats import pooled_sums, save_surface_stats, save_volume_stats
from .utils.vtk import read_mesh
from .utils.vtu_stream import AppendedVTU

NAME = "hiliftaeroml"
SEED = 0
VAL_FRAC = 0.10
_RUN_RE = re.compile(r"^geo_LHC(\d+)_AoA_(\d+)$")
STATS_POINTS = 1_000_000
FACTOR = 2   # collated/ keeps the leading n // 2 rows of each shuffled sample

MACH = 0.2
GAMMA = 1.4
QREF = 4.937856
UREF = 2679.5054741899685
CHORD_REF = 275.8
SPAN_REF = 1156.75
AREA_REF = 297360.0
P_INF = QREF / (0.5 * GAMMA * MACH ** 2)   # ~176.35, solver units
RHO_INF = 2.0 * QREF / UREF ** 2           # ~1.376e-6
T_INF = 518.67                             # deg Rankine, measured off the volume far field

P_FIELD = "PROJ(AVG(P))"
RHO_FIELD = "PROJ(AVG(RHO))"
TAU_FIELDS = ["AVG(TAU_WALL(0))", "AVG(TAU_WALL(1))", "AVG(TAU_WALL(2))"]
YPLUS_FIELD = "AVG(Y_PLUS)"
SURF_FIELDS = [P_FIELD, RHO_FIELD, *TAU_FIELDS, YPLUS_FIELD]
COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z", "rho_tilde", "yPlus"]
COND_COLS = ["geom_id", "aoa", "run_id", "qRef", "uRef", "chordRef", "spanRef", "areaRef", "mach"]

# Appended-array names, in increasing-offset order (the stream is forward-only).
POINTS, VEL_FIELD, T_FIELD, VOL_P_FIELD, VOL_RHO_FIELD = "Points", "avg(u)", "avg(T)", "avg(P)", "avg(rho)"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "cp"]
EXTRA_COLS = ["rho_tilde", "t_tilde"]
# x/y margins on the surface bbox; z = centre +/- 1.0 * Lz. Half domain: y >= 0.
CROP_NEG = np.array([0.15, 0.05])
CROP_POS = np.array([0.50, 0.15])
Z_HALF_FRAC = 1.00
KEEP_FRAC_MIN = 0.97      # measured 0.9989
KEEP_FRAC = 0.5           # Bernoulli thinning of the cropped points
FAR_RADIUS = 8000.0       # far-field gate shell (domain reaches ~26200)
FAR_MIN_POINTS = 20_000
FAR_MAX_GATE = 50_000
FREESTREAM_TOL = 1e-3     # measured 6e-8; one AoA step off is 0.035
BBOX_ROWS = 2_000_000     # surface rows read for the crop bbox (rows are shuffled)
_GATHER_ROWS = 1 << 22


def parse_stem(stem):
    """(geom_id, aoa, run_id = geom_id * 100 + aoa)."""
    m = _RUN_RE.match(stem)
    if not m:
        raise ValueError(f"not a hiliftaeroml stem: {stem!r}")
    g, a = int(m.group(1)), int(m.group(2))
    return g, a, g * 100 + a


def _row(stem, n, **kw):
    g, a, rid = parse_stem(stem)
    return dict(stem=stem, run_id=rid, geom_id=g, aoa=a, n_points=n, **kw)


def _vtu_path(root, stem):
    m = sorted(glob.glob(os.path.join(root, stem, "boundary_*.vtu")))
    return m[0] if m else None


def list_surface_runs(root):
    """Extracted samples only: a dir still holding a .tgz is mid-extraction and skipped."""
    runs = []
    for d in sorted(os.listdir(root)):
        if _RUN_RE.match(d) and not glob.glob(os.path.join(root, d, "*.tgz")):
            path = _vtu_path(root, d)
            if path is not None:
                runs.append((d, path))
    return sorted(runs, key=lambda r: parse_stem(r[0])[2])


def process_surface(stem, path, samples_dir, factor=FACTOR, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return _row(stem, n_rows(out_path))
    mesh = read_mesh(path, SURF_FIELDS)
    pts = np.ascontiguousarray(mesh.points, dtype=np.float32)
    p = np.asarray(mesh.point_data[P_FIELD], dtype=np.float32).reshape(-1)
    rho = np.asarray(mesh.point_data[RHO_FIELD], dtype=np.float32).reshape(-1)
    tau = np.stack([np.asarray(mesh.point_data[f], dtype=np.float32) for f in TAU_FIELDS], axis=1)
    yplus = np.asarray(mesh.point_data[YPLUS_FIELD], dtype=np.float32).reshape(-1, 1)
    out = np.concatenate([pts, ((p - P_INF) / QREF).reshape(-1, 1), (tau / QREF).astype(np.float32),
                          (rho / RHO_INF).reshape(-1, 1), yplus], axis=1).astype(np.float32)
    assert out.shape == (pts.shape[0], len(COLS)), f"{stem}: bad shape {out.shape}"
    out = shuffle_rows(out, SEED + parse_stem(stem)[2])
    out = np.ascontiguousarray(out[:out.shape[0] // factor])   # a prefix is a uniform subsample
    save_npy(out_path, out)
    return _row(stem, out.shape[0])


def surface(args):
    out = args.out or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    if args.only:
        path = _vtu_path(args.root, args.only)
        if path is None:
            print(f"SKIP {args.only}: no boundary_*.vtu (not extracted?)")
            return
        print(f"OK {process_surface(args.only, path, samples, args.factor, args.force)}")
        return
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    elif args.finalize:
        rows = write_manifest(out, rows_from_disk(samples, _RUN_RE, _row))
    else:
        runs = list_surface_runs(args.root)[:args.limit]
        res, errors = run_pool(process_surface, [(s, p, samples, args.factor, args.force)
                                                 for s, p in runs], args.workers, every=10)
        rows = merge_manifest(out, _RUN_RE, _row, res, errors)
    train, val = grouped_split(rows, "geom_id", VAL_FRAC, SEED)
    write_splits(os.path.join(out, "splits"), train=train, val=val)
    cond = np.array([[r["geom_id"], r["aoa"], r["run_id"], QREF, UREF, CHORD_REF, SPAN_REF,
                      AREA_REF, MACH] for r in rows], dtype=np.float32)
    np.savez(os.path.join(out, "metadata.npz"), stems=np.array([r["stem"] for r in rows]),
             cond=cond, cond_cols=np.array(COND_COLS))
    if all(os.path.exists(os.path.join(samples, s + ".npy")) for s in train):
        cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + parse_stem(t)[2])
                                  for t in train], args.stats_points, slice(None),
                                 args.stats_workers)
        save_surface_stats(os.path.join(out, "norm_stats.npz"), cnt, s, ss, COLS,
                           n_points=cnt, points_per_sample=args.stats_points)


# --- volume -------------------------------------------------------------------------------

def archive_path(root, stem):
    for pat in ("volume_*.vtu.tgz", "volume_*.vtu"):
        m = sorted(glob.glob(os.path.join(root, stem, pat)))
        if m:
            return m[0]
    return None


def list_volume_runs(root, surface_samples, include_unpaired=False):
    """(stem, path, paired); unpaired stems (no surface sample) only when asked."""
    have = os.path.isdir(surface_samples)
    runs = []
    for d in sorted(os.listdir(root)):
        path = archive_path(root, d) if _RUN_RE.match(d) else None
        if path is None:
            continue
        paired = have and os.path.exists(os.path.join(surface_samples, d + ".npy"))
        if paired or include_unpaired:
            runs.append((d, path, paired))
    return sorted(runs, key=lambda r: parse_stem(r[0])[2])


def _bbox(surface_samples, stem):
    a = np.load(os.path.join(surface_samples, stem + ".npy"), mmap_mode="r")
    w = np.asarray(a[:min(BBOX_ROWS, a.shape[0]), :3], dtype=np.float64)
    return w.min(0), w.max(0)


def surface_bbox(surface_samples, stem):
    """Surface bbox; an unpaired stem borrows the nearest AoA of the same geometry
    (the wing bbox agrees to ~1e-4 across AoA)."""
    if os.path.exists(os.path.join(surface_samples, stem + ".npy")):
        return _bbox(surface_samples, stem)
    g, aoa, _ = parse_stem(stem)
    sibs = [(abs(parse_stem(os.path.basename(f)[:-4])[1] - aoa), os.path.basename(f)[:-4])
            for f in sorted(glob.glob(os.path.join(surface_samples, f"geo_LHC{g:03d}_AoA_*.npy")))]
    if not sibs:
        raise FileNotFoundError(f"{stem}: no surface sample of geometry {g} to fall back on")
    return _bbox(surface_samples, min(sibs)[1])


def crop_box(surface_samples, stem):
    smn, smx = surface_bbox(surface_samples, stem)
    L = smx - smn
    zc = 0.5 * (smn[2] + smx[2])
    lo = np.array([smn[0] - CROP_NEG[0] * L[0], smn[1] - CROP_NEG[1] * L[1], zc - Z_HALF_FRAC * L[2]])
    hi = np.array([smx[0] + CROP_POS[0] * L[0], smx[1] + CROP_POS[1] * L[1], zc + Z_HALF_FRAC * L[2]])
    return lo.astype(np.float32), hi.astype(np.float32), 0.5 * (smn + smx)


def _crop_and_far_masks(pts, lo, hi, centre, r_far, chunk=_GATHER_ROWS):
    """(inside crop box, beyond r_far) masks, chunked to avoid full-size [n,3] temporaries."""
    n = pts.shape[0]
    inside, far = np.empty(n, bool), np.empty(n, bool)
    r2 = np.float32(r_far) ** 2
    for i in range(0, n, chunk):
        blk = pts[i:i + chunk]
        np.all((blk >= lo) & (blk <= hi), axis=1, out=inside[i:i + chunk])
        d = blk - centre
        np.greater((d * d).sum(1), r2, out=far[i:i + chunk])
    return inside, far


def _copy_masked(dst, src, mask, chunk=_GATHER_ROWS):
    """dst[:] = src[mask], chunk by chunk."""
    a = 0
    for i in range(0, mask.size, chunk):
        blk = mask[i:i + chunk]
        k = int(blk.sum())
        if k:
            dst[a:a + k] = src[i:i + chunk][blk]
            a += k
    assert a == dst.shape[0], f"masked copy filled {a} of {dst.shape[0]} rows"


def _thin(mask, keep_frac, rng, chunk=1 << 24):
    """AND `mask` in place with a chunked Bernoulli(keep_frac) draw."""
    if keep_frac >= 1.0:
        return mask
    for i in range(0, mask.size, chunk):
        blk = mask[i:i + chunk]
        blk &= rng.random(blk.size) < keep_frac
    return mask


def _save_shuffled(out_path, arr, rng, chunk=_GATHER_ROWS):
    """Write `arr` as an ordinary .npy with its rows permuted, gathering chunk by chunk."""
    m, ncol = arr.shape
    perm = rng.permutation(m)
    tmp = f"{out_path}.{os.getpid()}.tmp.npy"
    buf = np.empty((min(chunk, m), ncol), arr.dtype)
    with open(tmp, "wb") as fh:
        np.lib.format.write_array_header_1_0(
            fh, {"descr": np.lib.format.dtype_to_descr(arr.dtype),
                 "fortran_order": False, "shape": (m, ncol)})
        for a in range(0, m, chunk):
            k = min(chunk, m - a)
            np.take(arr, perm[a:a + k], axis=0, out=buf[:k])
            fh.write(memoryview(buf[:k]).cast("B"))
    os.replace(tmp, out_path)


def process_volume(stem, path, paired, samples_dir, surface_samples, keep_frac=KEEP_FRAC,
                   extra=False, force=False, delete_src=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    _, aoa, run_id = parse_stem(stem)
    if os.path.exists(out_path) and not force:
        return _row(stem, n_rows(out_path), n_raw=None, paired=paired)
    lo, hi, centre = crop_box(surface_samples, stem)
    cols = VOL_COLS + (EXTRA_COLS if extra else [])
    rng = np.random.default_rng(SEED + run_id)

    with AppendedVTU(path) as vtu:
        n_raw = vtu.n_points
        pts = vtu.read_array(POINTS)
        assert pts.shape == (n_raw, 3) and n_raw < 2 ** 31, f"{stem}: points {pts.shape}"
        # Gate rows (beyond FAR_RADIUS) are disjoint from the crop box, so one take-index
        # streams both and keep_mask splits them back apart.
        keep, far_mask = _crop_and_far_masks(pts, lo, hi, centre.astype(np.float32), FAR_RADIUS)
        far = np.flatnonzero(far_mask).astype(np.int32)
        del far_mask
        n_far = far.size
        if n_far > FAR_MAX_GATE:
            far = np.sort(rng.choice(far, FAR_MAX_GATE, replace=False))
        n_in = int(keep.sum())
        frac = n_in / n_raw
        assert frac >= KEEP_FRAC_MIN, f"{stem}: crop kept only {100 * frac:.2f}% of {n_raw}"
        sel = np.flatnonzero(_thin(keep, keep_frac, rng)).astype(np.int32)
        del keep
        m = sel.size
        assert m > 0, f"{stem}: nothing left after crop + thin"
        take = np.sort(np.concatenate([sel, far]))
        keep_mask = np.ones(take.size, bool)
        keep_mask[np.searchsorted(take, far)] = False
        assert int(keep_mask.sum()) == m, f"{stem}: gate rows overlap the crop box"

        xyz = np.empty((m, 3), np.float32)
        for a in range(0, m, _GATHER_ROWS):
            xyz[a:a + _GATHER_ROWS] = pts[sel[a:a + _GATHER_ROWS]]
        del pts, sel
        out = np.empty((m, len(cols)), np.float32)
        out[:, 0:3] = xyz
        del xyz

        vel = vtu.read_array(VEL_FIELD, take=take)
        assert n_far >= FAR_MIN_POINTS, f"{stem}: only {n_far} far-field points"
        far_vel = vel[~keep_mask]
        check_freestream_velocity(np.asarray(far_vel, dtype=np.float64).mean(0), UREF, aoa,
                                  FREESTREAM_TOL, stem, "far-field")
        _copy_masked(out[:, 3:6], vel, keep_mask)
        del vel
        if extra:
            t = vtu.read_array(T_FIELD, take=take)          # avg(T) precedes avg(P)
        p = vtu.read_array(VOL_P_FIELD, take=take)
        _copy_masked(out[:, 6:7], p, keep_mask)
        del p
        out[:, 6] -= np.float32(P_INF)
        out[:, 6] /= np.float32(QREF)
        if extra:
            rho = vtu.read_array(VOL_RHO_FIELD, take=take)
            _copy_masked(out[:, 7:8], rho, keep_mask)
            del rho
            out[:, 7] /= np.float32(RHO_INF)
            _copy_masked(out[:, 8:9], t, keep_mask)
            del t
            out[:, 8] /= np.float32(T_INF)

    assert np.isfinite(out).all(), f"{stem}: non-finite values in output"
    _save_shuffled(out_path, out, rng)
    del out
    if delete_src and path.endswith(".tgz"):
        os.unlink(path)
    print(f"  {stem}: {n_raw:,} -> {n_in:,} cropped ({100 * frac:.2f}%) -> {m:,} kept", flush=True)
    return _row(stem, m, n_raw=n_raw, paired=paired)


def probe(args):
    """Report one archive's block layout and far-field state; writes nothing."""
    surf = os.path.join(args.surface or os.path.join(args.root, "collated"), "samples")
    path = archive_path(args.root, args.stem)
    assert path is not None, f"{args.stem}: no archive"
    lo, hi, centre = crop_box(surf, args.stem)
    with AppendedVTU(path) as vtu:
        print(f"{args.stem}: {vtu.member} ({vtu.member_size / 1e9:.2f} GB), "
              f"{vtu.n_points:,} points / {vtu.n_cells:,} cells")
        for b in sorted(vtu.blocks.values(), key=lambda b: b.offset):
            print(f"    {b.name:14s} {b.dtype.str:5s} x{b.ncomp}  offset {b.offset / 1e9:8.2f} GB")
        pts = vtu.read_array(POINTS)
        inside, far_mask = _crop_and_far_masks(pts, lo, hi, centre.astype(np.float32), FAR_RADIUS)
        print(f"  crop keeps {100 * inside.mean():.4f}%; far field {int(far_mask.sum()):,} points")
        far = np.flatnonzero(far_mask)
        take = far if far.size <= FAR_MAX_GATE else np.sort(
            np.random.default_rng(SEED).choice(far, FAR_MAX_GATE, replace=False))
        del pts, inside, far_mask, far
        for name in (VEL_FIELD, T_FIELD, VOL_P_FIELD, VOL_RHO_FIELD):
            v = vtu.read_array(name, take=take).astype(np.float64)
            print(f"  {name:10s} far-field mean {np.round(v.mean(0), 6)} std {np.round(v.std(0), 6)}")


def volume(args):
    out = args.out or os.path.join(args.root, "volume_collated")
    surf = args.surface or os.path.join(args.root, "collated")
    samples, surf_samples = os.path.join(out, "samples"), os.path.join(surf, "samples")
    os.makedirs(samples, exist_ok=True)
    row = lambda s, n: _row(s, n, n_raw=None,  # noqa: E731
                            paired=os.path.exists(os.path.join(surf_samples, s + ".npy")))
    if args.only:
        path = archive_path(args.root, args.only)
        if path is None:
            print(f"SKIP {args.only}: no volume archive")
            return
        print(f"OK {process_volume(args.only, path, None, samples, surf_samples, args.keep_frac, args.extra, args.force, args.delete_src)}")
        return
    if args.finalize:
        write_manifest(out, rows_from_disk(samples, _RUN_RE, row))
    elif not args.stats_only:
        runs = list_volume_runs(args.root, surf_samples, args.include_unpaired)[:args.limit]
        res, errors = run_pool(process_volume, [(s, p, pr, samples, surf_samples, args.keep_frac,
                                                 args.extra, args.force, args.delete_src)
                                                for s, p, pr in runs], args.workers, every=1)
        merge_manifest(out, _RUN_RE, row, res, errors)

    # Coverage over the "convertible" train split: an archive is still there or its sample is.
    conv = {s for s, *_ in list_volume_runs(args.root, surf_samples, True)}
    conv |= {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(samples, "*.npy"))}
    split = read_split(os.path.join(surf, "splits"), "train")
    train = [s for s in split if os.path.exists(os.path.join(samples, s + ".npy"))]
    have = len(train) / max(1, len([s for s in split if s in conv]))
    if not train or have < args.stats_coverage:
        print(f"skipping norm_stats_volume ({100 * have:.1f}% of the convertible train split)")
        return
    ncol = int(np.load(os.path.join(samples, train[0] + ".npy"), mmap_mode="r").shape[1])
    cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + parse_stem(t)[2])
                              for t in train], args.stats_points, slice(3, ncol), args.stats_workers)
    save_volume_stats(os.path.join(out, "norm_stats_volume.npz"), cnt, s, ss,
                      (VOL_COLS + EXTRA_COLS)[3:ncol], extra=True,
                      points_per_sample=args.stats_points)


def _surface_args(p):
    cli.add_build_args(p, 8)
    p.add_argument("--only", default=None, help="process one stem (e.g. geo_LHC039_AoA_6)")
    p.add_argument("--finalize", action="store_true",
                   help="manifest/splits/metadata/stats from samples on disk")
    p.add_argument("--splits-only", action="store_true",
                   help="skip conversion; rebuild splits + metadata + stats from the manifest")
    p.add_argument("--factor", type=int, default=FACTOR,
                   help="keep the leading n // factor shuffled rows (collated/ uses 2)")
    p.add_argument("--stats-points", type=int, default=STATS_POINTS)
    p.add_argument("--stats-workers", type=int, default=8)


def _volume_args(p):
    cli.add_volume_args(p, 4)
    p.add_argument("--only", default=None, help="stream one stem and exit")
    p.add_argument("--finalize", action="store_true", help="manifest + stats from samples on disk")
    p.add_argument("--keep-frac", type=float, default=KEEP_FRAC,
                   help="fraction of cropped points kept (Bernoulli)")
    p.add_argument("--extra", action="store_true", help="also store rho_tilde, t_tilde ([n, 9])")
    p.add_argument("--include-unpaired", action="store_true",
                   help="also convert stems with no collated surface sample")
    p.add_argument("--delete-src", action="store_true",
                   help="DESTRUCTIVE: unlink each .tgz once its sample is written")
    p.add_argument("--stats-points", type=int, default=STATS_POINTS)
    p.add_argument("--stats-workers", type=int, default=8)
    p.add_argument("--stats-coverage", type=float, default=1.0)


def _probe_args(p):
    p.add_argument("stem")
    p.add_argument("--surface", default=None, help="paired surface tree (default <root>/collated)")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from boundary_*.vtu"),
        "volume": (_volume_args, volume, "stream volume_collated/ out of volume_*.vtu.tgz"),
        "probe": (_probe_args, probe, "print one archive's layout and far-field state"),
    })
