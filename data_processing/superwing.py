"""SuperWing (HF yunplus/SuperWing; native frame x chord, y vertical, z span).
surface <- data_surf.npy [28856, 9, 44096] -> [44096, 9] x y z cp cf_x cf_y cf_z rho T
volume  <- data_vol.<k>.npy [n_k, 8, 3086720] -> [n, 9] x y z ux uy uz cp rho_tilde layer
prune   -> 3-sigma cut on cf, columns named in the LOADER frame (loader cf_y = disk cf_z)
"""
import os
import re

import numpy as np

from .utils import cli
from .utils.geometry import check_keep, in_box, slab_box
from .utils.io import (all_on_disk, load_json, merge_manifest, n_rows, read_split,
                       rows_from_disk, save_npy, write_manifest, write_splits)
from .utils.pool import run_pool
from .utils.prune import run_prune
from .utils.shuffle import shuffle_rows
from .utils.splits import grouped_split
from .utils.stats import pooled_sums, save_surface_stats, save_volume_stats, serial_sums

NAME = "superwing"
SEED = 0
VAL_FRAC = 0.20
_STEM_RE = re.compile(r"^sample_(\d+)$")

RAW_COLS = ["x", "y", "z", "rho", "cp", "cf_x", "cf_y", "cf_z", "T"]
COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z", "rho", "T"]
REORDER = [RAW_COLS.index(c) for c in COLS]
INDEX_COLS = ["geom_id", "cond_id", "aoa", "mach", "s_half", "b_half",
              "cl_cfd", "cd_cfd", "cmz_cfd", "cl_ref", "cd_ref", "cmz_ref"]

N_SURF, N_LAYER = 44096, 70
N_VOL = N_SURF * N_LAYER
GAMMA = 1.4
VOL_RAW_COLS = ["x", "y", "z", "rho_tilde", "p_tilde", "ux", "uy", "uz"]
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "cp", "rho_tilde", "layer"]
VOL_REORDER = [VOL_RAW_COLS.index(c) for c in VOL_COLS[:6] + ["p_tilde", "rho_tilde"]]
CP_COL, RHO_COL, LAYER_COL = 6, 7, 8
# Structured blocks of the volume cloud: (surface offset, n_cells, volume offset); within a
# block the flattening is layer-major, vol_off + layer * n_cells + (cell - surf_off).
BLOCKS = [(0, 4480, 0), (4480, 320, 649600), (4800, 13888, 672000), (18688, 992, 1644160),
          (19680, 13888, 1713600), (33568, 992, 2685760), (34560, 1792, 2755200),
          (36352, 128, 2880640), (36480, 1792, 2889600), (38272, 128, 3015040),
          (38400, 896, 3024000), (39296, 320, 313600), (39616, 4480, 336000)]
SHARD_ROWS = [744, 739, 705, 678, 656, 689, 648, 666, 644, 654, 694, 647, 692, 694, 731,
              723, 700, 714, 671, 680, 680, 650, 667, 645, 667, 665, 674, 717, 598, 538,
              631, 575, 575, 605, 583, 643, 629, 520, 579, 572, 447, 469, 539, 500, 545, 187]
N_SHARD = len(SHARD_ROWS)
DUP_SHARD, MISSING_ROWS, N_SAMPLES = 32, 562, 28856
# x/z margins on the surface bbox; y = centre +/- 0.20 * max(Lx, Lz) (the wing is ~0.36 thick).
CROP_NEG = [0.25, None, 0.15]
CROP_POS = [0.75, None, 0.02]
Y_HALF_FRAC = 0.20
KEEP_FRAC_MIN = 0.75          # measured 0.814-0.838
ALIGN_POS_TOL, ALIGN_CP_TOL = 1e-4, 1e-3   # measured 1.1e-6 / 1.7e-7
STATS_COVERAGE_MIN = 0.95     # the shard-32 hole caps coverage at ~98.1%


def _stem(idx):
    return f"sample_{idx:05d}"


def _run_id(stem):
    return int(_STEM_RE.match(stem).group(1))


# --- surface ------------------------------------------------------------------------------

_SRC = None


def _init_surface(src):
    global _SRC
    _SRC = np.load(src, mmap_mode="r")


def process_surface(idx, samples_dir, force=False):
    stem = _stem(idx)
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem, idx, n_rows(out_path)
    raw = np.asarray(_SRC[idx], dtype=np.float32)                   # [9, 44096]
    out = shuffle_rows(np.ascontiguousarray(raw[REORDER].T, dtype=np.float32), SEED + idx)
    save_npy(out_path, out)
    return stem, idx, out.shape[0]


def _surface_row(index):
    def row(stem, n):
        meta = index[_run_id(stem)]
        return dict(stem=stem, run_id=_run_id(stem), n_points=n, geom_id=int(meta[0]),
                    cond_id=int(meta[1]), aoa=float(meta[2]), mach=float(meta[3]))
    return row


def surface(args):
    out = args.out or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    else:
        src = args.src or os.path.join(args.root, "data_surf", "data_surf.npy")
        index = np.load(args.index or os.path.join(args.root, "data_surf", "index.npy"))
        assert index.ndim == 2 and index.shape[1] == len(INDEX_COLS), f"index {index.shape}"
        n_total = int(np.load(src, mmap_mode="r").shape[0])
        assert index.shape[0] == n_total, f"index rows {index.shape[0]} != samples {n_total}"
        n = n_total if args.limit is None else min(args.limit, n_total)
        res, errors = run_pool(process_surface, [(i, samples, args.force) for i in range(n)],
                               args.workers, every=500, initializer=_init_surface,
                               initargs=(src,))
        row = _surface_row(index)
        rows = merge_manifest(out, _STEM_RE, row, [row(s, k) for s, _, k in res], errors)
        np.savez(os.path.join(out, "metadata.npz"), index=index, cols=np.array(INDEX_COLS))
    train, val = grouped_split(rows, "geom_id", VAL_FRAC, SEED)
    write_splits(os.path.join(out, "splits"), train=train, val=val)
    if all_on_disk(samples, train):
        save_surface_stats(os.path.join(out, "norm_stats.npz"),
                           *serial_sums([os.path.join(samples, s + ".npy") for s in train]), COLS)


# --- volume -------------------------------------------------------------------------------

def _build_tables():
    """LAYER[N_VOL] (col 8), WALL_ROWS[N_SURF] (volume row of each wall cell), SHARD_BASE."""
    layer = np.full(N_VOL, -1, np.int16)
    wall = np.full(N_SURF, -1, np.int32)
    for surf_off, n_b, vol_off in BLOCKS:
        layer[vol_off:vol_off + n_b * N_LAYER] = np.repeat(np.arange(N_LAYER, dtype=np.int16), n_b)
        wall[surf_off:surf_off + n_b] = np.arange(vol_off, vol_off + n_b, dtype=np.int32)
    assert (layer >= 0).all() and (wall >= 0).all() and sum(b[1] for b in BLOCKS) == N_SURF
    base, cur = [], 0
    for s in range(N_SHARD):
        if s == DUP_SHARD:
            base.append(None)
            cur += MISSING_ROWS
            continue
        base.append(cur)
        cur += SHARD_ROWS[s]
    assert sum(SHARD_ROWS) == 28869 and SHARD_ROWS[DUP_SHARD] == SHARD_ROWS[DUP_SHARD - 1]
    assert cur == N_SAMPLES, f"shard rows end at {cur}, expected {N_SAMPLES}"
    return layer, wall, base


LAYER, WALL_ROWS, SHARD_BASE = _build_tables()
_INDEX = None
_SHARD = {}


def _init_volume(metadata):
    global _INDEX
    _INDEX = np.load(metadata)["index"]
    _SHARD.clear()


def shard_path(src_dir, shard):
    return os.path.join(src_dir, f"data_vol.{shard}.npy")


def _shard_mm(src_dir, shard):
    """Memmap of one shard; at most one is held so --delete-src can actually free space."""
    if shard not in _SHARD:
        _SHARD.clear()
        _SHARD[shard] = np.load(shard_path(src_dir, shard), mmap_mode="r")
    return _SHARD[shard]


def shard_row(run_id):
    for s in range(N_SHARD):
        if SHARD_BASE[s] is not None and SHARD_BASE[s] <= run_id < SHARD_BASE[s] + SHARD_ROWS[s]:
            return s, run_id - SHARD_BASE[s]
    return None, None   # inside the shard-32 hole


def shard_stems(shard):
    return [_stem(SHARD_BASE[shard] + r) for r in range(SHARD_ROWS[shard])]


def _outputs_complete(samples_dir, shard):
    for stem in shard_stems(shard):
        try:
            shape = np.load(os.path.join(samples_dir, stem + ".npy"), mmap_mode="r").shape
        except Exception:  # noqa: BLE001 - missing / truncated
            return False
        if len(shape) != 2 or shape[1] != len(VOL_COLS) or shape[0] == 0:
            return False
    return True


def load_surface(surface_samples, stem):
    """Paired surface sample un-shuffled back to mesh order (needs the unpruned file)."""
    a = np.load(os.path.join(surface_samples, stem + ".npy"))
    assert a.shape == (N_SURF, len(COLS)), f"{stem}: surface shape {a.shape} (pruned?)"
    perm = np.random.default_rng(SEED + _run_id(stem)).permutation(N_SURF)
    return a[np.argsort(perm)]


def process_volume(shard, row, src_dir, samples_dir, surface_samples, force=False):
    run_id = SHARD_BASE[shard] + row
    stem = _stem(run_id)
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return stem, n_rows(out_path)
    surf = load_surface(surface_samples, stem)
    sp = np.asarray(surf[:, :3], dtype=np.float64)
    lo, hi = slab_box(sp.min(0), sp.max(0), CROP_NEG, CROP_POS, Y_HALF_FRAC, vert=1,
                      ref_axes=(0, 2))
    mach = float(_INDEX[run_id, 3])
    assert mach > 0, f"{stem}: bad Mach {mach}"

    raw = np.asarray(_shard_mm(src_dir, shard)[row], dtype=np.float32)   # [8, N_VOL]
    assert raw.shape == (8, N_VOL), f"{stem}: raw shape {raw.shape}"
    out = np.empty((N_VOL, len(VOL_COLS)), np.float32)
    for j, src in enumerate(VOL_REORDER):
        out[:, j] = raw[src]
    del raw
    out[:, LAYER_COL] = LAYER
    out[:, CP_COL] = (out[:, CP_COL] - 1.0) / np.float32(0.5 * GAMMA * mach ** 2)

    # The wall layer IS the paired surface sample: pins the (shard, row) -> stem mapping.
    wall = out[WALL_ROWS]
    dpos = float(np.abs(wall[:, :3] - surf[:, :3]).max())
    dcp = float(np.abs(wall[:, CP_COL] - surf[:, COLS.index("cp")]).max())
    drho = float(np.abs(wall[:, RHO_COL] - surf[:, COLS.index("rho")]).max())
    assert dpos < ALIGN_POS_TOL and dcp < ALIGN_CP_TOL and drho < ALIGN_CP_TOL, (
        f"{stem}: wall layer != surface (dpos={dpos:.2e} dcp={dcp:.2e} drho={drho:.2e})")
    del surf

    out = out[in_box(out[:, :3], lo, hi)]
    check_keep(stem, out.shape[0], N_VOL, KEEP_FRAC_MIN, lo, hi)
    assert np.isfinite(out).all(), f"{stem}: non-finite values in output"
    out = shuffle_rows(out, SEED + run_id)
    save_npy(out_path, out)
    return stem, out.shape[0]


def _volume_row(stem, n):
    run_id = _run_id(stem)
    meta = _INDEX[run_id]
    shard, row = shard_row(run_id)
    return dict(stem=stem, run_id=run_id, n_points=n, geom_id=int(meta[0]),
                cond_id=int(meta[1]), aoa=float(meta[2]), mach=float(meta[3]),
                shard=shard, row=row)


def build_volume(args, src_dir, out, samples, surface_samples, metadata, shards):
    """Shard by shard, a fresh pool each, so --delete-src can unlink a finished shard."""
    total = sum(SHARD_ROWS[s] for s in shards)
    total = total if args.limit is None else min(total, args.limit)
    rows, errors, done, freed = [], [], 0, 0
    for s in shards:
        todo = list(range(SHARD_ROWS[s]))
        if args.limit is not None:
            todo = todo[:max(0, args.limit - done)]
            if not todo:
                break
        if not os.path.exists(shard_path(src_dir, s)):
            if _outputs_complete(samples, s):   # consumed by an earlier --delete-src run
                rows += [_volume_row(st, n_rows(os.path.join(samples, st + ".npy")))
                         for st in shard_stems(s)]
                done += SHARD_ROWS[s]
                continue
            raise FileNotFoundError(f"shard {s} is missing and its samples are incomplete")
        shape = np.load(shard_path(src_dir, s), mmap_mode="r").shape
        assert shape == (SHARD_ROWS[s], 8, N_VOL), f"shard {s}: on-disk shape {shape}"
        res, err = run_pool(process_volume, [(s, r, src_dir, samples, surface_samples, args.force)
                                             for r in todo], args.workers, every=250,
                            initializer=_init_volume, initargs=(metadata,), label=f"shard {s}: ")
        rows += [_volume_row(st, n) for st, n in res]
        errors += err
        done += len(todo)
        if args.delete_src:
            if not err and _outputs_complete(samples, s):
                for k in ([s, DUP_SHARD] if s == DUP_SHARD - 1 else [s]):
                    if os.path.exists(shard_path(src_dir, k)):
                        freed += os.path.getsize(shard_path(src_dir, k))
                        os.remove(shard_path(src_dir, k))
                        print(f"  deleted data_vol.{k}.npy")
            else:
                print(f"  KEEPING shard {s}: outputs incomplete")
    if freed:
        print(f"freed {freed / 1e12:.2f} TB of source shards")
    return rows, errors


def volume(args):
    out = args.out or os.path.join(args.root, "volume_collated")
    surf = args.surface or os.path.join(args.root, "collated")
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    src_dir = os.path.join(args.root, "data_vol")
    metadata = os.path.join(surf, "metadata.npz")
    _init_volume(metadata)
    if args.only:
        shard, row = shard_row(_run_id(args.only))
        if shard is None:
            print(f"SKIP {args.only}: inside the shard-{DUP_SHARD} hole")
            return
        stem, n = process_volume(shard, row, src_dir, samples, os.path.join(surf, "samples"),
                                 args.force)
        print(f"OK {stem}: {n} points (shard {shard} row {row})")
        return
    if args.finalize:
        write_manifest(out, rows_from_disk(samples, _STEM_RE, _volume_row))
    elif not args.stats_only:
        shards = [s for s in range(N_SHARD) if SHARD_BASE[s] is not None
                  and (not args.shards or s in args.shards)]
        rows, errors = build_volume(args, src_dir, out, samples, os.path.join(surf, "samples"),
                                    metadata, shards)
        merge_manifest(out, _STEM_RE, _volume_row, rows, errors)

    train = read_split(os.path.join(surf, "splits"), "train")
    have = [s for s in train if os.path.exists(os.path.join(samples, s + ".npy"))]
    cover = len(have) / max(1, len(train))
    if cover < STATS_COVERAGE_MIN:
        print(f"skipping norm_stats_volume ({len(have)}/{len(train)} train samples on disk)")
        return
    cnt, s, ss = pooled_sums([(os.path.join(samples, t + ".npy"), SEED + _run_id(t))
                              for t in have], args.stats_points, slice(3, 8),
                             args.stats_workers, every=2000)
    save_volume_stats(os.path.join(out, "norm_stats_volume.npz"), cnt, s, ss, VOL_COLS[3:8],
                      extra=True, points_per_sample=args.stats_points)


# --- prune --------------------------------------------------------------------------------

def _loader_to_disk(m):
    """{loader cf name: disk cf name} for the signed permutation ORIENT (cf_new = cf @ m)."""
    assert np.allclose(m @ m.T, np.eye(3)) and (np.count_nonzero(m, axis=0) == 1).all()
    out = {"cp": "cp"}
    for j, name in enumerate(("cf_x", "cf_y", "cf_z")):
        out[name] = f"cf_{'xyz'[int(np.flatnonzero(m[:, j])[0])]}"
    return out


def _update_keep_mask(out_dir):
    """Compose this pass's keep masks into pruned_keep_mask.npy: [run_id, N_SURF/8] packed
    bits over the ORIGINAL disk rows (mesh cell of surviving row i = perm[keep][i])."""
    path = os.path.join(out_dir, "pruned_keep_mask.npy")

    def post(results, manifest):
        n_ids = max(r["run_id"] for r in manifest) + 1
        cum = (np.load(path) if os.path.exists(path)
               else np.full((n_ids, N_SURF // 8), 255, np.uint8))
        assert cum.shape == (n_ids, N_SURF // 8), f"{path} has shape {cum.shape}"
        for r in results.values():
            if r["skipped"] or r["keep"] is None:
                continue
            rid = _run_id(r["stem"])
            prev = np.unpackbits(cum[rid]).astype(bool)
            assert int(prev.sum()) == r["n_before"], f"{r['stem']}: keep mask out of sync"
            idx = np.flatnonzero(prev)
            prev[idx[~np.unpackbits(r["keep"]).astype(bool)[:r["n_before"]]]] = False
            cum[rid] = np.packbits(prev)
        save_npy(path, cum)
        print(f"wrote {path}")
    return post


def prune(args):
    from data.surface_volume_dataset import ORIENT
    out = args.out or os.path.join(args.root, "collated")
    cols = list(dict.fromkeys(args.columns))
    m2d = _loader_to_disk(ORIENT["superwing"].numpy())
    disk = [m2d[c] for c in cols]
    print("loader -> disk columns: " + ", ".join(f"{c}={d}" for c, d in zip(cols, disk)))
    run_prune(out, COLS, disk, args.sigma, log_columns=cols, log_meta=dict(disk_columns=disk),
              workers=args.workers, dry_run=args.dry_run, force=args.force, limit=args.limit,
              want_mask=True, post_fn=_update_keep_mask(out))


def _surface_args(p):
    cli.add_build_args(p, 48)
    p.add_argument("--src", default=None, help="default <root>/data_surf/data_surf.npy")
    p.add_argument("--index", default=None, help="default <root>/data_surf/index.npy")
    p.add_argument("--splits-only", action="store_true",
                   help="skip conversion; rebuild splits + stats from the manifest")


def _volume_args(p):
    cli.add_volume_args(p, 32)
    p.add_argument("--only", default=None, help="process one stem (e.g. sample_00000) and exit")
    p.add_argument("--shards", type=int, nargs="+", default=None, help="only these shards")
    p.add_argument("--finalize", action="store_true", help="manifest + stats from samples on disk")
    p.add_argument("--delete-src", action="store_true",
                   help="DESTRUCTIVE: unlink each shard once all its samples are written")
    p.add_argument("--stats-points", type=int, default=100_000,
                   help="rows per sample for norm_stats_volume.npz (0 = all)")
    p.add_argument("--stats-workers", type=int, default=16)


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from data_surf.npy + index.npy"),
        "volume": (_volume_args, volume, "build volume_collated/ from data_vol shards"),
        "prune": (lambda p: cli.add_prune_args(p, ["cf_x", "cf_y", "cf_z"],
                                               ["cp", "cf_x", "cf_y", "cf_z"], 3.0),
                  prune, "per-sample sigma cut of the surface Cf tail, in place (irreversible)"),
    })
