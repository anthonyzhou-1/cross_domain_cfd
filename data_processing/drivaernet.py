"""DrivAerNet++: separate pressure / wall-shear VTKs per design; CFD/<folder>/<stem>.vtk volumes.
surface -> [n_points, 7] x y z p tau_x tau_y tau_z   (p aligned to the wss point order)
volume  -> [n_points, 7] x y z ux uy uz p             (half-domain E_* runs mirrored)
"""
import glob
import os
import random
import re

import numpy as np
import pyvista as pv

from .utils import cli, simple
from .utils.geometry import box, detect_half, in_box, mirror_full, surface_bbox
from .utils.io import n_rows, save_npy
from .utils.shuffle import shuffle_rows, stem_seed
from .utils.splits import random_split
from .utils.vtk import f32

NAME = "drivaernet"
SEED = 0
VAL_FRAC = 0.20
COLS = ["x", "y", "z", "p", "tau_x", "tau_y", "tau_z"]
_NUM_RE = re.compile(r"_(\d+)$")
_STEM_RE = re.compile(r"^[A-Z]_.*_\d+$")

VEL_FIELD = "U"
VOL_P_FIELD = "p"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "p"]
# Crop as multiples of the surface bbox (x streamwise/wake +x, y width, z vertical, floor z~0).
CROP_NEG = np.array([0.5, 0.75, 0.0])
CROP_POS = np.array([2.0, 0.75, 0.5])


def _wss_dir(root, folder):
    return os.path.join(root, "wss", folder, "WallShearStressVTK_Updated", folder)


def _prs_dir(root, folder):
    return os.path.join(root, "pressure", folder, "PressureVTK", folder)


def list_surface_runs(root, folders=None):
    """(stem, folder) for stems present in both the wss and pressure trees."""
    if not folders:
        folders = sorted(d for d in os.listdir(os.path.join(root, "wss"))
                         if os.path.isdir(os.path.join(root, "wss", d)))
    runs = []
    for folder in folders:
        wdir, pdir = _wss_dir(root, folder), _prs_dir(root, folder)
        if not (os.path.isdir(wdir) and os.path.isdir(pdir)):
            continue
        w = {f[:-4] for f in os.listdir(wdir) if f.endswith(".vtk")}
        p = {f[:-4] for f in os.listdir(pdir) if f.endswith(".vtk")}
        runs += [(stem, folder, root) for stem in sorted(w & p)]
    return runs


def align_pressure(pts_w, pts_p, p):
    """p reordered to pts_w's point order by exact coordinate match."""
    def keys(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    vw, vp = keys(pts_w), keys(pts_p)
    n = len(vw)
    if len(np.unique(vw)) != n:  # duplicate coordinates make the sort ambiguous
        from scipy.spatial import cKDTree
        d, idx = cKDTree(pts_p).query(pts_w, k=1)
        assert d.max() == 0.0 and len(np.unique(idx)) == n, "non-bijective coordinate match"
        return p[idx]
    idx = np.empty(n, dtype=np.int64)
    idx[np.argsort(vw, kind="stable")] = np.argsort(vp, kind="stable")
    assert np.array_equal(pts_w, pts_p[idx]), "coordinate sets do not match"
    return p[idx]


def _row(stem, n, folder=None):
    m = _NUM_RE.search(stem)
    return dict(stem=stem, folder=folder, prefix=stem[0],
                numeric_id=int(m.group(1)) if m else -1, n_points=n)


def process_surface(stem, folder, root, samples_dir, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return _row(stem, n_rows(out_path), folder)
    mw = pv.read(os.path.join(_wss_dir(root, folder), stem + ".vtk"))
    mp = pv.read(os.path.join(_prs_dir(root, folder), stem + ".vtk"))
    pts_w = f32(mw.points)
    wss = f32(mw.point_data["wallShearStress"])
    pts_p = f32(mp.points)
    p = f32(mp.point_data["p"]).reshape(-1)
    out = np.concatenate([pts_w, align_pressure(pts_w, pts_p, p)[:, None], wss],
                         axis=1).astype(np.float32)
    out = shuffle_rows(out, stem_seed(stem, SEED))
    save_npy(out_path, out)
    return _row(stem, out.shape[0], folder)


def make_splits(rows, keep_test=False):
    """F/N stems split 80/20; the E stems are then appended 80/20 (random.Random(42) order),
    which is what collated/splits holds. keep_test leaves E as a separate test split."""
    test = sorted(r["stem"] for r in rows if r["prefix"] == "E")
    pool = sorted(r["stem"] for r in rows if r["prefix"] in ("F", "N"))
    train, val = random_split(pool, VAL_FRAC, SEED)
    if keep_test:
        return dict(train=train, val=val, test=test)
    order = list(test)
    random.Random(42).shuffle(order)
    n_tr = len(order) - int(round(VAL_FRAC * len(order)))
    return dict(train=train + order[:n_tr], val=val + order[n_tr:], _indent=2)


def list_volume_runs(root):
    """(stem, folder, path) for every CFD/<folder>/<stem>.vtk; part1/part2 folders are disjoint."""
    runs, seen = [], {}
    cfd = os.path.join(root, "CFD")
    if not os.path.isdir(cfd):
        return runs
    for folder in sorted(os.listdir(cfd)):
        if not os.path.isdir(os.path.join(cfd, folder)):
            continue
        for path in sorted(glob.glob(os.path.join(cfd, folder, "*.vtk"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in seen:
                print(f"  WARN duplicate stem {stem}: {seen[stem]} and {path}")
                continue
            seen[stem] = path
            runs.append((stem, folder, path))
    return sorted(runs, key=lambda r: r[0])


def process_volume(stem, folder, path, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, folder=folder, n_points=n_rows(out_path), mirrored=None)
    smn, smx = surface_bbox(surface_samples, stem)
    lo, hi, L = box(smn, smx, CROP_NEG, CROP_POS)
    assert abs(smn[2]) < 0.05 * L[2], f"{stem}: expected floor at z~0, got zmin={smn[2]}"

    mesh = pv.read(path)
    pts = f32(mesh.points)
    u = f32(mesh.point_data[VEL_FIELD], 3)
    p = f32(mesh.point_data[VOL_P_FIELD], 1)
    side = detect_half(pts[:, 1], L[1])  # E_* runs are half-domain (symmetry plane y = 0)
    out = np.concatenate([pts, u, p], axis=1)
    del mesh, pts, u, p
    out = out[in_box(out[:, :3], lo, hi)]
    if side is not None:
        out = mirror_full(out, side)
    out = shuffle_rows(out, stem_seed(stem, SEED))
    assert out.shape[1] == len(VOL_COLS), f"{stem}: bad shape {out.shape}"
    save_npy(out_path, out)
    return dict(stem=stem, folder=folder, n_points=out.shape[0], mirrored=side is not None)


def surface(args):
    simple.build_surface(args, lambda: list_surface_runs(args.root, args.folders),
                         process_surface, COLS, _STEM_RE, every=250, row_fn=_row,
                         split_fn=lambda rows: make_splits(rows, args.keep_test), key="stem")


def volume(args):
    simple.build_volume(args, lambda: list_volume_runs(args.root), process_volume, VOL_COLS,
                        _STEM_RE, every=25, key="stem",
                        row_fn=lambda s, n: dict(stem=s, folder=None, n_points=n, mirrored=None))


def _surface_args(p):
    simple.surface_args(48)(p)
    p.add_argument("--folders", nargs="*", default=None, help="subset of folders (default: all)")
    p.add_argument("--keep-test", action="store_true",
                   help="keep the E_* designs as a test split instead of folding them in")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (_surface_args, surface, "build collated/ from the wss + pressure VTKs"),
        "volume": (simple.volume_args(8), volume, "build volume_collated/ from CFD/*/*.vtk"),
    })
