"""DrivAerML: run_<N>/boundary_<N>.vtp (cell data), volume_<N>.vtu split into .part files.
surface -> [n_cells, 9]  x y z p tau_x tau_y tau_z Cp pPrime2   (kinematic)
volume  -> [n_points, 7] x y z ux uy uz p
"""
import glob
import os
import re
import shutil
import tempfile

import numpy as np
import pyvista as pv

from .utils import cli, simple
from .utils.geometry import box, detect_half, in_box, mirror_full, surface_bbox
from .utils.io import list_run_files, n_rows, save_npy
from .utils.shuffle import shuffle_rows
from .utils.vtk import f32

NAME = "drivaerml"
SEED = 0
_RUN_RE = re.compile(r"run_(\d+)$")

P_FIELD = "pMeanTrim"
WSS_FIELD = "wallShearStressMeanTrim"
EXTRA_FIELDS = ["CpMeanTrim", "pPrime2MeanTrim"]
COLS = ["x", "y", "z", "p", "tau_x", "tau_y", "tau_z", "Cp", "pPrime2"]

VEL_FIELD = "UMeanTrim"
VOL_P_FIELD = "pMeanTrim"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "p"]
# Crop as multiples of the surface bbox (x streamwise/wake +x, y width, z vertical; the
# floor is at z ~ -0.318 = surface zmin, so no downward extension).
CROP_NEG = np.array([0.5, 0.75, 0.0])
CROP_POS = np.array([2.0, 0.75, 0.5])
_COPY_BUF = 64 * 1024 * 1024


def process_surface(stem, run_id, path, samples_dir, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path))
    mesh = pv.read(path)
    pts = f32(mesh.cell_centers().points)
    p = f32(mesh.cell_data[P_FIELD], 1)
    wss = f32(mesh.cell_data[WSS_FIELD], 3)
    extra = [f32(mesh.cell_data[f], 1) for f in EXTRA_FIELDS]
    out = np.concatenate([pts, p, wss, *extra], axis=1).astype(np.float32)
    assert out.shape == (pts.shape[0], len(COLS)), f"{stem}: bad shape {out.shape}"
    out = shuffle_rows(out, SEED + run_id)
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_points=out.shape[0])


def list_volume_runs(root):
    """(stem, run_id, paths): a finalized volume_<N>.vtu, else its ordered .part files."""
    runs = []
    for d in sorted(os.listdir(root)):
        m = _RUN_RE.match(d)
        if not m:
            continue
        whole = sorted(glob.glob(os.path.join(root, d, "volume_*.vtu")))
        parts = sorted(glob.glob(os.path.join(root, d, "volume_*.vtu.*.part")))
        if whole or parts:
            runs.append((d, int(m.group(1)), [whole[0]] if whole else parts))
    return sorted(runs, key=lambda r: r[1])


def reconstruct(paths, scratch):
    """Concatenate .part files into a temp .vtu under `scratch` -> (path, is_temp)."""
    if len(paths) == 1 and paths[0].endswith(".vtu"):
        return paths[0], False
    os.makedirs(scratch, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".vtu", dir=scratch)
    try:
        with os.fdopen(fd, "wb") as dst:
            for part in paths:
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, dst, _COPY_BUF)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return tmp, True


def process_volume(stem, run_id, paths, samples_dir, surface_samples, scratch, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path), mirrored=None)
    lo, hi, L = box(*surface_bbox(surface_samples, stem), CROP_NEG, CROP_POS)
    vtu_path, is_temp = reconstruct(paths, scratch)
    try:
        mesh = pv.read(vtu_path)
        pts = f32(mesh.points)
        u = f32(mesh.point_data[VEL_FIELD], 3)
        p = f32(mesh.point_data[VOL_P_FIELD], 1)
        side = detect_half(pts[:, 1], L[1])
        out = np.concatenate([pts, u, p], axis=1)
        del mesh, pts, u, p
    finally:
        if is_temp and os.path.exists(vtu_path):
            os.remove(vtu_path)
    out = out[in_box(out[:, :3], lo, hi)]
    if side is not None:
        out = mirror_full(out, side)
    out = shuffle_rows(out, SEED + run_id)
    assert out.shape[1] == len(VOL_COLS), f"{stem}: bad shape {out.shape}"
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_points=out.shape[0], mirrored=side is not None)


def surface(args):
    simple.build_surface(args, lambda: list_run_files(args.root, _RUN_RE, "boundary_*.vtp"),
                         process_surface, COLS, _RUN_RE, seed=SEED)


def volume(args):
    simple.build_volume(args, lambda: list_volume_runs(args.root), process_volume, VOL_COLS,
                        _RUN_RE, extra_row=dict(mirrored=None), job_extra=(args.scratch,))


def _volume_args(p):
    simple.volume_args(2)(p)
    p.add_argument("--scratch", default=tempfile.gettempdir(),
                   help="local dir for the ~46 GB reconstructed .vtu (default $TMPDIR)")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (simple.surface_args(12), surface, "build collated/ from boundary_*.vtp"),
        "volume": (_volume_args, volume, "build volume_collated/ from volume_*.vtu[.part]"),
    })
