"""WindsorML (frame: x streamwise, y vertical, z width): run_<N>/{boundary,volume}_<N>.vtu.
surface -> [n_points, 9] x y z cp cfx cfy cfz cpvar yPlus   (point data, already coefficients)
volume  -> [n_cells, 7]  x y z ux uy uz p                   (cell centres)
"""
import os
import re

import numpy as np
import pyvista as pv

from .utils import cli, simple
from .utils.geometry import box, in_box, surface_bbox
from .utils.io import list_run_files, n_rows, save_npy
from .utils.shuffle import shuffle_rows
from .utils.vtk import f32

NAME = "windsorml"
SEED = 0
_RUN_RE = re.compile(r"run_(\d+)$")

P_FIELD = "cpavg"
WSS_FIELDS = ["cfxavg", "cfyavg", "cfzavg"]
EXTRA_FIELDS = ["cpvar", "yplusavg"]
COLS = ["x", "y", "z", "cp", "cfx", "cfy", "cfz", "cpvar", "yPlus"]

VEL_FIELDS = ["velocityxavg", "velocityyavg", "velocityzavg"]
VOL_P_FIELD = "pressureavg"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "p"]
# [xmin - Lx, xmax + 3Lx] x [ymin, ymax + Ly] x [zmin - Lz, zmax + Lz]: what volume_collated/ was
# built with (wider than the other car campaigns' 0.5/2.0, 0/0.5, 0.75 margins).
CROP_NEG = np.array([1.0, 0.0, 1.0])
CROP_POS = np.array([3.0, 1.0, 1.0])


def process_surface(stem, run_id, path, samples_dir, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path))
    mesh = pv.read(path)
    pts = f32(mesh.points)
    cp = f32(mesh.point_data[P_FIELD], 1)
    wss = np.stack([np.asarray(mesh.point_data[f], dtype=np.float32) for f in WSS_FIELDS], axis=1)
    extra = [f32(mesh.point_data[f], 1) for f in EXTRA_FIELDS]
    out = np.concatenate([pts, cp, wss, *extra], axis=1).astype(np.float32)
    assert out.shape == (pts.shape[0], len(COLS)), f"{stem}: bad shape {out.shape}"
    out = shuffle_rows(out, SEED + run_id)
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_points=out.shape[0])


def process_volume(stem, run_id, path, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_cells=n_rows(out_path))
    smn, smx = surface_bbox(surface_samples, stem)
    lo, hi, L = box(smn, smx, CROP_NEG, CROP_POS)
    assert abs(smn[1]) < 0.05 * L[1], f"{stem}: expected floor at y~0, got ymin={smn[1]}"

    mesh = pv.read(path)
    centers = mesh.cell_centers().points.astype(np.float32)
    u = np.stack([np.asarray(mesh.cell_data[f], dtype=np.float32) for f in VEL_FIELDS], axis=1)
    p = np.asarray(mesh.cell_data[VOL_P_FIELD], dtype=np.float32).reshape(-1, 1)
    out = np.concatenate([centers, u, p], axis=1)
    del mesh, centers, u, p
    out = shuffle_rows(out[in_box(out[:, :3], lo, hi)], SEED + run_id)
    assert out.shape[1] == len(VOL_COLS), f"{stem}: bad shape {out.shape}"
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_cells=out.shape[0])


def surface(args):
    simple.build_surface(args, lambda: list_run_files(args.root, _RUN_RE, "boundary_*.vtu"),
                         process_surface, COLS, _RUN_RE, seed=SEED)


def volume(args):
    simple.build_volume(args, lambda: list_run_files(args.root, _RUN_RE, "volume_*.vtu"),
                        process_volume, VOL_COLS, _RUN_RE, count_key="n_cells")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (simple.surface_args(32), surface, "build collated/ from boundary_*.vtu"),
        "volume": (simple.volume_args(4), volume, "build volume_collated/ from volume_*.vtu"),
    })
