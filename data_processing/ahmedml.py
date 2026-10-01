"""AhmedML: run_<N>/boundary_<N>.vtp (cell data), run_<N>/volume_<N>.vtu (point data).
surface -> [n_cells, 9]  x y z p tau_x tau_y tau_z Cp yPlus   (kinematic p / wss)
volume  -> [n_points, 7] x y z ux uy uz p
"""
import os
import re

import numpy as np
import pyvista as pv

from .utils import cli, simple
from .utils.geometry import box, detect_half, in_box, mirror_full, surface_bbox
from .utils.io import list_run_files, n_rows, save_npy
from .utils.shuffle import shuffle_rows
from .utils.vtk import f32

NAME = "ahmedml"
SEED = 0
_RUN_RE = re.compile(r"run_(\d+)$")

P_FIELD = "pMean"
WSS_FIELD = "wallShearStressMean"
EXTRA_FIELDS = ["static(p)_coeffMean", "yPlusMean"]
COLS = ["x", "y", "z", "p", "tau_x", "tau_y", "tau_z", "Cp", "yPlus"]

VEL_FIELD = "UMean"
VOL_P_FIELD = "pMean"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "p"]
# Crop as multiples of the surface bbox (x streamwise/wake +x, y width, z vertical).
CROP_NEG = np.array([0.5, 0.75, 0.0])
CROP_POS = np.array([2.0, 0.75, 0.5])
FLOOR_Z = 0.0  # the body sits on stilts above the floor; the crop reaches down to it


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


def process_volume(stem, run_id, path, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path), mirrored=None)
    smn, smx = surface_bbox(surface_samples, stem)
    lo, hi, L = box(smn, smx, CROP_NEG, CROP_POS)
    assert smn[2] > FLOOR_Z, f"{stem}: expected body above floor, got zmin={smn[2]}"
    lo[2] = FLOOR_Z

    mesh = pv.read(path)
    pts = f32(mesh.points)
    u = f32(mesh.point_data[VEL_FIELD], 3)
    p = f32(mesh.point_data[VOL_P_FIELD], 1)
    side = detect_half(pts[:, 1], L[1])  # before the crop removes the far field
    out = np.concatenate([pts, u, p], axis=1)
    del mesh, pts, u, p
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
    simple.build_volume(args, lambda: list_run_files(args.root, _RUN_RE, "volume_*.vtu"),
                        process_volume, VOL_COLS, _RUN_RE, extra_row=dict(mirrored=None))


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (simple.surface_args(32), surface, "build collated/ from boundary_*.vtp"),
        "volume": (simple.volume_args(8), volume, "build volume_collated/ from volume_*.vtu"),
    })
