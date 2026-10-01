"""Submarine: sample_<N>/{merged_surfaces.vtp, merged_volumes.vtu, metadata.json}, water (rho 998).
surface -> [n_points, 7] x y z cp cf_x cf_y cf_z   (3-sigma per-sample Cf cut)
volume  -> [n_points, 7] x y z ux uy uz p          (raw m/s, Pa)
"""
import os
import re

import numpy as np
import pyvista as pv

from .utils import cli, simple
from .utils.geometry import box, in_box, surface_bbox
from .utils.io import load_json, n_rows, save_npy
from .utils.shuffle import shuffle_rows
from .utils.vtk import f32

NAME = "submarine"
SEED = 0
_RUN_RE = re.compile(r"sample_(\d+)$")

P_FIELD = "Pressure (Pa)"
WSS_FIELD = "Wall Shear Stress (N/m²)"
COLS = ["x", "y", "z", "cp", "cf_x", "cf_y", "cf_z"]
RHO = 998.0     # water; q = 0.5 * RHO * U_inf^2 with U_inf from metadata.json
P_INF = 0.0
CF_CLIP_SIGMA = 3.0  # per-sample cut on every Cf component (sharp-edge spikes)
CF = slice(4, 7)

VEL_FIELD = "Velocity (m/s)"
VOL_COLS = ["x", "y", "z", "ux", "uy", "uz", "p"]
# Submerged body: no floor, so the cross-stream crop is symmetric.
CROP_NEG = np.array([0.5, 0.75, 0.75])
CROP_POS = np.array([2.0, 0.75, 0.75])


def list_runs(root, fname):
    runs = []
    for d in sorted(os.listdir(root)):
        m = _RUN_RE.match(d)
        if m and os.path.exists(os.path.join(root, d, fname)):
            runs.append((d, int(m.group(1)), os.path.join(root, d, fname)))
    return sorted(runs, key=lambda r: r[1])


def process_surface(stem, run_id, path, samples_dir, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path))
    meta = load_json(os.path.join(os.path.dirname(path), "metadata.json"))
    q = 0.5 * RHO * float(meta["parametrization"]["simulation_params"]["speed"]) ** 2
    mesh = pv.read(path)
    pts = f32(mesh.points)
    p = f32(mesh.point_data[P_FIELD], 1)
    wss = f32(mesh.point_data[WSS_FIELD], 3)
    out = np.concatenate([pts, (p - P_INF) / q, wss / q], axis=1).astype(np.float32)
    assert out.shape == (pts.shape[0], len(COLS)), f"{stem}: bad shape {out.shape}"
    cf = out[:, CF]
    mu, sd = cf.mean(0), cf.std(0)
    out = out[np.all(np.abs(cf - mu) <= CF_CLIP_SIGMA * sd, axis=1)]
    out = shuffle_rows(out, SEED + run_id)
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_points=out.shape[0])


def process_volume(stem, run_id, path, samples_dir, surface_samples, force=False):
    out_path = os.path.join(samples_dir, stem + ".npy")
    if os.path.exists(out_path) and not force:
        return dict(stem=stem, run_id=run_id, n_points=n_rows(out_path))
    lo, hi, _ = box(*surface_bbox(surface_samples, stem), CROP_NEG, CROP_POS)
    mesh = pv.read(path)
    pts = f32(mesh.points)
    u = f32(mesh.point_data[VEL_FIELD], 3)
    p = f32(mesh.point_data[P_FIELD], 1)
    out = np.concatenate([pts, u, p], axis=1)
    del mesh, pts, u, p
    out = shuffle_rows(out[in_box(out[:, :3], lo, hi)], SEED + run_id)
    assert out.shape[1] == len(VOL_COLS), f"{stem}: bad shape {out.shape}"
    save_npy(out_path, out)
    return dict(stem=stem, run_id=run_id, n_points=out.shape[0])


def surface(args):
    simple.build_surface(args, lambda: list_runs(args.root, "merged_surfaces.vtp"),
                         process_surface, COLS, _RUN_RE, seed=SEED)


def volume(args):
    simple.build_volume(args, lambda: list_runs(args.root, "merged_volumes.vtu"),
                        process_volume, VOL_COLS, _RUN_RE, stats_key="n_cells")


if __name__ == "__main__":
    cli.main(NAME, {
        "surface": (simple.surface_args(16), surface, "build collated/ from merged_surfaces.vtp"),
        "volume": (simple.volume_args(8, runs=False), volume,
                   "build volume_collated/ from merged_volumes.vtu"),
    })
