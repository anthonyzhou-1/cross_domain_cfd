"""Crop boxes, half-domain mirroring, freestream gates and force integrals."""
import os

import numpy as np

from .io import load_json

PLANE_EPS = 1e-6  # symmetry-plane nodes (y == 0) are kept once when mirroring


def surface_bbox(surface_samples, stem, rows=None):
    """(min, max) float64 of the paired surface sample (optionally its leading `rows`)."""
    path = os.path.join(surface_samples, stem + ".npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing paired surface sample {path}")
    a = np.load(path, mmap_mode="r")
    if rows is not None:
        a = a[:min(rows, a.shape[0])]
    sp = np.asarray(a[:, :3], dtype=np.float64)
    return sp.min(0), sp.max(0)


def box(smn, smx, neg, pos):
    """[smn - neg*L, smx + pos*L] per axis, L = smx - smn. Returns (lo f32, hi f32, L)."""
    L = smx - smn
    lo = smn - np.asarray(neg) * L
    hi = smx + np.asarray(pos) * L
    return lo.astype(np.float32), hi.astype(np.float32), L


def slab_box(smn, smx, neg, pos, half_frac, vert=2, ref_axes=None):
    """Box with the `vert` axis replaced by centre +/- half_frac * Lref.

    Lref = max(L[ref_axes]) (thin wings, where L[vert] is far smaller than the chord), or
    L[vert] when ref_axes is None. neg/pos entries on `vert` are ignored.
    """
    L = smx - smn
    half = half_frac * (max(L[a] for a in ref_axes) if ref_axes else L[vert])
    c = 0.5 * (smn[vert] + smx[vert])
    lo = np.array([c - half if a == vert else smn[a] - neg[a] * L[a] for a in range(3)])
    hi = np.array([c + half if a == vert else smx[a] + pos[a] * L[a] for a in range(3)])
    return lo.astype(np.float32), hi.astype(np.float32)


def in_box(pts, lo, hi):
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def check_keep(stem, n_kept, n_raw, min_frac, lo, hi):
    keep = n_kept / n_raw
    assert keep >= min_frac, (f"{stem}: crop kept only {100 * keep:.2f}% of {n_raw} points "
                              f"(lo={lo}, hi={hi})")
    return keep


def detect_half(y_raw, L_y):
    """'neg' / 'pos' for a one-sided (symmetry-plane) volume, None for a full domain."""
    ymin, ymax = float(y_raw.min()), float(y_raw.max())
    tol = 0.05 * L_y
    if ymax <= tol:
        return "neg"
    if ymin >= -tol:
        return "pos"
    return None


def mirror_full(out, side, y_col=1, uy_col=4):
    """Reflect a half-domain cloud across y = 0 (negating y and u_y) and append it."""
    if side == "neg":
        src = out[out[:, y_col] < -PLANE_EPS]
    else:
        src = out[out[:, y_col] > PLANE_EPS]
    mir = src.copy()
    mir[:, y_col] *= -1.0
    mir[:, uy_col] *= -1.0
    return np.concatenate([out, mir], axis=0)


def check_freestream_velocity(got, u_inf, aoa_deg, tol, stem, where):
    """Assert mean velocity `got` == u_inf * [cos a, 0, sin a] to relative `tol`."""
    a = np.deg2rad(float(aoa_deg))
    want = u_inf * np.array([np.cos(a), 0.0, np.sin(a)])
    got = np.asarray(got, dtype=np.float64)
    rel = float(np.linalg.norm(got - want) / u_inf)
    assert rel < tol, (f"{stem}: {where} velocity {np.round(got, 3)} vs U_inf@{aoa_deg}deg "
                       f"{np.round(want, 3)} (rel {rel:.2e} > {tol:g})")
    return rel


def lift_drag(force, aoa_deg, area_ref):
    """(CL, CD) of a force/q vector in a frame with x downstream, z up, AoA in x-z."""
    aoa = np.deg2rad(aoa_deg)
    drag_dir = np.array([np.cos(aoa), 0.0, np.sin(aoa)])
    lift_dir = np.array([-np.sin(aoa), 0.0, np.cos(aoa)])
    return float(force @ lift_dir / area_ref), float(force @ drag_dir / area_ref)


def block_force(arr):
    """F/q = sum((-cp n + cf) dA) over an [n, >=11] block (cp 3, cf 4:7, n 7:10, area 10)."""
    trac = -arr[:, 3:4] * arr[:, 7:10] + arr[:, 4:7]
    return (trac * arr[:, 10:11]).sum(0)


def load_forces(paths):
    """Numeric keys shared by every forces.json -> (cols, [N,k] f32), or (None, None)."""
    if not all(os.path.exists(p) for p in paths):
        return None, None
    recs = [load_json(p) for p in paths]
    cols = sorted(k for k in recs[0]
                  if all(k in r and isinstance(r[k], (int, float)) for r in recs))
    if not cols:
        return None, None
    return cols, np.array([[float(r[c]) for c in cols] for r in recs], np.float32)
