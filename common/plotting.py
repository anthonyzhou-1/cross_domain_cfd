import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_pointcloud(
    pos, # shape (n_points, 3) 
    color=None,
    title=None,
    alpha=0.5,
    num_points=None,
    figsize=(6, 6),
    save_path=None,
    norm_axes=False,
    view_rotation=[20, 125, 0],
    scale=None,
):
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(np.array(pos)) # convert to tensor if not already
    
    if color is not None and not isinstance(color, torch.Tensor):
        color = torch.tensor(np.array(color))

    if color is None:
        color = pos[:, -1] # use last coordinate as color if no color is provided
    if num_points is None or num_points > len(pos):
        num_points = len(pos)

    perm = torch.randperm(len(pos), generator=torch.Generator().manual_seed(0))[:num_points]
    plt.close()
    plt.clf()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = pos[perm].unbind(-1)

    if scale is not None:
        vmin, vmax = scale
    else:
        vmin, vmax = color.min(), color.max()

    if color is None:
        scatter = ax.scatter(x, y, z, s=3, c="k", alpha=alpha)
    else:
        scatter = ax.scatter(x, y, z, s=3, c=color[perm], cmap="coolwarm", alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    if norm_axes:
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1, top=1) 
    else:
        plt.axis("equal")
    ax.view_init(*view_rotation)
    if title is not None:
        ax.set_title(title)
    if color is not None:
        plt.colorbar(scatter, orientation="horizontal")
    plt.savefig(save_path, bbox_inches="tight", dpi=300) if save_path is not None else plt.show()


_AXIS = {"x": 0, "y": 1, "z": 2}

# For a slice with the given plane-normal, the two in-plane axes as
# (column index, label), ordered (horizontal, vertical) of the 2D image.
_PLANE = {
    "z": ((0, "x"), (1, "y")),   # XY plane -- side view (spanwise-normal centerplane)
    "y": ((0, "x"), (2, "z")),   # XZ plane -- top view  (vertical-normal)
    "x": ((2, "z"), (1, "y")),   # YZ plane -- front view (streamwise-normal cross-section)
}
_PLANE_NAME = {"z": "XY", "y": "XZ", "x": "YZ"}

DEFAULT_CROP = {"x": (-1.2, 2.2), "y": (0.0, 0.75), "z": (-0.45, 0.45)}


def load_volume_sample(path, max_points=4_000_000):
    """Load the first `max_points` rows of a (pre-shuffled) volume .npy -> (pos, vel, p)."""
    a = np.load(path, mmap_mode="r")
    m = a.shape[0] if max_points is None else min(max_points, a.shape[0])
    a = np.asarray(a[:m], dtype=np.float32)
    return a[:, :3], a[:, 3:6], a[:, 6]


def _robust_scale(values, diverging):
    """Robust (1-99 pct) color limits; symmetric about the median if diverging."""
    lo, hi = np.percentile(values, [1, 99])
    if diverging:
        c = float(np.median(values))
        half = max(c - lo, hi - c)
        lo, hi = c - half, c + half
    return float(lo), float(hi)


def volume_field_scale(sample, field="velocity"):
    """Robust (vmin, vmax) color limits for `field`, to share a colorbar across plots."""
    if isinstance(sample, str):
        _, vel, p = load_volume_sample(sample, None)
    else:
        arr = np.asarray(sample, dtype=np.float32)
        vel, p = arr[:, 3:6], arr[:, 6]
    values, _, _, diverging = _field_values(field, vel, p)
    return _robust_scale(values, diverging)


def _field_values(field, vel, p):
    """Map a field name to (values, label, cmap, diverging)."""
    f = field.lower()
    if f in ("velocity", "vel", "umag", "speed", "mag"):
        return np.linalg.norm(vel, axis=1), "|U| (m/s)", "viridis", False
    if f in ("pressure", "p"):
        return p, "p (Pa)", "RdBu_r", True
    if f in ("velocity_deficit", "vel_deficit", "dumag"):
        return np.linalg.norm(vel, axis=1), r"|u - u$_\infty$| / U$_\infty$", "viridis", False
    if f in ("cp", "pressure_coefficient"):
        return p, "Cp", "RdBu_r", True
    if f in ("ux", "uy", "uz"):
        return vel[:, {"ux": 0, "uy": 1, "uz": 2}[f]], f"{f} (m/s)", "RdBu_r", True
    if f in ("dux", "duy", "duz"):
        j = {"dux": 0, "duy": 1, "duz": 2}[f]
        return vel[:, j], f"({f[1:]} - u$_\infty$) / U$_\infty$", "RdBu_r", True
    raise ValueError(f"unknown field {field!r}")


def _slab_points(pos, values, normal, loc, thickness, crop):
    """Select the in-plane coords + field of points in a slab around a plane.

    Returns (h, v, val, (h0, h1, v0, v1), (hlab, vlab)): the horizontal/vertical
    coordinates and field values of points within `thickness` of the plane and
    inside the `crop` window, plus the view bounds and axis labels.
    """
    ax = _AXIS[normal]
    (hi, hlab), (vi, vlab) = _PLANE[normal]

    m = np.abs(pos[:, ax] - loc) <= 0.5 * thickness
    h, v, val = pos[m, hi], pos[m, vi], values[m]

    crop = crop or {}
    h0, h1 = crop.get(hlab, (h.min(), h.max())) if h.size else (0.0, 1.0)
    v0, v1 = crop.get(vlab, (v.min(), v.max())) if v.size else (0.0, 1.0)
    win = (h >= h0) & (h <= h1) & (v >= v0) & (v <= v1)
    h, v, val = h[win], v[win], val[win]
    if h.size < 16:
        raise ValueError(f"only {h.size} points in {_PLANE_NAME[normal]} slab at "
                         f"{normal}={loc:.4g}; widen `thickness`/`crop` or add points")
    return h, v, val, (h0, h1, v0, v1), (hlab, vlab)


def _default_loc(pos, normal):
    """Default slice plane: centerplane for z, through the body for y, near-wake for x."""
    axi = _AXIS[normal]
    if normal == "z":
        return 0.0                                   # spanwise symmetry plane
    if normal == "y":
        return float(np.percentile(pos[:, axi], 30))  # cut through the body
    return float(np.median(pos[:, axi]))              # x: just behind the body


def plot_volume_scatter(ax, pos, values, normal, loc=None, thickness=0.1,
                        crop=None, cmap="viridis", vmin=None, vmax=None,
                        s=1.0, max_scatter=1_000_000, rng_seed=0):
    """Scatter the slab points around one plane on `ax`, subsampled to `max_scatter`."""
    if loc is None:
        loc = _default_loc(pos, normal)
    h, v, val, (h0, h1, v0, v1), (hlab, vlab) = _slab_points(
        pos, values, normal, loc, thickness, crop)
    if max_scatter and h.size > max_scatter:
        idx = np.random.default_rng(rng_seed).choice(h.size, max_scatter, replace=False)
        h, v, val = h[idx], v[idx], val[idx]

    sc = ax.scatter(h, v, c=val, s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlim(h0, h1)
    ax.set_ylim(v0, v1)
    ax.set_xlabel(hlab)
    ax.set_ylabel(vlab)
    ax.set_title(f"{_PLANE_NAME[normal]} slice  ({normal} = {loc:.3g})")
    return sc


def plot_volume_scatter_slices(sample, field="velocity", save_path=None,
                               max_points=6_000_000, normals=("z", "y", "x"),
                               locs=None, thickness=0.02, crop="default",
                               s=1.0, max_scatter=400_000, scale=None, color_map=None):
    """Plot XY / XZ / YZ raw-point scatter slices of one volume sample, stacked.

    `sample` is a volume .npy path or an (n, >=7) array [x, y, z, ux, uy, uz, p].
    """
    if crop == "default":
        crop = DEFAULT_CROP
    if isinstance(sample, str):
        pos, vel, p = load_volume_sample(sample, max_points)
    else:
        arr = np.asarray(sample, dtype=np.float32)
        pos, vel, p = arr[:, :3], arr[:, 3:6], arr[:, 6]

    values, clabel, cmap, diverging = _field_values(field, vel, p)

    if color_map is not None:
        cmap = color_map

    vmin, vmax = scale if scale is not None else _robust_scale(values, diverging)

    locs = locs or {}
    fig, axs = plt.subplots(len(normals), 1,
                            figsize=(11, 3.4 * len(normals)),
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    sc = None
    for ax, nm in zip(axs, normals):
        sc = plot_volume_scatter(ax, pos, values, nm, loc=locs.get(nm),
                                 thickness=thickness, crop=crop, cmap=cmap,
                                 vmin=vmin, vmax=vmax, s=s, max_scatter=max_scatter)
    cbar = fig.colorbar(sc, ax=list(axs), shrink=0.8)
    cbar.set_label(clabel)
    fig.suptitle(f"{field} volume slices (raw points)", fontsize=13)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig
