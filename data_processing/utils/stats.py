"""Build-time per-column mean/std (norm_stats.npz / norm_stats_volume.npz)."""
import numpy as np

from .pool import run_pool


def moments(cnt, s, ss):
    mean = s / cnt
    std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 0.0))
    std[std == 0] = 1.0
    return mean, std


def serial_sums(paths, cols=slice(None), max_points=0):
    """(count, sum, sumsq) in float64 over each file's leading <=max_points rows (0 = all)."""
    cnt, s, ss = 0, 0.0, 0.0
    for path in paths:
        a = np.load(path, mmap_mode="r")
        if max_points and a.shape[0] > max_points:
            a = a[:max_points]
        a = np.asarray(a[:, cols], dtype=np.float64)
        cnt += a.shape[0]
        s = s + a.sum(0)
        ss = ss + (a * a).sum(0)
    return cnt, s, ss


def _window_sums(path, n_points, seed, cols):
    """Sums over a contiguous <=n_points window at a seeded start (rows are preshuffled)."""
    a = np.load(path, mmap_mode="r")
    n = a.shape[0]
    k = n if n_points <= 0 else min(n_points, n)
    start = int(np.random.default_rng(seed).integers(0, n - k + 1))
    w = np.asarray(a[start:start + k, cols], dtype=np.float64)
    return k, w.sum(0), (w * w).sum(0)


def pooled_sums(path_seeds, n_points, cols, workers, every=100):
    """(count, sum, sumsq) over a seeded window of every (path, seed), in a process pool."""
    every_row = "every row" if n_points <= 0 else f"<={n_points} rows"
    print(f"stats: {len(path_seeds)} samples x {every_row} ({workers} workers)")
    res, errors = run_pool(_window_sums, [(p, n_points, sd, cols) for p, sd in path_seeds],
                           workers, every=every)
    if errors:
        raise RuntimeError(f"{len(errors)} stats reads failed, e.g. {errors[0]}")
    cnt = sum(r[0] for r in res)
    s = np.sum([r[1] for r in res], axis=0)
    ss = np.sum([r[2] for r in res], axis=0)
    return cnt, s, ss


def save_surface_stats(path, cnt, s, ss, cols, **extra):
    mean, std = moments(cnt, s, ss)
    np.savez(path, mean=mean.astype(np.float32), std=std.astype(np.float32),
             cols=np.array(cols), **extra)
    print(f"norm_stats over {cnt:,} points: mean={np.round(mean, 4)} std={np.round(std, 4)}")


def save_volume_stats(path, cnt, s, ss, cols, n_field=4, extra=False, count_key="n_points",
                      points_per_sample=None):
    """mean/std hold the 4 loader channels [ux,uy,uz,p]; anything after goes to extra_*."""
    mean, std = moments(cnt, s, ss)
    assert mean[:n_field].shape == (4,), f"volume stats must be 4 wide, got {mean[:n_field].shape}"
    out = dict(mean=mean[:n_field].astype(np.float32), std=std[:n_field].astype(np.float32),
               vars=np.array(cols[:n_field]))
    if extra:
        out.update(extra_mean=mean[n_field:].astype(np.float32),
                   extra_std=std[n_field:].astype(np.float32),
                   extra_vars=np.array(cols[n_field:]))
    out[count_key] = cnt
    if points_per_sample is not None:
        out["points_per_sample"] = points_per_sample
    np.savez(path, **out)
    print(f"norm_stats_volume over {cnt:,} points: "
          f"mean={dict(zip(cols, np.round(mean, 4)))} std={dict(zip(cols, np.round(std, 4)))}")
