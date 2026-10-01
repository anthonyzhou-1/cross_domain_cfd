"""In-place per-sample sigma cut (|c - mean| > sigma * std on any column; optional cp bound),
with manifest n_points, the pruned_*.json log and norm_stats.npz updated. An identical
logged pass skips its stems; a column cut by a different pass aborts unless --force.
"""
import os

import numpy as np

from .io import load_json, read_split, save_npy, write_json
from .pool import run_pool
from .stats import moments


def sample_mask(a, col_idx, cols, sigma, side="both", cp_max=None, cp_col=3):
    """(keep, stats{col: (mean, std)}, degenerate cols, solo{col: n removed alone})."""
    n = a.shape[0]
    keep = np.ones(n, dtype=bool)
    stats, degenerate, solo = {}, [], {}
    for j in col_idx:
        c = a[:, j].astype(np.float64)
        finite = np.isfinite(c)
        mean = float(c[finite].mean())
        std = float(c[finite].std())
        stats[cols[j]] = (mean, std)
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            degenerate.append(cols[j])        # constant / non-finite: never mask on it
            continue
        dev = c - mean
        ok = (np.abs(dev) <= sigma * std if side == "both" else
              dev <= sigma * std if side == "hi" else dev >= -sigma * std)
        col_keep = finite & ok
        solo[cols[j]] = int(n - col_keep.sum())
        keep &= col_keep
    if cp_max is not None:
        cp = a[:, cp_col].astype(np.float64)
        keep &= np.isfinite(cp) & (cp <= cp_max)
    return keep, stats, degenerate, solo


def prune_one(path, stem, col_idx, cols, sigma, side, cp_max, skip, dry_run,
              stats_max_points, want_mask, extra_fn, extra_arg):
    a = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    before = a
    n_before = a.shape[0]
    n_out, stats, degenerate, solo, keep = 0, {}, [], {}, None
    if not skip:
        keep, stats, degenerate, solo = sample_mask(a, col_idx, cols, sigma, side, cp_max)
        n_out = int(n_before - keep.sum())
        if n_out:
            a = np.ascontiguousarray(a[keep])
            if not dry_run:
                save_npy(path, a)
    k = a.shape[0] if not stats_max_points else min(stats_max_points, a.shape[0])
    win = a[:k].astype(np.float64)
    out = dict(stem=stem, n_before=n_before, n_after=int(a.shape[0]), n_removed=n_out,
               stats=stats, degenerate=degenerate, solo=solo, skipped=bool(skip),
               count=k, sums=win.sum(0), sumsq=(win * win).sum(0),
               keep=np.packbits(keep) if (want_mask and keep is not None) else None)
    if extra_fn is not None:
        out.update(extra_fn(before, a, keep, extra_arg))
    return out


def _stems_of(p):
    s = p["samples"]
    return dict(s) if isinstance(s, dict) else {r["stem"]: r for r in s}


def run_prune(out_dir, cols, col_names, sigma, side="both", cp_max=None, log_name="pruned_cf.json",
              log_columns=None, log_meta=None, workers=8, dry_run=False, force=False, limit=None,
              stats_max_points=0, stats_count_key=False, want_mask=False, extra_fn=None,
              extra_args=None, summary_fn=None, post_fn=None, log_fields=()):
    """Prune `col_names` (on-disk names) in every train/val sample of `out_dir`; see module doc."""
    samples_dir = os.path.join(out_dir, "samples")
    splits_dir = os.path.join(out_dir, "splits")
    manifest_path = os.path.join(out_dir, "manifest.json")
    log_path = os.path.join(out_dir, log_name)
    if limit is not None:
        dry_run = True
    log_columns = list(log_columns if log_columns is not None else col_names)
    col_idx = [cols.index(c) for c in col_names]
    if not col_idx and cp_max is None:
        raise SystemExit("nothing to do: no columns and no cp bound")

    log = load_json(log_path) if os.path.exists(log_path) else {"passes": []}
    key = (log_columns, sigma, side, cp_max)
    same = [p for p in log["passes"]
            if (list(p["columns"]), p["sigma"], p.get("side", "both"), p.get("cp_max")) == key]
    other = [p for p in log["passes"] if p not in same]
    repeat = sorted(set(log_columns) & {c for p in other for c in p["columns"]})
    if repeat and not force:
        raise SystemExit(f"columns {repeat} were already pruned by an earlier pass "
                         f"{[(p['columns'], p['sigma']) for p in other]}; a second pass "
                         f"compounds. Pass --force if that is intended.")
    done = {}
    if not force:
        for p in same:
            done.update(_stems_of(p))
        if done:
            print(f"{len(done)} stems already pruned by this pass; skipping them")

    manifest = load_json(manifest_path)
    by_stem = {r["stem"]: r for r in manifest}
    train = read_split(splits_dir, "train")
    stems = train + read_split(splits_dir, "val")
    assert len(set(stems)) == len(stems), "train/val overlap"
    missing = [s for s in stems if s not in by_stem]
    assert not missing, f"stems absent from manifest.json: {missing[:5]}"
    train = set(train)
    if limit is not None:
        stems = stems[:limit]
    extra_args = extra_args or {}
    print(f"{len(stems)} samples ({len(train)} train); columns={log_columns} sigma={sigma} "
          f"side={side} cp_max={cp_max}; {workers} workers" + ("; DRY RUN" if dry_run else ""))

    jobs = [(os.path.join(samples_dir, s + ".npy"), s, col_idx, cols, sigma, side, cp_max,
             s in done, dry_run, stats_max_points, want_mask, extra_fn, extra_args.get(s))
            for s in stems]
    res, errors = run_pool(prune_one, jobs, workers, every=500)
    results = {r["stem"]: r for r in res}
    for r in res:
        if r["degenerate"]:
            print(f"  WARN {r['stem']}: degenerate columns {r['degenerate']}; not masked")
        if by_stem[r["stem"]]["n_points"] != r["n_before"]:
            print(f"  WARN {r['stem']}: manifest n_points {by_stem[r['stem']]['n_points']} "
                  f"!= file rows {r['n_before']}")

    fresh = [r for r in res if not r["skipped"]]
    tot_b = sum(r["n_before"] for r in fresh)
    tot_rm = sum(r["n_removed"] for r in fresh)
    frac = [r["n_removed"] / r["n_before"] for r in fresh] or [0.0]
    print(f"\nremoved {tot_rm:,} / {tot_b:,} points ({100 * tot_rm / max(tot_b, 1):.3f}%) "
          f"across {len(fresh)} samples; per-sample min={100 * min(frac):.2f}% "
          f"median={100 * float(np.median(frac)):.2f}% max={100 * max(frac):.2f}%")
    for c in col_names:
        solo = sum(r["solo"].get(c, 0) for r in fresh)
        print(f"  {c:>9s} alone would remove {solo:,} ({100 * solo / max(tot_b, 1):.3f}%)")
    if summary_fn is not None:
        summary_fn(res, by_stem)

    if errors:
        print(f"{len(errors)} errors -- nothing else updated")
        return results
    if dry_run:
        print("dry run: manifest / norm_stats / log unchanged")
        return results

    if post_fn is not None:
        post_fn(results, manifest)
    for r in manifest:
        if r["stem"] in results:
            r["n_points"] = results[r["stem"]]["n_after"]
    write_json(manifest_path, manifest, atomic=True)
    print(f"updated manifest.json ({len(results)} stems)")

    entries = []
    for r in sorted(results.values(), key=lambda r: r["stem"]):
        if r["skipped"] and r["stem"] in done:
            entries.append(dict(done[r["stem"]], stem=r["stem"]))
            continue
        e = dict(stem=r["stem"], n_before=r["n_before"], n_after=r["n_after"],
                 n_removed=r["n_removed"], **{k: r[k] for k in log_fields})
        for c, (m, s) in r["stats"].items():
            e[f"{c}_mean"], e[f"{c}_std"] = m, s
        entries.append(e)
    log["passes"] = other + [dict(columns=log_columns, sigma=sigma, side=side, cp_max=cp_max,
                                  **(log_meta or {}), samples=entries)]
    write_json(log_path, log, atomic=True)
    print(f"wrote {log_path} ({len(log['passes'])} passes)")

    tr = [r for r in res if r["stem"] in train]
    cnt = sum(r["count"] for r in tr)
    mean, std = moments(cnt, np.sum([r["sums"] for r in tr], axis=0),
                        np.sum([r["sumsq"] for r in tr], axis=0))
    extra = dict(n_points=np.int64(cnt)) if stats_count_key else {}
    np.savez(os.path.join(out_dir, "norm_stats.npz"), mean=mean.astype(np.float32),
             std=std.astype(np.float32), cols=np.array(cols), **extra)
    print(f"norm_stats.npz over {cnt:,} train points; now rerun "
          f"`python -m data_processing.norm_stats base|surface` for this dataset")
    return results
