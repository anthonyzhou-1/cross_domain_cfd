"""Drivers for the one-file-per-run builders; process(*run, samples_dir, [surface_samples,]
force) returns the sample's manifest row.
"""
import os

from . import cli
from .io import all_on_disk, load_json, merge_manifest, read_split, write_splits
from .pool import run_pool
from .splits import random_split
from .stats import save_surface_stats, save_volume_stats, serial_sums


def _id_row(run_re, count_key, extra=None):
    return lambda s, n: {"stem": s, "run_id": int(run_re.match(s).group(1)), count_key: n,
                         **(extra or {})}


def build_surface(args, runs_fn, process, cols, run_re, val_frac=0.20, seed=0, every=25,
                  row_fn=None, split_fn=None, key="run_id"):
    out, _ = cli.trees(args)
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    if args.splits_only:
        rows = load_json(os.path.join(out, "manifest.json"))
    else:
        runs = runs_fn()[:args.limit]
        res, errors = run_pool(process, [(*r, samples, args.force) for r in runs],
                               args.workers, every)
        rows = merge_manifest(out, run_re, row_fn or _id_row(run_re, "n_points"), res, errors,
                              key)
    if split_fn is None:
        stems = sorted((r["stem"] for r in rows), key=lambda s: int(run_re.match(s).group(1)))
        splits = dict(zip(("train", "val"), random_split(stems, val_frac, seed)))
    else:
        splits = split_fn(rows)
    indent = splits.pop("_indent", None)
    write_splits(os.path.join(out, "splits"), indent=indent, **splits)
    train = splits["train"]
    if all_on_disk(samples, train):
        save_surface_stats(os.path.join(out, "norm_stats.npz"),
                           *serial_sums([os.path.join(samples, s + ".npy") for s in train]), cols)
    else:
        print("skipping norm_stats (train samples not all present yet)")
    return rows


def build_volume(args, runs_fn, process, cols, run_re, count_key="n_points", every=1,
                 extra_row=None, job_extra=(), row_fn=None, key="run_id", stats_key=None):
    out, surf = cli.trees(args, volume=True)
    samples = os.path.join(out, "samples")
    os.makedirs(samples, exist_ok=True)
    if not args.stats_only:
        runs = runs_fn()
        if getattr(args, "runs", None):
            runs = [r for r in runs if r[0] in set(args.runs)]
        runs = runs[:args.limit]
        jobs = [(*r, samples, os.path.join(surf, "samples"), *job_extra, args.force)
                for r in runs]
        res, errors = run_pool(process, jobs, args.workers, every)
        merge_manifest(out, run_re, row_fn or _id_row(run_re, count_key, extra_row), res,
                       errors, key)
    train = read_split(os.path.join(surf, "splits"), "train")
    if train and all_on_disk(samples, train):
        save_volume_stats(os.path.join(out, "norm_stats_volume.npz"),
                          *serial_sums([os.path.join(samples, s + ".npy") for s in train],
                                       slice(3, 7)), cols[3:7], count_key=stats_key or count_key)
    else:
        print("skipping norm_stats_volume (train samples not all present yet)")


def surface_args(workers):
    def add(p):
        cli.add_build_args(p, workers)
        p.add_argument("--splits-only", action="store_true",
                       help="skip conversion; rebuild splits + stats from the manifest")
    return add


def volume_args(workers, runs=True):
    def add(p):
        cli.add_volume_args(p, workers)
        if runs:
            p.add_argument("--runs", nargs="+", default=None, help="only these run dirs")
    return add
