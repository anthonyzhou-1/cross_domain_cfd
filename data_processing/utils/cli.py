"""Subcommand CLI shared by the dataset modules."""
import argparse
import os

from .io import dataset_root


def main(name, commands, argv=None):
    """commands: {subcommand: (add_args(parser), run(args), help)}."""
    ap = argparse.ArgumentParser(prog=f"python -m data_processing.{name}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, (add, run, hlp) in commands.items():
        p = sub.add_parser(cmd, help=hlp, description=hlp)
        p.add_argument("--root", default=None,
                       help=f"{name} dataset directory (default $CFD_DATA_ROOT/{name})")
        add(p)
        p.set_defaults(_run=run)
    args = ap.parse_args(argv)
    args.root = dataset_root(args.root, name)
    args._run(args)


def add_build_args(p, workers, tree="collated", force=True, limit=True):
    p.add_argument("--workers", type=int, default=min(workers, os.cpu_count() or 8))
    p.add_argument("--out", default=None, help=f"output tree (default <root>/{tree})")
    if limit:
        p.add_argument("--limit", type=int, default=None, help="only the first N runs (smoke test)")
    if force:
        p.add_argument("--force", action="store_true", help="rebuild samples already on disk")


def add_volume_args(p, workers, force=True):
    add_build_args(p, workers, "volume_collated", force)
    p.add_argument("--surface", default=None,
                   help="paired surface tree for crop boxes and splits (default <root>/collated)")
    p.add_argument("--stats-only", action="store_true",
                   help="skip conversion; recompute norm_stats_volume.npz")


def add_prune_args(p, columns, choices, sigma, workers=32):
    p.add_argument("--out", default=None, help="collated tree to prune (default <root>/collated)")
    p.add_argument("--columns", nargs="*", default=columns, choices=choices,
                   help="columns to sigma-cut (default: what was applied on disk)")
    p.add_argument("--sigma", type=float, default=sigma)
    p.add_argument("--workers", type=int, default=min(workers, os.cpu_count() or 8))
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    p.add_argument("--force", action="store_true", help="allow a pass that compounds")
    p.add_argument("--limit", type=int, default=None, help="first N samples; implies --dry-run")


def trees(args, volume=False):
    """(out_dir, surface_dir) with defaults under --root."""
    surf = getattr(args, "surface", None) or os.path.join(args.root, "collated")
    if volume:
        return args.out or os.path.join(args.root, "volume_collated"), surf
    return args.out or surf, surf
