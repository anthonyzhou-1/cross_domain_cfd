"""Deterministic train/val splits (default_rng(seed) permutation)."""
import os

import numpy as np

from .io import load_json


def random_split(stems, val_frac, seed=0):
    """(train, val) from `stems` in the caller's order (sorted by run id in every builder)."""
    perm = np.random.default_rng(seed).permutation(len(stems))
    n_val = int(round(val_frac * len(stems)))
    val = sorted(stems[i] for i in perm[:n_val])
    train = sorted(stems[i] for i in perm[n_val:])
    return train, val


def grouped_split(rows, group_key, val_frac, seed=0):
    """(train, val) with every stem of a group (geometry) on one side."""
    by_group = {}
    for r in rows:
        by_group.setdefault(r[group_key], []).append(r["stem"])
    groups = sorted(by_group)
    perm = np.random.default_rng(seed).permutation(len(groups))
    n_val = int(round(val_frac * len(groups)))
    val_groups = {groups[i] for i in perm[:n_val]}
    train = sorted(s for g in groups if g not in val_groups for s in by_group[g])
    val = sorted(s for g in val_groups for s in by_group[g])
    print(f"  {len(groups)} groups, {len(val_groups)} in val")
    return train, val


def sticky_split(stems, splits_dir, val_frac, seed=0):
    """random_split over stems not yet assigned; stems already in splits/ keep their side."""
    old = {}
    for name in ("train", "val"):
        path = os.path.join(splits_dir, name + ".json")
        if os.path.exists(path):
            old.update({s: name for s in load_json(path)})
    known = [s for s in stems if s in old]
    fresh = [s for s in stems if s not in old]
    if old and fresh:
        print(f"splits: keeping {len(known)} existing assignments, drawing {len(fresh)} new")
    perm = np.random.default_rng(seed).permutation(len(fresh))
    n_val = int(round(val_frac * len(fresh)))
    val_idx = set(perm[:n_val].tolist())
    train = sorted([s for s in known if old[s] == "train"]
                   + [s for i, s in enumerate(fresh) if i not in val_idx])
    val = sorted([s for s in known if old[s] == "val"]
                 + [s for i, s in enumerate(fresh) if i in val_idx])
    return train, val
