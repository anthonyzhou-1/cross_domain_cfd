"""Per-sample row shuffles: a contiguous window of a shuffled file is a uniform subsample."""
import hashlib

import numpy as np


def stem_seed(stem, seed=0):
    """Stable 32-bit seed for string stems (md5 of the stem)."""
    return seed + int.from_bytes(hashlib.md5(stem.encode()).digest()[:4], "little")


def shuffle_rows(arr, seed):
    return arr[np.random.default_rng(seed).permutation(arr.shape[0])]
