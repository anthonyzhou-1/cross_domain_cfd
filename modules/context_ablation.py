"""Context ablations for in-context models: does the model actually read its demo?

    self_exact   the demo becomes the target's own crop (ceiling: error should be near zero).
    zero_val     demo values zeroed, i.e. the dataset mean (floor: no demo information).
"""

import torch

# The demo's value channels; positions are never touched by a value ablation.
CTX_VALUE_KEYS = ("context_surface_cp", "context_surface_cf",
                  "context_volume_p", "context_volume_vel")

# What `self_exact` copies over: the target's own clouds, in the demo's slots.
_SELF_EXACT = (("context_surface_pos", "surface_pos"),
               ("context_surface_cp", "surface_cp"),
               ("context_surface_cf", "surface_cf"),
               ("context_volume_pos", "volume_pos"),
               ("context_volume_p", "volume_p"),
               ("context_volume_vel", "volume_vel"),
               ("context_cond", "cond"))

MODES = ("self_exact", "zero_val")


def ablated_batch(batch, mode):
    """Shallow copy of `batch` with the context replaced per `mode`, or None if there is no context."""
    if "context_surface_pos" not in batch:
        return None
    if mode not in MODES:
        raise ValueError(f"unknown context ablation mode {mode!r}; expected one of {MODES}")

    out = dict(batch)

    if mode == "self_exact":
        for dst, src in _SELF_EXACT:
            if src in batch:            # `cond` is absent for unconditional runs
                out[dst] = batch[src]
        return out

    for k in CTX_VALUE_KEYS:
        out[k] = torch.zeros_like(batch[k])
    return out


def paired_forward(module, batch, seed, forward):
    """Run `forward(batch)` with every random draw pinned to `seed` (forked RNG, so training is unaffected)."""
    device = batch["surface_pos"].device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        return forward(batch)


def seed_for(batch_idx):
    """A per-batch seed, stable across variants and across epochs."""
    return 0x5EED + 9176 * batch_idx
