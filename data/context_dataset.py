"""SurfaceVolumeDataset plus the one-geometry / many-physics features.

  env_by / env_id / env_values / env_base   environment identity (per-sample or per-entry row)
  env_keep / env_drop                       held-out environments (filters the split AND the demo pool)
  in_context / context_split / context_match  one solved demo run of the same campaign per item
  freestream_scale / tag / cond_const       the twin control (Reynolds-similar rescale) and oracle cond

With none of these set, `__getitem__` returns exactly what `SurfaceVolumeDataset` does.
"""
import json
import os

import numpy as np
import torch

from data.surface_volume_dataset import (
    P, PHYS, PHYSICAL_ZERO_MODES, POS, SurfaceVolumeDataset, V_P, V_VEL, WSS,
)


def _as_key(value):
    """A metadata match key: a tuple of floats rounded to 6 decimals."""
    if np.isscalar(value) or isinstance(value, (int, float)):
        value = [value]
    return tuple(float(v) for v in np.round(np.asarray(value, dtype=np.float64), 6))


class ContextSurfaceVolumeDataset(SurfaceVolumeDataset):
    # Class-level defaults: some are read by overridden hooks during the base __init__.
    context_match = None      # metadata columns a demo must agree with the target on
    _match_key = None         # per-item operating-point key   (aligned to self.stems)
    _pool_key = None          # per-pool-run key               (aligned to context_stems)
    _pool_groups = None       # key -> ascending pool positions
    _pool_slot = None         # pool position -> its slot inside its own group
    in_context = False
    freestream_scale = 1.0
    gain_vel = 1.0
    gain_dyn = 1.0
    cond_const = None
    env_by = None
    _env_ids = None
    _env_map = None
    _env_id_override = False
    _full_split = None

    def __init__(self, surf_dir, vol_dir, split,
                 in_context=False,
                 n_context_surface=None, n_context_volume=None,
                 context_split="train", context_match=None,
                 env_by=None, env_values=None, env_base=0, env_id=None,
                 env_keep=None, env_drop=None,
                 freestream_scale=None, tag=None, cond_const=None,
                 **kw):
        # Read by the overridden `_read_split`, which the base __init__ calls.
        self.env_by = (None if not env_by else
                       [env_by] if isinstance(env_by, str) else [str(c) for c in env_by])
        self.env_values = None if env_values is None else [_as_key(v) for v in env_values]
        self.env_base = int(env_base)
        self._env_map = None
        self._full_split = {}
        self._env_keep = None if env_keep is None else {_as_key(v) for v in env_keep}
        self._env_drop = None if env_drop is None else {_as_key(v) for v in env_drop}
        if (self._env_keep or self._env_drop) and not self.env_by:
            raise ValueError(
                "env_keep/env_drop select environments by metadata value, so they need "
                "`env_by` to say which metadata columns define an environment."
            )

        self.in_context = bool(in_context)
        self.context_match = (None if not context_match else
                              [context_match] if isinstance(context_match, str) else
                              [str(c) for c in context_match])
        # Demos default to the target's environment; `context_match: []` opts out.
        if self.in_context and context_match is None and self.env_by:
            self.context_match = list(self.env_by)
        n_surface = kw.get("n_surface", 128000)
        n_volume = kw.get("n_volume", 128000)
        self.n_context_surface = n_context_surface if n_context_surface is not None else n_surface
        self.n_context_volume = n_context_volume if n_context_volume is not None else n_volume

        super().__init__(surf_dir, vol_dir, split, **kw)

        # Twin control: the same runs at s * U_inf with Re fixed, so u ~ s and p, tau ~ s^2.
        self.freestream_scale = 1.0 if freestream_scale is None else float(freestream_scale)
        if self.freestream_scale <= 0:
            raise ValueError(f"freestream_scale must be > 0, got {freestream_scale}")
        if self.field_norm in PHYSICAL_ZERO_MODES and freestream_scale is not None:
            raise ValueError(
                f"field_norm={self.field_norm!r} is incompatible with freestream_scale: an "
                "exact Reynolds-similarity rescale is invisible in coefficient form, so the "
                "K twin campaigns would be identical. Use field_norm='dataset' for twin runs."
            )
        if self.freestream_scale != 1.0 and not self.precomputed_coef and self.p_inf != 0.0:
            raise ValueError(
                f"{self.dataset}: freestream_scale needs p_inf == 0 (got {self.p_inf}); the "
                "gain is applied to the raw pressure column and would not commute with a "
                "non-zero freestream offset."
            )
        self.gain_vel = self.freestream_scale
        self.gain_dyn = self.freestream_scale ** 2

        self.tag = str(tag) if tag else self.dataset

        # Oracle arm: a constant cond overriding the emitted one (target and demo).
        self.cond_const = (None if cond_const is None else
                           torch.tensor(list(cond_const), dtype=torch.float32))
        if self.cond_const is not None and self.return_cond:
            want = self.cond.shape[-1] if self.cond is not None else self.pad_cond.shape[-1]
            if self.cond_const.shape[-1] != want:
                raise ValueError(
                    f"cond_const has {self.cond_const.shape[-1]} channels but "
                    f"{self.dataset} emits {want}; they must match or the model's "
                    "parameter_channels means two different things across entries."
                )

        # Environment row: explicit `env_id` > per-sample `env_by` > the ENV_ID registry.
        self._env_id_override = env_id is not None
        if env_id is not None:
            if self.env_by:
                raise ValueError(
                    "env_id and env_by both set: one pins the whole dataset to a row, the "
                    "other derives a row per sample. Pick one."
                )
            self.env_id = int(env_id)
        if self.env_by:
            keys = self._stem_env_keys(surf_dir, self.stems)
            self._env_ids = np.array([self._env_row(k) for k in keys], dtype=np.int64)
            self.env_id = int(np.bincount(self._env_ids).argmax())   # modal row
            self._check_num_val_stride(surf_dir, kw.get("num_val"))

        # Demo pool: never num_val-truncated.
        if self.context_match and not self.in_context:
            raise ValueError(
                f"context_match={self.context_match} is set but in-context mode is off; it "
                "only constrains how a DEMO is drawn, so on its own it does nothing. Set "
                "`in_context: True`, or drop context_match."
            )
        if self.in_context:
            if not (self.load_surface and self.load_volume):
                raise ValueError("in-context mode needs load_surface and load_volume: "
                                 "a demo is encoded from both of its clouds")
            by_stem = {r["stem"]: r for r in self._load_manifest(surf_dir)}
            vol_len = ({r["stem"]: (r["n_cells"] if "n_cells" in r else r["n_points"])
                        for r in self._load_manifest(vol_dir)} if self.load_volume else {})
            cs = split if context_split is None else context_split
            self.context_split = cs
            self.context_stems = self._read_split(surf_dir, cs, by_stem, vol_len)
            if not self.context_stems:
                raise ValueError(f"in_context=True but the {cs!r} context pool is empty")
            ctx_meta, ctx_cols = self._load_metadata(surf_dir, self.context_stems, by_stem)
            self.context_cond, _ = self._load_cond(self.context_stems, ctx_meta, ctx_cols)
            if self.dataset in PHYS:
                self._register_phys(self.context_stems, ctx_meta, ctx_cols)
            self._init_context_match(ctx_meta, ctx_cols)
            self._index_pool()

        self._log_setup()

    # --- environment identity ------------------------------------------------------------

    def _check_num_val_stride(self, surf_dir, num_val):
        """Refuse a `num_val` stride that serves only a subset of the split's environments."""
        if num_val is None or self.split == "train" or self._env_ids is None:
            return
        full = self._full_split.get(self.split)
        if full is None or len(full) == len(self.stems):
            return
        served = set(self._env_ids.tolist())
        available = {self._env_row(k) for k in self._stem_env_keys(surf_dir, full)}
        if len(served) < len(available):
            raise ValueError(
                f"{self.dataset} {self.split}: num_val={num_val} strides the split down to "
                f"{len(served)} of {len(available)} environments ({sorted(served)} of "
                f"{sorted(available)}). The split files are geometry-major with the operating "
                "point inner, so a stride commensurate with the number of environments serves "
                "one regime and the val metric stops being a metric. Use num_val: null, or a "
                "value coprime with the environment count."
            )

    def _stem_env_keys(self, surf_dir, stems, by_stem=None):
        """The environment key of each stem in `stems`, from the `env_by` metadata columns."""
        if by_stem is None:
            by_stem = {r["stem"]: r for r in self._load_manifest(surf_dir)}
        meta, cols = self._load_metadata(surf_dir, stems, by_stem)
        if meta is None:
            raise ValueError(
                f"env_by={self.env_by} needs metadata.npz, which dataset {self.dataset!r} "
                "does not have; the env columns are metadata columns"
            )
        missing = [c for c in self.env_by if c not in cols]
        if missing:
            raise ValueError(f"env_by columns {missing} are not metadata columns of "
                             f"{self.dataset!r}; available: {cols}")
        idx = [cols.index(c) for c in self.env_by]
        return [_as_key(r) for r in np.asarray(meta)[:, idx]]

    def _build_env_map(self, surf_dir):
        """key -> env row, from `env_values` or the union of all split files (split-independent)."""
        if self._env_map is not None:
            return
        if self.env_values is not None:
            keys = list(self.env_values)
        else:
            by_stem = {r["stem"]: r for r in self._load_manifest(surf_dir)}
            allstems = []
            for name in ("train", "val", "test"):
                path = os.path.join(surf_dir, "splits", f"{name}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        allstems.extend(json.load(f))
            allstems = [s for s in dict.fromkeys(allstems) if s in by_stem]
            keys = sorted(set(self._stem_env_keys(surf_dir, allstems, by_stem)))
        self._env_map = {k: self.env_base + i for i, k in enumerate(keys)}

    def _env_row(self, key):
        row = self._env_map.get(key)
        if row is None:
            raise ValueError(
                f"{self.dataset}: environment {key} (env_by={self.env_by}) has no row in the "
                f"context table {sorted(self._env_map)}. Set `env_values` explicitly if the "
                "split files do not cover every environment."
            )
        return row

    def _read_split(self, surf_dir, split, by_stem, vol_len):
        """`SurfaceVolumeDataset._read_split` plus the env_keep/env_drop filter."""
        stems = super()._read_split(surf_dir, split, by_stem, vol_len)
        if not self.env_by:
            return stems
        self._full_split[split] = stems
        self._build_env_map(surf_dir)
        if self._env_keep is None and self._env_drop is None:
            return stems
        keys = self._stem_env_keys(surf_dir, stems, by_stem)
        keep = [s for s, k in zip(stems, keys)
                if (self._env_keep is None or k in self._env_keep)
                and (self._env_drop is None or k not in self._env_drop)]
        if not keep:
            raise ValueError(
                f"{self.dataset} {split}: env_keep={sorted(self._env_keep or [])} "
                f"env_drop={sorted(self._env_drop or [])} selected 0 of {len(stems)} stems; "
                f"environments present: {sorted(set(keys))}"
            )
        kept_envs = sorted({k for k in keys
                            if (self._env_keep is None or k in self._env_keep)
                            and (self._env_drop is None or k not in self._env_drop)})
        print(f"env filter ({self.dataset} {split}): {len(keep)}/{len(stems)} stems, "
              f"{len(kept_envs)} environment(s) kept: {kept_envs}")
        return keep

    # --- twin control: gains on the raw columns (valid since p_inf == 0) ------------------

    def _surf_coef(self, data):
        if self.gain_dyn != 1.0:
            data = data.clone()
            data[:, P] *= self.gain_dyn
            data[:, WSS] *= self.gain_dyn
        return super()._surf_coef(data)

    def _volume_coef(self, vol, stem=None):
        if self.gain_dyn != 1.0:
            vol = vol.clone()
            vol[:, V_VEL] *= self.gain_vel
            vol[:, V_P] *= self.gain_dyn
        return super()._volume_coef(vol, stem)

    # --- in-context demo pool ------------------------------------------------------------

    def _init_context_match(self, ctx_meta, ctx_cols):
        """Per-item and per-pool-run operating-point keys, for `context_match`."""
        self._match_key = self._pool_key = None
        if not self.context_match:
            return
        if self.meta is None or ctx_meta is None:
            raise ValueError(
                f"context_match={self.context_match} needs metadata.npz, which dataset "
                f"{self.dataset!r} does not have; the match columns are metadata columns"
            )
        missing = sorted({c for c in self.context_match
                          if c not in self.meta_cols or c not in ctx_cols})
        if missing:
            raise ValueError(
                f"context_match columns {missing} are not metadata columns of "
                f"{self.dataset!r}; available: {self.meta_cols}"
            )
        tgt = [self.meta_cols.index(c) for c in self.context_match]
        pool = [ctx_cols.index(c) for c in self.context_match]
        self._match_key = [_as_key(r) for r in np.asarray(self.meta)[:, tgt]]
        self._pool_key = [_as_key(r) for r in np.asarray(ctx_meta)[:, pool]]

    def _index_pool(self):
        """stem -> pool position; under `context_match` also the per-key groups and slots."""
        self._pool_pos = {s: j for j, s in enumerate(self.context_stems)}
        self._pool_groups = self._pool_slot = None
        if not self.context_match:
            return
        groups = {}
        slot = [0] * len(self._pool_key)
        for j, key in enumerate(self._pool_key):
            g = groups.setdefault(key, [])
            slot[j] = len(g)
            g.append(j)
        self._pool_groups, self._pool_slot = groups, slot
        orphan = {k for k in set(self._match_key) if k not in groups}
        if orphan:
            raise ValueError(
                f"{self.split}: {len(orphan)} operating point(s) in this split have no run "
                f"at the same point in the {self.context_split!r} context pool "
                f"(context_match={self.context_match}); e.g. {sorted(orphan)[:4]}. Either the "
                "match columns take values the pool never does, or `context_split` names a "
                "split that does not cover this one's regimes."
            )

    def _warn_small_pool(self):
        """Say so when the pool (or its smallest match group) holds no run besides the target."""
        n = len(self.context_stems)
        if self._pool_groups is not None:
            sizes = {k: len(v) for k, v in self._pool_groups.items()}
            n = min(sizes.values())
            smallest = min(sizes, key=lambda k: sizes[k])
            print(f"[context] match={self.context_match}: {len(sizes)} operating-point "
                  f"group(s), sizes {min(sizes.values())}..{max(sizes.values())} "
                  f"(smallest at {smallest})")
        if n < 2:
            where = "smallest group" if self._pool_groups is not None else "pool"
            print(f"[context] WARNING: {self.split}: {where} of {n} run(s) cannot supply a "
                  "demo distinct from the target")

    def _pick_context(self, idx, rng):
        """Pool index of the demo for item `idx`, excluding the target itself.

        Random on train; `idx % n` on val/test so the pairing is fixed across epochs. A pool
        holding nothing but the target falls back to self-context.
        """
        cand, self_slot = self._candidates(idx)
        n = len(cand)
        if n - (self_slot is not None) <= 0:
            return cand[self_slot]
        if self.split != "train":
            j = idx % n
            while j == self_slot:
                j = (j + 1) % n
            return cand[j]
        if n <= 4:
            return [cand[int(j)] for j in rng.permutation(n) if j != self_slot][0]
        while True:
            j = int(rng.integers(0, n))
            if j != self_slot:
                return cand[j]

    def _candidates(self, idx):
        """(pool positions item `idx` may draw from, the target's slot among them or None)."""
        self_pos = self._pool_pos.get(self.stems[idx])
        if self._pool_groups is None:
            return range(len(self.context_stems)), self_pos
        cand = self._pool_groups[self._match_key[idx]]
        return cand, (None if self_pos is None else self._pool_slot[self_pos])

    def _context_crop(self, stem, rng):
        """One demo's surface and volume crops, in its own frame."""
        arr = np.load(os.path.join(self.surf_samples, stem + ".npy"), mmap_mode="r")
        data = torch.from_numpy(self._surface_window(arr, self.n_context_surface, rng))
        pos, c = self._frame(data[:, POS])
        cp, cf = self._surf_coef(data)
        c_thin = c
        if (self.pos_norm != "dataset" and self.normalize
                and self.load_volume and self.volume_thin is not None):
            c_thin = self._orient(data[:, POS]).mean(dim=0, keepdim=True)
        v_pos, vel, vp = self._volume_crop(stem, self.n_context_volume, rng, c, c_thin)
        return pos, cp, cf, c, v_pos, vel, vp

    def _context(self, idx, rng):
        """The flat `context_*` entries of item `idx` (one demo run)."""
        j = self._pick_context(idx, rng)
        stem = self.context_stems[j]
        pos, cp, cf, c, v_pos, vel, vp = self._context_crop(stem, rng)
        row = {"context_surface_pos": pos, "context_surface_cp": cp,
               "context_surface_cf": cf, "context_volume_pos": v_pos,
               "context_volume_vel": vel, "context_volume_p": vp,
               "context_centroid": c.squeeze(0) if c is not None else torch.zeros(3)}
        if self.return_cond:
            row["context_cond"] = (self.cond_const.clone()
                                   if self.cond_const is not None
                                   else self.context_cond[j].clone()
                                   if self.context_cond is not None
                                   else self.pad_cond.clone())
        return row

    def _add_cond_meta(self, out, idx):
        """`SurfaceVolumeDataset._add_cond_meta`, with `cond_const` overriding the target's cond."""
        out = super()._add_cond_meta(out, idx)
        if self.return_cond and self.cond_const is not None:
            out["cond"] = self.cond_const.clone()
        return out

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if self._env_ids is not None:
            out["env_id"] = torch.tensor(int(self._env_ids[idx]), dtype=torch.long)
        if self.in_context:
            # A second rng stream, so the target crop matches the base class's.
            out.update(self._context(idx, self._rng()))
        return out

    def _log_setup(self):
        if self.freestream_scale != 1.0 or self.cond_const is not None:
            print(f"tag={self.tag} freestream_scale={self.freestream_scale} "
                  f"(gain: vel x{self.gain_vel:g}, p/tau x{self.gain_dyn:g}) "
                  f"cond_const={None if self.cond_const is None else self.cond_const.tolist()}")
        if self.env_by:
            rows = sorted(set(self._env_ids.tolist()))
            print(f"env_by={self.env_by}: {len(rows)} row(s) in this split {rows} "
                  f"(campaign table: {len(self._env_map)} rows, base {self.env_base})")
        elif self._env_id_override:
            print(f"env_id override: {self.dataset} -> row {self.env_id}")
        if self.in_context:
            print(f"in_context: pool={self.context_split} ({len(self.context_stems)} runs),",
                  f"match={self.context_match},",
                  f"n_context_surface={self.n_context_surface}",
                  f"n_context_volume={self.n_context_volume}")
            self._warn_small_pool()
