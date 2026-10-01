import os
import lightning as L
import torch

from data.context_dataset import ContextSurfaceVolumeDataset
from data.surface_volume_dataset import _infer_dataset
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

# Optional ContextSurfaceVolumeDataset kwargs, passed through only when named in the config
# (and per-entry overridable on a joint run).
CONTEXT_KEYS = (
    "in_context", "n_context_surface", "n_context_volume", "context_split", "context_match",
    "env_by", "env_values", "env_base", "env_id", "env_keep", "env_drop",
    "freestream_scale", "tag", "cond_const",
)


def _split_kw(shared, overrides, split):
    """The context kwargs for `split`: the shared block with `split_overrides[split]` applied."""
    return {**shared, **(overrides or {}).get(split, {})}


# Misspellings of `heldout: True` (an entry validated on but never trained on), refused by name.
HELDOUT_ALIASES = ("held_out", "hold_out", "holdout", "val_only", "eval_only", "no_train")


def resolve_dir(path, root=None):
    """Join a relative dataset dir onto `root` (or $CFD_DATA_ROOT)."""
    if path is None or os.path.isabs(path):
        return path
    root = root or os.environ.get("CFD_DATA_ROOT", ".")
    return os.path.join(root, path)


class SurfaceVolumeDataModule(L.LightningDataModule):
    def __init__(self, dataconfig):
        super().__init__()
        c = dataconfig
        self.data_root = c.get("data_root")
        self.surf_dir = resolve_dir(c["surface_dir"], self.data_root)
        self.vol_dir = resolve_dir(c["volume_dir"], self.data_root)
        self.n_geometry = c.get("n_geometry", None)
        self.n_surface = c.get("n_surface", c.get("n_crop", 128000))
        self.n_volume = c.get("n_volume", 128000)
        self.load_surface = c.get("load_surface", True)
        self.load_volume = c.get("load_volume", True)
        self.dataset = c.get("dataset") or _infer_dataset(c["surface_dir"])
        self.normalize = c.get("normalize", True)
        self.batch_size = c.get("batch_size", 1)
        self.num_workers = c.get("num_workers", 8)
        self.pin_memory = c.get("pin_memory", True)
        self.prefetch_factor = c.get("prefetch_factor", 4)
        self.return_metadata = c.get("return_metadata", False)
        self.return_cond = c.get("return_cond", False)
        self.global_cond_norm = c.get("global_cond_norm", True)
        self.num_val = c.get("num_val", None)
        self.field_norm = c.get("field_norm", "dataset")
        self.pos_norm = c.get("pos_norm", "dataset")
        
        self.volume_thin = c.get("volume_thin", None)
        self.surface_thin = c.get("surface_thin", None)
        self.surf_stats_name = c.get("surf_stats_name", "norm_stats_centered.npz")
        self.vol_stats_name = c.get("vol_stats_name", "norm_stats_volume.npz")

        self.include_prefixes = c.get("include_prefixes", None)
        self.exclude_prefixes = c.get("exclude_prefixes", None)
        # Constant channels appended to the emitted cond (see _init_cond_extra).
        self.cond_extra = c.get("cond_extra", None)
        self.context_kw = {k: c[k] for k in CONTEXT_KEYS if k in c}
        self.split_overrides = c.get("split_overrides", None)

        self.train_dataset = self._ds("train")
        self.val_dataset = self._ds("val")

    def _ds(self, split):
        return ContextSurfaceVolumeDataset(
            self.surf_dir, self.vol_dir, split,
            **_split_kw(self.context_kw, self.split_overrides, split),
            n_geometry=self.n_geometry, n_surface=self.n_surface, n_volume=self.n_volume,
            load_surface=self.load_surface, load_volume=self.load_volume,
            dataset=self.dataset, normalize=self.normalize,
            return_metadata=self.return_metadata, return_cond=self.return_cond,
            num_val=self.num_val, global_cond_norm=self.global_cond_norm,
            cond_extra=self.cond_extra,
            field_norm=self.field_norm, pos_norm=self.pos_norm,
            volume_thin=self.volume_thin, surface_thin=self.surface_thin,
            surf_stats_name=self.surf_stats_name, vol_stats_name=self.vol_stats_name,
            include_prefixes=self.include_prefixes,
            exclude_prefixes=self.exclude_prefixes,
        )

    @property
    def env_names(self):
        """The campaign tag this run trains on, as a list (matching the joint datamodule)."""
        return [self.train_dataset.tag]

    def setup(self, stage=None):
        if stage in ("test", None):
            self.test_dataset = self._ds("test")

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)


class JointSurfaceVolumeDataModule(L.LightningDataModule):

    def __init__(self, dataconfig):
        super().__init__()
        c = dataconfig
        self.entries = c["datasets"]
        self.data_root = c.get("data_root")
        # Shared defaults; each entry may override any of these.
        self.shared = dict(
            n_geometry=c.get("n_geometry", None),
            n_surface=c.get("n_surface", 128000),
            n_volume=c.get("n_volume", 128000),
            load_surface=c.get("load_surface", True),
            load_volume=c.get("load_volume", True),
            normalize=c.get("normalize", True),
            return_metadata=c.get("return_metadata", False),
            return_cond=c.get("return_cond", False),
            global_cond_norm=c.get("global_cond_norm", True),
            # Per-entry campaign label; every entry must pass the same number of channels.
            cond_extra=c.get("cond_extra", None),
            field_norm=c.get("field_norm", "dataset"),
            pos_norm=c.get("pos_norm", "dataset"),
            num_val=c.get("num_val", None),

            volume_thin=c.get("volume_thin", None),
            surface_thin=c.get("surface_thin", None),
            surf_stats_name=c.get("surf_stats_name", "norm_stats_centered.npz"),
            vol_stats_name=c.get("vol_stats_name", "norm_stats_volume.npz"),
            include_prefixes=c.get("include_prefixes", None),
            exclude_prefixes=c.get("exclude_prefixes", None),
        )
        self.context_kw = {k: c[k] for k in CONTEXT_KEYS if k in c}
        self.split_overrides = c.get("split_overrides", None)
        self.batch_size = c.get("batch_size", 1)
        self.num_workers = c.get("num_workers", 8)
        self.pin_memory = c.get("pin_memory", True)
        self.prefetch_factor = c.get("prefetch_factor", 4)
        self.samples_per_epoch = c.get("samples_per_epoch", None)

        # `heldout: True` entries are validated on, never trained on.
        for e in self.entries:
            bad = [k for k in HELDOUT_ALIASES if k in e]
            if bad:
                raise ValueError(
                    f"dataset entry {e.get('tag') or e.get('surface_dir')} sets {bad}, which "
                    "no datamodule reads -- the key that holds an entry out of training is "
                    "`heldout: True`.")
        self.train_entries = [e for e in self.entries if not e.get("heldout", False)]
        if not self.train_entries:
            raise ValueError(
                f"every one of the {len(self.entries)} dataset entries is `heldout: True`, so "
                "there is nothing to train on.")
        self.train_datasets = [self._ds(e, "train") for e in self.train_entries]
        self.val_datasets = [self._ds(e, "val") for e in self.entries]
        self.train_dataset = ConcatDataset(self.train_datasets)
        self._check_cond_width()
        if len(self.train_entries) != len(self.entries):
            held = [e.get("tag") or e.get("dataset") for e in self.entries
                    if e.get("heldout", False)]
            print(f"joint: training on {len(self.train_entries)}/{len(self.entries)} entries; "
                  f"HELD OUT of training, validated only: {held}")
        # Train entry tags in env_id order.
        self.env_names = [n for _, n in sorted(
            {(ds.env_id, ds.tag) for ds in self.train_datasets})]

    def _ds(self, entry, split):
        p = {**self.shared, **{k: entry[k] for k in self.shared if k in entry}}
        ctx = _split_kw(self.context_kw, self.split_overrides, split)
        ctx = {**ctx, **{k: entry[k] for k in CONTEXT_KEYS if k in entry}}
        return ContextSurfaceVolumeDataset(
            resolve_dir(entry["surface_dir"], self.data_root),
            resolve_dir(entry["volume_dir"], self.data_root), split,
            **ctx,
            n_geometry=p["n_geometry"], n_surface=p["n_surface"], n_volume=p["n_volume"],
            load_surface=p["load_surface"], load_volume=p["load_volume"],
            dataset=entry.get("dataset") or _infer_dataset(entry["surface_dir"]),
            normalize=p["normalize"],
            return_metadata=p["return_metadata"], return_cond=p["return_cond"],
            num_val=p["num_val"], global_cond_norm=p["global_cond_norm"],
            cond_extra=p["cond_extra"],
            field_norm=p["field_norm"], pos_norm=p["pos_norm"],
            volume_thin=p["volume_thin"], surface_thin=p["surface_thin"],
            surf_stats_name=p["surf_stats_name"], vol_stats_name=p["vol_stats_name"],
            include_prefixes=p["include_prefixes"],
            exclude_prefixes=p["exclude_prefixes"],
        )

    def _check_cond_width(self):
        """Refuse a mixture whose entries emit cond vectors of different widths."""
        widths = {}
        for ds in self.val_datasets:
            w = ds.cond.shape[-1] if ds.cond is not None else ds.pad_cond.shape[-1]
            widths.setdefault(int(w), []).append(ds.tag)
        if len(widths) > 1:
            raise ValueError(
                "joint entries emit cond vectors of different widths: "
                + "; ".join(f"{w} -> {tags}" for w, tags in sorted(widths.items()))
                + ". Every entry must pass the same number of `cond_extra` channels."
            )

    def _train_sampler(self):
        """WeightedRandomSampler over the train entries: P(entry) = weight / sum(weights)."""
        weights = []
        for entry, ds in zip(self.train_entries, self.train_datasets):
            w = float(entry.get("weight", 1.0))
            n = len(ds)
            weights.extend([w / n] * n)  # per-sample weight -> P(dataset) = w / sum(w)
        num_samples = self.samples_per_epoch or sum(len(ds) for ds in self.train_datasets)
        return WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=int(num_samples), replacement=True,
        )

    def _loader(self, dataset, shuffle=False, sampler=None):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, sampler=self._train_sampler())

    def val_dataloader(self):
        # One loader per dataset -> TrainModule selects denorm stats via dataloader_idx.
        return [self._loader(ds, shuffle=False) for ds in self.val_datasets]
