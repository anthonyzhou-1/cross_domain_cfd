"""Training module for the in-context model (SMART_IC): one target plus one solved demo run.

Adds the `smart_ic` forward path and the `val/<tag>/ctx_gap_<mode>` guardrail (ablated error
minus a paired baseline; positive means the ablation hurt). See modules/context_ablation.py.
"""
import torch

from modules import context_ablation as ctx_ablate
from modules.train_module import TrainModule

# Model names that consume a demonstration run, and so have a context to ablate.
IN_CONTEXT_MODELS = {"smart_ic"}


class ContextTrainModule(TrainModule):
    def __init__(self, config: dict, ds_train=None):
        super().__init__(config, ds_train=ds_train)
        modelconfig = config["model"]
        # Costs len(modes) extra forwards on the first `context_ablation_batches` val batches.
        self.context_ablation = modelconfig.get(
            "context_ablation", ["self_exact", "zero_val"])
        self.context_ablation_batches = modelconfig.get("context_ablation_batches", 8)
        if self.model_name not in IN_CONTEXT_MODELS:
            self.context_ablation = []

    def _context_fields(self, batch):
        """Demo clouds as [pos, cp, cf] (surface) and [pos, p, vel] (volume)."""
        if "context_surface_pos" not in batch:
            raise KeyError(
                f"{self.model_name} needs the context_* keys; set `in_context: True` in the "
                "config's data block"
            )
        surface_field = torch.cat(
            [batch["context_surface_pos"], batch["context_surface_cp"],
             batch["context_surface_cf"]], dim=-1)
        volume_field = torch.cat(
            [batch["context_volume_pos"], batch["context_volume_p"],
             batch["context_volume_vel"]], dim=-1)
        return surface_field, volume_field

    def forward_smart_ic(self, batch):
        """Assemble inputs, run SMART_IC, and return (preds, labels, aux).

        `query_params` is the target's operating point, `context_params` the demo's.
        """
        geometry_position, surf, vol = self._smart_queries(batch)
        surface_field, volume_field = self._context_fields(batch)   # at full resolution

        surface_pred, volume_pred = self.model(
            geo=geometry_position,          # full res
            surface_field=surface_field,    # demo run, full res
            volume_field=volume_field,
            surf_query_pos=surf[0],         # optionally capped during training
            vol_query_pos=vol[0],
            query_params=batch.get("cond", None),
            context_params=batch.get("context_cond", None),
        )
        return self._smart_outputs(surface_pred, volume_pred, surf, vol, batch["centroid"])

    def _forward_batch(self, batch):
        if self.model_name == "smart_ic":
            return self.forward_smart_ic(batch)
        return super()._forward_batch(batch)

    def _mean_error(self, preds, labels, ds):
        """Mean relative L2 over the four fields, in physical units."""
        return float(torch.stack([
            self.criterion(getattr(ds, denorm)(preds[k]), getattr(ds, denorm)(labels[k]))
            for k, (_, denorm) in self.FIELDS.items()]).mean())

    def _log_context_ablation(self, batch, batch_idx, ds, tag):
        """Log how much the prediction depends on the demo, against a baseline under the same seed."""
        seed = ctx_ablate.seed_for(batch_idx)
        fwd = self._forward_batch
        base_preds, base_labels, _ = ctx_ablate.paired_forward(self, batch, seed, fwd)
        base = self._mean_error(base_preds, base_labels, ds)
        self.log(f"val/{tag}ctx_baseline", base, on_step=False, on_epoch=True,
                 sync_dist=self.ddp, add_dataloader_idx=False)

        for mode in self.context_ablation:
            ablated = ctx_ablate.ablated_batch(batch, mode)
            if ablated is None:
                return                      # no context in this batch; nothing to ablate
            preds, labels, _ = ctx_ablate.paired_forward(self, ablated, seed, fwd)
            self.log(f"val/{tag}ctx_gap_{mode}",
                     self._mean_error(preds, labels, ds) - base,
                     on_step=False, on_epoch=True, sync_dist=self.ddp,
                     add_dataloader_idx=False)

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval=False):
        out = super().validation_step(batch, batch_idx, dataloader_idx, eval=eval)
        # Skipped under eval=True: val.py drives that path and wants the plain triple.
        if not eval and self.context_ablation and batch_idx < self.context_ablation_batches:
            ds = self.val_datasets[dataloader_idx] if self.val_datasets else self.ds
            tag = f"{getattr(ds, 'tag', ds.dataset)}/" if self.val_datasets else ""
            self._log_context_ablation(batch, batch_idx, ds, tag)
        return out
