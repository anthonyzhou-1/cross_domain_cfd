import lightning as L
import torch
from common.loss import ScaledLpLoss
from common.utils import (COND_PARAM_KEYS, adam_param_groups, build_lr_scheduler,
                          build_muon_optimizer, finetune_optim_config, step_lr_config)
from common.plotting import (plot_pointcloud, plot_volume_scatter_slices,
                             volume_field_scale)
from modules.models.upt.model import AnchoredBranchedUPT
from modules.models.smart.model import SMART, SMART_IC
from modules.models.transolver.model import Transolver_plus

# a default for normalize volume coords
DEFAULT_PLOT_REGION = {
    "crop": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.0, 1.0)},
    "locs": {"z": 0.0, "y": 0.0, "x": 0.0},
}

class TrainModule(L.LightningModule):

    def __init__(self, config: dict, ds_train=None):
        super().__init__()

        modelconfig = config["model"]
        self.modelconfig = modelconfig
        self.config = config
        self.model_name = modelconfig["model_name"]
        self.lr = modelconfig["lr"]
        self.log_dir = modelconfig["log_dir"]
        self.ds = ds_train
        self.plot_every = modelconfig.get("plot_every", 1)
        # Counts completed validations; drives the plot gate under step-based validation.
        self._val_count = 0
        # Set by train.py for joint runs: one dataset per val loader.
        self.val_datasets = None
        self.optimizer_type = modelconfig.get("optimizer", "adam")
        self.step_lr = step_lr_config(config)
        self.cond_lr_mult = modelconfig.get("cond_lr_mult", None)
        self.ft_betas, self.ft_lr_schedule = finetune_optim_config(config)

        self.criterion = ScaledLpLoss()

        self.loss_weights = modelconfig.get("loss_weights", {}) or {}

        self.max_surface_points = modelconfig.get("max_surface_points", None)
        self.max_volume_points = modelconfig.get("max_volume_points", None)

        if self.model_name == "upt":
            self.num_supernodes = modelconfig["num_supernodes"]
            self.num_surface_anchors = modelconfig["num_surface_anchors"]
            self.num_volume_anchors = modelconfig["num_volume_anchors"]
            self.model = AnchoredBranchedUPT(**modelconfig["model_params"])
        elif self.model_name == "transolver":
            self.model = Transolver_plus(**modelconfig["model_params"])
        elif self.model_name == "smart":
            self.model = SMART(**modelconfig["model_params"])
        elif self.model_name == "smart_ic":
            # Fed by ContextTrainModule (modules/context_module.py).
            self.model = SMART_IC(**modelconfig["model_params"])
        else:
            raise ValueError("Model not found")

        self.FIELDS = {
            "surface_cp": ("surface_cp", "denormalize_cp"),
            "surface_cf": ("surface_cf", "denormalize_cf"),
            "volume_p": ("volume_p", "denormalize_volume_p"),
            "volume_vel": ("volume_vel", "denormalize_velocity"),
        }

        strategy = config["training"]["strategy"]
        self.ddp = strategy in ("ddp", "ddp_find_unused_parameters_true")

        self.save_hyperparameters(ignore=["ds_train"])

    # --- input assembly -------------------------------------------------------
    def _sample_supernode_idx(self, num_points, device):
        """Sorted random subset of geometry indices (kept so the RNG stream matches trained runs)."""
        k = min(self.num_supernodes, num_points)
        return torch.randperm(num_points, device=device)[:k].sort().values

    @staticmethod
    def _split_anchors_queries(pos, fields, num_anchors, max_points):
        """Split (optionally subsampled) points into anchor/query tensors.

        Args:
            pos: [1, N, 3] positions.
            fields: list of [1, N, C] label tensors aligned with `pos`.
            num_anchors: number of anchor tokens (full attention).
            max_points: cap on total points used this step (None -> all N).

        Returns anchors [1,na,3], queries [1,nq,3], and the label lists split the
        same way: (field_anchors, field_queries).
        """

        n_total = pos.size(1)
        perm = torch.randperm(n_total, device=pos.device)
        if max_points is not None and max_points < n_total:
            perm = perm[:max_points]
        n = perm.numel()
        na = min(num_anchors, n)
        a_idx, q_idx = perm[:na], perm[na:]
        anchors, queries = pos[:, a_idx], pos[:, q_idx]
        f_a = [f[:, a_idx] for f in fields]
        f_q = [f[:, q_idx] for f in fields]
        return anchors, queries, f_a, f_q

    def forward_upt(self, batch):
        """Assemble inputs, run the model, and return (preds, labels, aux).

        preds/labels are dicts keyed by FIELDS (each [1, na+nq, C], anchors then
        queries). aux carries the matching {surface,volume}_pos for plotting.
        """

        cat = lambda a, b: torch.cat([a, b], dim=1)

        geometry_position = batch.get("geometry_pos", None)
        if geometry_position is None:
            geometry_position = batch["surface_pos"]

        cond = batch.get('cond', None)
        geometry_position = geometry_position[0]  # [N_g, 3] (bs=1 -> drop batch dim)
        supernode_idx = self._sample_supernode_idx(geometry_position.size(0), geometry_position.device)

        s_anchor, s_query, (s_cp_a, s_cf_a), (s_cp_q, s_cf_q) = self._split_anchors_queries(
            batch["surface_pos"], [batch["surface_cp"], batch["surface_cf"]],
            self.num_surface_anchors, self.max_surface_points,
        )

        v_anchor, v_query, (v_p_a, v_vel_a), (v_p_q, v_vel_q) = self._split_anchors_queries(
            batch["volume_pos"], [batch["volume_p"], batch["volume_vel"]],
            self.num_volume_anchors, self.max_volume_points,
        )

        out = self.model(
            geometry_position=geometry_position,
            geometry_supernode_idx=supernode_idx,
            geometry_batch_idx=None,
            surface_anchor_position=s_anchor,
            volume_anchor_position=v_anchor,
            surface_query_position=s_query,
            volume_query_position=v_query,
            cond=cond,
        )

        preds = {
            "surface_cp": cat(out["surface_anchor_cp"], out["surface_query_cp"]),
            "surface_cf": cat(out["surface_anchor_cf"], out["surface_query_cf"]),
            "volume_p": cat(out["volume_anchor_cp"], out["volume_query_cp"]),
            "volume_vel": cat(out["volume_anchor_velocity"], out["volume_query_velocity"]),
        }
        labels = {
            "surface_cp": cat(s_cp_a, s_cp_q),
            "surface_cf": cat(s_cf_a, s_cf_q),
            "volume_p": cat(v_p_a, v_p_q),
            "volume_vel": cat(v_vel_a, v_vel_q),
        }
        aux = {
            "surface_pos": cat(s_anchor, s_query),
            "volume_pos": cat(v_anchor, v_query),
            "centroid": batch["centroid"],  # per-sample centroid -> native coords for plotting
        }

        return preds, labels, aux

    @staticmethod
    def _cap(pos, fields, max_points):
        """Randomly subsample `pos` and its aligned label tensors to `max_points`.

        Args:
            pos: [1, N, 3] query positions.
            fields: list of [1, N, C] label tensors aligned with `pos`.
            max_points: cap on the points used this step (None / >= N -> no-op).

        Returns (pos, fields) with the same random subset applied to all of them.
        """
        n = pos.size(1)
        if max_points is None or n <= max_points:
            return pos, fields
        perm = torch.randperm(n, device=pos.device)[:max_points]
        return pos[:, perm], [f[:, perm] for f in fields]

    def _smart_queries(self, batch):
        """(geometry, surface queries+labels, volume queries+labels), with optional point caps."""
        geometry_position = batch.get("geometry_pos", None)
        if geometry_position is None:
            geometry_position = batch["surface_pos"]

        surf_q, (surf_cp, surf_cf) = self._cap(
            batch["surface_pos"], [batch["surface_cp"], batch["surface_cf"]],
            self.max_surface_points)
        vol_q, (vol_p, vol_vel) = self._cap(
            batch["volume_pos"], [batch["volume_p"], batch["volume_vel"]],
            self.max_volume_points)

        return geometry_position, (surf_q, surf_cp, surf_cf), (vol_q, vol_p, vol_vel)

    @staticmethod
    def _smart_outputs(surface_pred, volume_pred, surf, vol, centroid):
        """Pack the two prediction tensors into (preds, labels, aux); surface is [cp, cf], volume [p, vel]."""
        surf_q, surf_cp, surf_cf = surf
        vol_q, vol_p, vol_vel = vol
        preds = {
            "surface_cp": surface_pred[..., :1],
            "surface_cf": surface_pred[..., 1:],
            "volume_p": volume_pred[..., :1],
            "volume_vel": volume_pred[..., 1:],
        }
        labels = {
            "surface_cp": surf_cp,
            "surface_cf": surf_cf,
            "volume_p": vol_p,
            "volume_vel": vol_vel,
        }
        aux = {
            "surface_pos": surf_q,
            "volume_pos": vol_q,
            "centroid": centroid,  # per-sample centroid -> native coords for plotting
        }
        return preds, labels, aux

    def forward_smart(self, batch):
        """Assemble inputs, run SMART / Transolver++, and return (preds, labels, aux)."""
        geometry_position, surf, vol = self._smart_queries(batch)

        surface_pred, volume_pred = self.model(
            geo=geometry_position,          # full res
            surf_query_pos=surf[0],         # optionally capped during training
            vol_query_pos=vol[0],
            params=batch.get("cond", None),
        )
        return self._smart_outputs(surface_pred, volume_pred, surf, vol, batch["centroid"])

    def _forward_batch(self, batch):
        """Dispatch to the right input-assembly path for this model."""
        if self.model_name in ("smart", "transolver"):
            return self.forward_smart(batch)
        if self.model_name == "upt":
            return self.forward_upt(batch)
        raise ValueError(f"no forward path for model_name={self.model_name!r}")

    # --- steps ----------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        preds, labels, _ = self._forward_batch(batch)

        losses = {k: self.criterion(preds[k], labels[k]) for k in preds}
        loss = sum(self.loss_weights.get(k, 1.0) * v for k, v in losses.items())

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        for k, v in losses.items():
            self.log(f"train/loss_{k}", v, on_step=True, on_epoch=True, sync_dist=self.ddp)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0, eval=False):
        ds = self.val_datasets[dataloader_idx] if self.val_datasets else self.ds
        tag = f"{getattr(ds, 'tag', ds.dataset)}/" if self.val_datasets else ""

        preds, labels, aux = self._forward_batch(batch)

        # per-field loss in physical (denormalized) units
        loss_dict = {}
        for k, (_, denorm) in self.FIELDS.items():
            fn = getattr(ds, denorm)
            loss_dict[k] = self.criterion(fn(preds[k]), fn(labels[k]))
            if not eval:
                self.log(f"val/{tag}loss_{k}", loss_dict[k], on_step=False, on_epoch=True, sync_dist=self.ddp)
                # Relative L2 on the standardized tensors (what training_step optimizes).
                self.log(f"val/{tag}loss_norm_{k}", self.criterion(preds[k], labels[k]),
                         on_step=False, on_epoch=True, sync_dist=self.ddp)

        # Per-environment breakdown (datasets with `env_by`); batch_size is 1 there.
        if not eval and getattr(ds, "_env_ids", None) is not None and "env_id" in batch:
            env = batch["env_id"].reshape(-1)
            if env.numel() == 1:
                etag = f"{tag}env{int(env[0])}/"
                for k in self.FIELDS:
                    self.log(f"val/{etag}loss_norm_{k}", self.criterion(preds[k], labels[k]),
                             on_step=False, on_epoch=True, sync_dist=self.ddp,
                             add_dataloader_idx=False)

        avg_canon_loss = 0
        if not eval and "u_inf" in batch:
            ref = (batch["u_inf"], batch["vol_q"], batch["u_ref"])
            for k, canon in (("surface_cp", "canonical_cp"), ("surface_cf", "canonical_cf"),
                             ("volume_p", "canonical_volume_p"),
                             ("volume_vel", "canonical_velocity")):
                if k not in preds:
                    continue
                fn = getattr(ds, canon)
                args = () if k.startswith("surface") else (ref,)
                canon_loss = self.criterion(fn(preds[k], *args), fn(labels[k], *args))
                avg_canon_loss += canon_loss
                self.log(f"val/{tag}loss_canon_{k}",
                         canon_loss,
                         on_step=False, on_epoch=True, sync_dist=self.ddp)

        avg_canon_loss /= len(self.FIELDS)
        self.log(f"val/{tag}loss_canon_avg", avg_canon_loss, on_step=False, on_epoch=True, sync_dist=self.ddp)

        if batch_idx == 0 and (not self.ddp or self.global_rank == 0): # only plot on rank 0
            gate = self._plot_gate()
            if gate is not None:
                self._plot(preds, labels, aux, ds, *gate)

        if eval:
            return loss_dict, preds, labels, aux
        return

    def on_validation_epoch_end(self):
        if self._trainer is not None and not self.trainer.sanity_checking:
            self._val_count += 1

    def _plot_gate(self):
        """(filename index, first-validation flag) for this validation, or None to skip.

        Epoch-based validation indexes plots by epoch; step-based validation counts
        validations for `plot_every` and names files by global step.
        """
        trainer = self._trainer # `.trainer` raises when the module isn't attached
        if trainer is not None and trainer.check_val_every_n_epoch is None:
            n = self._val_count
            return (self.global_step, n == 0) if n % self.plot_every == 0 else None
        ep = self.current_epoch
        # `or ep == 0` enables plotting for zero-shot
        return (ep, ep == 0) if ((ep + 1) % self.plot_every == 0 or ep == 0) else None

    def _plot(self, preds, labels, aux, ds, ep, first):
        """Surface cp/cf scatter + volume slice plots; `first` also dumps ground truth."""
        # Joint runs plot every dataset at batch_idx 0; prefix filenames to avoid clobber.
        pfx = f"{getattr(ds, 'tag', ds.dataset)}_" if self.val_datasets else ""

        # --- surface: cp point cloud ---
        # Scale predictions to the ground-truth range so pred/label colorbars match.
        surf_pos = aux["surface_pos"][0].detach().cpu() # don't denormalize
        cp = lambda x: ds.denormalize_cp(x)[0].detach().cpu()
        cf = lambda x: ds.denormalize_cf(x)[0, ..., 0].detach().cpu() # x-dir
        cp_gt, cf_gt = cp(labels["surface_cp"]), cf(labels["surface_cf"])
        cp_scale = (float(cp_gt.min()), float(cp_gt.max()))
        cf_scale = (float(cf_gt.min()), float(cf_gt.max()))
        if first:
            plot_pointcloud(surf_pos, color=cp_gt, scale=cp_scale,
                            save_path=f"{self.log_dir}{pfx}{ep}_gt_surface_cp.png")
            plot_pointcloud(surf_pos, color=cf_gt, scale=cf_scale,
                            save_path=f"{self.log_dir}{pfx}{ep}_gt_surface_cf.png")
        plot_pointcloud(surf_pos, color=cp(preds["surface_cp"]), scale=cp_scale,
                        save_path=f"{self.log_dir}{pfx}{ep}_pred_surface_cp.png")
        plot_pointcloud(surf_pos, color=cf(preds["surface_cf"]), scale=cf_scale,
                        save_path=f"{self.log_dir}{pfx}{ep}_pred_surface_cf.png")

        def vol_array(p_field, vel_field):
            pos = aux["volume_pos"] # don't denorm this
            vel = ds.denormalize_velocity(vel_field)
            p = ds.denormalize_volume_p(p_field)
            return torch.cat([pos, vel, p], dim=-1)[0].detach().cpu().numpy()

        gt_arr = vol_array(labels["volume_p"], labels["volume_vel"])
        pred_arr = vol_array(preds["volume_p"], preds["volume_vel"])
        region = DEFAULT_PLOT_REGION
        phys = getattr(ds, "field_norm", "dataset") == "physical"
        for field in (("velocity_deficit", "cp") if phys else ("velocity", "pressure")):
                scale = volume_field_scale(gt_arr, field)
                if first:
                    self._safe_slices(gt_arr, field, f"{self.log_dir}{pfx}{ep}_gt_volume_{field}.png",
                                    scale=scale, **region)
                self._safe_slices(pred_arr, field, f"{self.log_dir}{pfx}{ep}_pred_volume_{field}.png",
                                scale=scale, **region)

    def _safe_slices(self, arr, field, save_path, scale=None, crop=None, locs=None):
        """Volume raw-point scatter slices; slabs can be too sparse under subsampling -> skip, don't crash val."""
        try:
            plot_volume_scatter_slices(arr, field=field, save_path=save_path,
                                       scale=scale, crop=crop, locs=locs, s=10)
        except Exception as e:  # noqa: BLE001 - a bad slab must not kill validation
            print(f"[val plot] skipped {field} slices ({save_path}): {e}")

    # --- optimization ---------------------------------------------------------
    # Attribute names under which each model keeps its transformer body.
    BODY_MODULE_NAMES = ("geometry_blocks", "blocks", "surface_blocks", "volume_blocks",
                         "encoder_blocks", "decoder_blocks")

    # Body parameters that stay on AdamW under Muon (identity-initialized conditioning modulators).
    NO_MUON_PARAM_KEYS = COND_PARAM_KEYS + ("modulator",)

    def get_muon_param_groups(self):
        """Split the model into the Muon body group (hidden 2D+ body weights) and the AdamW rest.

        Deduplicated by id, since weight-shared SMART decoder blocks reuse encoder modules.
        """
        body_modules = [getattr(self.model, name) for name in self.BODY_MODULE_NAMES
                        if hasattr(self.model, name)]
        if not body_modules:
            raise ValueError(
                f"No transformer body modules found on {type(self.model).__name__} "
                f"(looked for {self.BODY_MODULE_NAMES}); cannot build Muon param groups. "
                "Add this model's body attribute to BODY_MODULE_NAMES."
            )

        hidden_weights, hidden_ids = [], set()
        for mod in body_modules:
            for name, p in mod.named_parameters():
                if (p.requires_grad and p.ndim >= 2 and id(p) not in hidden_ids
                        and not any(k in name for k in self.NO_MUON_PARAM_KEYS)):
                    hidden_weights.append(p)
                    hidden_ids.add(id(p))
        if not hidden_weights:
            raise ValueError(
                f"No 2D body weights found on {type(self.model).__name__}; "
                "cannot build Muon param groups"
            )

        other_params = [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in hidden_ids]

        groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=self.lr * 10, weight_decay=0.01),
            dict(params=other_params, use_muon=False,
                 lr=self.lr, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        return groups

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                adam_param_groups(self.model, self.lr, cond_lr_mult=self.cond_lr_mult),
                lr=self.lr, betas=self.ft_betas)
        elif self.optimizer_type == "muon":
            optimizer = build_muon_optimizer(self.get_muon_param_groups())
        else:
            raise ValueError(f"Optimizer not found: {self.optimizer_type}")

        # The finetune cosine schedule (lr_schedule) takes precedence over step_lr when both are set.
        return [optimizer], [build_lr_scheduler(optimizer, lr_schedule=self.ft_lr_schedule,
                                                step_lr=self.step_lr)]
