import argparse
import math
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from common.callbacks import EMAWeightAveraging
from common.utils import get_yaml, save_yaml
from data.datamodule import SurfaceVolumeDataModule, JointSurfaceVolumeDataModule

# Dataset names recognised in a checkpoint path, for run naming when --source_name is absent.
KNOWN_DATASETS = ("drivaernet", "drivaerml", "windsorml", "ahmedml", "joint", "combined", "emmi_wing", "double_delta",
                  "blendednet", "shift_pump", "shift_suv", "submarine", "shift_cca", "superwing", "hiliftaeroml")

# Model-config keys that describe the architecture and must agree with the source checkpoint.
ARCH_KEYS = ("model_name", "model_params", "num_supernodes", "num_surface_anchors")

# Parameter-name substrings allowed to be missing from a source checkpoint (cond path of a no-cond source).
FROM_SCRATCH_KEYS = ("adaLN_modulation", "cond_proj", "cond_gate")


def source_tag(checkpoint_path):
    """Best-effort source-dataset tag from a checkpoint path (for the run name)."""
    p = checkpoint_path.lower()
    for name in KNOWN_DATASETS:
        if name in p:
            return name
    return "scratch" if "scratch" in p else "unknown"


def _leaf_diff(want, got, prefix=""):
    """Per-leaf differences between two nested config values, as (dotted key, want, got)."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = []
        for k in dict.fromkeys(list(want) + list(got)):
            out += _leaf_diff(want.get(k, "<absent>"), got.get(k, "<absent>"),
                              f"{prefix}.{k}" if prefix else k)
        return out
    return [] if want == got else [(prefix, want, got)]


def warn_config_drift(modelconfig, checkpoint_path):
    """Warn (never raise) if the architecture differs from the config.yml beside the checkpoint."""
    if checkpoint_path == "scratch":
        return
    src_path = os.path.join(os.path.dirname(checkpoint_path), "config.yml")
    if not os.path.isfile(src_path):
        print(f"[drift] no config.yml beside the checkpoint ({src_path}); skipping arch check")
        return
    src_model = (get_yaml(src_path) or {}).get("model", {})
    diffs = []
    for k in ARCH_KEYS:
        diffs += _leaf_diff(src_model.get(k, "<absent>"), modelconfig.get(k, "<absent>"), k)
    if not diffs:
        print(f"[drift] architecture matches the source snapshot ({src_path})")
        return
    print(f"[drift] WARNING: architecture differs from the source snapshot ({src_path}):")
    for k, want, got in diffs:
        print(f"  {k}: source={want!r} here={got!r}")


def build_model(config, ds_train):
    from modules.context_module import IN_CONTEXT_MODELS, ContextTrainModule
    from modules.train_module import TrainModule
    cls = (ContextTrainModule if config["model"]["model_name"] in IN_CONTEXT_MODELS
           else TrainModule)
    return cls(config=config, ds_train=ds_train)


def main(args):
    config = get_yaml(args.config)
    dataconfig = config["data"]
    trainconfig = config["training"]

    if args.data_root is not None:
        dataconfig["data_root"] = args.data_root

    # Relative dirs are resolved against data.data_root / $CFD_DATA_ROOT by the datamodule.
    data_type = dataconfig.get("type", "combined")
    if args.target:
        dataconfig["surface_dir"] = args.surface_dir or f"{args.target}/collated"
        dataconfig["volume_dir"] = args.volume_dir or f"{args.target}/volume_collated"
        trainconfig["description"] = args.target
    else:
        if args.surface_dir:
            dataconfig["surface_dir"] = args.surface_dir
        if args.volume_dir:
            dataconfig["volume_dir"] = args.volume_dir

    warn_config_drift(config["model"], args.checkpoint)

    target = trainconfig["description"]

    seed = args.seed
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    if data_type == "combined":
        datamodule = SurfaceVolumeDataModule(dataconfig=dataconfig)
    elif data_type == "joint":
        datamodule = JointSurfaceVolumeDataModule(dataconfig=dataconfig)
    else:
        raise ValueError(f"Unknown data.type: {data_type!r} (expected 'combined' or 'joint')")

    # S <= 0 means every train run; a positive S larger than the pool is an error.
    n_total = len(datamodule.train_dataset)
    if args.num_samples > n_total:
        raise ValueError(
            f"num_samples={args.num_samples} > available train runs ({n_total}). "
            "Pass --num_samples -1 to use every train run."
        )
    num_samples = n_total if args.num_samples <= 0 else args.num_samples
    if args.num_samples <= 0:
        print(f"[samples] --num_samples {args.num_samples} -> all {n_total} train runs")

    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    src = args.source_name or source_tag(args.checkpoint)
    desc = args.description or f"ft_{src}2{target}"
    name = f"{desc}_S{num_samples}_{seed}_{now}"
    wandb_mode = args.wandb_mode or trainconfig.get("wandb_mode", "online")
    wandb_logger = WandbLogger(project=trainconfig["project"], name=name, mode=wandb_mode)

    path = trainconfig["log_dir"] + name + "/"
    path_zeroshot = trainconfig["log_dir"] + name + "/" + "zeroshot/"
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_zeroshot, exist_ok=True)
    print(f"Logging to: {path}")
    config['model']['log_dir'] = path

    config["model"]["lr"] = args.lr

    # The budget is in optimizer steps, independent of S; the cosine is built over it.
    batch_size = dataconfig.get("batch_size", 1)
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = args.max_steps
    val_every_steps = args.val_every_steps or trainconfig.get("val_every_n_steps") or total_steps
    val_every_steps = max(1, min(int(val_every_steps), total_steps))
    warmup_steps = round(args.warmup_frac * total_steps)
    warmup_steps = min(max(1, warmup_steps) if args.warmup_frac > 0 else 0, total_steps)
    print(f"lr schedule: cosine over {total_steps} steps "
          f"({steps_per_epoch}/epoch, {total_steps / steps_per_epoch:.1f} epochs), "
          f"{warmup_steps} warmup, floor {args.min_lr_ratio} x lr, betas {tuple(args.betas)}")

    # Read by common.utils.finetune_optim_config (betas, lr_schedule).
    config["finetune"] = {
        "checkpoint": args.checkpoint,
        "num_samples": num_samples,
        "num_samples_arg": args.num_samples,
        "num_samples_available": n_total,
        "max_steps": total_steps,
        "val_every_steps": val_every_steps,
        "lr": args.lr,
        "ema_decay": args.ema_decay,
        "eval_zero_shot": bool(args.eval_zero_shot or num_samples == 2),
        "betas": [float(b) for b in args.betas],
        "lr_schedule": {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "seed": seed,
        "subset_seed": args.subset_seed,
        "source": src,
        "source_explicit": args.source_name is not None,
        "target": target,
    }
    save_yaml(config, path + "config.yml")

    model = build_model(config, datamodule.train_dataset)
    if getattr(datamodule, "val_datasets", None) is not None:
        model.val_datasets = datamodule.val_datasets

    if args.checkpoint != 'scratch':
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        weights = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(weights, strict=False)
        print(f"loaded checkpoint weights from: {args.checkpoint} "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
        if missing:
            print("  missing[:8]:", missing[:8])
        if unexpected:
            print("  unexpected[:8]:", unexpected[:8])
        surprises = [k for k in missing if not any(t in k for t in FROM_SCRATCH_KEYS)]
        if surprises:
            raise RuntimeError(
                f"{len(surprises)} checkpoint keys are missing and are NOT part of a known "
                f"from-scratch path: {surprises[:8]}. The source architecture likely differs "
                f"from this config (see the drift warning above)."
            )

    model.lr = args.lr
    model.log_dir = path_zeroshot

    # Nested subsets: the S=2 subset is a prefix of the S=4 subset, and so on.
    g = torch.Generator().manual_seed(args.subset_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    idx = perm[: num_samples]
    ft_train = Subset(datamodule.train_dataset, idx)
    stems = getattr(datamodule.train_dataset, "stems", None)
    print(f"Finetuning on {len(ft_train)} / {n_total} train runs"
          + (f" (stems: {[stems[i] for i in idx]})"
             if stems is not None and num_samples <= 64 else ""))

    nw = min(dataconfig.get("num_workers", 8), num_samples)
    train_loader = DataLoader(
        ft_train,
        batch_size=dataconfig.get("batch_size", 1),
        shuffle=True,
        num_workers=nw,
        pin_memory=dataconfig.get("pin_memory", True),
        persistent_workers=nw > 0,
        prefetch_factor=dataconfig.get("prefetch_factor", 4) if nw > 0 else None,
    )
    val_loader = datamodule.val_dataloader()

    devices = [int(d) for d in args.devices] if args.devices else trainconfig["devices"]
    callbacks = [LearningRateMonitor(logging_interval="step")]

    if args.ema_decay is not None:
        if not 0.0 < args.ema_decay < 1.0:
            raise ValueError(f"--ema_decay must be in (0, 1), got {args.ema_decay}")
        horizon = 1.0 / (1.0 - args.ema_decay)
        callbacks.append(EMAWeightAveraging(args.ema_decay))
        print(f"[ema] averaging weights at decay {args.ema_decay} "
              f"(~{horizon:.0f}-step horizon over a {total_steps}-step run); "
              "val metrics and final.ckpt are the averaged weights")
        if horizon > total_steps:
            print("[ema] WARNING: the averaging horizon is longer than the whole run. "
                  "Lower --ema_decay or raise --max_steps.")
    else:
        print("[ema] --ema_decay unset: training the raw weights (no averaging)")

    # A single final.ckpt is written by hand below, so Lightning's checkpointing stays off.
    trainer = L.Trainer(
        devices=devices,
        accelerator=trainconfig["accelerator"],
        strategy="auto",
        check_val_every_n_epoch=None,
        val_check_interval=val_every_steps,
        log_every_n_steps=trainconfig.get("log_every_n_steps", 10),
        max_epochs=-1,
        max_steps=total_steps,
        default_root_dir=path,
        callbacks=callbacks,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    # Before fit(), so this validates the unaveraged source weights.
    if args.eval_zero_shot or num_samples == 2:
        print("=== zero-shot validation ===")
        trainer.validate(model, dataloaders=val_loader)

    model.log_dir = path

    print(f"=== finetuning {total_steps} steps on S={num_samples} "
          f"(val every {val_every_steps} steps) ===")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_ckpt = path + "final.ckpt"
    trainer.save_checkpoint(final_ckpt)
    print(f"saved final checkpoint to: {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot + few-sample finetuning")
    parser.add_argument("--config", default="configs/finetuning/cond_smart.yaml", help="finetune config (configs/finetuning/*.yaml)")
    parser.add_argument("--checkpoint", required=True, help="pretrained source checkpoint, or 'scratch'")
    parser.add_argument("--data_root", default=None, help="dataset root (overrides data.data_root and $CFD_DATA_ROOT)")
    parser.add_argument("--target", default=None,
                        help="target dataset: sets surface_dir/volume_dir to <target>/{collated,volume_collated}")
    parser.add_argument("--surface_dir", default=None, help="explicit surface dir (overrides --target)")
    parser.add_argument("--volume_dir", default=None, help="explicit volume dir (overrides --target)")
    parser.add_argument("--source_name", default=None,
                        help="source tag for the run name (default: matched from the checkpoint path)")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="S: number of target train runs; -1 (or 0) uses all of them")
    parser.add_argument("--max_steps", type=int, default=400, help="total finetuning optimizer steps")
    parser.add_argument("--val_every_steps", type=int, default=None,
                        help="validate every k steps (default: config val_every_n_steps, else once at the end)")
    parser.add_argument("--lr", type=float, default=1e-4, help="finetuning learning rate")
    parser.add_argument("--ema_decay", type=float, default=None,
                        help="EMA weight-averaging decay (default: off)")
    parser.add_argument("--eval_zero_shot", action="store_true",
                        help="validate the source weights before training (implied by --num_samples 2)")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95),
                        metavar=("BETA1", "BETA2"), help="Adam betas")
    parser.add_argument("--warmup_frac", type=float, default=0.05,
                        help="fraction of the step budget spent on linear warmup")
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="cosine floor as a fraction of the initial lr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_seed", type=int, default=0,
                        help="seed for the nested train-subset permutation (independent of --seed)")
    parser.add_argument("--description", default=None, help="run name prefix (default ft_<src>2<target>)")
    parser.add_argument("--devices", nargs="+", default=[], help="GPU indices (default: config value)")
    parser.add_argument("--wandb_mode", default=None, help="online | offline | disabled")
    args = parser.parse_args()

    main(args)
