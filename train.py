import argparse
import os
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from common.callbacks import EMAWeightAveraging
from common.utils import get_yaml, save_yaml
from data.datamodule import SurfaceVolumeDataModule, JointSurfaceVolumeDataModule
from modules.context_module import IN_CONTEXT_MODELS, ContextTrainModule
from modules.train_module import TrainModule


def process_args(args, config):
    modelconfig = config['model']
    trainconfig = config['training']
    dataconfig = config['data']

    if len(args.devices) > 0:
        trainconfig["devices"] = [int(device) for device in args.devices]
    if args.seed is not None:
        trainconfig["seed"] = args.seed
    if args.wandb_mode is not None:
        trainconfig["wandb_mode"] = args.wandb_mode
    if args.model_name is not None:
        modelconfig["model_name"] = args.model_name
    if args.checkpoint is not None:
        trainconfig["checkpoint"] = args.checkpoint
    if args.description is not None:
        trainconfig["description"] = args.description
    if args.data_root is not None:
        dataconfig["data_root"] = args.data_root
    if args.max_steps is not None:
        trainconfig["max_steps"] = args.max_steps
    if args.save_every_n_train_steps is not None:
        trainconfig["save_every_n_train_steps"] = args.save_every_n_train_steps
    if args.archive_every_n_train_steps is not None:
        trainconfig["archive_every_n_train_steps"] = args.archive_every_n_train_steps
    if args.val_every_n_steps is not None:
        trainconfig["val_every_n_steps"] = args.val_every_n_steps

    return config, modelconfig, trainconfig, dataconfig


def main(args):
    config = get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    description = trainconfig.get("description", "")
    name = modelconfig["model_name"] + "_" + description + "_" + str(seed) + "_" + now
    wandb_logger = WandbLogger(project=trainconfig["project"],
                               name=name,
                               mode=trainconfig["wandb_mode"])
    path = trainconfig["log_dir"] + name + "/"
    modelconfig['log_dir'] = path

    os.makedirs(path, exist_ok=True)
    print(f"Logging to: {path}")

    if dataconfig.get("type", "combined") == "joint":
        datamodule = JointSurfaceVolumeDataModule(dataconfig=dataconfig)
    else:
        datamodule = SurfaceVolumeDataModule(dataconfig=dataconfig)

    save_yaml(config, path + "config.yml")

    module_cls = (ContextTrainModule if modelconfig["model_name"] in IN_CONTEXT_MODELS
                  else TrainModule)
    model = module_cls(config=config, ds_train=datamodule.train_dataset)
    # Joint runs denormalize val metrics with each dataset's own stats.
    if getattr(datamodule, "val_datasets", None) is not None:
        model.val_datasets = datamodule.val_datasets

    callbacks = []

    save_every_n_train_steps = trainconfig.get("save_every_n_train_steps", 1000)
    archive_every_n_train_steps = trainconfig.get("archive_every_n_train_steps", 50000)

    max_steps = trainconfig.get("max_steps", None)
    if max_steps is not None and int(max_steps) > 0:
        max_steps = int(max_steps)
        max_epochs = -1
        print(f"Training for {max_steps} gradient steps (max_epochs disabled); "
              f"lr x{trainconfig.get('lr_decay_gamma', 0.99)} every "
              f"{trainconfig.get('lr_decay_every_n_steps', 10000)} steps")
    else:
        max_steps = -1
        max_epochs = trainconfig["max_epochs"]
        print(f"Training for {max_epochs} epochs")

    callbacks.append(ModelCheckpoint(
        dirpath=path,
        every_n_train_steps=save_every_n_train_steps,
        save_last=True,
        save_top_k=0,
    ))

    if archive_every_n_train_steps is not None and int(archive_every_n_train_steps) > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=path,
            filename="model_step_{step:08d}",
            auto_insert_metric_name=False,
            every_n_train_steps=int(archive_every_n_train_steps),
            save_last=False,
            save_top_k=-1,
        ))
        print(f"Archiving a permanent checkpoint every {int(archive_every_n_train_steps)} "
              f"gradient steps; last.ckpt refreshed every {save_every_n_train_steps}")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    accumulate_grad_batches = trainconfig.get("accumulate_grad_batches", 1)

    # val_check_interval counts batches, so scale by accumulation to get gradient steps.
    val_every_n_steps = trainconfig.get("val_every_n_steps", None)
    if val_every_n_steps is not None and int(val_every_n_steps) > 0:
        val_every_n_steps = int(val_every_n_steps)
        val_kwargs = {"check_val_every_n_epoch": None,
                      "val_check_interval": val_every_n_steps * accumulate_grad_batches}
        print(f"Validating every {val_every_n_steps} gradient steps")
    else:
        val_kwargs = {"check_val_every_n_epoch": trainconfig["check_val_every_n_epoch"]}
        print(f"Validating every {trainconfig['check_val_every_n_epoch']} epoch(s)")

    precision = trainconfig.get("precision", None)
    trainer_kwargs = {} if precision is None else {"precision": precision}

    trainer = L.Trainer(devices=trainconfig["devices"],
                        num_nodes=trainconfig.get("num_nodes", 1),
                        accelerator=trainconfig["accelerator"],
                        strategy=trainconfig["strategy"],
                        log_every_n_steps=trainconfig["log_every_n_steps"],
                        max_epochs=max_epochs,
                        max_steps=max_steps,
                        default_root_dir=path,
                        callbacks=callbacks + [lr_monitor, EMAWeightAveraging(trainconfig["ema_decay"])],
                        logger=wandb_logger,
                        accumulate_grad_batches=accumulate_grad_batches,
                        num_sanity_val_steps=trainconfig.get("num_sanity_val_steps", 1),
                        **val_kwargs,
                        **trainer_kwargs)

    if trainconfig["checkpoint"] is not None:
        trainer.fit(model=model, datamodule=datamodule,
                    ckpt_path=trainconfig["checkpoint"], weights_only=False)
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", required=True)
    parser.add_argument('--data_root', default=None, help='Dataset root (overrides data.data_root and $CFD_DATA_ROOT).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', default=[], help='GPU indices.')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None, help='online | offline | disabled')
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to resume training from.')
    parser.add_argument('--max_steps', type=int, default=None, help='Stop after N gradient steps (overrides training.max_epochs).')
    parser.add_argument('--save_every_n_train_steps', type=int, default=None, help='Refresh last.ckpt every N gradient steps.')
    parser.add_argument('--archive_every_n_train_steps', type=int, default=None, help='Keep a permanent checkpoint every N gradient steps (0 disables).')
    parser.add_argument('--val_every_n_steps', type=int, default=None, help='Validate every N gradient steps instead of per epoch.')
    args = parser.parse_args()

    main(args)
