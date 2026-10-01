import argparse
import os
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

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

    return config, modelconfig, trainconfig, dataconfig


def main(args):
    config = get_yaml(args.config)
    config, modelconfig, trainconfig, dataconfig = process_args(args, config)

    seed = trainconfig["seed"]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    description = trainconfig.get("description", "")
    name = "VAL_" + modelconfig["model_name"] + "_" + description + "_" + str(seed) + "_" + now
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
    if getattr(datamodule, "val_datasets", None) is not None:
        model.val_datasets = datamodule.val_datasets

    trainer = L.Trainer(devices=trainconfig["devices"],
                        num_nodes=trainconfig.get("num_nodes", 1),
                        accelerator=trainconfig["accelerator"],
                        strategy=trainconfig["strategy"],
                        check_val_every_n_epoch=trainconfig["check_val_every_n_epoch"],
                        log_every_n_steps=trainconfig["log_every_n_steps"],
                        default_root_dir=path,
                        logger=wandb_logger)

    # last.ckpt stores the EMA weights, so this validates the averaged model.
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=trainconfig["checkpoint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a model')
    parser.add_argument("--config", required=True)
    parser.add_argument('--data_root', default=None, help='Dataset root (overrides data.data_root and $CFD_DATA_ROOT).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--devices', nargs='+', default=[], help='GPU indices.')
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--wandb_mode', default=None, help='online | offline | disabled')
    parser.add_argument('--description', default=None)
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to validate.')
    args = parser.parse_args()

    main(args)
