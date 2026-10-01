import yaml
import torch 
import math 

def get_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def save_yaml(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

# Parameter-name substrings of the conditioning path (cond encoder, AdaLN modulation, gates).
COND_PARAM_KEYS = ("cond_proj", "adaLN_modulation", "cond_gate")


def split_cond_params(model, keys=None):
    """Partition a model's trainable parameters into (conditioning, everything else)."""
    if keys is None:
        keys = getattr(model, "COND_PARAM_KEYS", COND_PARAM_KEYS)
    cond, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (cond if any(k in name for k in keys) else rest).append(p)
    return cond, rest


def cond_param_group(model, lr, cond_lr_mult):
    """The conditioning path as its own parameter group (lr * cond_lr_mult, no weight decay), or None."""
    if cond_lr_mult is None:
        return None
    cond_params, _ = split_cond_params(model)
    if not cond_params:
        raise ValueError(
            "model.cond_lr_mult is set but the model has no conditioning parameters "
            "(num_cond=0?)"
        )
    return dict(params=cond_params, lr=lr * cond_lr_mult,
                betas=(0.9, 0.95), weight_decay=0.0)


def adam_param_groups(model, lr, cond_lr_mult=None):
    """Adam groups: one flat group, plus the conditioning path when cond_lr_mult is set."""
    cond = cond_param_group(model, lr, cond_lr_mult)
    if cond is None:
        return [dict(params=[p for p in model.parameters() if p.requires_grad])]
    taken = {id(p) for p in cond["params"]}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in taken]
    return [dict(params=base, lr=lr), dict(params=cond["params"], lr=cond["lr"])]


def finetune_optim_config(config):
    """(adam betas, lr_schedule) from the config's `finetune` block (written by finetune.py)."""
    ftconfig = config.get("finetune") or {}
    betas = ftconfig.get("betas") or (0.9, 0.999)
    return tuple(betas), ftconfig.get("lr_schedule")


def cosine_warmup_scheduler(optimizer, total_steps, warmup_steps=0, min_lr_ratio=0.0):
    """Per-step linear warmup then cosine decay to min_lr_ratio * lr, as a LambdaLR."""
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def step_lr_config(config):
    """StepLR settings for a run capped by training.max_steps, or None for an epoch budget."""
    trainconfig = config.get("training") or {}
    max_steps = trainconfig.get("max_steps", None)
    if max_steps is None or int(max_steps) <= 0:
        return None
    return {"step_size": int(trainconfig.get("lr_decay_every_n_steps", 1000)),
            "gamma": float(trainconfig.get("lr_decay_gamma", 0.99))}


def build_lr_scheduler(optimizer, lr_schedule=None, name=None, step_lr=None):
    """Lightning scheduler config: finetune cosine (lr_schedule) > per-N-step StepLR > per-epoch StepLR."""
    if lr_schedule is not None:
        config = {"scheduler": cosine_warmup_scheduler(optimizer, **lr_schedule),
                  "interval": "step"}
    elif step_lr is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_lr["step_size"], gamma=step_lr["gamma"])
        config = {"scheduler": scheduler, "interval": "step"}
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        config = {"scheduler": scheduler, "interval": "epoch"}
    if name is not None:
        config["name"] = name
    return config


def build_muon_optimizer(param_groups):
    """Muon + auxiliary AdamW: the distributed variant only when world_size > 1; empty groups dropped."""
    import torch.distributed as dist
    from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

    groups = [g for g in param_groups if len(g["params"]) > 0]
    if not groups:
        raise ValueError("Muon optimizer got an empty parameter list")

    distributed = (dist.is_available() and dist.is_initialized()
                   and dist.get_world_size() > 1)
    cls = MuonWithAuxAdam if distributed else SingleDeviceMuonWithAuxAdam
    return cls(groups)
