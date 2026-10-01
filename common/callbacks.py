from torch.optim.swa_utils import get_ema_avg_fn

from lightning.pytorch.callbacks import WeightAveraging


class EMAWeightAveraging(WeightAveraging):
    """EMA of the weights, updated every optimizer step; validation and checkpoints use the EMA."""

    def __init__(self, decay=0.99):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))
        self.decay = decay

    def should_update(self, step_idx=None, epoch_idx=None):
        return True
