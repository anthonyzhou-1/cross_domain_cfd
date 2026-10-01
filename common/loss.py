import torch


def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)


class ScaledLpLoss(torch.nn.Module):
    """Relative Lp loss per sample, ||input - target||_p / ||target||_p, then reduced."""

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)
