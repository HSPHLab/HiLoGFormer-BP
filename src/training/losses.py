"""Losses on standardised targets."""
import torch
import torch.nn.functional as F


def main_loss(pred, y, kind="huber", delta=1.0):
    pred = pred.float()
    if kind == "huber":
        return F.huber_loss(pred, y, delta=delta)
    if kind == "l1":
        return F.l1_loss(pred, y)
    if kind == "mse":
        return F.mse_loss(pred, y)
    raise ValueError(f"unknown loss {kind}")


def masked_aux_loss(pred, target, mask, delta=1.0):
    """Huber loss averaged over valid auxiliary entries only."""
    if pred is None or mask.numel() == 0 or mask.sum() == 0:
        return torch.zeros((), device=target.device)
    l = F.huber_loss(pred.float(), target, delta=delta, reduction="none")
    return (l * mask).sum() / mask.sum().clamp_min(1.0)
