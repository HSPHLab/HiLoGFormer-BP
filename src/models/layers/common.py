"""Small building blocks shared by models."""
import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: randomly drops the whole residual branch per sample during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.base_prob = float(drop_prob)   # rate from the depth schedule
        self.drop_prob = float(drop_prob)   # current rate (scaled during warm-up)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)
