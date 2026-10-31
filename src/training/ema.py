"""Exponential moving average of model weights (parameters AND buffers, e.g. BatchNorm stats)."""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            s = src[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(s, alpha=1.0 - self.decay)
            else:
                v.copy_(s)
