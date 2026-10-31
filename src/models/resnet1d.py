"""
ResNet1D reference baseline (same architecture as the first experiments in train_bp.py):
a residual 1D CNN on the stacked input channels, global average pooling, MLP head.
"""
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = nn.Identity()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))


@register_model("resnet1d")
class ResNet1D(nn.Module):
    def __init__(self, input_names, n_targets=2, n_aux=0, width=32, dropout=0.2, blocks=(2, 2, 2, 2),
                 kernel=7):
        super().__init__()
        in_ch = len(input_names)
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 15, 2, 7, bias=False), nn.BatchNorm1d(width),
                                  nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        layers, cin = [], width
        for i, nb in enumerate(blocks):
            cout = width * 2 ** i
            for j in range(nb):
                layers.append(ResBlock1D(cin, cout, stride=2 if (j == 0 and i > 0) else 1, k=kernel))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, n_targets))
        self.aux_head = nn.Linear(cin, n_aux) if n_aux > 0 else None

    def forward(self, x):
        h = self.body(self.stem(x)).mean(dim=-1)
        out = {"bp": self.head(h)}
        if self.aux_head is not None:
            out["aux"] = self.aux_head(h)
        return out
