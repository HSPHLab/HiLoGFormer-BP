"""
HGCTNet (Liu et al., IEEE JBHI 28(7), 2024) - ported from the official notebook
https://github.com/zdzdliu/HGCTNet (HGCTNet.ipynb, commit 6d5103cac4a6fe77d5033ae96fc4e9e68f2aee30).

Kept exactly: CNN (make_layers + BasicBlock), additive attention (HGA), transformer encoder (pre-LN, attention scaled
by sqrt(emb_size)), fusion, regressor, calibration BP = alpha * pred + (1 - alpha) * baseline with alpha ~ randn,
Kaiming init. Changed only through constructor arguments (docs/reimplementation/HGCTNet.md):
  in_channels  3 -> 2 (ECG, PPG)            n_feats 47 -> 31 (ECG/PPG features)
  key_size     111 -> time steps after the CNN for our 1,250-sample input (computed, 12)
The notebook's `import math`, `import os`, `import torch` (used but not imported in the code cells) are added.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CFG = {"A": [32, 32, "M", 64, 64, "M"]}


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super().__init__()
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries, keys):
        queries = self.w_q(queries.unsqueeze(1))
        keys = self.w_k(keys)
        return self.softmax(self.w_v(torch.tanh(queries + keys)))


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1, dilation=1)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)] if batch_norm else [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, shortcut=True):
        super().__init__()
        self.downsample = None
        self.conv1, self.bn1 = conv3x3(inplanes, planes), nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2, self.bn2 = conv3x3(planes, planes), nn.BatchNorm1d(planes)
        self.conv3, self.bn3 = conv3x3(planes, planes), nn.BatchNorm1d(planes)
        self.shortcut = shortcut
        if inplanes != planes:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes), nn.BatchNorm1d(planes))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.shortcut:
            out += identity
        return self.relu(out)


class CNN_Net(nn.Module):
    def __init__(self, inplanes=3):
        super().__init__()
        self.features = nn.Sequential(
            make_layers(CFG["A"], in_channels=inplanes, batch_norm=True),
            BasicBlock(inplanes=64, planes=128), nn.MaxPool1d(kernel_size=2, stride=3),
            BasicBlock(inplanes=128, planes=256), nn.MaxPool1d(kernel_size=2, stride=3),
            BasicBlock(inplanes=256, planes=256), nn.MaxPool1d(kernel_size=2, stride=3))

    def forward(self, x):
        return self.features(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size, self.num_heads = emb_size, num_heads
        self.keys, self.queries, self.values = (nn.Linear(emb_size, emb_size) for _ in range(3))
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", q, k)
        att = self.att_drop(F.softmax(energy / self.emb_size ** 0.5, dim=-1))      # sic: sqrt(emb_size), as published
        out = rearrange(torch.einsum("bhal, bhlv -> bhav ", att, v), "b h n d -> b n (h d)")
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(nn.Linear(emb_size, expansion * emb_size), nn.ReLU(), nn.Dropout(drop_p),
                         nn.Linear(expansion * emb_size, emb_size))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=4, drop_p=0.1, forward_expansion=2, forward_drop_p=0.1):
        super().__init__(
            ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size), MultiHeadAttention(emb_size, num_heads, drop_p), nn.Dropout(drop_p))),
            ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                      FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                                      nn.Dropout(drop_p))))


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads, pos=True):
        super().__init__()
        self.emb = PositionalEncoding(d_model=emb_size, dropout=0.1, max_len=256)
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size, num_heads=num_heads) for _ in range(depth)])
        self.pos = pos

    def forward(self, x):
        if self.pos:
            x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        return x


class HGCTNet(nn.Module):
    """feas = [baseline SBP, baseline DBP (mmHg), n_demo demographics, n_feats handcrafted features] (as the notebook)."""

    def __init__(self, in_channels=2, seq_len=1250, n_feats=31, n_demo=4, num_classes=2, num_hid=8, depth=2,
                 num_heads=2, hga=True, trans=True, init_weights=True):
        super().__init__()
        self.hga, self.trans, self.n_demo, self.n_feats = hga, trans, n_demo, n_feats
        self.cnn = CNN_Net(inplanes=in_channels)
        with torch.no_grad():
            key_size = self.cnn(torch.zeros(1, in_channels, seq_len)).shape[-1]       # 111 in the notebook; 12 here
        self.key_size = key_size
        self.regressor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(), nn.Linear(128, num_classes))
        self.cnn_attention = AdditiveAttention(key_size=key_size, query_size=n_feats, num_hiddens=num_hid)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.emblay = nn.Sequential(nn.Linear(n_demo + n_feats, 128, bias=True), nn.ReLU(True), nn.Dropout(0.5),
                                    nn.Linear(128, 256, bias=True))
        self.rnn = TransformerEncoder(depth=depth, emb_size=256, num_heads=num_heads, pos=True)
        self.balance_alpha = nn.Parameter(torch.randn(2))
        if init_weights:
            self._initialize_weights()

    def forward(self, x, feas):
        basebp = feas[:, 0:2]
        demos = feas[:, 2:2 + self.n_demo]
        hand_feas = feas[:, 2 + self.n_demo:2 + self.n_demo + self.n_feats]
        x_s = self.cnn(x)
        if self.hga:
            x_s = torch.mul(self.cnn_attention(hand_feas, x_s), x_s)
        x_res = self.avgpool(x_s).view(x_s.size(0), -1)
        if self.trans:
            x_s = self.rnn(x_s.permute(0, 2, 1)).permute(0, 2, 1)
        x_s = self.avgpool(x_s)[:, :, 0]
        x = torch.mul(x_s + x_res, self.emblay(torch.cat([demos, hand_feas], dim=1)))
        x = self.regressor(x)
        return x * self.balance_alpha + basebp * (1 - self.balance_alpha)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()
