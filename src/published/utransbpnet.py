"""
UTransBPNet (Zheng et al., Scientific Reports 15:17654, 2025) - reconstructed from the paper (no official code).
Specification, decisions T1-T9 and every choice where the paper is silent: docs/reimplementation/UTransBPNet.md (section 4).

  python -m src.published.utransbpnet      # self-test: encoder-layer equivalence, shape after every stage, parameters

Input:  x (B, 2, 1250) = processed ECG, PPG (10 s at 125 Hz); inside the model: VPPG, APPG = 1st / 2nd derivative of the
        PPG, each of the 4 channels min-max scaled to (0, 1) per window (T3).
Output: (B, 2) = [SBP, DBP] in mmHg (the network predicts min-max-scaled values; y_min / y_range buffers convert back, T5).
Architecture: AllTune variant trained end-to-end on SBP/DBP (T1): U-Net -> 1-channel output at input resolution -> FC -> 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 125
WIDTHS = (128, 256, 384, 512)        # Fig. 2 (approximate)
POOL = 5                             # three x5 mean-poolings: 1250 -> 250 -> 50 -> 10


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv1d(cin, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class SEBlock(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.f = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(c, max(c // reduction, 1), 1), nn.ReLU(inplace=True),
                               nn.Conv1d(max(c // reduction, 1), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.f(x)


class ConvSE(nn.Module):
    """3 Conv_blocks -> SE -> + residual (1x1 projection if the channel number changes)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.blocks = nn.Sequential(ConvBlock(cin, cout), ConvBlock(cout, cout), ConvBlock(cout, cout))
        self.se = SEBlock(cout)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.se(self.blocks(x)) + self.res(x)


class MHCA(nn.Module):
    """Multi-head cross attention (Petit et al. 2021, as adopted by the paper): Q, K from the deeper map Y, V from the skip
    map S pooled to Y's resolution; output -> 1x1 conv -> BN -> sigmoid -> up-sampled = filter Z; returns S * Z."""

    def __init__(self, cy, cs, heads=8):
        super().__init__()
        self.h = heads
        self.q, self.k = nn.Conv1d(cy, cs, 1), nn.Conv1d(cy, cs, 1)
        self.v = nn.Conv1d(cs, cs, 1)
        self.z = nn.Sequential(nn.Conv1d(cs, cs, 1), nn.BatchNorm1d(cs), nn.Sigmoid())

    def forward(self, y, s):
        b, c, t = y.shape[0], s.shape[1], y.shape[2]
        q = self.q(y).view(b, self.h, c // self.h, t)
        k = self.k(y).view(b, self.h, c // self.h, t)
        v = self.v(F.avg_pool1d(s, s.shape[2] // t)).view(b, self.h, c // self.h, t)
        a = torch.softmax(torch.einsum("bhdt,bhds->bhts", q, k) * (c // self.h) ** -0.5, dim=-1)
        o = torch.einsum("bhts,bhds->bhdt", a, v).reshape(b, c, t)
        z = F.interpolate(self.z(o), size=s.shape[2], mode="nearest")
        return s * z


class UpBlock(nn.Module):
    """Decoder scale: up-sample Y (x5) + Conv_block; MHCA-filtered skip S; concat -> Conv_SE."""

    def __init__(self, cy, cs):
        super().__init__()
        self.up = ConvBlock(cy, cs)
        self.att = MHCA(cy, cs)
        self.fuse = ConvSE(2 * cs, cs)

    def forward(self, y, s):
        yu = self.up(F.interpolate(y, size=s.shape[2], mode="nearest"))
        return self.fuse(torch.cat([self.att(y, s), yu], dim=1))


class EncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer written out explicitly (same parameters and computation as
    nn.TransformerEncoderLayer(norm_first=True, activation=relu) in its standard / training path). Used instead of the
    PyTorch module because its fused inference fast paths (layer and nn.MultiheadAttention) returned non-finite outputs
    in PyTorch 2.0.1 (docs/reimplementation/UTransBPNet.md, section 7)."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.h = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop_att, self.drop1, self.drop2, self.drop_ff = (nn.Dropout(dropout) for _ in range(4))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = self.drop_att(torch.softmax(q @ k.transpose(-2, -1) * (d // self.h) ** -0.5, dim=-1))
        x = x + self.drop1(self.proj((a @ v).transpose(1, 2).reshape(b, n, d)))
        return x + self.drop2(self.linear2(self.drop_ff(F.relu(self.linear1(self.norm2(x))))))


class UTransBPNet(nn.Module):
    def __init__(self, seq_len=1250, depth=12, heads=8, ff=1024, dropout=0.1, n_side=1):
        super().__init__()
        c1, c2, c3, c4 = WIDTHS
        self.e1, self.e2, self.e3, self.e4 = ConvSE(4, c1), ConvSE(c1 + 4, c2), ConvSE(c2 + 4, c3), ConvSE(c3 + 4, c4)
        n_tok = seq_len // POOL ** 3
        self.pos = nn.Parameter(torch.zeros(1, n_tok, c4))                       # learned positional encoding
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.transformer = nn.Sequential(*[EncoderLayer(c4, heads, ff, dropout) for _ in range(depth)])
        self.d3, self.d2, self.d1 = UpBlock(c4, c3), UpBlock(c3, c2), UpBlock(c2, c1)
        self.out = nn.Conv1d(c1, 1, 1)                                           # 1-channel output at input resolution
        self.head = nn.Linear(seq_len, 2)                                         # AllTune FC
        self.register_buffer("y_min", torch.zeros(2))
        self.register_buffer("y_range", torch.ones(2))

    @staticmethod
    def inputs(x):
        """[ECG, PPG] -> [ECG, PPG, VPPG, APPG], each min-max scaled to (0, 1) per window (T3)."""
        ppg = x[:, 1:2]
        v = torch.gradient(ppg, dim=-1)[0] * FS
        a = torch.gradient(v, dim=-1)[0] * FS
        z = torch.cat([x[:, 0:1], ppg, v, a], dim=1)
        lo, hi = z.amin(-1, keepdim=True), z.amax(-1, keepdim=True)
        return (z - lo) / (hi - lo).clamp_min(1e-6)

    def forward_scaled(self, x, side=None, trace=None):
        z = self.inputs(x.float())
        p = lambda t: F.avg_pool1d(t, POOL)
        f1 = self.e1(z)
        z2 = p(z); f2 = self.e2(torch.cat([p(f1), z2], 1))
        z3 = p(z2); f3 = self.e3(torch.cat([p(f2), z3], 1))
        z4 = p(z3); f4 = self.e4(torch.cat([p(f3), z4], 1))
        t = self.transformer(f4.transpose(1, 2) + self.pos).transpose(1, 2)
        y = self.d3(t, f3)
        y = self.d2(y, f2)
        y = self.d1(y, f1)
        w = self.out(y)
        if trace is not None:
            trace += [("input (4 ch)", z.shape), ("enc1", f1.shape), ("enc2", f2.shape), ("enc3", f3.shape), ("enc4", f4.shape),
                      ("transformer", t.shape), ("dec out", y.shape), ("1-ch output", w.shape)]
        return self.head(w.flatten(1))                                           # min-max-scaled [SBP, DBP]

    def forward(self, x, side=None):
        return self.forward_scaled(x, side) * self.y_range + self.y_min          # mmHg


def check_encoder_layer(d=512, heads=8, ff=1024):
    """Our EncoderLayer == nn.TransformerEncoderLayer (standard path, dropout 0) with copied weights."""
    ref = nn.TransformerEncoderLayer(d, heads, ff, 0.0, batch_first=True, norm_first=True).train()
    ours = EncoderLayer(d, heads, ff, 0.0).train()
    ours.qkv.weight.data.copy_(ref.self_attn.in_proj_weight); ours.qkv.bias.data.copy_(ref.self_attn.in_proj_bias)
    ours.proj.load_state_dict(ref.self_attn.out_proj.state_dict())
    for a in ("linear1", "linear2", "norm1", "norm2"):
        getattr(ours, a).load_state_dict(getattr(ref, a).state_dict())
    x = torch.randn(4, 10, d)
    return (ours(x) - ref(x)).abs().max().item()


if __name__ == "__main__":
    print(f"encoder layer vs nn.TransformerEncoderLayer (standard path): max |difference| {check_encoder_layer():.2e}")
    m = UTransBPNet()
    trace = []
    with torch.no_grad():
        out = m.forward_scaled(torch.randn(2, 2, 1250), None, trace)
    for name, shp in trace:
        print(f"{name:14s} {tuple(shp)}")
    print(f"output         {tuple(out.shape)}")
    n = sum(p.numel() for p in m.parameters())
    print(f"parameters     {n:,}  (paper: 32.55 M)")
