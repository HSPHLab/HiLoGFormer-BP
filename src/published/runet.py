"""
rU-Net (Chen et al., IEEE JBHI 29(1), 2025) - reconstructed from the paper (no official code).
Specification, decisions U1-U8 and every choice where the paper is silent: docs/reimplementation/rU-Net.md (section 4).

  python -m src.published.runet      # self-test: prints the shape after every stage and the parameter count

Input:  x (B, 2, 1250) = processed ECG, PPG (10 s at 125 Hz); demo (B, 5) = standardised sex, age, height, weight, BMI
Output: (B, 2) = [SBP, DBP] in mmHg (the network predicts standardised values; y_mean / y_std buffers convert back, U5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

C1, C2, C3 = 32, 64, 128             # filters per level (paper)
D = 128                              # branch / fusion token width (paper: 250 x 128 outputs)


def conv(dim, *a, **k):
    return (nn.Conv1d if dim == 1 else nn.Conv2d)(*a, **k)


class RSB(nn.Module):
    """Residual block: conv3(stride) -> ReLU -> conv3 -> + conv1x1(stride) shortcut -> ReLU (Fig. 2c; no BN shown)."""

    def __init__(self, cin, cout, stride, dim):
        super().__init__()
        self.c1 = conv(dim, cin, cout, 3, stride=stride, padding=1)
        self.c2 = conv(dim, cout, cout, 3, padding=1)
        self.sc = conv(dim, cin, cout, 1, stride=stride)

    def forward(self, x):
        return F.relu(self.c2(F.relu(self.c1(x))) + self.sc(x))


class CBAM(nn.Module):
    """Channel then spatial attention (Woo et al. 2018 defaults: reduction 16, kernel 7)."""

    def __init__(self, c, dim, reduction=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // reduction, 1)), nn.ReLU(), nn.Linear(max(c // reduction, 1), c))
        self.spatial = conv(dim, 2, 1, k, padding=k // 2)
        self.dims = tuple(range(2, 2 + dim))

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean(self.dims)) + self.mlp(x.amax(self.dims)))
        x = x * w.view(*w.shape, *([1] * len(self.dims)))
        s = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)))
        return x * s


class ScaleAttention(nn.Module):
    """'ATT' at one rU-Net scale: Q = K = V = [features || tiled demographics] (paper Eq. 3), single head over time
    positions, projections back to C channels, residual. On 2-D maps the attention weights come from the
    frequency-averaged tokens and are applied to every frequency row (ours, D7)."""

    def __init__(self, c, n_demo, dim):
        super().__init__()
        self.dim = dim
        self.demo = nn.Sequential(nn.Linear(n_demo, 16), nn.ReLU(), nn.Linear(16, c))
        self.q, self.k = nn.Linear(2 * c, c), nn.Linear(2 * c, c)
        self.v = conv(dim, 2 * c, c, 1)
        self.o = conv(dim, c, c, 1)
        self.scale = c ** -0.5

    def forward(self, x, demo):
        d = self.demo(demo)                                            # (B, C)
        d = d.view(*d.shape, *([1] * self.dim)).expand_as(x)           # tiled over positions
        z = torch.cat([x, d], dim=1)                                   # (B, 2C, T[, F])
        tok = z.transpose(1, 2) if self.dim == 1 else z.mean(3).transpose(1, 2)    # (B, T, 2C)
        a = torch.softmax(self.q(tok) @ self.k(tok).transpose(1, 2) * self.scale, dim=-1)   # (B, T, T)
        v = self.v(z)
        out = torch.einsum("bts,bcs->bct", a, v) if self.dim == 1 else torch.einsum("bts,bcsf->bctf", a, v)
        return x + self.o(out)


class RUNet(nn.Module):
    """1-D or 2-D rU-Net (Fig. 1b): 3 encoder levels (32/64/128, strides 1, 2, 5 / (5,2)), CBAM at the deepest scale,
    attention at every scale, decoder with up-sampling and skip concatenation; output C1 channels at full resolution."""

    def __init__(self, cin, n_demo, dim):
        super().__init__()
        s2, s3 = (2, 5) if dim == 1 else ((2, 2), (5, 2))
        self.e1, self.e2, self.e3 = RSB(cin, C1, 1, dim), RSB(C1, C2, s2, dim), RSB(C2, C3, s3, dim)
        self.cbam = CBAM(C3, dim)
        self.a1, self.a2, self.a3 = ScaleAttention(C1, n_demo, dim), ScaleAttention(C2, n_demo, dim), ScaleAttention(C3, n_demo, dim)
        self.up3 = nn.Upsample(scale_factor=s3, mode="nearest")
        self.up2 = nn.Upsample(scale_factor=s2, mode="nearest")
        self.d2, self.d1 = RSB(C3 + C2, C2, 1, dim), RSB(C2 + C1, C1, 1, dim)

    def forward(self, x, demo, trace=None):
        f1 = self.e1(x)
        f2 = self.e2(f1)
        f3 = self.cbam(self.e3(f2))
        y = self.a3(f3, demo)
        y = self.d2(torch.cat([self.up3(y), self.a2(f2, demo)], dim=1))      # 125 -> 625 (1-D); 25x32 -> 125x64 (2-D)
        y = self.d1(torch.cat([self.up2(y), self.a1(f1, demo)], dim=1))      # 625 -> 1250;       125x64 -> 250x128
        if trace is not None:
            trace += [("enc1", f1.shape), ("enc2", f2.shape), ("enc3+CBAM", f3.shape), ("dec out", y.shape)]
        return y


class TokenAttention(nn.Module):
    """'Concat -> ATT' after each branch: tokens [branch (250), demographic token (1)] -> attention + residual."""

    def __init__(self, d, heads=1):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, tok):
        return tok + self.att(tok, tok, tok, need_weights=False)[0]


class rUNetBP(nn.Module):
    def __init__(self, n_demo=5, seq_len=1250, n_fft=256, hop=5, frames=250, bins=128, reg_hidden=64):
        super().__init__()
        self.n_fft, self.hop, self.frames, self.bins = n_fft, hop, frames, bins
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))
        self.unet1d, self.unet2d = RUNet(2, n_demo, 1), RUNet(2, n_demo, 2)
        self.pool = nn.AvgPool1d(seq_len // frames)                    # 1250 -> 250
        self.gru_t, self.res_t = nn.GRU(C1, D // 2, batch_first=True, bidirectional=True), nn.Linear(C1, D)
        self.fc1 = nn.Linear(C1, 1)                                     # "FC with units = 1" over the channels
        self.gru_f, self.res_f = nn.GRU(bins, D // 2, batch_first=True, bidirectional=True), nn.Linear(bins, D)
        self.demo_tok = nn.Linear(n_demo, D)
        self.att_t, self.att_f = TokenAttention(D), TokenAttention(D)
        self.fusion = nn.MultiheadAttention(D, 2, batch_first=True)    # paper: 2 heads
        n_tok = 2 * frames + 1
        self.heads = nn.ModuleList([nn.Sequential(nn.Flatten(), nn.Linear(n_tok * D, reg_hidden), nn.ReLU(), nn.Linear(reg_hidden, 1))
                                    for _ in range(2)])

    def stft(self, x):
        b = x.shape[0]
        s = torch.stft(x.reshape(b * 2, -1).float(), n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                       center=True, return_complex=True).abs()        # (2B, 129, 251)
        return s[:, :self.bins, :self.frames].reshape(b, 2, self.bins, self.frames).transpose(2, 3)   # (B, 2, 250, 128)

    def forward_std(self, x, demo, trace=None):
        dt = self.demo_tok(demo).unsqueeze(1)                          # (B, 1, D)
        h1 = self.unet1d(x, demo, trace)                               # (B, 32, 1250)
        t = self.pool(h1).transpose(1, 2)                              # (B, 250, 32)
        t = self.gru_t(t)[0] + self.res_t(t)                           # (B, 250, 128)
        t = self.att_t(torch.cat([t, dt], dim=1))[:, :self.frames]
        S = self.stft(x)
        h2 = self.unet2d(S, demo, trace)                               # (B, 32, 250, 128)
        f = self.fc1(h2.permute(0, 2, 3, 1)).squeeze(-1)               # (B, 250, 128)
        f = self.gru_f(f)[0] + self.res_f(f)
        f = self.att_f(torch.cat([f, dt], dim=1))[:, :self.frames]
        tok = torch.cat([f, t, dt], dim=1)                             # (B, 501, 128)
        tok = tok + self.fusion(tok, tok, tok, need_weights=False)[0]
        if trace is not None:
            trace += [("STFT", S.shape), ("time branch", t.shape), ("STFT branch", f.shape), ("fusion tokens", tok.shape)]
        return torch.cat([h(tok) for h in self.heads], dim=1)          # standardised [SBP, DBP]

    def forward(self, x, demo):
        return self.forward_std(x, demo) * self.y_std + self.y_mean    # mmHg


if __name__ == "__main__":
    m = rUNetBP()
    trace = []
    with torch.no_grad():
        out = m.forward_std(torch.randn(2, 2, 1250), torch.randn(2, 5), trace)
    for name, shp in trace:
        print(f"{name:16s} {tuple(shp)}")
    print(f"output           {tuple(out.shape)}")
    print(f"parameters       {sum(p.numel() for p in m.parameters()):,}")
