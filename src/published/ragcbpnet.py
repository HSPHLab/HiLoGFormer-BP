"""
RAGCBPNet (Huang et al., medRxiv 2025, doi 10.1101/2025.10.30.25339198) - reconstructed from the preprint (no code).
Specification, decisions G1-G9 and every choice where the preprint is silent: docs/reimplementation/RAGCBPNet.md.

  python -m src.published.ragcbpnet      # self-test: shapes and parameter count

Input:  x (B, 2, 1250) = our processed ECG, PPG; only the PPG is used (G2): PPG, VPPG, APPG, each min-max scaled per window.
        side (B, 9 + n_feats) = [standardised sex, age, height, weight, BMI, cos(hour), sin(hour), baseline SBP, baseline DBP,
                                 standardised handcrafted features]
Output: (B, 2) = [SBP, DBP] in mmHg (standardised internally; y_mean / y_std buffers convert back, G8).
Regression part only (the preprint's contrastive hypertension classifier is a separate task).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 125
N_SIDE = 9            # 5 demographics + cos/sin hour + baseline SBP/DBP
D = 256               # fusion width (ours)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, width, stride):
        super().__init__()
        cout = width * self.expansion
        self.c1, self.b1 = nn.Conv1d(cin, width, 1, bias=False), nn.BatchNorm1d(width)
        self.c2, self.b2 = nn.Conv1d(width, width, 3, stride=stride, padding=1, bias=False), nn.BatchNorm1d(width)
        self.c3, self.b3 = nn.Conv1d(width, cout, 1, bias=False), nn.BatchNorm1d(cout)
        self.down = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        return F.relu(self.b3(self.c3(y)) + self.down(x))


class ResNet50_1D(nn.Module):
    """ResNet50 (bottleneck blocks 3-4-6-3, widths 64-128-256-512, expansion 4) with 1-D convolutions."""

    def __init__(self, cin=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(cin, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool1d(3, stride=2, padding=1))
        layers, c = [], 64
        for width, n, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
            for i in range(n):
                layers.append(Bottleneck(c, width, stride if i == 0 else 1))
                c = width * Bottleneck.expansion
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.layers(self.stem(x))


class CBAM1D(nn.Module):
    """Channel then spatial attention (Woo et al. 2018 defaults: reduction 16, kernel 7)."""

    def __init__(self, c, reduction=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, c // reduction), nn.ReLU(), nn.Linear(c // reduction, c))
        self.spatial = nn.Conv1d(2, 1, k, padding=k // 2)

    def forward(self, x):
        w = torch.sigmoid(self.mlp(x.mean(-1)) + self.mlp(x.amax(-1))).unsqueeze(-1)
        x = x * w
        return x * torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class SelfAttention(nn.Module):
    """Multi-head self-attention written out explicitly (no PyTorch fused inference path; cf. UTransBPNet section 7)."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-2, -1) * (d // self.h) ** -0.5, dim=-1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, d))


class Gate(nn.Module):
    """Preprint Eq. (1): Gate = FC1(G) * sigmoid(FC2(G)) + G * (1 - sigmoid(FC2(G)))."""

    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, g):
        s = torch.sigmoid(self.fc2(g))
        return self.fc1(g) * s + g * (1 - s)


class RAGCBPNet(nn.Module):
    def __init__(self, n_feats=31, heads=4, reg_hidden=128):
        super().__init__()
        self.n_feats = n_feats
        self.backbone = ResNet50_1D(3)
        self.cbam = CBAM1D(self.backbone.out_channels)
        self.deep_proj = nn.Linear(self.backbone.out_channels, D)          # deep feature tokens
        self.hand_proj = nn.Linear(n_feats, D)                              # handcrafted feature token
        self.att = SelfAttention(D, heads)                                  # stage 1: self-attention fusion
        self.norm = nn.LayerNorm(D)
        self.gate = Gate(D + N_SIDE)                                        # stage 2: gating with side information
        self.head = nn.Sequential(nn.Linear(D + N_SIDE + 2, reg_hidden), nn.ReLU(), nn.Linear(reg_hidden, 2))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    @staticmethod
    def inputs(x):
        """[ECG, PPG] -> [PPG, VPPG, APPG] (PPG only, G2), each min-max scaled to (0, 1) per window."""
        ppg = x[:, 1:2].float()
        v = torch.gradient(ppg, dim=-1)[0] * FS
        a = torch.gradient(v, dim=-1)[0] * FS
        z = torch.cat([ppg, v, a], dim=1)
        lo, hi = z.amin(-1, keepdim=True), z.amax(-1, keepdim=True)
        return (z - lo) / (hi - lo).clamp_min(1e-6)

    def forward_std(self, x, side, trace=None):
        z = self.inputs(x)
        fmap = self.cbam(self.backbone(z))                                  # (B, 2048, T)
        tok = torch.cat([self.deep_proj(fmap.transpose(1, 2)),
                         self.hand_proj(side[:, N_SIDE:N_SIDE + self.n_feats]).unsqueeze(1)], dim=1)
        tok = self.norm(tok + self.att(tok))
        fused = tok.mean(1)                                                 # (B, D)
        g = self.gate(torch.cat([fused, side[:, :N_SIDE]], dim=1))
        out = self.head(torch.cat([g, side[:, N_SIDE - 2:N_SIDE]], dim=1))  # baseline SBP/DBP fed again (preprint 3.3.3)
        if trace is not None:
            trace += [("input (PPG, VPPG, APPG)", z.shape), ("ResNet50 + CBAM", fmap.shape), ("attention tokens", tok.shape),
                      ("gate output", g.shape)]
        return out

    def forward(self, x, side):
        return self.forward_std(x, side) * self.y_std + self.y_mean


if __name__ == "__main__":
    m = RAGCBPNet()
    trace = []
    with torch.no_grad():
        out = m.eval().forward_std(torch.randn(2, 2, 1250), torch.randn(2, N_SIDE + 31), trace)
    for name, shp in trace:
        print(f"{name:24s} {tuple(shp)}")
    print(f"output                   {tuple(out.shape)}")
    print(f"parameters               {sum(p.numel() for p in m.parameters()):,}")
