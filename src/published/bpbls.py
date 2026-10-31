"""
BpBLS (Huang et al., IEEE JBHI 30(6), 2026) - ported from the official code
https://github.com/Huangzx1023/BpBLS-for-BP-Estimation (main.py, utils.py; zip sha256 5ac31aaf...).
Specification and paper-code discrepancies: docs/reimplementation/BpBLS.md (rule B0: the code is followed).

Kept exactly: BLS network (input BatchNorm -> 460 sigmoid feature nodes with BatchNorm -> 28 tanh enhancement nodes with
BatchNorm -> linear output without bias), output columns [DBP, SBP], closed-form output weights with pulse-pressure
regularisation (ridge_regression_pseudo_inverse_reg), calibration BP = 0.29 * output + 0.71 * baseline.
Changed only through arguments: input size 74 -> 37 (31 features + 4 demographics + 2 baseline values).
"""
import torch
import torch.nn as nn


class BLS(nn.Module):
    """Official `BLS` class (input size made an argument; the unused `y` argument of forward kept optional)."""

    def __init__(self, in_dim, feature_nodes=10, feature_windows=46, enhancement_nodes=28, num_classes=2, normalization="batch"):
        super().__init__()
        self.feature_nodes, self.feature_windows = feature_nodes, feature_windows
        self.bn_in = nn.BatchNorm1d(in_dim, track_running_stats=True)
        self.feature_layer = nn.Linear(in_dim, feature_nodes * feature_windows)
        self.feature_bn = nn.BatchNorm1d(feature_nodes * feature_windows, track_running_stats=True) if normalization == "batch" else nn.Identity()
        self.fc31 = nn.Linear(feature_nodes * feature_windows, enhancement_nodes)
        self.bn_en = nn.BatchNorm1d(enhancement_nodes, track_running_stats=True)
        self.final = nn.Linear(feature_nodes * feature_windows + enhancement_nodes, num_classes, bias=False)

    def forward(self, x, y=None):
        x = self.bn_in(x)
        feature_nodes = torch.sigmoid(self.feature_bn(self.feature_layer(x)))
        enhancement_nodes = torch.tanh(self.bn_en(self.fc31(feature_nodes)))
        fea_and_enhance = torch.cat([feature_nodes, enhancement_nodes], dim=1)
        return self.final(fea_and_enhance), fea_and_enhance


def ridge_regression_pseudo_inverse_reg(X, y, alpha, beta, pp):
    """Official closed form (utils.py). y columns [DBP, SBP]; pp per sample; returns (B_plus, W) with W columns [DBP, SBP]."""
    identity_matrix = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    XTX = X.T @ X
    B_plus = torch.linalg.pinv((1 + beta) * XTX + alpha * identity_matrix) @ X.T
    Beta = beta / (1 + beta)
    C1 = 1.0 / (1.0 - Beta ** 2.0)
    delta = Beta * pp
    W_s = C1 * B_plus @ (Beta * y[:, 0:1] + y[:, 1:2] + delta)
    W_d = C1 * B_plus @ (y[:, 0:1] + Beta * y[:, 1:2] - delta)
    return B_plus, torch.cat((W_d, W_s), dim=1)


class BpBLS(nn.Module):
    """Inference wrapper used by our evaluation scripts: forward(signals (ignored), feas) -> [SBP, DBP] in mmHg.
    feas = [demographics, handcrafted features (min-max scaled), baseline DBP, baseline SBP (mmHg)]."""

    def __init__(self, in_dim, feature_nodes=10, feature_windows=46, enhancement_nodes=28, beta_cal=0.29):
        super().__init__()
        self.bls = BLS(in_dim, feature_nodes, feature_windows, enhancement_nodes, num_classes=2)
        self.register_buffer("W", torch.zeros(feature_nodes * feature_windows + enhancement_nodes, 2))
        self.beta_cal = float(beta_cal)

    def forward(self, x_signals, feas):
        _, A = self.bls(feas)
        out = A @ self.W                                     # [DBP, SBP]
        bp = self.beta_cal * out + (1 - self.beta_cal) * feas[:, -2:]
        return bp[:, [1, 0]]                                 # -> [SBP, DBP]
