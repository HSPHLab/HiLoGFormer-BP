"""Convolutional and encoder blocks (pre-LayerNorm, residual, DropPath)."""
import torch
import torch.nn as nn

from .attention import Attention
from .common import MLP, DropPath


class ResConv1D(nn.Module):
    """Residual conv block: conv-BN-GELU-conv-BN + (strided 1x1) skip, GELU."""

    def __init__(self, c_in, c_out, k, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False), nn.BatchNorm1d(c_out))
        self.skip = (nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))
                     if (c_in != c_out or stride != 1) else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class EncoderBlock(nn.Module):
    """Self-attention encoder block: x + DropPath(Attn(LN(x))), then x + DropPath(MLP(LN(x)))."""

    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0, drop_path=0.0, rope=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, attn_dropout, dropout, rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), d_model, dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x, pos, mask=None):
        x = x + self.dp(self.attn(self.ln1(x), pos_q=pos, pos_k=pos, mask=mask))
        return x + self.dp(self.mlp(self.ln2(x)))


class CrossBlock(nn.Module):
    """Cross-attention encoder block: x attends to ctx (another signal) on a shared time grid."""

    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0, drop_path=0.0, rope=None):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, attn_dropout, dropout, rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), d_model, dropout)
        self.dp = DropPath(drop_path)

    def forward(self, x, ctx, pos_x, pos_ctx):
        x = x + self.dp(self.attn(self.ln_q(x), ctx=self.ln_kv(ctx), pos_q=pos_x, pos_k=pos_ctx))
        return x + self.dp(self.mlp(self.ln2(x)))


class AttentionPool(nn.Module):
    """Learned weighted average over tokens: (B, L, d) -> (B, d)."""

    def __init__(self, d_model):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.ln(x)
        w = self.score(h).float().softmax(dim=1).to(h.dtype)      # (B, L, 1)
        return (w * h).sum(dim=1)
