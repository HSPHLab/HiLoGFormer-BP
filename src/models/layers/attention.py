"""
Multi-head attention with optional RoPE, usable for self-attention (ctx=None) and cross-attention.

Scores and softmax are computed in float32 for stability under bf16 autocast. A boolean mask
(Lq, Lk), True = may attend, implements local (banded) attention.
Set `keep_attn = True` to store the last attention map (for interpretability analyses).
"""
import math

import torch
import torch.nn as nn

from .rope import apply_rope


def band_mask(L, window, device):
    """True where |i - j| <= window // 2: each token sees about `window` neighbours (local attention)."""
    i = torch.arange(L, device=device)
    return (i[:, None] - i[None, :]).abs() <= window // 2


def sinusoidal_embedding(L, d, device):
    """Classic absolute sinusoidal positions (used only for the 'no RoPE' ablation)."""
    pos = torch.arange(L, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, proj_dropout=0.0, rope=None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.scale = self.dh ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.rope = rope
        self.keep_attn = False
        self.last_attn = None

    def forward(self, x, ctx=None, pos_q=None, pos_k=None, mask=None):
        ctx = x if ctx is None else ctx
        B, Lq, D = x.shape
        Lk = ctx.shape[1]
        q = self.q(x).view(B, Lq, self.h, self.dh).transpose(1, 2)                  # (B, H, Lq, Dh)
        k, v = self.kv(ctx).view(B, Lk, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 2 x (B, H, Lk, Dh)
        if self.rope is not None:
            cq, sq = self.rope.cos_sin(pos_q)
            ck, sk = (cq, sq) if pos_k is pos_q else self.rope.cos_sin(pos_k)
            q, k = apply_rope(q, cq, sq), apply_rope(k, ck, sk)
        scores = (q @ k.transpose(-2, -1)).float() * self.scale                       # (B, H, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        if self.keep_attn:
            self.last_attn = attn.detach()
        out = self.attn_drop(attn).to(v.dtype) @ v                                     # (B, H, Lq, Dh)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.proj_drop(self.proj(out))
