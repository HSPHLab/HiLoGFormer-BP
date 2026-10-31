"""
Rotary positional embedding (RoPE) with EXPLICIT positions.

Positions are passed in, not derived from sequence length, so queries and keys of two different
signals can share one time grid. In cross-attention (PPG tokens attending to ECG tokens) the score
then depends on the relative time lag between the two tokens - i.e. the pulse arrival time.

`base` sets the wavelengths: smaller base -> faster rotation -> finer resolution at short lags.
Our lags of interest are ~2-6 tokens (0.1-0.4 s at 64 ms per token), so the default is 1000 rather
than the 10000 used for long sequences.
"""
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=1000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, positions):
        """positions: (L,) float tensor (token index on the shared time grid) -> cos, sin of shape (L, Dh)."""
        ang = positions.float()[:, None] * self.inv_freq[None, :]
        ang = ang.repeat_interleave(2, dim=-1)       # interleaved pairs (x0,x1), (x2,x3), ...
        return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, L, Dh). Rotates each (even, odd) feature pair by its position angle (in float32)."""
    xf = x.float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return (xf * cos + rot * sin).to(x.dtype)
