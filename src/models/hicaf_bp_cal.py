"""
HiCAF-BP-Cal: calibration-based HiCAF-BP (siamese).

The target recording and a calibration recording of the SAME person (with its known, standardised BP)
pass through one shared HiCAF-BP encoder. A head compares the two embeddings
[z_t, z_c, z_t - z_c, z_t * z_c] (fused, and PPG-only when available) together with an embedding of the
calibration BP (and, if ctx_dim > 0, an embedding of the time context: time of day of both readings and
time since calibration), and predicts the BP CHANGE. Output = calibration BP + predicted change.

The last head layer is zero-initialised, so training starts exactly at the "calibration value only"
baseline and learns the change from there. The encoder is normally initialised from a trained
HiCAF-BP of the same fold (config `init`), so no new subjects leak in.
"""
import torch
import torch.nn as nn

from .hicaf_bp import HiCAFBP
from .registry import register_model


@register_model("hicaf_bp_cal")
class HiCAFBPCal(nn.Module):
    def __init__(self, input_names, n_targets=2, n_aux=0, cal_head_hidden=256, cal_dropout=0.1,
                 bp_embed_dim=64, ctx_dim=0, ctx_embed_dim=32, demo_dim=0, demo_embed_dim=32, **encoder_kwargs):
        super().__init__()
        self.encoder = HiCAFBP(input_names, n_targets=n_targets, n_aux=0, **encoder_kwargs)
        d = self.encoder.d_model
        self.use_ppg = self.encoder.use_gate            # PPG-only embedding exists when both signals are used
        n_emb = 2 if self.use_ppg else 1
        self.bp_embed = nn.Sequential(nn.Linear(n_targets, bp_embed_dim), nn.GELU(),
                                      nn.Linear(bp_embed_dim, bp_embed_dim))
        self.ctx_dim = int(ctx_dim)
        self.ctx_embed = (nn.Sequential(nn.Linear(self.ctx_dim, ctx_embed_dim), nn.GELU(),
                                        nn.Linear(ctx_embed_dim, ctx_embed_dim)) if self.ctx_dim > 0 else None)
        self.demo_dim = int(demo_dim)
        self.demo_embed = (nn.Sequential(nn.Linear(self.demo_dim, demo_embed_dim), nn.GELU(),
                                         nn.Linear(demo_embed_dim, demo_embed_dim)) if self.demo_dim > 0 else None)
        in_dim = (n_emb * 4 * d + bp_embed_dim + (ctx_embed_dim if self.ctx_dim > 0 else 0)
                  + (demo_embed_dim if self.demo_dim > 0 else 0))
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cal_head_hidden), nn.GELU(),
                                  nn.Dropout(cal_dropout), nn.Linear(cal_head_hidden, cal_head_hidden // 2),
                                  nn.GELU(), nn.Linear(cal_head_hidden // 2, n_targets))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def set_drop_path(self, scale):
        self.encoder.set_drop_path(scale)

    def set_keep_attn(self, flag=True):
        self.encoder.set_keep_attn(flag)

    @staticmethod
    def _compare(a, b):
        return [a, b, a - b, a * b]

    def forward(self, x, x_cal=None, bp_cal=None, ctx=None, demo=None):
        if x_cal is None or bp_cal is None:
            raise ValueError("HiCAF-BP-Cal needs x_cal and bp_cal (use calibration.enabled: true)")
        if self.ctx_embed is not None and ctx is None:
            raise ValueError("model.ctx_dim > 0 needs the time context (calibration.time_context: true)")
        B = x.shape[0]
        z, zp = self.encoder.encode(torch.cat([x, x_cal], dim=0))     # one pass for target + calibration
        feats = self._compare(z[:B], z[B:])
        if self.use_ppg:
            feats += self._compare(zp[:B], zp[B:])
        feats.append(self.bp_embed(bp_cal.to(z.dtype)))
        if self.ctx_embed is not None:
            feats.append(self.ctx_embed(ctx.to(z.dtype)))
        if self.demo_embed is not None:
            if demo is None:
                raise ValueError("model.demo_dim > 0 needs demographics (calibration.demographics)")
            feats.append(self.demo_embed(demo.to(z.dtype)))
        delta = self.head(torch.cat(feats, dim=-1)).float()
        return {"bp": bp_cal.float() + delta, "delta": delta}
