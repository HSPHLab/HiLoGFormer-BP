"""
HiCAF-BP: Hierarchical Cross-modal Attention Fusion for cuffless blood pressure.

Input  x: (B, C, T) z-scored signals in the order of `input_names` (e.g. ["ecg", "ppg"]), T = 1250 (10 s, 125 Hz)

L1  Multi-scale residual tokenizer per signal (kernels 7/15/31 ~ 56/120/250 ms, separate weights),
    total stride 8 -> L ~ 157 tokens of 64 ms. The time axis is kept: timing is the BP information.
L2  Intra-modal encoder per signal: RoPE self-attention, local (~1 s window, beat level) blocks
    followed by global blocks (rhythm level), alternating local -> global.
L3  Cross-modal encoder: bidirectional RoPE cross-attention (PPG <- ECG and ECG <- PPG) on a shared
    time grid, so attention depends on the ECG->PPG time lag (pulse arrival time).
L4  Fusion encoder on the time-aligned, concatenated tokens -> attention pooling -> z_fused.
Heads  per target t (SBP, DBP): gate g_t = sigmoid(MLP[z_fused, z_ppg]) mixes the fused embedding with a
    PPG-only embedding (attention-pooled L2 PPG tokens): h_t = g_t * z_fused + (1 - g_t) * z_ppg.
    Motivation: our ablation showed SBP gains from ECG+PPG while DBP is best from PPG alone.
Optional auxiliary head: physiological targets (heart rate, pulse arrival time) from z_fused.

Ablation switches (config): pos_encoding (rope | sinusoidal | none), cross_attention, gate,
local_window (0 = all global), tokenizer_kernels, depths, single-signal inputs.
"""
import torch
import torch.nn as nn

from .layers.attention import band_mask, sinusoidal_embedding
from .layers.blocks import AttentionPool, CrossBlock, EncoderBlock, ResConv1D
from .layers.common import MLP, DropPath
from .layers.rope import RotaryEmbedding
from .registry import register_model


class MultiScaleTokenizer(nn.Module):
    """(B, 1, T) -> (B, L, d): parallel multi-scale conv branches, 1x1 mix, strided residual stages."""

    def __init__(self, d_model, channels=64, kernels=(7, 15, 31), n_stages=2, stage_kernel=7):
        super().__init__()
        self.branches = nn.ModuleList([ResConv1D(1, channels, k, stride=2) for k in kernels])
        self.mix = nn.Sequential(nn.Conv1d(channels * len(kernels), d_model, 1, bias=False),
                                 nn.BatchNorm1d(d_model), nn.GELU())
        self.stages = nn.Sequential(*[ResConv1D(d_model, d_model, stage_kernel, stride=2) for _ in range(n_stages)])

    def forward(self, x):
        h = torch.cat([b(x) for b in self.branches], dim=1)
        return self.stages(self.mix(h)).transpose(1, 2)


@register_model("hicaf_bp")
class HiCAFBP(nn.Module):
    def __init__(self, input_names, n_targets=2, n_aux=0,
                 d_model=128, n_heads=4, mlp_ratio=4.0,
                 tokenizer_channels=64, tokenizer_kernels=(7, 15, 31), tokenizer_stages=2,
                 intra_depth=4, cross_depth=2, fusion_depth=3,
                 local_window=16, global_every=2,
                 pos_encoding="rope", rope_base=1000.0,
                 cross_attention=True, gate=True,
                 dropout=0.1, attn_dropout=0.0, drop_path=0.1, head_hidden=128):
        super().__init__()
        self.mods = list(input_names)
        if not set(self.mods) <= {"ecg", "ppg"}:
            raise ValueError(f"HiCAF-BP expects ecg and/or ppg, got {self.mods}")
        two = len(self.mods) == 2
        self.use_cross = bool(cross_attention) and two
        self.use_gate = bool(gate) and two
        self.pos_encoding = pos_encoding
        self.local_window = int(local_window or 0)
        self.global_every = max(1, int(global_every))
        self.n_targets, self.n_aux = n_targets, n_aux
        self.d_model = d_model

        if pos_encoding not in ("rope", "sinusoidal", "none"):
            raise ValueError("pos_encoding must be rope | sinusoidal | none")
        rope = RotaryEmbedding(d_model // n_heads, rope_base) if pos_encoding == "rope" else None

        # depth-wise DropPath schedule over every encoder block on the longest path
        n_blocks = intra_depth + (cross_depth if self.use_cross else 0) + fusion_depth
        rates = torch.linspace(0, drop_path, max(n_blocks, 1)).tolist()
        enc = dict(n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout, rope=rope)

        # L1 + L2 (separate weights per signal)
        self.tokenizer = nn.ModuleDict({m: MultiScaleTokenizer(d_model, tokenizer_channels, tuple(tokenizer_kernels),
                                                               tokenizer_stages) for m in self.mods})
        self.intra = nn.ModuleDict({m: nn.ModuleList([EncoderBlock(d_model, drop_path=rates[i], **enc)
                                                      for i in range(intra_depth)]) for m in self.mods})
        # L3
        off = intra_depth
        self.cross = nn.ModuleList([nn.ModuleDict({m: CrossBlock(d_model, drop_path=rates[off + i], **enc)
                                                   for m in self.mods}) for i in range(cross_depth)]) \
            if self.use_cross else nn.ModuleList()
        off += len(self.cross)
        # L4
        self.fuse_proj = nn.Sequential(nn.Linear(d_model * len(self.mods), d_model), nn.LayerNorm(d_model))
        self.fusion = nn.ModuleList([EncoderBlock(d_model, drop_path=rates[off + i], **enc)
                                     for i in range(fusion_depth)])
        self.fusion_norm = nn.LayerNorm(d_model)
        self.pool_fused = AttentionPool(d_model)
        # heads
        if self.use_gate:
            self.pool_ppg = AttentionPool(d_model)
            self.gate_mlp = nn.ModuleList([MLP(2 * d_model, d_model // 2, 1) for _ in range(n_targets)])
        self.heads = nn.ModuleList([MLP(d_model, head_hidden, 1, dropout) for _ in range(n_targets)])
        self.aux_head = MLP(d_model, head_hidden, n_aux, dropout) if n_aux > 0 else None

        self._drop_paths = [m for m in self.modules() if isinstance(m, DropPath)]
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- trainer / analysis hooks
    def set_drop_path(self, scale):
        """Called by the trainer each epoch: DropPath warm-up (scale goes 0 -> 1)."""
        for dp in self._drop_paths:
            dp.drop_prob = dp.base_prob * float(scale)

    def set_keep_attn(self, flag=True):
        """Store the last attention maps (e.g. cross-attention PPG <- ECG to inspect learned time lags)."""
        for m in self.modules():
            if hasattr(m, "keep_attn"):
                m.keep_attn = bool(flag)

    def _is_global(self, i):
        return self.local_window <= 0 or (i % self.global_every) == self.global_every - 1

    # ---------------------------------------------------------------- forward
    def encode(self, x):
        """L1-L4 -> (z_fused (B, d), z_ppg (B, d) or None). Used by forward() and by HiCAF-BP-Cal."""
        toks = {m: self.tokenizer[m](x[:, j:j + 1]) for j, m in enumerate(self.mods)}     # L1: (B, L, d)
        L = min(t.shape[1] for t in toks.values())
        toks = {m: t[:, :L] for m, t in toks.items()}
        pos = torch.arange(L, device=x.device, dtype=torch.float32)                        # shared time grid
        if self.pos_encoding == "sinusoidal":
            pe = sinusoidal_embedding(L, toks[self.mods[0]].shape[-1], x.device)
            toks = {m: t + pe.to(t.dtype) for m, t in toks.items()}
        local = band_mask(L, self.local_window, x.device) if self.local_window > 0 else None

        for m in self.mods:                                                                 # L2
            for i, blk in enumerate(self.intra[m]):
                toks[m] = blk(toks[m], pos, mask=None if self._is_global(i) else local)

        z_ppg = self.pool_ppg(toks["ppg"]) if self.use_gate else None                       # PPG-only path

        for layer in self.cross:                                                            # L3 (simultaneous update)
            e, p = toks["ecg"], toks["ppg"]
            toks = {"ecg": layer["ecg"](e, p, pos, pos), "ppg": layer["ppg"](p, e, pos, pos)}

        f = self.fuse_proj(torch.cat([toks[m] for m in self.mods], dim=-1))                 # L4
        for blk in self.fusion:
            f = blk(f, pos)
        z = self.pool_fused(self.fusion_norm(f))
        return z, z_ppg

    def forward(self, x):
        z, z_ppg = self.encode(x)
        preds, gates = [], []
        for t in range(self.n_targets):
            if self.use_gate:
                g = torch.sigmoid(self.gate_mlp[t](torch.cat([z, z_ppg], dim=-1)))           # (B, 1)
                h = g * z + (1 - g) * z_ppg
                gates.append(g)
            else:
                h = z
            preds.append(self.heads[t](h))
        out = {"bp": torch.cat(preds, dim=-1)}
        if gates:
            out["gates"] = torch.cat(gates, dim=-1)
        if self.aux_head is not None:
            out["aux"] = self.aux_head(z)
        return out
