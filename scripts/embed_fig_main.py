#!/usr/bin/env python
"""
Single-column embedding figure for the main paper (plotting only, no model inference).

  python scripts/embed_fig_main.py

Reads runs/embeddings/emb_fold0.npz (written by scripts/embed_viz.py --extract) and draws
  (a) t-SNE of the fused embeddings z of all test recordings of fold 0; the 8 subjects with the most recordings in color,
  (b) t-SNE of dz = z_target - mean z of the calibration recordings, colored by the true SBP change since calibration.
t-SNE exactly as in embed_viz.py: z-scored features, perplexity 30, PCA init, seed 2026. Both panels have plot boxes of
the same size. Output: paper/figures/emb_main.pdf (+ preview png).
"""
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from src.viz import style  # noqa: E402

SEED = 2026


def tsne(X):
    from sklearn.manifold import TSNE
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(Xs)


def main():
    import matplotlib.pyplot as plt
    if hasattr(style, "setup"):
        style.setup()
    z = np.load(CODE_DIR / "runs/embeddings/emb_fold0.npz", allow_pickle=False)
    pid = z["a_pid"]
    Pa = tsne(z["a_z"])
    mk = np.isfinite(z["a_dz"]).all(1)
    Pb = tsne(z["a_dz"][mk])
    v = z["a_dbp"][mk, 0]
    lim = np.nanpercentile(np.abs(v), 95)
    ids, cnt = np.unique(pid, return_counts=True)
    top = ids[np.argsort(-cnt, kind="stable")[:8]]

    fig = plt.figure(figsize=(style.WIDTH["single"], 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.045], wspace=0.06, hspace=0.12,
                          left=0.07, right=0.84, top=0.97, bottom=0.04)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    fig.add_subplot(gs[0, 1]).axis("off")                                  # empty slot: same box size as (b)
    cax = fig.add_subplot(gs[1, 1])

    axa.scatter(Pa[:, 0], Pa[:, 1], s=2, lw=0, color="#D9D9D9", rasterized=True)
    cmap = plt.get_cmap("tab10")
    for j, p in enumerate(top):
        k = pid == p
        axa.scatter(Pa[k, 0], Pa[k, 1], s=5, lw=0, color=cmap(j), rasterized=True)
    axa.text(0.02, 0.98, "8 subjects with the most recordings", transform=axa.transAxes, va="top", fontsize=7,
             bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1})
    sc = axb.scatter(Pb[:, 0], Pb[:, 1], c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=2, lw=0, rasterized=True)
    fig.colorbar(sc, cax=cax, label="true ΔSBP since calibration (mmHg)")
    for ax in (axa, axb):
        ax.set(xticks=[], yticks=[], xlabel="t-SNE 1", ylabel="t-SNE 2")
        ax.grid(False)
    style.panel_labels([axa, axb], x=-0.06, y=1.0)
    plt.rcParams["savefig.bbox"] = "standard"                               # keep the fixed layout
    out = Path("/tmp/emb_main.pdf"); fig.savefig(out, dpi=600); fig.savefig(out.with_suffix(".png"), dpi=300); print("saved", out)
    print(f"(a) {len(Pa):,} recordings of {len(ids)} subjects; top-8 subjects: {', '.join(f'{c}' for c in np.sort(cnt)[::-1][:8])} recordings")
    print(f"(b) {mk.sum():,} readings of {len(np.unique(pid[mk]))} subjects; colour limit ±{lim:.1f} mmHg")


if __name__ == "__main__":
    main()
