#!/usr/bin/env python
"""
Embedding visualisation of HiCAF-BP-Cal (docs/EMBEDDING_PLAN.md). Inference only, frozen models.

  python scripts/embed_viz.py --extract --gpu 2      # embeddings of every fold model (Aurora test subjects + PulseDB sample)
  python scripts/embed_viz.py --analyze              # metrics (all folds) + figures (fold 0)

Model: the interpretability model (no-time; runs/hicaf_bp_cal_notime). Each fold model has its own embedding space, so
every map and metric uses ONE fold model; maps are drawn for fold 0, metrics are reported for all 5 folds (mean ± SD).
Embeddings: z = fused ECG+PPG embedding of the shared encoder (the representation the calibration head compares);
dz = z_target − mean z of the subject's (first 3) calibration recordings (calibration-relative, what the head sees).
Aurora: all test recordings of the fold (targets and calibration recordings). PulseDB (zero-shot): 100 random subjects per
source (seed 2026), the first 3 segments as calibration and 40 evenly spaced later segments per subject.
Maps: PCA (linear), t-SNE (perplexity 30, PCA init, seed 2026), UMAP (only if the umap-learn package is already installed).
Metrics (full-dimensional embedding, 10 nearest neighbours, cosine-free Euclidean): subject purity (share of neighbours from
the same subject vs chance); cross-subject k-NN R^2 of SBP / DBP from z and of the true change from dz (neighbours restricted
to OTHER subjects); domain separability (balanced k-NN accuracy Aurora vs PulseDB; 0.5 = indistinguishable).
Outputs: runs/embeddings/emb_fold{f}.npz; paper/tables/embeddings/EMBEDDINGS.md (+ .json);
         paper/figures/emb_aurora.pdf, emb_delta.pdf, emb_domain.pdf
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
MODEL_RUN = "runs/hicaf_bp_cal_notime"
OUT = CODE_DIR / "runs/embeddings"
T = ["sbp", "dbp"]
SEED, K = 2026, 10
N_SUB_PULSEDB, N_SEG_PULSEDB = 100, 40


# ------------------------------------------------------------------------------------------------ extraction (GPU)
def extract(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import BPCropDataset
    from src.data.pairs import calibration_split
    from src.data.splits import assign_roles, load_metadata
    from src.models import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    # PulseDB sample (identical for every fold model)
    pdb = CODE_DIR / "data_processed/pulsedb_v1"
    pmeta = pd.read_csv(pdb / "metadata.csv", low_memory=False).sort_values("sample_id").reset_index(drop=True)
    picks = []
    for src in ["MIMIC", "VitalDB"]:
        pids = pmeta.loc[pmeta.protocol == src, "pid"].unique()
        for pid in rng.choice(pids, min(N_SUB_PULSEDB, len(pids)), replace=False):
            rows = np.flatnonzero(pmeta.pid.to_numpy() == pid)
            later = rows[3:]
            sel = later[np.linspace(0, len(later) - 1, min(N_SEG_PULSEDB, len(later))).round().astype(int)] if len(later) else []
            picks += [(r, "cal") for r in rows[:3]] + [(r, "target") for r in np.unique(sel)]
    pm = pmeta.iloc[[r for r, _ in picks]].reset_index(drop=True)
    pm["role_emb"] = [k for _, k in picks]
    pchan = json.load(open(pdb / "config.json")).get("channels", ["ecg", "ppg"])
    psig = np.load(pdb / "signals.npy", mmap_mode="r")
    for fold in range(5):
        t0 = time.time()
        ck = torch.load(CODE_DIR / MODEL_RUN / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        model = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        meta, fs, ch = load_metadata(cfg)
        d = assign_roles(meta, cfg, fold)
        d = d[d.role == "test"].sort_values("sample_id").reset_index(drop=True)
        cal, tg = calibration_split(d, cfg["calibration"]["phase"], cfg["calibration"]["conditions"], int(cfg["calibration"]["k_max"]))

        def embed(df, sig, chans):
            ds = BPCropDataset(df, sig, [chans.index(c) for c in cfg["data"]["inputs"]], int(round(cfg["data"]["crop_s"] * fs)),
                               y_mean, y_std, T, [], None, None, train=False, augment_cfg=None)
            zs = []
            with torch.no_grad():
                for b in DataLoader(ds, batch_size=512, shuffle=False, num_workers=8):
                    z, _ = model.encoder.encode(b["x"].to(dev))
                    zs.append(z.float().cpu().numpy())
            return np.concatenate(zs)

        za = embed(d, np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r"), ch)
        is_cal = np.zeros(len(d), bool)
        cal_mean_z, cal_bp = {}, {}
        for pid, pos in cal.items():
            pos = list(pos)[:3]
            is_cal[pos] = True
            cal_mean_z[pid] = za[pos].mean(0)
            cal_bp[pid] = d.loc[pos, T].to_numpy(float).mean(0)
        tgt = np.zeros(len(d), bool)
        tgt[np.asarray(tg, int)] = True
        dz = np.stack([za[i] - cal_mean_z[p] if (tgt[i] and p in cal_mean_z) else np.full(za.shape[1], np.nan) for i, p in enumerate(d.pid)])
        dbp = np.stack([d.loc[i, T].to_numpy(float) - cal_bp[p] if (tgt[i] and p in cal_bp) else [np.nan, np.nan] for i, p in enumerate(d.pid)])
        zp = embed(pm, psig, pchan)
        pcal = pm.role_emb.to_numpy() == "cal"
        pz_c = {pid: zp[np.flatnonzero((pm.pid.to_numpy() == pid) & pcal)].mean(0) for pid in pm.pid.unique()}
        pb_c = {pid: pm.loc[(pm.pid == pid) & pcal, T].to_numpy(float).mean(0) for pid in pm.pid.unique()}
        pdz = np.stack([zp[i] - pz_c[p] if not pcal[i] else np.full(zp.shape[1], np.nan) for i, p in enumerate(pm.pid)])
        pdbp = np.stack([pm.loc[i, T].to_numpy(float) - pb_c[p] if not pcal[i] else [np.nan, np.nan] for i, p in enumerate(pm.pid)])
        np.savez_compressed(OUT / f"emb_fold{fold}.npz",
                            a_z=za.astype(np.float32), a_dz=dz.astype(np.float32), a_bp=d[T].to_numpy(np.float32), a_dbp=dbp.astype(np.float32),
                            a_pid=d.pid.to_numpy().astype(str), a_phase=d.phase.astype(str).to_numpy().astype("U"), a_condition=d.condition.astype(str).to_numpy().astype("U"),
                            a_is_cal=is_cal, a_is_target=tgt,
                            p_z=zp.astype(np.float32), p_dz=pdz.astype(np.float32), p_bp=pm[T].to_numpy(np.float32), p_dbp=pdbp.astype(np.float32),
                            p_pid=pm.pid.to_numpy().astype(str), p_source=pm.protocol.to_numpy().astype(str), p_is_cal=pcal)
        print(f"fold {fold}: Aurora {len(d):,} recordings ({d.pid.nunique()} subjects), PulseDB {len(pm):,} segments, d = {za.shape[1]} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ analysis (CPU)
def knn_cross_subject_r2(X, y, groups, k=K):
    """R^2 of predicting y by the mean of the k nearest neighbours from OTHER subjects."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k * 8 + 1, len(X))).fit(X)
    _, idx = nn.kneighbors(X)
    pred = np.empty(len(X))
    for i in range(len(X)):
        other = [j for j in idx[i, 1:] if groups[j] != groups[i]][:k]
        pred[i] = y[other].mean() if other else np.nan
    m = np.isfinite(pred)
    return 1 - np.sum((y[m] - pred[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)


def subject_purity(X, groups, k=K):
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    same = (groups[idx[:, 1:]] == groups[:, None]).mean()
    counts = pd.Series(groups).value_counts()
    chance = float(((counts - 1) * counts).sum() / (len(groups) * (len(groups) - 1)))
    return float(same), chance


def domain_knn_accuracy(Xa, Xp, k=K, seed=SEED):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n = min(len(Xa), len(Xp))
    X = np.vstack([Xa[rng.choice(len(Xa), n, replace=False)], Xp[rng.choice(len(Xp), n, replace=False)]])
    lab = np.r_[np.zeros(n), np.ones(n)]
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return float(((lab[idx[:, 1:]].mean(1) > 0.5) == lab).mean())


def maps(X, seed=SEED):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    out = {"PCA": (pca.transform(Xs), pca.explained_variance_ratio_)}
    out["t-SNE"] = (TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(Xs), None)
    try:
        import umap                                                           # only if already installed (no environment change)
        out["UMAP"] = (umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Xs), None)
    except ImportError:
        pass
    return out


def analyze():
    from src.viz.style import DATASET, SETTING, figure, panel_labels, save, setup
    tab = CODE_DIR / "paper/tables/embeddings"
    tab.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in range(5):
        z = np.load(OUT / f"emb_fold{fold}.npz", allow_pickle=True)
        g = z["a_pid"]
        r = {"fold": fold, "n_aurora": len(g), "d": z["a_z"].shape[1]}
        r["purity"], r["purity_chance"] = subject_purity(z["a_z"], g)
        for j, t in enumerate(T):
            r[f"r2_z_{t}"] = knn_cross_subject_r2(z["a_z"], z["a_bp"][:, j], g)
            m = np.isfinite(z["a_dz"]).all(1)
            r[f"r2_dz_d{t}"] = knn_cross_subject_r2(z["a_dz"][m], z["a_dbp"][m, j], g[m])
        home = z["a_phase"].astype(str) == "ambulatory"
        for st, sm in [("lab", ~home), ("home", home)]:
            mm = m & sm
            for j, t in enumerate(T):
                r[f"r2_dz_d{t}_{st}"] = knn_cross_subject_r2(z["a_dz"][mm], z["a_dbp"][mm, j], g[mm])
        r["domain_acc"] = domain_knn_accuracy(z["a_z"], z["p_z"])
        for src in ["MIMIC", "VitalDB"]:
            r[f"domain_acc_{src}"] = domain_knn_accuracy(z["a_z"], z["p_z"][z["p_source"] == src])
        rows.append(r)
        print(f"fold {fold}: purity {r['purity']:.2f} (chance {r['purity_chance']:.3f}), R2 z->SBP {r['r2_z_sbp']:.2f}, "
              f"dz->dSBP {r['r2_dz_dsbp']:.2f}, domain acc {r['domain_acc']:.2f}", flush=True)
    m = pd.DataFrame(rows)
    ms = lambda c: f"{m[c].mean():.2f} ± {m[c].std(ddof=1):.2f}"
    L = ["# Embeddings of HiCAF-BP-Cal (no-time model) — quantitative summary (5 fold models, mean ± SD)", "",
         f"10 nearest neighbours in the full {int(m.d.iloc[0])}-dimensional embedding; Aurora test recordings of each fold; "
         "cross-subject = neighbours restricted to other subjects.", "",
         "| Property | Metric | Value |", "|---|---|---|",
         f"| Subject identity | share of neighbours from the same subject | {ms('purity')} (chance {m.purity_chance.mean():.3f}) |",
         f"| Absolute BP in z | cross-subject k-NN R², SBP / DBP | {ms('r2_z_sbp')} / {ms('r2_z_dbp')} |",
         f"| BP change in dz (calibration-relative) | cross-subject k-NN R², ΔSBP / ΔDBP | {ms('r2_dz_dsbp')} / {ms('r2_dz_ddbp')} |",
         f"| BP change within setting (robustness) | cross-subject k-NN R² in-lab only, ΔSBP / ΔDBP | {ms('r2_dz_dsbp_lab')} / {ms('r2_dz_ddbp_lab')} |",
         f"| | home only, ΔSBP / ΔDBP | {ms('r2_dz_dsbp_home')} / {ms('r2_dz_ddbp_home')} |",
         f"| Domain shift (Aurora vs PulseDB) | balanced k-NN accuracy (0.5 = indistinguishable) | {ms('domain_acc')} "
         f"(MIMIC {ms('domain_acc_MIMIC')}, VitalDB {ms('domain_acc_VitalDB')}) |"]
    (tab / "EMBEDDINGS.md").write_text("\n".join(L))
    m.to_json(tab / "EMBEDDINGS.json", orient="records", indent=2)
    print("\n".join(L))

    # figures: fold 0
    z = np.load(OUT / "emb_fold0.npz", allow_pickle=True)
    setup()
    rng = np.random.default_rng(SEED)
    phase = z["a_phase"]
    setting = np.where(phase == "ambulatory", "home", "lab")
    subj = pd.Series(z["a_pid"]).value_counts()
    top = list(subj.index[:8])
    A = maps(z["a_z"])
    meths = list(A)
    # Fig E1: Aurora z by SBP, setting, subject
    fig, axes = figure("double", height=2.2 * len(meths), nrows=len(meths), ncols=3)
    axes = np.atleast_2d(axes)
    for i, mth in enumerate(meths):
        P, ev = A[mth]
        ax = axes[i, 0]
        sc = ax.scatter(P[:, 0], P[:, 1], c=z["a_bp"][:, 0], cmap="viridis", s=2, lw=0, vmin=90, vmax=170, rasterized=True)
        fig.colorbar(sc, ax=ax, label="SBP (mmHg)", shrink=0.85)
        ax = axes[i, 1]
        order = rng.permutation(len(P))                                       # random drawing order: no group hides another
        ax.scatter(P[order, 0], P[order, 1], s=2, lw=0, alpha=0.7, rasterized=True,
                   color=[SETTING[st]["color"] for st in setting[order]])
        for st in ["lab", "home"]:
            ax.scatter([], [], s=8, color=SETTING[st]["color"], label=SETTING[st]["label"])
        ax.legend(loc="best", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none")
        ax = axes[i, 2]
        ax.scatter(P[:, 0], P[:, 1], s=2, lw=0, color="#D9D9D9", rasterized=True)
        cmap = __import__("matplotlib").colormaps["tab10"]
        for j, p in enumerate(top):
            k = z["a_pid"] == p
            ax.scatter(P[k, 0], P[k, 1], s=5, lw=0, color=cmap(j), rasterized=True)
        ax.text(0.02, 0.98, "8 subjects with most recordings", transform=ax.transAxes, va="top", fontsize=6,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1})
        for ax in axes[i]:
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(f"{mth} 1" + (f" ({ev[0]:.0%})" if ev is not None else ""))
            ax.set_ylabel(f"{mth} 2" + (f" ({ev[1]:.0%})" if ev is not None else ""))
    panel_labels(axes.ravel())
    save(fig, "emb_aurora", source=__file__)
    # Fig E2: calibration-relative dz by true change
    mk = np.isfinite(z["a_dz"]).all(1)
    D = maps(z["a_dz"][mk])
    fig, axes = figure("double", height=2.2 * len(D), nrows=len(D), ncols=2)
    axes = np.atleast_2d(axes)
    for i, mth in enumerate(D):
        P, ev = D[mth]
        for j, t in enumerate(T):
            ax = axes[i, j]
            v = z["a_dbp"][mk, j]
            lim = np.nanpercentile(np.abs(v), 95)
            sc = ax.scatter(P[:, 0], P[:, 1], c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=2, lw=0, rasterized=True)
            fig.colorbar(sc, ax=ax, label=f"true Δ{t.upper()} from calibration (mmHg)", shrink=0.85)
            ax.set(xticks=[], yticks=[], xlabel=f"{mth} 1" + (f" ({ev[0]:.0%})" if ev is not None else ""),
                   ylabel=f"{mth} 2" + (f" ({ev[1]:.0%})" if ev is not None else ""))
    panel_labels(axes.ravel())
    save(fig, "emb_delta", source=__file__)
    # Fig E3: domain (Aurora targets vs PulseDB) by dataset and SBP
    na = min(4000, len(z["a_z"]))
    ia = rng.choice(len(z["a_z"]), na, replace=False)
    X = np.vstack([z["a_z"][ia], z["p_z"]])
    lab = np.r_[np.full(na, "Aurora"), z["p_source"]]
    sbp = np.r_[z["a_bp"][ia, 0], z["p_bp"][:, 0]]
    G = maps(X)
    fig, axes = figure("double", height=2.2 * len(G), nrows=len(G), ncols=2)
    axes = np.atleast_2d(axes)
    for i, mth in enumerate(G):
        P, ev = G[mth]
        ax = axes[i, 0]
        keymap = {"Aurora": "aurora", "MIMIC": "mimic", "VitalDB": "vitaldb"}
        order = rng.permutation(len(P))
        ax.scatter(P[order, 0], P[order, 1], s=2, lw=0, alpha=0.7, rasterized=True, color=[DATASET[keymap[n]]["color"] for n in lab[order]])
        for name, key in keymap.items():
            ax.scatter([], [], s=8, color=DATASET[key]["color"], label=DATASET[key]["label"])
        ax.legend(fontsize=6, frameon=True, framealpha=0.85, edgecolor="none")
        ax = axes[i, 1]
        sc = ax.scatter(P[:, 0], P[:, 1], c=sbp, cmap="viridis", vmin=80, vmax=180, s=2, lw=0, rasterized=True)
        fig.colorbar(sc, ax=ax, label="SBP (mmHg)", shrink=0.85)
        for ax in axes[i]:
            ax.set(xticks=[], yticks=[], xlabel=f"{mth} 1" + (f" ({ev[0]:.0%})" if ev is not None else ""),
                   ylabel=f"{mth} 2" + (f" ({ev[1]:.0%})" if ev is not None else ""))
    panel_labels(axes.ravel())
    save(fig, "emb_domain", source=__file__)
    print("maps drawn with:", ", ".join(meths), "" if "UMAP" in meths else "(UMAP not installed - PCA and t-SNE only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    sys.path.insert(0, str(CODE_DIR))
    if args.extract:
        extract(args.gpu)
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
