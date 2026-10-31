#!/usr/bin/env python
"""
Interpretability C3 - beat-level saliency, reported only if faithful (docs/INTERPRETABILITY_PLAN.md, section 5, C3). GPU.

  python scripts/interp_C3_saliency.py --gpu 2            # 2,000 random test readings (seed 2026)
  python scripts/interp_C3_saliency.py --gpu 2 --n 20     # quick test

Interpretability model (plan v1.2): the no-time fold models; each test reading uses its own fold's model and its k = 3
calibration readings (as in the frozen evaluation). Target of the attribution: the predicted change d_hat (mean over the
3 calibration readings), separately for SBP and DBP.
Integrated gradients: 50 steps, zero baseline (= the mean of the z-scored signals), w.r.t. the target ECG and PPG.
Faithfulness (pre-registered): replace the top 10 % / 20 % most salient samples (|IG|, both channels) by linear interpolation
vs the same number of random samples; |change of d_hat| compared by a paired one-sided Wilcoxon test (salient > random).
Saliency maps are reported only if p < 0.05 at both deletion levels; otherwise the negative result is reported.
Beat alignment: ECG on R-peak-to-R-peak cycles, PPG on foot-to-foot cycles (src/features/pwa_features.py), 100 bins.
Outputs: paper/tables/interpretability/C3_saliency.md (+ .json), runs/interpretability/C3_saliency.npz,
         paper/figures/interp_C3_saliency.pdf (Fig. I4: saliency maps if faithful, else the faithfulness test)
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
sys.path.insert(0, str(CODE_DIR))
T = ["sbp", "dbp"]
BINS = 100


def interp_delete(x, idx):
    """Replace samples idx (per channel) by linear interpolation from the remaining samples. x: (C, L)."""
    y = x.copy()
    keep = np.ones(x.shape[1], bool)
    keep[idx] = False
    for c in range(x.shape[0]):
        y[c, ~keep] = np.interp(np.flatnonzero(~keep), np.flatnonzero(keep), x[c, keep])
    return y


def cycle_profile(values, marks, bins=BINS):
    """Average of `values` over cycles between consecutive marks, each resampled to `bins` points."""
    prof = []
    for a, b in zip(marks[:-1], marks[1:]):
        if b - a >= 10:
            prof.append(np.interp(np.linspace(a, b, bins, endpoint=False), np.arange(len(values)), values))
    return np.array(prof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--plot_only", action="store_true", help="redraw Fig. I4 from runs/interpretability/C3_saliency.npz (+ .json)")
    args = ap.parse_args()
    if args.plot_only:
        z = np.load(CODE_DIR / "runs/interpretability/C3_saliency.npz")
        js = json.load(open(CODE_DIR / "paper/tables/interpretability/C3_saliency.json"))
        prof = {(t, c): z[f"{t}_{c}"] for t in T for c in ("ecg", "ppg")}
        wave = {c: z[f"wave_{c}"] for c in ("ecg", "ppg")}
        return draw(js["faithful"], prof, wave, js["tests"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from scipy.stats import wilcoxon

    from src.data.dataset import BPCropDataset
    from src.data.pairs import calibration_split, eval_pairs
    from src.data.splits import assign_roles, load_metadata
    from src.features.pwa_features import detect_points
    from src.models import build_model
    from src.viz.style import TARGET, figure, panel_labels, save, setup

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(2026)
    t0 = time.time()
    # draw the readings: n test targets at random over all folds (seed 2026)
    per_fold = {}
    for fold in range(5):
        ck = torch.load(CODE_DIR / MODEL_RUN / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg = ck["config"]
        meta, fs, ch = load_metadata(cfg)
        df = assign_roles(meta, cfg, fold)
        d = df[df.role == "test"].reset_index(drop=True)
        cal, tg = calibration_split(d, cfg["calibration"]["phase"], cfg["calibration"]["conditions"], int(cfg["calibration"]["k_max"]))
        per_fold[fold] = (ck, cfg, d, cal, tg, fs, ch)
    pool = [(f, int(t)) for f in range(5) for t in per_fold[f][4]]
    pick = [pool[i] for i in rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)]
    print(f"{len(pick)} test readings drawn from {len(pool):,}", flush=True)

    rec = {k: [] for k in ["fold", "sample_id", "setting"]}
    prof = {(t, c): [] for t in T for c in ("ecg", "ppg")}
    wave = {c: [] for c in ("ecg", "ppg")}
    faith = {t: {lvl: {"salient": [], "random": []} for lvl in (10, 20)} for t in T}
    for fold in range(5):
        items = [t for f, t in pick if f == fold]
        if not items:
            continue
        ck, cfg, d, cal, tg, fs, ch = per_fold[fold]
        st = ck["stats"]
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        model = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        for prm in model.parameters():
            prm.requires_grad_(False)                                            # gradients w.r.t. the input only
        signals = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r")
        ch_idx = [ch.index(c) for c in cfg["data"]["inputs"]]
        ds = BPCropDataset(d, signals, ch_idx, int(round(cfg["data"]["crop_s"] * fs)), y_mean, y_std, T, [], None, None,
                           train=False, augment_cfg=None)
        pids = d.pid.to_numpy()
        i_ecg, i_ppg = list(cfg["data"]["inputs"]).index("ecg"), list(cfg["data"]["inputs"]).index("ppg")

        def delta(x, zc, pc, bc):
            """Standardised predicted change, mean over the calibration readings. x: (B, C, L)."""
            z, zp = model.encoder.encode(x)
            outs = []
            for j in range(zc.shape[0]):
                B = x.shape[0]
                feats = model._compare(z, zc[j].expand(B, -1))
                if model.use_ppg:
                    feats += model._compare(zp, pc[j].expand(B, -1))
                feats.append(model.bp_embed(bc[j].expand(B, -1)))
                outs.append(model.head(torch.cat(feats, dim=-1)).float())
            return torch.stack(outs).mean(0)                                     # (B, 2): change in standardised units

        for t_pos in items:
            x = ds[t_pos]["x"].numpy()                                           # (C, L), z-scored
            cpos = cal[pids[t_pos]][:3]
            with torch.no_grad():
                xc = torch.stack([ds[int(c)]["x"] for c in cpos]).to(dev)
                zc, pc = model.encoder.encode(xc)
                bc = torch.stack([ds[int(c)]["y"] for c in cpos]).to(dev)
            xs = torch.from_numpy(x).to(dev)
            alphas = torch.linspace(1.0 / args.steps, 1.0, args.steps, device=dev).view(-1, 1, 1)
            ig = {}
            for j, t in enumerate(T):
                path = (alphas * xs.unsqueeze(0)).requires_grad_(True)
                out = delta(path, zc, pc, bc)[:, j].sum()
                g, = torch.autograd.grad(out, path)
                ig[t] = (xs * g.mean(0)).detach().cpu().numpy() * y_std[j]       # mmHg per sample
            # faithfulness: deletion of salient vs random samples
            with torch.no_grad():
                base = delta(xs.unsqueeze(0), zc, pc, bc)[0].cpu().numpy() * y_std
            for t in T:
                score = np.abs(ig[t]).max(axis=0)
                for lvl in (10, 20):
                    k = int(x.shape[1] * lvl / 100)
                    sal = np.argsort(-score)[:k]
                    rnd = rng.choice(x.shape[1], k, replace=False)
                    xb = torch.from_numpy(np.stack([interp_delete(x, sal), interp_delete(x, rnd)]).astype(np.float32)).to(dev)
                    with torch.no_grad():
                        dd = delta(xb, zc, pc, bc).cpu().numpy() * y_std
                    j = T.index(t)
                    faith[t][lvl]["salient"].append(abs(dd[0, j] - base[j]))
                    faith[t][lvl]["random"].append(abs(dd[1, j] - base[j]))
            # beat-aligned profiles
            cycles, rpk, _, _ = detect_points(x[i_ecg], x[i_ppg], fs)
            feet = np.array([c["s"] for c in cycles] + ([cycles[-1]["v"]] if cycles else []))
            for c_name, marks, ci in [("ecg", np.asarray(rpk), i_ecg), ("ppg", feet, i_ppg)]:
                if len(marks) >= 3:
                    for t in T:
                        prof[(t, c_name)].append(cycle_profile(np.abs(ig[t][ci]), marks).mean(0))
                    wave[c_name].append(cycle_profile(x[ci], marks).mean(0))
            rec["fold"].append(fold); rec["sample_id"].append(int(d.sample_id.iloc[t_pos]))
            rec["setting"].append("home" if d.phase.iloc[t_pos] == "ambulatory" else "lab")
        print(f"fold {fold}: {len(items)} readings ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model
        torch.cuda.empty_cache()

    # faithfulness test
    L = ["# Interpretability C3 — beat-level saliency (integrated gradients) and faithfulness test", "",
         f"{len(rec['sample_id'])} random test readings (seed 2026); no-time model; {args.steps} IG steps, zero baseline. "
         "Deletion: top 10 % / 20 % samples by |IG| (max over channels) vs the same number of random samples, replaced by linear "
         "interpolation; |change of the predicted Δ| compared with a paired one-sided Wilcoxon test (salient > random).", "",
         "| Target | Deleted | Median |ΔΔ̂| salient (mmHg) | Median |ΔΔ̂| random (mmHg) | Salient > random in | p (one-sided) |",
         "|---|---|---|---|---|---|"]
    res, faithful = {}, True
    for t in T:
        for lvl in (10, 20):
            s, r_ = np.array(faith[t][lvl]["salient"]), np.array(faith[t][lvl]["random"])
            p = wilcoxon(s, r_, alternative="greater").pvalue
            res[f"{t}/{lvl}"] = {"median_salient": float(np.median(s)), "median_random": float(np.median(r_)),
                                 "share_salient_greater": float(np.mean(s > r_)), "p": float(p)}
            faithful &= p < 0.05
            L.append(f"| {t.upper()} | {lvl} % | {np.median(s):.2f} | {np.median(r_):.2f} | {np.mean(s > r_):.0%} | {p:.2g} |")
    L += ["", f"**Faithfulness criterion (p < 0.05 at both deletion levels, both targets): {'PASSED - saliency maps are reported' if faithful else 'NOT PASSED - saliency maps are not reported; this negative result is reported instead'}.**"]
    tab = CODE_DIR / "paper/tables/interpretability"
    (tab / "C3_saliency.md").write_text("\n".join(L))
    json.dump({"faithful": bool(faithful), "tests": res, "n": len(rec["sample_id"])}, open(tab / "C3_saliency.json", "w"), indent=2)
    np.savez_compressed(CODE_DIR / "runs/interpretability/C3_saliency.npz", **{f"{t}_{c}": np.array(prof[(t, c)]) for t in T for c in ("ecg", "ppg")},
                        wave_ecg=np.array(wave["ecg"]), wave_ppg=np.array(wave["ppg"]), sample_id=np.array(rec["sample_id"]),
                        fold=np.array(rec["fold"]), setting=np.array(rec["setting"]))
    print("\n".join(L))

    draw(faithful, prof, wave, res)


def draw(faithful, prof, wave, res):
    """Fig. I4: saliency maps if faithful, else the faithfulness test."""
    from src.viz.style import TARGET, figure, panel_labels, save, setup
    setup()
    if faithful:
        fig, axes = figure("double", height=2.3, ncols=2)
        ph = np.linspace(0, 100, BINS, endpoint=False)
        for ax, c, lab in [(axes[0], "ecg", "ECG cycle (R peak to R peak, %)"), (axes[1], "ppg", "PPG cycle (foot to foot, %)")]:
            w = np.mean(wave[c], axis=0)
            ax2 = ax.twinx()
            hw, = ax2.plot(ph, w, color="#B0B0B0", lw=0.8, label=f"mean {c.upper()} cycle (scaled)")
            ax2.set_yticks([]); ax2.spines["right"].set_visible(False); ax2.grid(False)
            hs = [ax.plot(ph, np.mean(prof[(t, c)], axis=0), color=TARGET[t]["color"], label=f"{TARGET[t]['label']} change")[0] for t in T]
            ax.set(xlabel=lab, ylabel="|attribution| (mmHg per sample)")
            ax.legend(handles=hs + [hw], loc="upper right")
    else:
        fig, axes = figure("single", height=2.3, ncols=1)
        axes = np.atleast_1d(axes)
        x = np.arange(4)
        keys = [(t, lvl) for t in T for lvl in (10, 20)]
        axes[0].bar(x - 0.19, [res[f"{t}/{l}"]["median_salient"] for t, l in keys], width=0.36, color="#4D4D4D", label="salient samples deleted")
        axes[0].bar(x + 0.19, [res[f"{t}/{l}"]["median_random"] for t, l in keys], width=0.36, color="#B0B0B0", label="random samples deleted")
        axes[0].set(xticks=x, xticklabels=[f"{t.upper()} {l} %" for t, l in keys], ylabel="Median change of predicted Δ (mmHg)")
        axes[0].legend()
    panel_labels(axes)
    save(fig, "interp_C3_saliency", source=__file__)


if __name__ == "__main__":
    main()
