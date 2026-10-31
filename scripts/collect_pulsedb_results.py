#!/usr/bin/env python
"""
Collect the PulseDB adaptation study (plan: docs/PULSEDB_FINETUNING_PLAN.md). Read-only on all run folders.

  python scripts/collect_pulsedb_results.py

Checks first (stops if any fails): all 100 runs finished; every run used the frozen split file (hash); every run's
test subjects = the frozen 30% test set; every run predicted exactly the same test segments; the zero-shot
predictions cover exactly these segments.
Then, per dataset (MIMIC, VitalDB) and k (1, 3), on the SAME test segments:
  PRIMARY          mean ± SD over the 5 draws (each model used exactly its fraction); ME ± SD, AAMI and IEEE
                   from the draw means
  ensemble         average prediction of the 5 models, ONLY at 0% (zero-shot) and 100% - below 100% the 5 draws
                   together contain more subjects than the fraction, so an ensemble would overstate it
  reference        calibration value only
  paired tests     per draw: fine-tuned model r vs its from-scratch twin r (same subjects) and vs Aurora model r
                   zero-shot; per-subject MAE averaged over the 5 draws; Wilcoxon signed-rank
Output: paper/tables/finetuning/  RESULTS_<dataset>.md, summary.csv, best_epochs.csv
        paper/figures/finetuning_learning_curve.png (working figure)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import regression_metrics  # noqa: E402

T = ["sbp", "dbp"]
DATASETS, FRACTIONS, DRAWS, KS, MODES = ["MIMIC", "VitalDB"], [5, 10, 20, 50, 100], [0, 1, 2, 3, 4], [1, 3], ["finetune", "scratch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--zero_shot", default="runs/hicaf_bp_cal_notime/external_pulsedb")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    from scipy.stats import wilcoxon
    runs, data, zs = Path(args.runs), Path(args.data), Path(args.zero_shot)
    out_t, out_f = Path(args.out) / "tables" / "finetuning", Path(args.out) / "figures"
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(data / "splits" / "splits_manifest.json"))

    # ------------------------------------------------------------------ checks
    problems, preds, epochs = [], {}, []
    for ds in DATASETS:
        split = pd.read_csv(data / "splits" / f"splits_{ds}.csv")
        test_pids = set(split.pid[split.test])
        sha = manifest["files"][f"splits_{ds}.csv"]
        ref = {}
        for mode in MODES:
            for f in FRACTIONS:
                for r in DRAWS:
                    d = runs / f"pulsedb_{mode}" / ds / f"f{f}" / f"d{r}"
                    if not (d / "metrics.json").exists():
                        problems.append(f"missing run: {d}")
                        continue
                    info = json.load(open(d / "metrics.json"))["info"]
                    if info.get("split_sha256") != sha:
                        problems.append(f"split hash mismatch: {d}")
                    epochs.append({"dataset": ds, "mode": mode, "fraction": f, "draw": r,
                                   "best_epoch": info["best_epoch"], "best_val_score": info["best_val_score"],
                                   "train_subjects": info["n_subjects"]["train"], "val_subjects": info["n_subjects"]["val"]})
                    for k in KS:
                        p = pd.read_csv(d / f"predictions_test_k{k}.csv").sort_values("sample_id").reset_index(drop=True)
                        if set(p.pid) != test_pids:
                            problems.append(f"test subjects differ from the frozen test set: {d} k={k}")
                        if k not in ref:
                            ref[k] = p.sample_id.to_numpy()
                        elif not np.array_equal(ref[k], p.sample_id.to_numpy()):
                            problems.append(f"test segments differ between runs: {d} k={k}")
                        preds[(ds, mode, f, r, k)] = p
        for k in KS:
            ens = pd.read_csv(zs / f"predictions_ensemble_k{k}.csv")
            ens = ens[ens.pid.isin(test_pids)].sort_values("sample_id").reset_index(drop=True)
            if k in ref and not np.array_equal(ens.sample_id.to_numpy(), ref[k]):
                problems.append(f"zero-shot predictions do not cover exactly the test segments: {ds} k={k}")
            preds[(ds, "zeroshot", 0, "ens", k)] = ens
            for r in DRAWS:
                z = np.load(zs / f"predictions_fold{r}_k{k}.npz")
                zp = pd.DataFrame(z["pred"], columns=[f"{t}_pred" for t in T]).assign(sample_id=z["sample_id"])
                zz = ens.drop(columns=[f"{t}_pred" for t in T]).merge(zp, on="sample_id", how="left")
                if zz[[f"{t}_pred" for t in T]].isna().any().any():
                    problems.append(f"zero-shot fold {r} predictions missing segments: {ds} k={k}")
                preds[(ds, "zeroshot", 0, r, k)] = zz
    if problems:
        print("CHECKS FAILED:\n  " + "\n  ".join(problems[:30]))
        sys.exit(1)
    print("All checks passed: 100 runs, frozen splits, identical test subjects and segments, zero-shot aligned.")
    pd.DataFrame(epochs).to_csv(out_t / "best_epochs.csv", index=False)

    # ------------------------------------------------------------------ results
    rows = []
    for ds in DATASETS:
        L = [f"# PulseDB adaptation — {ds}", "",
             f"Test set: {preds[(ds, 'finetune', 5, 0, 1)].pid.nunique():,} subjects (fixed 30%, never used for training "
             f"or model selection). Fractions are % of the 70% pool; subject counts in brackets. PRIMARY = mean ± SD over the 5 "
             f"draws (each model used exactly its fraction); ME ± SD, AAMI and IEEE from the draw means. Ensemble (average "
             f"of the 5 models) only at 0% and 100%, where all models used the same subjects (below 100%, the 5 draws "
             f"together contain more subjects than the fraction).", ""]
        ep = pd.DataFrame(epochs)
        sel = ep[(ep.dataset == ds) & (ep["mode"] == "finetune")]
        n_train = {f: int(sel[sel.fraction == f].train_subjects.iloc[0]) for f in FRACTIONS}
        n_val = {f: int(sel[sel.fraction == f].val_subjects.iloc[0]) for f in FRACTIONS}
        for k in KS:
            cal = preds[(ds, "zeroshot", 0, "ens", k)].copy()
            for t in T:
                cal[f"{t}_pred"] = cal[f"{t}_cal_mean"]
            mc = regression_metrics(cal, T, 5)
            L += [f"## k = {k}", "",
                  "| Method | Fraction (train + val subjects) | SBP MAE mean ± SD | SBP ME ± SD | DBP MAE mean ± SD | DBP ME ± SD | AAMI SBP / DBP | IEEE SBP / DBP | Ensemble MAE SBP / DBP |",
                  "|---|---|---|---|---|---|---|---|---|"]
            L.append(f"| Calibration value only | – | {mc['sbp']['MAE']:.2f} | {mc['sbp']['ME']:+.2f} ± {mc['sbp']['SDE']:.2f} | "
                     f"{mc['dbp']['MAE']:.2f} | {mc['dbp']['ME']:+.2f} ± {mc['dbp']['SDE']:.2f} | – | – | – |")
            rows.append({"dataset": ds, "k": k, "method": "calibration_only", "fraction": None,
                         **{f"{t}_mae_ens": mc[t]["MAE"] for t in T}})
            draw_level = {}
            for mode, f in [("zeroshot", 0)] + [(m, f) for m in MODES for f in FRACTIONS]:
                per = [preds[(ds, mode, f, r, k)] for r in DRAWS]
                ms = [regression_metrics(p, T, 5) for p in per]                    # one model = one draw = exactly f
                draw_level[(mode, f)] = per
                g = {t: {x: np.array([m[t][x] for m in ms]) for x in ("MAE", "ME", "SDE")} for t in T}
                ens_txt = {t: "–" for t in T}
                if f in (0, 100):                                                  # all models saw the same subjects
                    ens = per[0].copy()
                    for t in T:
                        ens[f"{t}_pred"] = np.mean([p[f"{t}_pred"].to_numpy() for p in per], axis=0)
                    me = regression_metrics(ens, T, 5)
                    ens_txt = {t: f"{me[t]['MAE']:.2f}" for t in T}
                name = {"zeroshot": "Zero-shot (Aurora only)", "finetune": "Fine-tuned Aurora model", "scratch": "From scratch (no Aurora)"}[mode]
                frac = "0%" if f == 0 else f"{f}% ({n_train[f]} + {n_val[f]})"
                mean = {t: {x: g[t][x].mean() for x in g[t]} for t in T}
                aami = {t: "pass" if abs(mean[t]["ME"]) <= 5 and mean[t]["SDE"] <= 8 else "fail" for t in T}
                ieee = {t: "A" if mean[t]["MAE"] <= 5 else "B" if mean[t]["MAE"] <= 6 else "C" if mean[t]["MAE"] <= 7 else "D" for t in T}
                L.append(f"| {name} | {frac} | {g['sbp']['MAE'].mean():.2f} ± {g['sbp']['MAE'].std(ddof=1):.2f} | "
                         f"{mean['sbp']['ME']:+.2f} ± {mean['sbp']['SDE']:.2f} | {g['dbp']['MAE'].mean():.2f} ± {g['dbp']['MAE'].std(ddof=1):.2f} | "
                         f"{mean['dbp']['ME']:+.2f} ± {mean['dbp']['SDE']:.2f} | {aami['sbp']} / {aami['dbp']} | {ieee['sbp']} / {ieee['dbp']} | "
                         f"{ens_txt['sbp']} / {ens_txt['dbp']} |")
                rows.append({"dataset": ds, "k": k, "method": mode, "fraction": f,
                             **{f"{t}_mae_mean": g[t]["MAE"].mean() for t in T}, **{f"{t}_mae_sd": g[t]["MAE"].std(ddof=1) for t in T},
                             **{f"{t}_me_mean": mean[t]["ME"] for t in T}, **{f"{t}_sde_mean": mean[t]["SDE"] for t in T},
                             **{f"{t}_mae_ens": (float(ens_txt[t]) if ens_txt[t] != "–" else np.nan) for t in T}})
            L += ["", "Paired subject-level comparison, PER DRAW (fine-tuned model r vs its from-scratch twin r trained on the "
                  "same subjects, and vs Aurora model r zero-shot); per-subject MAE averaged over the 5 draws; Wilcoxon; "
                  "% = subjects where fine-tuned is better:", "",
                  "| Fraction | Fine-tuned vs from scratch: SBP | DBP | Fine-tuned vs zero-shot: SBP | DBP |", "|---|---|---|---|---|"]

            def draw_avg(mode, f):
                per = draw_level[(mode, f)]
                return {t: pd.concat([(p[f"{t}_pred"] - p[t]).abs().groupby(p.pid).mean() for p in per], axis=1).mean(axis=1) for t in T}

            z0 = draw_avg("zeroshot", 0)
            for f in FRACTIONS:
                a, b = draw_avg("finetune", f), draw_avg("scratch", f)
                cells = []
                for other in (b, z0):
                    for t in T:
                        p = wilcoxon(a[t], other[t]).pvalue
                        cells.append(f"{a[t].mean() - other[t].mean():+.2f} mmHg, {(a[t] < other[t]).mean():.0%}, "
                                     f"p {'< 0.001' if p < 0.001 else f'{p:.3f}'}")
                L.append(f"| {f}% | " + " | ".join(cells) + " |")
            L.append("")
        (out_t / f"RESULTS_{ds}.md").write_text("\n".join(L))
        print("\n".join(L))
    summ = pd.DataFrame(rows)
    summ.to_csv(out_t / "summary.csv", index=False)

    # ------------------------------------------------------------------ working figure: learning curve (k = 1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = np.arange(len(FRACTIONS) + 1)
    for i, ds in enumerate(DATASETS):
        s = summ[(summ.dataset == ds) & (summ.k == 1)]
        for j, t in enumerate(T):
            a = ax[i, j]
            z = s[s.method == "zeroshot"].iloc[0]
            for mode, c in [("finetune", "C0"), ("scratch", "C1")]:
                q = s[s.method == mode].sort_values("fraction")
                mu = np.r_[z[f"{t}_mae_mean"], q[f"{t}_mae_mean"]] if mode == "finetune" else np.r_[np.nan, q[f"{t}_mae_mean"]]
                sd = np.r_[z[f"{t}_mae_sd"], q[f"{t}_mae_sd"]] if mode == "finetune" else np.r_[np.nan, q[f"{t}_mae_sd"]]
                a.plot(x, mu, "o-", color=c, label={"finetune": "Fine-tuned Aurora model", "scratch": "From scratch"}[mode])
                a.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.2)
            a.axhline(s[s.method == "calibration_only"][f"{t}_mae_ens"].iloc[0], ls="--", color="grey", label="Calibration value only")
            a.set_xticks(x)
            a.set_xticklabels(["0%"] + [f"{f}%" for f in FRACTIONS])
            a.set_title(f"{ds} — {t.upper()} (k = 1)")
            a.set_ylabel("MAE (mmHg)")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    for a in ax[1]:
        a.set_xlabel("Fraction of the fine-tuning pool (0% = zero-shot)")
    plt.tight_layout()
    plt.savefig(out_f / "finetuning_learning_curve.png", dpi=150)
    print(f"\nSaved {out_t}/ and {out_f}/finetuning_learning_curve.png")


if __name__ == "__main__":
    main()
