#!/usr/bin/env python
"""
Ablation report (plan: docs/ABLATION_PLAN.md). Read-only on all run folders.

  python scripts/ablation_report.py

Checks first (stops if any fails): every ablation has all 5 folds; every fold used the same frozen folds file as
v1 (hash in provenance); every ablation predicted exactly v1's test recordings (A1: v1's recordings are a subset
of the uncalibrated predictions and are selected).
Then, for k = 1 and 3, on exactly v1's test recordings (pooled over the 5 folds, every subject once):
  MAE and ME ± SD for SBP / DBP, Δ MAE vs v1 (positive = ablation worse), % of subjects where v1 is better,
  Wilcoxon signed-rank on per-subject MAE, parameters, mean validation score over the folds.
A1 (uncalibrated) has no calibration readings, so its k = 1 and k = 3 rows use the same predictions.
Output: paper/tables/ablation/ ABLATION_RESULTS.md, ablation_summary.csv, ablation_delta.png (working figure)
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
SETS = {   # --set main (default): reference = the reported model; --set v1: the earlier table (reference with time context)
    "main": (("ref", "HiCAF-BP-Cal (full model)", "hicaf_bp_cal_notime"), [
        ("A1", "− calibration (uncalibrated model)", "hicaf_bp_main"),
        ("A2", "+ time context", "hicaf_bp_cal_t_scratch"),
        ("A3", "− cross-modal attention", "abl_nt_no_cross"),
        ("A4", "− RoPE (sinusoidal instead)", "abl_nt_sinusoidal"),
        ("A5", "− PPG-only comparison (fused embedding only)", "abl_nt_no_gate"),
        ("A6", "− multi-scale tokenizer (single 120 ms kernel)", "abl_nt_single_scale"),
        ("A7", "− local/global attention (all global)", "abl_nt_all_global"),
        ("A8", "− ECG input (PPG only)", "abl_nt_ppg_only"),
        ("A9", "− PPG input (ECG only)", "abl_nt_ecg_only"),
    ], "ablation_main"),
    "v1": (("ref", "Full model (v1)", "hicaf_bp_cal_t_scratch"), [
    ("A1", "− calibration (uncalibrated model)", "hicaf_bp_main"),
    ("A2", "− time context", "hicaf_bp_cal_notime"),
    ("A3", "− cross-modal attention", "abl_no_cross"),
    ("A4", "− RoPE (sinusoidal instead)", "abl_sinusoidal"),
    ("A5", "− target-specific gates", "abl_no_gate"),
    ("A6", "− multi-scale tokenizer (single 120 ms kernel)", "abl_single_scale"),
    ("A7", "− local/global attention (all global)", "abl_all_global"),
    ("A8", "− ECG input (PPG only)", "abl_ppg_only"),
    ("A9", "− PPG input (ECG only)", "abl_ecg_only"),
], "ablation"),
}


def load_fold_preds(run_dir, k, uncal=False):
    name = "predictions_test.csv" if (k == 1 or uncal) else f"predictions_test_k{k}.csv"
    return pd.concat([pd.read_csv(run_dir / f"fold_{i}" / name) for i in range(5)], ignore_index=True)


def run_info(run_dir):
    p, v = [], []
    for i in range(5):
        m = json.load(open(run_dir / f"fold_{i}" / "metrics.json"))
        info = m.get("info", m)
        p.append(info.get("n_params", m.get("n_params", np.nan)))
        v.append(info.get("best_val_score", m.get("best_val_score", np.nan)))
    return float(np.nanmean(p)) / 1e6, float(np.nanmean(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--folds", default="data_processed/v1/folds.csv")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--set", default="main", choices=list(SETS))
    args = ap.parse_args()
    REF, ABLATIONS, sub = SETS[args.set]
    from scipy.stats import wilcoxon
    runs = Path(args.runs)
    out = Path(args.out) / "tables" / sub
    out.mkdir(parents=True, exist_ok=True)
    folds_hash = hashlib.sha256(Path(args.folds).read_bytes()).hexdigest()

    # ------------------------------------------------------------------ checks
    problems = []
    for aid, _, exp in [REF] + ABLATIONS:
        for i in range(5):
            d = runs / exp / f"fold_{i}"
            if not (d / "metrics.json").exists():
                problems.append(f"{aid}: missing {d}")
                continue
            prov = (d / "provenance.json").read_text() if (d / "provenance.json").exists() else ""
            if folds_hash not in prov and folds_hash[:16] not in prov:
                problems.append(f"{aid}: fold {i} provenance does not contain the frozen folds hash")
    if problems:
        print("CHECKS FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    ref = {k: load_fold_preds(runs / REF[2], k).sort_values("sample_id").reset_index(drop=True) for k in (1, 3)}
    preds = {}
    for aid, _, exp in ABLATIONS:
        for k in (1, 3):
            p = load_fold_preds(runs / exp, k, uncal=(aid == "A1"))
            if p.sample_id.duplicated().any():
                problems.append(f"{aid} k={k}: a recording is predicted twice")
            p = p.set_index("sample_id")
            missing = set(ref[k].sample_id) - set(p.index)
            if missing:
                problems.append(f"{aid} k={k}: {len(missing)} of the reference's test recordings are missing")
                continue
            if aid != "A1" and len(p) != len(ref[k]):
                problems.append(f"{aid} k={k}: predicts {len(p)} recordings, reference {len(ref[k])}")
            q = ref[k][["sample_id", "pid"] + T].copy()
            for t in T:
                q[f"{t}_pred"] = p.loc[q.sample_id, f"{t}_pred"].to_numpy()
            preds[(aid, k)] = q
    if problems:
        print("CHECKS FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"All checks passed: {REF[2]} + {len(ABLATIONS)} ablations x 5 folds, frozen folds (hash {folds_hash[:12]}...), "
          f"identical test recordings ({len(ref[1]):,} for k = 1).")

    # ------------------------------------------------------------------ results
    params_ref, val_ref = run_info(runs / REF[2])
    rows, L = [], ["# Ablation study (Aurora-BP, 5-fold CV; leave-one-out from the final model)", "",
                   f"Pooled over the 5 folds: {len(ref[1]):,} test recordings from {ref[1].pid.nunique():,} subjects, "
                   "identical for every row. Δ = ablation − full model (positive = removing the component makes it worse). "
                   "'full better' = % of subjects whose per-subject MAE is lower with the full model; p = Wilcoxon signed-rank "
                   "on per-subject MAE. VAL = mean best validation score over the 5 folds (for information only).", ""]
    for k in (1, 3):
        mr = regression_metrics(ref[k], T, 5)
        per_ref = {t: (ref[k][f"{t}_pred"] - ref[k][t]).abs().groupby(ref[k].pid).mean() for t in T}
        L += [f"## k = {k}", "",
              "| # | Model | Params (M) | SBP MAE | Δ SBP | full better / p (SBP) | DBP MAE | Δ DBP | full better / p (DBP) | SBP ME ± SD | DBP ME ± SD | VAL |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|",
              f"| {REF[0]} | **{REF[1]}** | {params_ref:.2f} | **{mr['sbp']['MAE']:.2f}** | – | – | **{mr['dbp']['MAE']:.2f}** | – | – | "
              f"{mr['sbp']['ME']:+.2f} ± {mr['sbp']['SDE']:.2f} | {mr['dbp']['ME']:+.2f} ± {mr['dbp']['SDE']:.2f} | {val_ref:.3f} |"]
        for aid, name, exp in ABLATIONS:
            q = preds[(aid, k)]
            m = regression_metrics(q, T, 5)
            params, val = run_info(runs / exp)
            cells = {}
            for t in T:
                per = (q[f"{t}_pred"] - q[t]).abs().groupby(q.pid).mean()
                p = wilcoxon(per, per_ref[t]).pvalue
                cells[t] = (m[t]["MAE"] - mr[t]["MAE"], (per_ref[t] < per).mean(), p)
            fmt = lambda c: f"{c[1]:.0%} / {'< 0.001' if c[2] < 0.001 else f'{c[2]:.3f}'}"
            L.append(f"| {aid} | {name} | {params:.2f} | {m['sbp']['MAE']:.2f} | {cells['sbp'][0]:+.2f} | {fmt(cells['sbp'])} | "
                     f"{m['dbp']['MAE']:.2f} | {cells['dbp'][0]:+.2f} | {fmt(cells['dbp'])} | "
                     f"{m['sbp']['ME']:+.2f} ± {m['sbp']['SDE']:.2f} | {m['dbp']['ME']:+.2f} ± {m['dbp']['SDE']:.2f} | "
                     f"{val:.3f} |")
            rows.append({"id": aid, "ablation": name, "experiment": exp, "k": k, "params_M": params, "val_score": val,
                         **{f"{t}_mae": m[t]["MAE"] for t in T}, **{f"{t}_delta": cells[t][0] for t in T},
                         **{f"{t}_full_better": cells[t][1] for t in T}, **{f"{t}_p": cells[t][2] for t in T},
                         **{f"{t}_me": m[t]["ME"] for t in T}, **{f"{t}_sde": m[t]["SDE"] for t in T}})
        rows.append({"id": REF[0], "ablation": REF[1], "experiment": REF[2], "k": k, "params_M": params_ref, "val_score": val_ref,
                     **{f"{t}_mae": mr[t]["MAE"] for t in T}, **{f"{t}_me": mr[t]["ME"] for t in T}, **{f"{t}_sde": mr[t]["SDE"] for t in T}})
        L.append("")
    L += ["A1: the uncalibrated model has no calibration input, so its k = 1 and k = 3 rows are identical. "
          "A5 also removes the PPG-only branch of the calibration head; A8/A9 have one input, so cross-attention and gates are "
          "inherently absent. Parameter counts differ where a component carries parameters. "
          "A2 (main set) ADDS the time context to the full model: Δ = with − without."]
    (out / "ABLATION_RESULTS.md").write_text("\n".join(L))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "ablation_summary.csv", index=False)
    print("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ[(summ.k == 1) & (summ.id != REF[0])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for a, t in zip(ax, T):
        a.barh(s.id + "  " + s.ablation.str.replace("− ", ""), s[f"{t}_delta"], color=["C3" if v > 0 else "C2" for v in s[f"{t}_delta"]])
        a.axvline(0, color="k", lw=0.8)
        a.set_xlabel(f"Δ {t.upper()} MAE vs full model (mmHg, k = 1)")
        a.grid(alpha=0.3, axis="x")
    ax[0].invert_yaxis()
    plt.tight_layout()
    plt.savefig(out / "ablation_delta.png", dpi=150)
    print(f"\nSaved {out}/")


if __name__ == "__main__":
    main()