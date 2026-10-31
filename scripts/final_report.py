#!/usr/bin/env python
"""
Final evaluation of one experiment on the POOLED 5-fold CV test set (every test recording predicted once).

  python scripts/final_report.py --experiment hicaf_bp_cal_t_scratch

For each number of calibration readings k (1 and 3) it reports, for SBP and DBP:
  Regression    MAE (+ 95% CI, subject-level bootstrap), RMSE, Pearson r, R^2,
                ME +/- SD of error (AAMI / ISO 81060-2: |ME| <= 5 and SD <= 8 mmHg),
                Bland-Altman 95% limits of agreement,
                % of |errors| within 5 / 10 / 15 mmHg + BHS grade, IEEE 1708 grade (MAE),
                median within-subject correlation (BP tracking), per-fold MAE mean +/- SD
  Classification (thresholds applied to predicted SBP/DBP; 4-class, 3-class, binary 130/80)
                accuracy, balanced accuracy, macro-F1, quadratic-weighted kappa, AUROC/AUPRC (binary),
                per-class sensitivity / specificity / precision / F1, confusion matrix
  Comparison    the same metrics for "calibration value only" and "uncalibrated model + shift" on the
                same recordings (context for the paper)
Grades are descriptive (this is not a formal clinical validation protocol).

Output: runs/<experiment>/final/  FINAL_RESULTS.md, final_metrics.json, figures/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.evaluation.metrics import classification_metrics, regression_metrics  # noqa: E402
from src.evaluation.report import paper_figures  # noqa: E402

K_FOLDS = 5
TARGETS = ["sbp", "dbp"]


def load_pooled(exp_dir, k):
    name = "predictions_test.csv" if k == 1 else f"predictions_test_k{k}.csv"
    files = [exp_dir / f"fold_{i}" / name for i in range(K_FOLDS)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing prediction files: {missing}")
    p = pd.concat([pd.read_csv(f).assign(fold=i) for i, f in enumerate(files)], ignore_index=True)
    if p["sample_id"].duplicated().any():
        sys.exit("a recording appears in more than one fold - folds are broken")
    return p


def bootstrap_mae_ci(p, n_boot=1000, seed=0):
    """95% CI of MAE by resampling SUBJECTS (recordings of one person stay together)."""
    rng = np.random.default_rng(seed)
    ae = {t: (p[f"{t}_pred"] - p[t]).abs() for t in TARGETS}
    by = {t: ae[t].groupby(p["pid"]).agg(["sum", "count"]) for t in TARGETS}
    pids = by["sbp"].index.to_numpy()
    out = {}
    for t in TARGETS:
        s, c = by[t]["sum"].to_numpy(), by[t]["count"].to_numpy()
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(pids), len(pids))
            boots.append(s[idx].sum() / c[idx].sum())
        out[t] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def per_fold(p):
    rows = [{t: (g[f"{t}_pred"] - g[t]).abs().mean() for t in TARGETS} for _, g in p.groupby("fold")]
    d = pd.DataFrame(rows)
    return {t: (float(d[t].mean()), float(d[t].std(ddof=1))) for t in TARGETS}


def with_preds(p, col):
    b = p.copy()
    for t in TARGETS:
        b[f"{t}_pred"] = p[f"{t}_{col}"]
    return b


def fmt_regression(m, ci, pf):
    L = ["| Metric | SBP | DBP |", "|---|---|---|"]
    s, d = m["sbp"], m["dbp"]
    L.append(f"| MAE (mmHg) [95% CI] | **{s['MAE']:.2f}** [{ci['sbp'][0]:.2f}, {ci['sbp'][1]:.2f}] | "
             f"**{d['MAE']:.2f}** [{ci['dbp'][0]:.2f}, {ci['dbp'][1]:.2f}] |")
    L.append(f"| MAE per fold, mean ± SD | {pf['sbp'][0]:.2f} ± {pf['sbp'][1]:.2f} | {pf['dbp'][0]:.2f} ± {pf['dbp'][1]:.2f} |")
    rows = [("RMSE (mmHg)", "RMSE", "{:.2f}"), ("Pearson r", "r", "{:.3f}"), ("R²", "R2", "{:.3f}")]
    for name, key, f in rows:
        L.append(f"| {name} | {f.format(s[key])} | {f.format(d[key])} |")
    L.append(f"| ME ± SD (mmHg) | {s['ME']:+.2f} ± {s['SDE']:.2f} | {d['ME']:+.2f} ± {d['SDE']:.2f} |")
    L.append(f"| 95% limits of agreement (mmHg) | {s['LoA_low']:+.1f} to {s['LoA_high']:+.1f} | "
             f"{d['LoA_low']:+.1f} to {d['LoA_high']:+.1f} |")
    L.append(f"| AAMI / ISO 81060-2 (\\|ME\\| ≤ 5 and SD ≤ 8) | {'pass' if s['AAMI_pass'] else 'fail'} | "
             f"{'pass' if d['AAMI_pass'] else 'fail'} |")
    L.append(f"| \\|error\\| ≤ 5 / 10 / 15 mmHg | {s['within5']:.1%} / {s['within10']:.1%} / {s['within15']:.1%} | "
             f"{d['within5']:.1%} / {d['within10']:.1%} / {d['within15']:.1%} |")
    L.append(f"| BHS grade (A: 60/85/95 %, B: 50/75/90 %, C: 40/65/85 %) | {s['BHS_grade']} | {d['BHS_grade']} |")
    L.append(f"| IEEE 1708 grade (A ≤ 5, B ≤ 6, C ≤ 7, D > 7 mmHg MAE) | {s['IEEE1708_grade']} | {d['IEEE1708_grade']} |")
    L.append(f"| Within-subject r, median [IQR] | {s['within_subject_r_median']:.2f} "
             f"[{s['within_subject_r_q1']:.2f}, {s['within_subject_r_q3']:.2f}] | {d['within_subject_r_median']:.2f} "
             f"[{d['within_subject_r_q1']:.2f}, {d['within_subject_r_q3']:.2f}] |")
    return L


def fmt_classification(cls):
    L = ["| Scheme | Accuracy | Balanced acc. | Macro-F1 | Kappa (quadratic) | AUROC | AUPRC |",
         "|---|---|---|---|---|---|---|"]
    for name, r in cls.items():
        auc = f"{r['auroc']:.3f}" if "auroc" in r else "–"
        apr = f"{r['auprc']:.3f}" if "auprc" in r else "–"
        L.append(f"| {name} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['macro_f1']:.3f} | "
                 f"{r['kappa']:.3f} | {auc} | {apr} |")
    for name, r in cls.items():
        L += ["", f"**{name}**", "", "| Class | Support | Sensitivity | Specificity | Precision | F1 |",
              "|---|---|---|---|---|---|"]
        for lab, c in r["per_class"].items():
            L.append(f"| {lab} | {c['support']:,} | {c['sensitivity']:.3f} | {c['specificity']:.3f} | "
                     f"{c['precision']:.3f} | {c['f1']:.3f} |")
        labs = r["labels"]
        L += ["", "Confusion matrix (rows = true, columns = predicted):", "",
              "| true \\ pred | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
        for lab, row in zip(labs, r["confusion_matrix"]):
            L.append(f"| {lab} | " + " | ".join(f"{v:,}" for v in row) + " |")
    return L


def fmt_comparison(rows):
    L = ["| Method (same recordings) | SBP MAE | SBP ME ± SD | DBP MAE | DBP ME ± SD |", "|---|---|---|---|---|"]
    for name, m in rows:
        L.append(f"| {name} | {m['sbp']['MAE']:.2f} | {m['sbp']['ME']:+.2f} ± {m['sbp']['SDE']:.2f} | "
                 f"{m['dbp']['MAE']:.2f} | {m['dbp']['ME']:+.2f} ± {m['dbp']['SDE']:.2f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="hicaf_bp_cal_t_scratch")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3], help="numbers of calibration readings")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--config", default="base.yaml", help="only used for paths")
    args = ap.parse_args()

    runs = Path(load_config(args.config)["paths"]["runs_dir"])
    exp_dir = runs / args.experiment
    out = exp_dir / "final"
    out.mkdir(exist_ok=True)

    L = [f"# Final results — {args.experiment}", "",
         "Pooled 5-fold subject-level cross-validation test set: every test recording predicted exactly once, "
         "by a model that never saw that subject during training. Grades are descriptive.", ""]
    all_metrics = {}
    for k in args.ks:
        p = load_pooled(exp_dir, k)
        m = regression_metrics(p, TARGETS, min_rec=5)
        ci = bootstrap_mae_ci(p, args.n_boot)
        pf = per_fold(p)
        cls = classification_metrics(p)
        comp = [("**HiCAF-BP-Cal (ours)**", m)]
        if all(f"{t}_cal_mean" in p for t in TARGETS):
            comp.append(("Calibration value only", regression_metrics(with_preds(p, "cal_mean"), TARGETS, 5)))
        if all(f"{t}_uncal_shift_pred" in p for t in TARGETS):
            comp.append(("Uncalibrated model + calibration shift", regression_metrics(with_preds(p, "uncal_shift_pred"), TARGETS, 5)))
        all_metrics[f"k{k}"] = {"n_recordings": int(len(p)), "n_subjects": int(p.pid.nunique()),
                                "regression": m, "mae_ci95": ci, "mae_per_fold_mean_sd": pf,
                                "classification": cls, "comparison": {n: x for n, x in comp}}
        L += [f"## {k} calibration reading{'s' if k > 1 else ''} (k = {k})", "",
              f"{len(p):,} test recordings from {p.pid.nunique():,} subjects.", "",
              "### Regression", ""] + fmt_regression(m, ci, pf) + \
             ["", "### Classification (from predicted SBP/DBP)", ""] + fmt_classification(cls) + \
             ["", "### Comparison on the same recordings", ""] + fmt_comparison(comp) + [""]
        paper_figures(p, out / "figures" / f"k{k}", title=f"{args.experiment}, k = {k}")

    (out / "FINAL_RESULTS.md").write_text("\n".join(L))
    json.dump(all_metrics, open(out / "final_metrics.json", "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\nSaved {out}/FINAL_RESULTS.md, final_metrics.json and figures/")


if __name__ == "__main__":
    main()
