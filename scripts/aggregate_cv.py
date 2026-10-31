#!/usr/bin/env python
"""
Combine the 5 CV folds of one experiment into the final (paper) results.

  python scripts/aggregate_cv.py --experiment hicaf_bp

Reads   runs/<experiment>/fold_0..4/predictions_test.csv  (+ metrics.json, config.yaml)
Writes  runs/<experiment>/cv/
          cv_predictions_test.csv   every recording predicted exactly once, with demographics
          cv_metrics.json           pooled metrics, per-fold mean ± SD, subgroups, baselines
          CV_RESULTS.md             readable tables
          figures/                  Bland-Altman, error histograms, scatter (SBP, DBP)
and one line in runs/EXPERIMENTS_CV.csv / .md
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.splits import assign_roles  # noqa: E402
from src.evaluation.metrics import (classification_metrics, evaluate_predictions,  # noqa: E402
                                    regression_metrics)
from src.evaluation.report import (classification_section, paper_figures, regression_section,  # noqa: E402
                                   write_md_table)

K = 5
DEMOGRAPHICS = ["age_group", "gender", "bmi_group", "fitz_group", "self_report_htn"]
GROUPS = ["protocol", "phase", "is_exercise"] + DEMOGRAPHICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default="base.yaml", help="only used for paths")
    args = ap.parse_args()

    base = load_config(args.config)
    exp_dir = Path(base["paths"]["runs_dir"]) / args.experiment
    fold_dirs = [exp_dir / f"fold_{k}" for k in range(K)]
    missing = [str(d) for d in fold_dirs if not (d / "predictions_test.csv").exists()]
    if missing:
        sys.exit(f"Missing folds: {missing}")

    cfg = yaml.safe_load(open(fold_dirs[0] / "config.yaml"))
    meta = pd.read_csv(Path(cfg["paths"]["data_dir"]) / "metadata.csv", low_memory=False)
    min_rec = cfg["eval"]["min_recordings_within_subject"]
    targets = cfg["data"]["targets"]

    preds, fold_rows, base_preds = [], [], []
    for k, d in enumerate(fold_dirs):
        p = pd.read_csv(d / "predictions_test.csv")
        p["fold"] = k
        preds.append(p)
        m = json.load(open(d / "metrics.json"))
        o = m["test"]["overall"]
        fold_rows.append({"fold": k, "n": o["n"], "subjects": o["subjects"],
                          **{f"{t}_{s}": o[t][s] for t in targets for s in ["MAE", "ME", "SDE", "r", "RMSE"]},
                          "best_epoch": m["info"]["best_epoch"]})
        # training-mean baseline of THIS fold (its own training subjects)
        fcfg = yaml.safe_load(open(d / "config.yaml"))
        tr = assign_roles(meta, fcfg, k)
        tr = tr[tr.role == "train"]
        b = p.copy()
        for t in targets:
            b[f"{t}_pred"] = tr[t].mean()
        base_preds.append(b)

    pred = pd.concat(preds, ignore_index=True)
    if pred["sample_id"].duplicated().any():
        sys.exit("A recording appears in more than one test fold — folds are broken.")
    demo_cols = [c for c in DEMOGRAPHICS if c in meta]
    pred = pred.merge(meta[["sample_id"] + demo_cols], on="sample_id", how="left")
    for c in demo_cols:
        pred[c] = pred[c].fillna("unknown").astype(str)

    out = exp_dir / "cv"
    out.mkdir(exist_ok=True)
    pred.to_csv(out / "cv_predictions_test.csv", index=False)

    pooled = evaluate_predictions(pred, None, group_cols=[g for g in GROUPS if g in pred], min_rec=min_rec)
    pooled["baseline_training_mean"] = regression_metrics(pd.concat(base_preds), targets, min_rec)
    orc = pred.copy()
    for t in targets:
        orc[f"{t}_pred"] = pred.groupby("pid")[t].transform("mean")
    pooled["reference_subject_mean_oracle"] = regression_metrics(orc, targets, min_rec)
    for key, col in [("baseline_calibration_only", "cal_mean"), ("baseline_uncalibrated_shift", "uncal_shift_pred")]:
        if all(f"{t}_{col}" in pred for t in targets):
            b = pred.copy()
            for t in targets:
                b[f"{t}_pred"] = pred[f"{t}_{col}"]
            pooled[key] = regression_metrics(b, targets, min_rec)

    folds = pd.DataFrame(fold_rows)
    num = folds.drop(columns=["fold"])
    per_fold = {"table": folds.to_dict(orient="records"),
                "mean": num.mean().to_dict(), "sd": num.std(ddof=1).to_dict()}
    json.dump({"experiment": args.experiment, "pooled": pooled, "per_fold": per_fold,
               "classification_per_fold": {k: classification_metrics(g) for k, g in pred.groupby("fold")}},
              open(out / "cv_metrics.json", "w"), indent=2, default=float)

    # ---- readable report ----
    L = [f"# {args.experiment} — 5-fold subject-level cross-validation", "",
         f"{len(pred):,} test recordings from {pred.pid.nunique():,} subjects (each predicted exactly once).", "",
         "## Per-fold mean ± SD", "", "| Metric | Mean ± SD |", "|---|---|"]
    for t in targets:
        for s in ["MAE", "ME", "SDE", "r"]:
            L.append(f"| {t.upper()} {s} | {per_fold['mean'][f'{t}_{s}']:.2f} ± {per_fold['sd'][f'{t}_{s}']:.2f} |")
    L += [""] + regression_section(pooled, "Pooled over all folds (overall, baselines, subgroups)")
    L += [""] + classification_section(pooled["classification"])
    (out / "CV_RESULTS.md").write_text("\n".join(L))
    paper_figures(pred, out / "figures", title=args.experiment)
    print("\n".join(L[:40]))

    # ---- CV registry ----
    o, c = pooled["overall"], pooled["classification"]
    line = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "experiment": args.experiment,
            "model": cfg["model"]["name"], "inputs": "+".join(cfg["data"]["inputs"]),
            "subjects": int(pred.pid.nunique()),
            "sbp_MAE": round(o["sbp"]["MAE"], 2),
            "sbp_MAE_fold_sd": round(per_fold["sd"]["sbp_MAE"], 2),
            "dbp_MAE": round(o["dbp"]["MAE"], 2),
            "dbp_MAE_fold_sd": round(per_fold["sd"]["dbp_MAE"], 2),
            "sbp_ME_SD": f"{o['sbp']['ME']:+.1f}±{o['sbp']['SDE']:.1f}",
            "dbp_ME_SD": f"{o['dbp']['ME']:+.1f}±{o['dbp']['SDE']:.1f}",
            "sbp_r": round(o["sbp"]["r"], 3), "sbp_within_subj_r": round(o["sbp"]["within_subject_r_median"], 3),
            "binary_acc": round(c["binary_130_80"]["accuracy"], 3),
            "binary_auroc": round(c["binary_130_80"].get("auroc", float("nan")), 3),
            "3class_acc": round(c["3class"]["accuracy"], 3)}
    reg = Path(base["paths"]["runs_dir"]) / "EXPERIMENTS_CV.csv"
    pd.DataFrame([line]).to_csv(reg, mode="a", header=not reg.exists(), index=False)
    write_md_table(pd.read_csv(reg), reg.with_suffix(".md"))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
