#!/usr/bin/env python
"""
Calibrated-group comparison report (plan: docs/REIMPLEMENTATION_PLAN.md, R8 / R11). Read-only on all run folders.

  python scripts/report_calibrated.py        # rerun after each new calibrated method (RAGCBPNet is picked up automatically)

Aurora-BP (5-fold CV, pooled; the same test recordings for every method, checked), k = 1 and k = 3:
  our model v1 (with time context), our no-time model, HGCTNet, BpBLS, RAGCBPNet (when available);
  reference: calibration value only. Paired subject-level Wilcoxon tests against v1. Also split by setting:
  in-lab (initial and return visits) and ambulatory (home) readings.
Zero-shot PulseDB (ensemble of the 5 fold models; identical target segments, checked), k = 1 and k = 3, per source:
  our no-time model (v1 needs time of day, unavailable in PulseDB), HGCTNet, BpBLS, RAGCBPNet; paired tests vs ours.
Output: runs/report_calibrated/CALIBRATED_COMPARISON.md, metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
T = ["sbp", "dbp"]
AURORA = [("hicaf_bp_cal_t_scratch", "HiCAF-BP-Cal + time context"), ("hicaf_bp_cal_notime", "HiCAF-BP-Cal (ours)"),
          ("pub_hgctnet", "HGCTNet"), ("pub_bpbls", "BpBLS"), ("pub_ragcbpnet", "RAGCBPNet")]
ZERO = [("hicaf_bp_cal_notime", "HiCAF-BP-Cal (ours)"), ("pub_hgctnet", "HGCTNet"), ("pub_bpbls", "BpBLS"),
        ("pub_ragcbpnet", "RAGCBPNet")]


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(rows, ref, regression_metrics, wilcoxon, metrics, key):
    """rows: [(label, frame with pid, sbp, dbp, sbp_pred, dbp_pred, fold-MAE array or None)]; ref: label of the paired reference."""
    L = ["| Method | SBP MAE | SBP ME ± SD | SBP r | DBP MAE | DBP ME ± SD | DBP r | subj. MAE SBP / DBP | "
         f"vs {ref}: Δ subj. MAE SBP / DBP (% better), p |", "|---|---|---|---|---|---|---|---|---|"]
    per = {lab: {t: (q[f"{t}_pred"] - q[t]).abs().groupby(q.pid).mean() for t in T} for lab, q, _ in rows}
    for lab, q, fm in rows:
        m = regression_metrics(q, T, 5)
        metrics[f"{key}/{lab}"] = m
        vs = "–"
        if lab != ref and ref in per and not lab.startswith("Calibration"):
            cells = []
            for t in T:
                a, b = per[lab][t], per[ref][t]
                cells.append(f"{a.mean() - b.mean():+.2f} ({(a < b).mean():.0%}), p {fmt_p(wilcoxon(a, b).pvalue)}")
            vs = "; ".join(cells)
        spread = ""
        if fm is not None and len(fm) > 1:
            spread = f" (5 models {fm[:, 0].mean():.2f} ± {fm[:, 0].std(ddof=1):.2f} / {fm[:, 1].mean():.2f} ± {fm[:, 1].std(ddof=1):.2f})"
        L.append(f"| {lab}{spread} | {m['sbp']['MAE']:.2f} | {m['sbp']['ME']:+.2f} ± {m['sbp']['SDE']:.2f} | {m['sbp']['r']:.2f} | "
                 f"{m['dbp']['MAE']:.2f} | {m['dbp']['ME']:+.2f} ± {m['dbp']['SDE']:.2f} | {m['dbp']['r']:.2f} | "
                 f"{per[lab]['sbp'].mean():.2f} / {per[lab]['dbp'].mean():.2f} | {vs} |")
    return L


def main():
    from scipy.stats import wilcoxon

    from src.evaluation.metrics import regression_metrics

    runs = CODE_DIR / "runs"
    out = runs / "report_calibrated"
    out.mkdir(exist_ok=True)
    metrics = {}
    L = ["# Calibrated methods — comparison (Aurora-BP 5-fold CV and zero-shot PulseDB)", "",
         "Every method receives the same calibration readings (k = first 1 or 3 calibration recordings / segments) and is "
         "scored on the same recordings (checked). Paired tests: Wilcoxon signed-rank on per-subject MAE.", ""]

    # ------------------------------------------------------------------ Aurora
    for k in (1, 3):
        name = "predictions_test.csv" if k == 1 else f"predictions_test_k{k}.csv"
        frames = {}
        for run, lab in AURORA:
            if all((runs / run / f"fold_{i}" / name).exists() for i in range(5)):
                frames[lab] = pd.concat([pd.read_csv(runs / run / f"fold_{i}" / name) for i in range(5)]).set_index("sample_id").sort_index()
        ref = frames["HiCAF-BP-Cal (ours)"]
        ids = ref.index
        for lab, d in frames.items():
            if not np.array_equal(d.index.to_numpy(), ids.to_numpy()):
                sys.exit(f"Aurora k={k}: {lab} was not scored on the same test recordings - stop")
        cal_src = next((d for lab, d in frames.items() if "sbp_cal_mean" in d), None)
        L += [f"## Aurora-BP, k = {k} ({len(ids):,} test recordings, {ref.pid.nunique():,} subjects)", ""]
        subsets = [("all readings", ids)]
        if "phase" in ref:
            subsets += [("in-lab (initial + return visits)", ids[ref.phase.isin(["initial", "return"]).to_numpy()]),
                        ("ambulatory (home)", ids[(ref.phase == "ambulatory").to_numpy()])]
        for sname, sid in subsets:
            if len(sid) == 0:
                continue
            rows = [(lab, d.loc[sid], None) for lab, d in frames.items()]
            if cal_src is not None:
                c = cal_src.loc[sid, ["pid", *T]].copy()
                for t in T:
                    c[f"{t}_pred"] = cal_src.loc[sid, f"{t}_cal_mean"]
                rows.append(("Calibration value only (reference)", c, None))
            L += [f"### {sname} ({len(sid):,} recordings)", ""]
            L += table(rows, "HiCAF-BP-Cal (ours)", regression_metrics, wilcoxon, metrics, f"aurora/k{k}/{sname}")
            L.append("")
        L.append(f"Methods included: {', '.join(frames)}.")
        L.append("")

    # ------------------------------------------------------------------ zero-shot PulseDB
    for k in (1, 3):
        ens = {}
        for run, lab in ZERO:
            d = runs / run / "external_pulsedb"
            if (d / f"predictions_ensemble_k{k}.csv").exists():
                e = pd.read_csv(d / f"predictions_ensemble_k{k}.csv")
                folds = [np.load(d / f"predictions_fold{f}_k{k}.npz")["pred"] for f in range(5) if (d / f"predictions_fold{f}_k{k}.npz").exists()]
                ens[lab] = (e, folds)
        ref_lab = "HiCAF-BP-Cal (ours)"
        ref = ens[ref_lab][0]
        for lab, (e, _) in ens.items():
            if not np.array_equal(e.sample_id.to_numpy(), ref.sample_id.to_numpy()):
                sys.exit(f"zero-shot k={k}: {lab} target segments differ - stop")
        L += [f"## Zero-shot PulseDB, k = {k} ({len(ref):,} target segments, {ref.pid.nunique():,} subjects)", ""]
        for sname, ix in [("combined", ref.index)] + [(s, ref.index[ref.protocol == s]) for s in sorted(ref.protocol.unique())]:
            rows = []
            for lab, (e, folds) in ens.items():
                fm = np.array([[np.abs(pf[ix.to_numpy(), j] - e.loc[ix, t].to_numpy()).mean() for j, t in enumerate(T)] for pf in folds]) if folds else None
                rows.append((lab, e.loc[ix], fm))
            c = ref.loc[ix, ["pid", *T]].copy()
            for t in T:
                c[f"{t}_pred"] = ref.loc[ix, f"{t}_cal_mean"]
            rows.append(("Calibration value only (reference)", c, None))
            L += [f"### {sname}", ""]
            L += table(rows, ref_lab, regression_metrics, wilcoxon, metrics, f"zeroshot/k{k}/{sname}")
            L.append("")
        L.append(f"Methods included: {', '.join(ens)}.")
        L.append("")
    (out / "CALIBRATED_COMPARISON.md").write_text("\n".join(L))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\nSaved {out}/CALIBRATED_COMPARISON.md")


if __name__ == "__main__":
    main()
