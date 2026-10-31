#!/usr/bin/env python
"""
Additional analyses for the paper, computed from the FROZEN v1 predictions (read-only; nothing is retrained).

  python scripts/paper_analyses.py

  1. Comparison table: overall / in-lab (both protocols) / in-lab auscultatory (subset matched to RAGCBPNet),
     k = 1 and 3, with AAMI, IEEE 1708 and BHS, plus RAGCBPNet and HGCTNet rows copied from the RAGCBPNet preprint.
  2. Significance: subject-level paired tests of absolute error (Wilcoxon signed-rank on per-subject MAE) and
     subject-bootstrap 95% CI of the MAE difference: model vs calibration-only, model vs uncalibrated+shift,
     k=3 vs k=1.
  3. Hypertension detection at 140/90 (SBP, DBP, either), both '>' (RAGCBPNet definition) and '>=' (guideline),
     on all test recordings and on the matched subset.
  4. Error vs time since calibration (k = 1): MAE of the model and of 'calibration value only' per time bin.

Output: paper/  tables/*.md, tables/*.csv, figures/*.png, PAPER_ANALYSES.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import regression_metrics  # noqa: E402

T = ["sbp", "dbp"]
RAGCBPNET_ROWS = [  # copied from the RAGCBPNet preprint (medRxiv 2025.10.30.25339198), Table 2 / Table 7
    ("RAGCBPNet (preprint)", "in-lab, auscultatory: 9,433 rec / 626 subj", 7.08, 0.96, 10.29, 4.90, -0.46, 6.78),
    ("HGCTNet, retrained by RAGCBPNet authors (preprint)", "same subset", 7.61, 1.24, 11.52, 5.21, -0.90, 7.43),
]


def load(run, k):
    f = "predictions_test.csv" if k == 1 else f"predictions_test_k{k}.csv"
    return pd.concat([pd.read_csv(Path(run) / f"fold_{i}" / f).assign(fold=i) for i in range(5)], ignore_index=True)


def ieee(mae):
    return "A" if mae <= 5 else "B" if mae <= 6 else "C" if mae <= 7 else "D"


def aami(me, sd):
    return "pass" if abs(me) <= 5 and sd <= 8 else "fail"


# ------------------------------------------------------------------ 1. comparison table
def comparison_table(run):
    L = ["| Method | Test data | SBP MAE | SBP ME ± SD | SBP AAMI | SBP IEEE | SBP BHS | DBP MAE | DBP ME ± SD | DBP AAMI | DBP IEEE | DBP BHS |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for k in (1, 3):
        p = load(run, k)
        for name, q in [("overall", p), ("in-lab, both protocols", p[p.phase != "ambulatory"]),
                        ("in-lab, auscultatory (matched to RAGCBPNet)", p[p.protocol == "auscultatory"])]:
            m = regression_metrics(q, T, 5)
            s, d = m["sbp"], m["dbp"]
            data = f"{name}: {len(q):,} rec / {q.pid.nunique()} subj"
            L.append(f"| **HiCAF-BP-Cal, k = {k}** | {data} | {s['MAE']:.2f} | {s['ME']:+.2f} ± {s['SDE']:.2f} | "
                     f"{aami(s['ME'], s['SDE'])} | {s['IEEE1708_grade']} | {s['BHS_grade']} | {d['MAE']:.2f} | "
                     f"{d['ME']:+.2f} ± {d['SDE']:.2f} | {aami(d['ME'], d['SDE'])} | {d['IEEE1708_grade']} | {d['BHS_grade']} |")
            rows.append({"method": f"HiCAF-BP-Cal k={k}", "data": name, "n_rec": len(q), "n_subj": q.pid.nunique(),
                         **{f"{t}_{x}": m[t][x] for t in T for x in ["MAE", "ME", "SDE"]}})
    for name, data, sm, sme, ssd, dm, dme, dsd in RAGCBPNET_ROWS:
        L.append(f"| {name} | {data} | {sm:.2f} | {sme:+.2f} ± {ssd:.2f} | {aami(sme, ssd)} | {ieee(sm)} | – | "
                 f"{dm:.2f} | {dme:+.2f} ± {dsd:.2f} | {aami(dme, dsd)} | {ieee(dm)} | – |")
        rows.append({"method": name, "data": data, "sbp_MAE": sm, "sbp_ME": sme, "sbp_SDE": ssd,
                     "dbp_MAE": dm, "dbp_ME": dme, "dbp_SDE": dsd})
    L += ["", "RAGCBPNet / HGCTNet values are copied from the RAGCBPNet preprint (not peer-reviewed); BHS cannot be "
          "computed from the reported numbers. Subsets are matched by protocol but not identical (quality thresholds "
          "differ; calibration readings are excluded from our test set)."]
    return L, pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. significance
def paired_subject_test(p, a_cols, b_cols, n_boot=2000, seed=0):
    """a/b: dict target -> prediction column. Per-subject MAE, Wilcoxon signed-rank, bootstrap CI of MAE difference."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    out = {}
    for t in T:
        ea = (p[a_cols[t]] - p[t]).abs()
        eb = (p[b_cols[t]] - p[t]).abs()
        g = pd.DataFrame({"a": ea, "b": eb, "pid": p["pid"]}).groupby("pid")
        per = g.mean()
        sums, cnt = g.sum(), g.size().to_numpy()
        stat = wilcoxon(per["a"], per["b"])
        diffs = []
        sa, sb = sums["a"].to_numpy(), sums["b"].to_numpy()
        for _ in range(n_boot):
            i = rng.integers(0, len(cnt), len(cnt))
            diffs.append((sa[i].sum() - sb[i].sum()) / cnt[i].sum())
        out[t] = {"mae_a": float(ea.mean()), "mae_b": float(eb.mean()), "diff": float(ea.mean() - eb.mean()),
                  "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
                  "n_subjects": int(len(per)), "subjects_better": float((per["a"] < per["b"]).mean()),
                  "p_wilcoxon": float(stat.pvalue)}
    return out


def significance(run):
    p1, p3 = load(run, 1), load(run, 3)
    tests = []
    for k, p in [(1, p1), (3, p3)]:
        model = {t: f"{t}_pred" for t in T}
        if all(f"{t}_cal_mean" in p for t in T):
            tests.append((f"Model vs calibration value only (k={k})", p, model, {t: f"{t}_cal_mean" for t in T}))
        if all(f"{t}_uncal_shift_pred" in p for t in T):
            tests.append((f"Model vs uncalibrated model + shift (k={k})", p, model, {t: f"{t}_uncal_shift_pred" for t in T}))
    m = p1[["sample_id", "pid"] + T + [f"{t}_pred" for t in T]].merge(
        p3[["sample_id"] + [f"{t}_pred" for t in T]], on="sample_id", suffixes=("_k1", "_k3"))
    tests.append(("Model k=3 vs model k=1", m, {t: f"{t}_pred_k3" for t in T}, {t: f"{t}_pred_k1" for t in T}))

    L = ["| Comparison | Target | MAE (A) | MAE (B) | Difference A − B [95% CI] | Subjects where A better | p (Wilcoxon, per-subject MAE) |",
         "|---|---|---|---|---|---|---|"]
    rows = []
    for name, p, a, b in tests:
        r = paired_subject_test(p, a, b)
        for t in T:
            x = r[t]
            pv = "< 0.001" if x["p_wilcoxon"] < 0.001 else f"{x['p_wilcoxon']:.3f}"
            L.append(f"| {name} | {t.upper()} | {x['mae_a']:.2f} | {x['mae_b']:.2f} | {x['diff']:+.2f} "
                     f"[{x['ci_low']:+.2f}, {x['ci_high']:+.2f}] | {x['subjects_better']:.0%} ({x['n_subjects']}) | {pv} |")
            rows.append({"comparison": name, "target": t, **x})
    L += ["", "A = first method in the comparison. Tests are paired and at the SUBJECT level (one value per subject), "
          "because recordings of the same person are not independent. CI: subject-level bootstrap (2,000 resamples)."]
    return L, pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. hypertension 140/90
def auroc(y, s):
    y = np.asarray(y, bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def auprc(y, s):
    y = np.asarray(y, bool)
    o = np.argsort(-np.asarray(s, float), kind="stable")
    yt = y[o]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / max(yt.sum(), 1))


def binary_metrics(y, yhat, score):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp, tn = (y & yhat).sum(), (~y & ~yhat).sum()
    fp, fn = (~y & yhat).sum(), (y & ~yhat).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"prevalence": y.mean(), "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": 2 * prec * sens / max(prec + sens, 1e-12), "balanced_accuracy": (sens + spec) / 2,
            "auroc": auroc(y, score), "auprc": auprc(y, score), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def hypertension(run):
    L = ["| k | Test data | Task | Definition | Prevalence | Accuracy | Sensitivity | Specificity | Precision | F1 | AUROC | AUPRC |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for k in (1, 3):
        p = load(run, k)
        for data, q in [("all test recordings", p), ("in-lab auscultatory (matched)", p[p.protocol == "auscultatory"])]:
            for strict in (True, False):
                op = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
                sym = ">" if strict else "≥"
                tasks = {f"systolic (SBP {sym} 140)": (op(q.sbp, 140), op(q.sbp_pred, 140), q.sbp_pred),
                         f"diastolic (DBP {sym} 90)": (op(q.dbp, 90), op(q.dbp_pred, 90), q.dbp_pred),
                         f"either (SBP {sym} 140 or DBP {sym} 90)": (op(q.sbp, 140) | op(q.dbp, 90),
                                                                    op(q.sbp_pred, 140) | op(q.dbp_pred, 90),
                                                                    np.maximum(q.sbp_pred - 140, q.dbp_pred - 90))}
                for task, (y, yh, sc) in tasks.items():
                    m = binary_metrics(y, yh, sc)
                    L.append(f"| {k} | {data} | {task} | {'RAGCBPNet' if strict else 'guideline'} | {m['prevalence']:.1%} | "
                             f"{m['accuracy']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['precision']:.3f} | "
                             f"{m['f1']:.3f} | {m['auroc']:.3f} | {m['auprc']:.3f} |")
                    rows.append({"k": k, "data": data, "task": task, "definition": sym, **m})
    L += ["", "Hypertension is detected by thresholding the PREDICTED SBP/DBP (no separate classifier). AUROC/AUPRC use the "
          "predicted value as the score. RAGCBPNet (preprint) reports for its in-lab auscultatory subset: systolic accuracy "
          "0.8952, AUROC 0.8627; diastolic accuracy 0.9442, AUROC 0.8301 (hypertension defined as SBP > 140 or DBP > 90; "
          "a dedicated classifier with contrastive learning)."]
    return L, pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. error vs time since calibration
def time_since_calibration(run, meta_path, cfg_phase="initial", cfg_conditions=("calibration", "sitting", "supine")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = load(run, 1)
    meta = pd.read_csv(meta_path, low_memory=False)
    if "include_main" in meta:
        meta = meta[meta["include_main"].astype(bool)]
    meta = meta[meta.pid.isin(p.pid.unique())].copy()
    meta["t"] = pd.to_datetime(meta["date_time"], errors="coerce")
    cand = meta[(meta.phase == cfg_phase) & meta.condition.isin(list(cfg_conditions))].sort_values("sample_id")
    cal_t = cand.groupby("pid").head(1).set_index("pid")["t"]              # k=1 calibration = first candidate
    q = p.merge(meta[["sample_id", "t"]], on="sample_id", how="left")
    q["hours"] = (q["t"] - q.pid.map(cal_t)).dt.total_seconds() / 3600
    q = q[q["hours"].notna()]
    bins = [-np.inf, 1, 3, 6, 12, 24, 24 * 7, np.inf]
    labels = ["< 1 h", "1–3 h", "3–6 h", "6–12 h", "12–24 h", "1–7 days", "> 7 days"]
    q["bin"] = pd.cut(q["hours"], bins=bins, labels=labels, right=False)
    rows = []
    for b, g in q.groupby("bin", observed=True):
        r = {"time since calibration": b, "n": len(g), "subjects": g.pid.nunique()}
        for t in T:
            r[f"{t}_MAE_model"] = (g[f"{t}_pred"] - g[t]).abs().mean()
            if f"{t}_cal_mean" in g:
                r[f"{t}_MAE_cal_only"] = (g[f"{t}_cal_mean"] - g[t]).abs().mean()
        rows.append(r)
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(d))
    for i, t in enumerate(T):
        ax[i].plot(x, d[f"{t}_MAE_model"], "o-", label="HiCAF-BP-Cal")
        if f"{t}_MAE_cal_only" in d:
            ax[i].plot(x, d[f"{t}_MAE_cal_only"], "s--", color="grey", label="Calibration value only")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(d["time since calibration"], rotation=30)
        ax[i].set_ylabel(f"{t.upper()} MAE (mmHg)")
        ax[i].set_xlabel("Time since calibration")
        ax[i].grid(alpha=0.3)
        ax[i].legend()
        for xi, n in zip(x, d["n"]):
            ax[i].annotate(f"n={n:,}", (xi, d[f"{t}_MAE_model"].iloc[xi]), textcoords="offset points",
                           xytext=(0, 8), ha="center", fontsize=7)
    plt.tight_layout()
    L = ["| Time since calibration | Recordings | Subjects | SBP MAE (model) | SBP MAE (cal. only) | DBP MAE (model) | DBP MAE (cal. only) |",
         "|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        L.append(f"| {r['time since calibration']} | {r['n']:,} | {r['subjects']:,} | {r['sbp_MAE_model']:.2f} | "
                 f"{r.get('sbp_MAE_cal_only', np.nan):.2f} | {r['dbp_MAE_model']:.2f} | {r.get('dbp_MAE_cal_only', np.nan):.2f} |")
    L += ["", f"k = 1; calibration = first resting initial-visit reading. {len(q):,} of {len(p):,} test recordings had a "
          "parsable time stamp and a calibration time. '< 1 h' includes readings taken shortly before the calibration "
          "reading during the same visit."]
    return L, d, fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hicaf_bp_cal_t_scratch")
    ap.add_argument("--meta", default="data_processed/v1/metadata.csv")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    doc = ["# Paper analyses (from frozen v1 predictions: " + args.run + ")", ""]
    for title, fn, name in [("1. Comparison with prior work (overall and matched subsets)", comparison_table, "comparison"),
                            ("2. Statistical significance (subject-level, paired)", significance, "significance"),
                            ("3. Hypertension detection at 140/90", hypertension, "hypertension_140_90")]:
        print(f"... {title}", flush=True)
        L, df = fn(args.run)
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
        (out / "tables" / f"{name}.md").write_text("\n".join(L))
        doc += [f"## {title}", ""] + L + [""]
    print("... 4. Error vs time since calibration", flush=True)
    L, df, fig = time_since_calibration(args.run, args.meta)
    df.to_csv(out / "tables" / "error_vs_time_since_calibration.csv", index=False)
    (out / "tables" / "error_vs_time_since_calibration.md").write_text("\n".join(L))
    fig.savefig(out / "figures" / "error_vs_time_since_calibration.png", dpi=200)
    doc += ["## 4. Error vs time since calibration", ""] + L + ["", "![](figures/error_vs_time_since_calibration.png)"]
    (out / "PAPER_ANALYSES.md").write_text("\n".join(doc))
    print("\n".join(doc))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
