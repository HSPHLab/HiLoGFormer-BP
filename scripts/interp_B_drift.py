#!/usr/bin/env python
"""
Interpretability B - "When does the patient need a new cuff calibration?" (docs/INTERPRETABILITY_PLAN.md, section 4).

  python scripts/interp_B_drift.py

Drift score D = ||z_t - z_c|| of the shared encoder's fused embedding (PPG-only embedding as secondary), mean over the
calibration readings used for the estimate. Aurora-BP (no-time model, test subjects; thresholds and the logistic model from each
fold's validation subjects): deciles of D vs time since calibration; AUROC for an unacceptable reading (|error| > 10 mmHg)
of D, time and both; recalibration rules (D > T; time rule at the same flag rate; D > T or time > N) with sensitivity,
specificity, PPV and the MAE of non-flagged readings. PulseDB (zero-shot, ensemble; threshold-free): Spearman, AUROC, deciles.
Outputs: paper/tables/interpretability/B_drift.md (+ .json), paper/figures/interp_B_drift.pdf (Fig. I2)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

CODE_DIR = Path(__file__).resolve().parents[1]
MODEL_RUN = "runs/hicaf_bp_cal_notime"        # interpretability model (plan v1.2): the no-time model, Aurora and PulseDB
CACHE = "runs/interpretability/pairs_notime.csv.gz"
sys.path.insert(0, str(CODE_DIR))
from src.interp.stats import auc_ci, spearman_ci  # noqa: E402

T = ["sbp", "dbp"]
UNACC = 10.0


def aurora_readings():
    c = pd.read_csv(CODE_DIR / CACHE)
    g = c.groupby(["fold", "split", "sample_id"])
    r = g.agg(pid=("pid", "first"), n_cal=("cal_rank", "size"), hours=("hours_since_cal", "mean"),
              d_fused=("drift_fused", "mean"), d_ppg=("drift_ppg", "mean"), sbp=("sbp", "first"), dbp=("dbp", "first"),
              sbp_est=("sbp_pred_full", "mean"), dbp_est=("dbp_pred_full", "mean")).reset_index()
    for t in T:
        r[f"{t}_abs"] = (r[f"{t}_est"] - r[t]).abs()
        r[f"{t}_bad"] = (r[f"{t}_abs"] > UNACC).astype(int)
    r["hours"] = r["hours"].fillna(0.0)
    ph = []
    for f in sorted(r.fold.unique()):
        for s in ("val", "test"):
            p = CODE_DIR / MODEL_RUN / f"fold_{f}" / f"predictions_{s}_k3.csv"
            if p.exists():
                ph.append(pd.read_csv(p, usecols=["sample_id", "phase"]).assign(fold=f, split=s))
    r = r.merge(pd.concat(ph), on=["fold", "split", "sample_id"], how="left")
    r["setting"] = np.where(r.phase == "ambulatory", "home", "lab")
    return r


def rule_metrics(flag, bad, absr):
    tp, fp = int((flag & bad).sum()), int((flag & ~bad).sum())
    fn, tn = int((~flag & bad).sum()), int((~flag & ~bad).sum())
    return {"flag_rate": flag.mean(), "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "mae_not_flagged": absr[~flag].mean(), "mae_flagged": absr[flag].mean() if flag.any() else np.nan}


def main():
    from src.viz.style import DATASET, METHOD, NEUTRAL, TARGET, figure, panel_labels, save, setup
    r = aurora_readings()
    val, test = r[r.split == "val"], r[r.split == "test"].copy()
    out = {"n_test": len(test), "n_subjects": int(test.pid.nunique())}
    L = ["# Interpretability B — \"When does the patient need a new cuff calibration?\"", "",
         "Drift D = ||z_t − z_c|| (fused embedding; mean over the calibration readings of the estimate). Aurora-BP: no-time model, test "
         "subjects; thresholds (80th percentiles) and the logistic model (D, log time) from each fold's validation subjects. "
         f"Unacceptable reading: |error| > {UNACC:.0f} mmHg. 95 % CI: subject-level bootstrap (1000).", "",
         f"## Aurora-BP ({len(test):,} test readings, {test.pid.nunique():,} subjects)", ""]
    # logistic model per fold (validation) -> test probabilities
    roc = {}
    for t in T:
        p_both = np.zeros(len(test))
        for f in sorted(test.fold.unique()):
            v, te = val[val.fold == f], test.fold.to_numpy() == f
            X = lambda d: np.column_stack([d.d_fused, np.log1p(d.hours.clip(lower=0))])
            mu, sd = X(v).mean(0), X(v).std(0) + 1e-9
            lr = LogisticRegression().fit((X(v) - mu) / sd, v[f"{t}_bad"])
            p_both[te] = lr.predict_proba((X(test[te]) - mu) / sd)[:, 1]
        test[f"{t}_p_both"] = p_both
        bad = test[f"{t}_bad"].to_numpy().astype(bool)
        L += [f"### {t.upper()} (MAE {test[f'{t}_abs'].mean():.2f}; unacceptable {bad.mean():.1%})", "",
              "| Score | Spearman ρ with |error| (95 % CI) | AUROC for unacceptable (95 % CI) | AUROC in-lab | AUROC home |", "|---|---|---|---|---|"]
        for lab, col in [("drift D (fused)", "d_fused"), ("drift (PPG-only, secondary)", "d_ppg"),
                         ("time since calibration", "hours"), ("D + time (logistic, validation-fitted)", f"{t}_p_both")]:
            rho, rci = spearman_ci(test[col], test[f"{t}_abs"], test.pid)
            auc, aci = auc_ci(bad, test[col], test.pid)
            sub = []
            for st in ["lab", "home"]:
                m = (test.setting == st).to_numpy()
                sub.append(auc_ci(bad[m], test[col].to_numpy()[m], test.pid.to_numpy()[m], n=200)[0])
            out[f"{t}/{lab}"] = {"spearman": rho, "spearman_ci": rci.tolist(), "auroc": auc, "auroc_ci": aci.tolist(),
                                 "auroc_lab": sub[0], "auroc_home": sub[1]}
            L.append(f"| {lab} | {rho:.3f} ({rci[0]:.3f}–{rci[1]:.3f}) | {auc:.3f} ({aci[0]:.3f}–{aci[1]:.3f}) | {sub[0]:.3f} | {sub[1]:.3f} |")
            if col in ("d_fused", "hours", f"{t}_p_both"):
                roc[(t, lab)] = roc_curve(bad, test[col])
        # recalibration rules
        T_D = test.fold.map(val.groupby("fold").d_fused.quantile(0.8)).to_numpy()
        N_h = test.fold.map(val.groupby("fold").hours.quantile(0.8)).to_numpy()
        flag_d = test.d_fused.to_numpy() > T_D
        flag_t = np.zeros(len(test), bool)
        flag_t[np.argsort(-test.hours.to_numpy(), kind="stable")[:int(flag_d.sum())]] = True
        flag_c = flag_d | (test.hours.to_numpy() > N_h)
        absr = test[f"{t}_abs"].to_numpy()
        L += ["", "| Rule (thresholds from validation) | Flag rate | Sensitivity | Specificity | PPV | MAE not flagged | MAE flagged |",
              "|---|---|---|---|---|---|---|"]
        for lab, fl in [("recalibrate if D > T", flag_d), ("time rule, same flag rate", flag_t),
                        ("recalibrate if D > T or time > N", flag_c)]:
            m = rule_metrics(fl, bad, absr)
            out[f"{t}/rule/{lab}"] = m
            L.append(f"| {lab} | {m['flag_rate']:.1%} | {m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['ppv']:.1%} | "
                     f"{m['mae_not_flagged']:.2f} | {m['mae_flagged']:.2f} |")
        L.append(f"| (no rule) | 0 % | | | {bad.mean():.1%} | {absr.mean():.2f} | |")
        L.append("")
    # deciles (descriptive)
    dec = {}
    for col in ["d_fused", "hours"]:
        q = pd.qcut(test[col].rank(method="first"), 10, labels=False)
        dec[col] = test.groupby(q).agg(mae_sbp=("sbp_abs", "mean"), mae_dbp=("dbp_abs", "mean"), bad_sbp=("sbp_bad", "mean"),
                                       median=(col, "median"))
    L += ["### By decile (test, descriptive): MAE SBP / DBP and unacceptable SBP", "",
          "| Decile | D: median | D: MAE SBP / DBP | D: unacceptable SBP | time: median (h) | time: MAE SBP / DBP | time: unacceptable SBP |",
          "|---|---|---|---|---|---|---|"]
    for i in range(10):
        a, b = dec["d_fused"].iloc[i], dec["hours"].iloc[i]
        L.append(f"| {i + 1} | {a['median']:.3g} | {a.mae_sbp:.2f} / {a.mae_dbp:.2f} | {a.bad_sbp:.1%} | {b['median']:.1f} | "
                 f"{b.mae_sbp:.2f} / {b.mae_dbp:.2f} | {b.bad_sbp:.1%} |")
    L.append("")

    # PulseDB (threshold-free)
    z = np.load(CODE_DIR / "runs/interpretability/pulsedb_v1notime.npz")
    d_ens = z["drift_fused"].mean(axis=0)
    hours = np.nan_to_num(z["hours_since_cal"], nan=0.0)
    L += ["## PulseDB (zero-shot, no-time model; D = mean over the 5 models; threshold-free)", "",
          "| Data | Target | Spearman ρ(D, |error|) (95 % CI) | AUROC D (95 % CI) | AUROC time | MAE in lowest / highest D decile |", "|---|---|---|---|---|---|"]
    pdec = {}
    for sname in ["combined", "MIMIC", "VitalDB"]:
        m = np.ones(len(d_ens), bool) if sname == "combined" else (z["protocol"] == sname)
        for j, t in enumerate(T):
            a = np.abs(z["ens_est"][m, j] - z[t][m])
            bad = a > UNACC
            rho, rci = spearman_ci(d_ens[m], a, z["pid"][m], n=200)
            auc, aci = auc_ci(bad, d_ens[m], z["pid"][m], n=200)
            auc_t = auc_ci(bad, hours[m], z["pid"][m], n=50)[0]
            q = pd.qcut(pd.Series(d_ens[m]).rank(method="first"), 10, labels=False).to_numpy()
            dm = pd.Series(a).groupby(q).mean()
            pdec[(sname, t)] = dm.to_numpy()
            out[f"pulsedb/{sname}/{t}"] = {"spearman": rho, "spearman_ci": rci.tolist(), "auroc": auc, "auroc_ci": aci.tolist(),
                                           "auroc_time": auc_t, "decile_mae": dm.tolist()}
            L.append(f"| {sname} | {t.upper()} | {rho:.3f} ({rci[0]:.3f}–{rci[1]:.3f}) | {auc:.3f} ({aci[0]:.3f}–{aci[1]:.3f}) | "
                     f"{auc_t:.3f} | {dm.iloc[0]:.2f} / {dm.iloc[-1]:.2f} |")
    tab = CODE_DIR / "paper/tables/interpretability"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "B_drift.md").write_text("\n".join(L))
    json.dump(out, open(tab / "B_drift.json", "w"), indent=2, default=float)
    print("\n".join(L))

    # Figure I2
    setup()
    fig, axes = figure("double", height=2.4, ncols=3)
    x = np.arange(1, 11)
    ax = axes[0]
    for col, ls, lab in [("d_fused", "-", "drift score D"), ("hours", ":", "time since calibration")]:
        for t in T:
            ax.plot(x, dec[col][f"mae_{t}"], color=TARGET[t]["color"], ls=ls, marker="o" if col == "d_fused" else "s",
                    label=f"{TARGET[t]['label']}: {lab}")
    ax.set(xticks=x, xlabel="Decile (1 = lowest)", ylabel="MAE (mmHg)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
    ax = axes[1]
    for (t, lab), (fpr, tpr, _) in roc.items():
        if t != "sbp":
            continue
        sty = {"drift D (fused)": ("-", METHOD["ours"]["color"]), "time since calibration": (":", NEUTRAL["line"]),
               "D + time (logistic, validation-fitted)": ("--", TARGET["sbp"]["color"])}[lab]
        short = {"drift D (fused)": "drift score D", "time since calibration": "time since calibration",
                 "D + time (logistic, validation-fitted)": "D + time"}[lab]
        ax.plot(fpr, tpr, ls=sty[0], color=sty[1], label=f"{short} (AUC {out[f'sbp/{lab}']['auroc']:.2f})")
    ax.plot([0, 1], [0, 1], color="#B0B0B0", lw=0.6)
    ax.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6, title="Unacceptable SBP reading:", title_fontsize=6)
    ax = axes[2]
    for sname, key in [("MIMIC", "mimic"), ("VitalDB", "vitaldb")]:
        ax.plot(x, pdec[(sname, "sbp")], color=DATASET[key]["color"], marker="o", label=f"{DATASET[key]['label']} SBP")
        ax.plot(x, pdec[(sname, "dbp")], color=DATASET[key]["color"], marker="s", ls="--", label=f"{DATASET[key]['label']} DBP")
    ax.set(xticks=x, xlabel="Decile of drift score D (1 = lowest)", ylabel="MAE (mmHg)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
    panel_labels(axes)
    save(fig, "interp_B_drift", source=__file__)


if __name__ == "__main__":
    main()
