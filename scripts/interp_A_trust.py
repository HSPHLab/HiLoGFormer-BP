#!/usr/bin/env python
"""
Interpretability A - "Can I trust this reading?" (docs/INTERPRETABILITY_PLAN.md, section 3). Aurora-BP part.

  python scripts/interp_A_trust.py

Input: runs/interpretability/pairs_notime.csv.gz (frozen cache of the no-time model; verified against its frozen predictions).
Score A1 (calibration consistency, CC): SD of the three single-calibration estimates of a reading (per target SBP / DBP);
the reported estimate is their mean (= the frozen k = 3 estimate). Readings with fewer than 3 calibration readings have no
CC and are excluded from A (count reported).
Thresholds: from each fold's VALIDATION subjects, applied to the same fold's TEST subjects (never chosen on test).
Outputs: paper/tables/interpretability/A_trust.md (+ A_trust.json), paper/figures/interp_A_trust.pdf (Fig. I1, Aurora panels)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parents[1]
MODEL_RUN = "runs/hicaf_bp_cal_notime"        # interpretability model (plan v1.2): the no-time model, Aurora and PulseDB
CACHE = "runs/interpretability/pairs_notime.csv.gz"
sys.path.insert(0, str(CODE_DIR))
T = ["sbp", "dbp"]
UNACC = 10.0                    # unacceptable reading: |error| > 10 mmHg (plan, section 2)
COVERAGES = [0.9, 0.8, 0.7]
N_BOOT, SEED = 1000, 2026


def per_reading(cache, run_dir):
    """One row per target reading: k = 3 estimate (mean of the 3), CC (SD of the 3), error, setting."""
    g = cache.groupby(["fold", "split", "sample_id"])
    r = g.agg(pid=("pid", "first"), n_cal=("cal_rank", "size"), hours=("hours_since_cal", "mean"),
              sbp=("sbp", "first"), dbp=("dbp", "first"),
              sbp_est=("sbp_pred_full", "mean"), dbp_est=("dbp_pred_full", "mean"),
              sbp_cc=("sbp_pred_full", "std"), dbp_cc=("dbp_pred_full", "std")).reset_index()
    for t in T:
        r[f"{t}_err"] = r[f"{t}_est"] - r[t]
        r[f"{t}_abs"] = r[f"{t}_err"].abs()
    phase = []                                                       # setting from the frozen prediction files
    for f in sorted(r.fold.unique()):
        for s in ("val", "test"):
            p = run_dir / f"fold_{f}" / f"predictions_{s}_k3.csv"
            if p.exists():
                phase.append(pd.read_csv(p, usecols=["sample_id", "phase"]).assign(fold=f, split=s))
    r = r.merge(pd.concat(phase), on=["fold", "split", "sample_id"], how="left")
    r["setting"] = np.where(r.phase == "ambulatory", "home", "lab")
    return r


def boot_ci(df, fn, n=N_BOOT, seed=SEED):
    """Subject-level bootstrap 95 % CI of fn(df)."""
    rng = np.random.default_rng(seed)
    groups = {p: ix for p, ix in df.groupby("pid").indices.items()}
    pids = np.array(list(groups))
    vals = []
    for _ in range(n):
        take = np.concatenate([groups[p] for p in rng.choice(pids, len(pids), replace=True)])
        vals.append(fn(df.iloc[take]))
    return np.nanpercentile(vals, [2.5, 97.5])


def risk_coverage(score, abs_err, grid):
    """MAE of retained readings when keeping the lowest-score fraction c of readings, for c in grid."""
    order = np.argsort(score, kind="stable")
    cum = np.cumsum(abs_err[order]) / np.arange(1, len(order) + 1)
    return np.array([cum[max(int(np.ceil(c * len(order))) - 1, 0)] for c in grid])


def main():
    from src.viz.style import NEUTRAL, TARGET, TRAFFIC, figure, panel_labels, save, setup
    cache = pd.read_csv(CODE_DIR / CACHE)
    r = per_reading(cache, CODE_DIR / MODEL_RUN)
    n_all = r.groupby("split").size().to_dict()
    r = r[r.n_cal == 3].copy()
    excl = {s: n_all[s] - int((r.split == s).sum()) for s in n_all}
    val, test = r[r.split == "val"], r[r.split == "test"].copy()
    rng = np.random.default_rng(SEED)
    out = {"n_readings": {"val": len(val), "test": len(test)}, "excluded_fewer_than_3_calibrations": excl}
    L = ["# Interpretability A — \"Can I trust this reading?\" (Aurora-BP, test subjects)", "",
         f"Score: calibration consistency (CC) = SD of the three single-calibration estimates; estimate = their mean (frozen k = 3 "
         f"estimate). Thresholds from each fold's validation subjects, applied to that fold's test subjects. Readings with fewer "
         f"than 3 calibration readings excluded (test: {excl.get('test', 0):,}). Unacceptable reading: |error| > {UNACC:.0f} mmHg. "
         f"95 % CI: subject-level bootstrap ({N_BOOT} resamples).", "",
         f"Test: {len(test):,} readings, {test.pid.nunique():,} subjects.", ""]

    grid = np.linspace(0.01, 1.0, 100)
    curves = {}
    for t in T:
        s, a = test[f"{t}_cc"].to_numpy(), test[f"{t}_abs"].to_numpy()
        rho = spearmanr(s, a).correlation
        rho_ci = boot_ci(test, lambda d: spearmanr(d[f"{t}_cc"], d[f"{t}_abs"]).correlation)
        rc = risk_coverage(s, a, grid)
        oracle = risk_coverage(a, a, grid)
        timec = risk_coverage(test["hours"].fillna(0).to_numpy(), a, grid)
        curves[t] = {"cc": rc, "oracle": oracle, "time": timec, "overall": a.mean()}
        aurc, aurc_or = rc.mean(), oracle.mean()
        out[t] = {"spearman": rho, "spearman_ci": rho_ci.tolist(), "aurc": aurc, "aurc_oracle": aurc_or, "mae_all": a.mean()}
        L += [f"## {t.upper()}", "",
              f"- Spearman ρ(CC, |error|) = {rho:.3f} (95 % CI {rho_ci[0]:.3f}–{rho_ci[1]:.3f})",
              f"- MAE of all readings {a.mean():.2f} mmHg; unacceptable readings {np.mean(a > UNACC):.1%}",
              f"- AURC {aurc:.2f} (oracle {aurc_or:.2f}, random = {a.mean():.2f})", "",
              "| Target coverage | Achieved coverage (test) | MAE retained (95 % CI) | MAE flagged | Unacceptable: retained / flagged | "
              "Unacceptable readings captured by flag | Same flag rate by time since calibration: MAE retained | Random: MAE retained |",
              "|---|---|---|---|---|---|---|---|"]
        for c in COVERAGES:
            thr = val.groupby("fold")[f"{t}_cc"].quantile(c)                      # per-fold validation threshold
            keep = test[f"{t}_cc"].to_numpy() <= test.fold.map(thr).to_numpy()
            ach = keep.mean()
            mr, mf = a[keep].mean(), a[~keep].mean()
            ci = boot_ci(test.assign(_k=keep), lambda d: d.loc[d._k, f"{t}_abs"].mean())
            cap = (a[~keep] > UNACC).sum() / max((a > UNACC).sum(), 1)
            n_flag = int((~keep).sum())
            tkeep = np.ones(len(test), bool)
            tkeep[np.argsort(-test["hours"].fillna(0).to_numpy(), kind="stable")[:n_flag]] = False
            rnd = np.mean([a[rng.permutation(len(a))[n_flag:]].mean() for _ in range(200)])
            out[t][f"cov{int(c * 100)}"] = {"achieved": ach, "mae_retained": mr, "mae_retained_ci": ci.tolist(), "mae_flagged": mf,
                                            "unacc_retained": float(np.mean(a[keep] > UNACC)), "unacc_flagged": float(np.mean(a[~keep] > UNACC)),
                                            "unacc_captured": float(cap), "time_rule_mae_retained": float(a[tkeep].mean()),
                                            "random_mae_retained": float(rnd)}
            L.append(f"| {c:.0%} | {ach:.1%} | {mr:.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {mf:.2f} | {np.mean(a[keep] > UNACC):.1%} / "
                     f"{np.mean(a[~keep] > UNACC):.1%} | {cap:.1%} | {a[tkeep].mean():.2f} | {rnd:.2f} |")
        # traffic light (validation 50th / 80th percentiles, per fold)
        q50 = test.fold.map(val.groupby("fold")[f"{t}_cc"].quantile(0.5)).to_numpy()
        q80 = test.fold.map(val.groupby("fold")[f"{t}_cc"].quantile(0.8)).to_numpy()
        light = np.where(s <= q50, "green", np.where(s <= q80, "amber", "red"))
        test[f"{t}_light"] = light
        L += ["", "| Traffic light | Share of readings | MAE (mmHg) | Unacceptable | Share in-lab / home |", "|---|---|---|---|---|"]
        out[t]["light"] = {}
        for lg in ["green", "amber", "red"]:
            m = light == lg
            lab = np.mean(test.setting.to_numpy()[m] == "lab")
            out[t]["light"][lg] = {"share": float(m.mean()), "mae": float(a[m].mean()), "unacc": float(np.mean(a[m] > UNACC)),
                                   "share_lab": float(lab)}
            L.append(f"| {lg} | {m.mean():.1%} | {a[m].mean():.2f} | {np.mean(a[m] > UNACC):.1%} | {lab:.0%} / {1 - lab:.0%} |")
        L.append("")

    tab = CODE_DIR / "paper/tables/interpretability"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "A_trust.md").write_text("\n".join(L))
    json.dump(out, open(tab / "A_trust.json", "w"), indent=2, default=float)
    test.to_csv(CODE_DIR / "runs/interpretability/A_readings_test_notime.csv.gz", index=False, compression="gzip")
    print("\n".join(L))

    # ---------------------------------------------------------------- Figure I1 (Aurora panels)
    setup()
    fig, axes = figure("double", height=2.4, ncols=3)
    ax = axes[0]
    for t in T:
        c = curves[t]
        ax.plot(grid * 100, c["cc"], color=TARGET[t]["color"], label=f"{TARGET[t]['label']}: consistency score")
        ax.plot(grid * 100, c["time"], color=TARGET[t]["color"], ls=":", lw=0.9, label=f"{TARGET[t]['label']}: time since calibration")
        ax.axhline(c["overall"], color=TARGET[t]["color"], ls="--", lw=0.6)
    ax.set(xlim=(50, 100), xlabel="Readings retained (%)", ylabel="MAE of retained readings (mmHg)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
    for j, t in enumerate(T):
        ax = axes[1 + j]
        x = np.arange(3)
        lt = out[t]["light"]
        ax.bar(x, [lt[g]["mae"] for g in ["green", "amber", "red"]], color=[TRAFFIC[g] for g in ["green", "amber", "red"]], width=0.62)
        for i, g in enumerate(["green", "amber", "red"]):
            ax.text(i, lt[g]["mae"] / 2, f"{lt[g]['unacc']:.0%}\n> {UNACC:.0f}\nmmHg", ha="center", va="center", fontsize=6,
                    color="black" if g == "amber" else "white")
        ax.axhline(out[t]["mae_all"], color=NEUTRAL["line"], ls="--", lw=0.7, label="MAE of all readings")
        ax.set(xticks=x, xticklabels=[f"{g.capitalize()}\n{lt[g]['share']:.0%}" for g in ["green", "amber", "red"]],
               xlabel="Trust level (share of readings)", ylabel=f"{TARGET[t]['label']} MAE (mmHg)", xlim=(-0.5, 2.5),
               ylim=(0, max(lt[g]["mae"] for g in lt) * 1.3))
        ax.legend(loc="upper left", fontsize=6)
        ax.grid(axis="x", visible=False)
    panel_labels(axes)
    save(fig, "interp_A_trust", source=__file__)


if __name__ == "__main__":
    main()
