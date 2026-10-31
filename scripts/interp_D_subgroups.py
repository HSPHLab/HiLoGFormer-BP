#!/usr/bin/env python
"""
Interpretability D - "For which patients does it work?" (docs/INTERPRETABILITY_PLAN.md, section 6). CPU only.

  python scripts/interp_D_subgroups.py

Interpretability model (plan v1.2): no-time model; its k = 3 test estimates from the frozen cache
(runs/interpretability/pairs_notime.csv.gz; verified against the frozen predictions). Comparator: the best published
calibrated method on Aurora-BP (RAGCBPNet, k = 3, identical test recordings - checked).
Subgroups (test subjects): age, sex, BMI, self-reported hypertension, setting, true |change| from calibration, |change in
pulse rate| from calibration (frozen PWA features), signal quality (Aurora's optical_quality; thirds cut on validation
readings). Per subgroup and target: MAE (95 % CI, subject bootstrap), ME ± SD, share within 5 / 10 / 15 mmHg (BHS),
AAMI criterion (|ME| <= 5 and SD <= 8 mmHg), paired subject-level difference to RAGCBPNet (Wilcoxon).
Decision rule: "use" when the AAMI criterion holds, otherwise "use with caution".
Outputs: paper/tables/interpretability/D_subgroups.md (+ .json), paper/figures/interp_D_subgroups.pdf (Fig. I5)
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CODE_DIR = Path(__file__).resolve().parents[1]
MODEL_RUN = "runs/hicaf_bp_cal_notime"
CACHE = "runs/interpretability/pairs_notime.csv.gz"
COMPARATOR = ("runs/pub_ragcbpnet", "RAGCBPNet")
FEATS = "data_processed/features_pwa/aurora_v1.csv"
sys.path.insert(0, str(CODE_DIR))

T = ["sbp", "dbp"]


def load():
    from src.config import load_config
    from src.data.splits import load_metadata
    c = pd.read_csv(CODE_DIR / CACHE)
    r = c.groupby(["fold", "split", "sample_id"]).agg(
        pid=("pid", "first"), sbp=("sbp", "first"), dbp=("dbp", "first"), sbp_cal=("sbp_cal", "mean"), dbp_cal=("dbp_cal", "mean"),
        sbp_est=("sbp_pred_full", "mean"), dbp_est=("dbp_pred_full", "mean"), cal_ids=("cal_sample_id", lambda s: tuple(s))).reset_index()
    meta, _, _ = load_metadata(load_config("experiments/hicaf_bp_cal_t_scratch.yaml"))
    keep = [k for k in ["sample_id", "age", "gender", "height", "weight", "self_report_htn", "phase", "measurement", "optical_quality"] if k in meta]
    r = r.merge(meta[keep], on="sample_id", how="left")
    if "optical_quality" not in r:                                         # merge from Aurora's measurement index if needed
        cands = [p for p in glob.glob(str(CODE_DIR / "**" / "measurement_index.csv"), recursive=True) if "/runs/" not in p][:1]
        if cands:
            mi = pd.read_csv(cands[0], usecols=["pid", "phase", "measurement", "optical_quality"]).drop_duplicates(["pid", "phase", "measurement"])
            r = r.merge(mi, on=["pid", "phase", "measurement"], how="left")
            print(f"optical_quality merged from {cands[0]} (available for {r.optical_quality.notna().mean():.1%} of readings)")
        else:
            r["optical_quality"] = np.nan
            print("optical_quality not found - signal-quality subgroup skipped")
    h, w = r.height.astype(float), r.weight.astype(float)
    r["bmi"] = 703.0 * w / h ** 2 if np.nanmedian(h) < 100 else w / (h / 100.0) ** 2
    F = pd.read_csv(CODE_DIR / FEATS).set_index("sample_id")["pulse_rate_ppg"]
    r["d_hr"] = F.reindex(r.sample_id).to_numpy() - np.array([F.reindex(list(i)).mean() for i in r.cal_ids])
    for t in T:
        r[f"{t}_err"] = r[f"{t}_est"] - r[t]
        r[f"{t}_dtrue"] = (r[t] - r[f"{t}_cal"]).abs()
    comp = pd.concat([pd.read_csv(CODE_DIR / COMPARATOR[0] / f"fold_{f}" / "predictions_test_k3.csv") for f in range(5)]).set_index("sample_id")
    test = r[r.split == "test"].copy()
    assert set(test.sample_id) == set(comp.index), "comparator not scored on the same test recordings"
    for t in T:
        test[f"{t}_err_comp"] = comp.loc[test.sample_id, f"{t}_pred"].to_numpy() - test[t].to_numpy()
    return r[r.split == "val"], test


def subgroups(val, test):
    q = val.optical_quality.quantile([1 / 3, 2 / 3]).to_numpy() if val.optical_quality.notna().any() else None
    g = {"age": pd.cut(test.age, [0, 40, 60, 200], right=False, labels=["< 40 y", "40–60 y", "≥ 60 y"]),
         "sex": pd.Categorical(test.gender.map({"F": "female", "M": "male"}), categories=["female", "male"]),
         "BMI": pd.cut(test.bmi, [0, 25, 30, 100], right=False, labels=["< 25", "25–30", "≥ 30"]),
         "hypertension (self-report)": pd.Categorical(test.self_report_htn.astype(str).str.lower().map(
             {"norm": "normotensive", "managed": "managed", "unmanaged": "unmanaged"}), categories=["normotensive", "managed", "unmanaged"]),
         "setting": pd.Categorical(test.phase.map({"initial": "in-lab, initial", "return": "in-lab, return", "ambulatory": "ambulatory (home)"}),
                                   categories=["in-lab, initial", "in-lab, return", "ambulatory (home)"]),
         "pulse-rate change from calibration (absolute)": pd.cut(test.d_hr.abs(), [0, 5, 15, 1e9], right=False, labels=["< 5 bpm", "5–15 bpm", "≥ 15 bpm"])}
    if q is not None:
        g["signal quality (thirds)"] = pd.cut(test.optical_quality, [-np.inf, q[0], q[1], np.inf], labels=["low", "middle", "high"])
    return g


def stats(d, t, n_boot=1000):
    e, ec = d[f"{t}_err"].to_numpy(), d[f"{t}_err_comp"].to_numpy()
    a = np.abs(e)
    codes, uniq = pd.factorize(d.pid.to_numpy())
    S, N = np.bincount(codes, weights=a, minlength=len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(2026)
    boots = []
    for _ in range(n_boot):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))
        boots.append((w @ S) / (w @ N))
    me, sd = e.mean(), e.std(ddof=1)
    sa = pd.Series(a).groupby(d.pid.to_numpy()).mean()
    sc = pd.Series(np.abs(ec)).groupby(d.pid.to_numpy()).mean()
    p = wilcoxon(sa, sc).pvalue if len(sa) > 10 else np.nan
    return {"n": len(d), "subjects": len(uniq), "mae": a.mean(), "mae_ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "me": me, "sd": sd, "w5": np.mean(a <= 5), "w10": np.mean(a <= 10), "w15": np.mean(a <= 15),
            "aami": bool(abs(me) <= 5 and sd <= 8), "mae_comp": np.abs(ec).mean(), "delta_vs_comp": (sa - sc).mean(),
            "better_share": float((sa < sc).mean()), "p": p}


def main():
    from src.viz.style import METHOD, figure, panel_labels, save, setup
    val, test = load()
    groups = subgroups(val, test)
    out, L = {}, ["# Interpretability D — \"For which patients does it work?\" (Aurora-BP test subjects; no-time model, k = 3)", "",
                  f"Comparator: {COMPARATOR[1]} (best published calibrated method; identical test recordings). AAMI criterion: |ME| ≤ 5 and "
                  "SD ≤ 8 mmHg. 95 % CI: subject-level bootstrap (1000). Δ vs comparator: mean per-subject MAE difference (ours − "
                  f"{COMPARATOR[1]}), share of subjects where ours is better, Wilcoxon p.", ""]
    for t in T:
        groups[f"true {t.upper()} change from calibration (absolute)"] = pd.cut(test[f"{t}_dtrue"], [0, 10, 20, 1e9], right=False,
                                                                  labels=["< 10 mmHg", "10–20 mmHg", "≥ 20 mmHg"])
    for t in T:
        L += [f"## {t.upper()}", "", "| Subgroup | Level | Readings (subjects) | MAE (95 % CI) | ME ± SD | Within 5 / 10 / 15 mmHg | AAMI | "
              f"{COMPARATOR[1]} MAE | Δ vs {COMPARATOR[1]} (ours better), p | Guidance |", "|---|---|---|---|---|---|---|---|---|---|"]
        allr = stats(test, t)
        out[f"{t}/all"] = allr
        L.append(f"| all | all | {allr['n']:,} ({allr['subjects']:,}) | {allr['mae']:.2f} ({allr['mae_ci'][0]:.2f}–{allr['mae_ci'][1]:.2f}) | "
                 f"{allr['me']:+.2f} ± {allr['sd']:.2f} | {allr['w5']:.0%} / {allr['w10']:.0%} / {allr['w15']:.0%} | {'yes' if allr['aami'] else 'no'} | "
                 f"{allr['mae_comp']:.2f} | {allr['delta_vs_comp']:+.2f} ({allr['better_share']:.0%}), {allr['p']:.1g} | "
                 f"{'use' if allr['aami'] else 'use with caution'} |")
        for gname, lab in groups.items():
            if gname.startswith("true ") and t.upper() not in gname:
                continue
            lab = pd.Series(lab, index=test.index) if not isinstance(lab, pd.Series) else lab
            for lev in lab.cat.categories:
                m = (lab == lev).to_numpy()
                if m.sum() < 50:
                    continue
                s = stats(test[m], t)
                out[f"{t}/{gname}/{lev}"] = s
                L.append(f"| {gname} | {lev} | {s['n']:,} ({s['subjects']:,}) | {s['mae']:.2f} ({s['mae_ci'][0]:.2f}–{s['mae_ci'][1]:.2f}) | "
                         f"{s['me']:+.2f} ± {s['sd']:.2f} | {s['w5']:.0%} / {s['w10']:.0%} / {s['w15']:.0%} | {'yes' if s['aami'] else 'no'} | "
                         f"{s['mae_comp']:.2f} | {s['delta_vs_comp']:+.2f} ({s['better_share']:.0%}), {s['p']:.1g} | "
                         f"{'use' if s['aami'] else 'use with caution'} |")
        L.append("")
    tab = CODE_DIR / "paper/tables/interpretability"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "D_subgroups.md").write_text("\n".join(L))
    json.dump(out, open(tab / "D_subgroups.json", "w"), indent=2, default=float)
    print("\n".join(L))

    # Figure I5: forest plot
    setup()
    rows = [k.split("/", 1)[1] for k in out if k.startswith("sbp/")]
    rows = [r_ for r_ in rows if not r_.startswith("true ")] + [r_ for r_ in rows if r_.startswith("true SBP")]
    fig, axes = figure("double", height=0.16 * len(rows) + 0.9, ncols=2, sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, t in zip(axes, T):
        for yi, key in zip(y, rows):
            k = f"{t}/{key}" if f"{t}/{key}" in out else f"{t}/{key.replace('SBP', t.upper())}"
            if k not in out:
                continue
            s = out[k]
            ax.errorbar(s["mae"], yi, xerr=[[s["mae"] - s["mae_ci"][0]], [s["mae_ci"][1] - s["mae"]]], fmt="o",
                        color=METHOD["ours"]["color"], mfc=METHOD["ours"]["color"] if s["aami"] else "white", ms=3.5, lw=0.8)
            ax.plot(s["mae_comp"], yi, METHOD["ragcbpnet"]["marker"], color=METHOD["ragcbpnet"]["color"], ms=3)
        ax.axvline(out[f"{t}/all"]["mae"], color="#808080", ls="--", lw=0.6)
        ax.set(xlabel=f"{t.upper()} MAE (mmHg)")
    axes[0].set(yticks=y, yticklabels=[r_.replace("/", ": ").replace("true SBP change", "true BP change") for r_ in rows])
    axes[0].tick_params(axis="y", labelsize=6)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color=METHOD["ours"]["color"], ls="", label="HiCAF-BP-Cal (ours), AAMI met"),
         Line2D([], [], marker="o", color=METHOD["ours"]["color"], mfc="white", ls="", label="HiCAF-BP-Cal (ours), AAMI not met"),
         Line2D([], [], marker=METHOD["ragcbpnet"]["marker"], color=METHOD["ragcbpnet"]["color"], ls="", label=COMPARATOR[1])]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6)
    panel_labels(axes, x=-0.05)
    save(fig, "interp_D_subgroups", source=__file__)


if __name__ == "__main__":
    main()
