#!/usr/bin/env python
"""
Interpretability C1 + C2 - "Why did the BP estimate change?" (docs/INTERPRETABILITY_PLAN.md, section 5). CPU only.

  python scripts/interp_C_explain.py

Interpretability model: the no-time model (plan v1.2); cache runs/interpretability/pairs_notime.csv.gz (frozen, verified).
C1  dF = handcrafted features of the target window - mean features of its calibration windows (frozen 31 ECG/PPG features,
    data_processed/features_pwa/aurora_v1.csv; same 10 s windows the model sees). Per fold, a Lasso surrogate (standardised dF,
    missing -> validation median, penalty by 5-fold subject-grouped CV) is fitted on VALIDATION subjects to the model's change
    d_hat = estimate - calibration BP, and separately to the true change; R^2 on TEST subjects. Pre-registered sign expectations:
    pulse arrival times (ptt_rs1, ptt_rm1, ptt_rp1) negative with dSBP; pulse rate positive with dSBP and dDBP (a sign is judged only for features selected in >= 3 of the 5 folds).
    Plausibility flag: |d_hat - surrogate| above the validation 90th percentile (per fold).
    Example explanations: 4 test readings drawn at random (seed 2026) among those with |d_hat_SBP| >= 10 mmHg (2 in-lab, 2 home).
C2  Pathway contributions from the cache: c_fused = estimate - estimate with the fused comparison set to "no change";
    c_ppg likewise for the PPG-only comparison; fused share = |c_fused| / (|c_fused| + |c_ppg|). (No time pathway: the
    interpretability model has no time input.)
Outputs: paper/tables/interpretability/C_explain.md (+ .json, C_examples.md), paper/figures/interp_C_explain.pdf (Fig. I3)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold

CODE_DIR = Path(__file__).resolve().parents[1]
MODEL_RUN = "runs/hicaf_bp_cal_notime"
CACHE = "runs/interpretability/pairs_notime.csv.gz"
FEATS = "data_processed/features_pwa/aurora_v1.csv"
sys.path.insert(0, str(CODE_DIR))
from src.features.pwa_features import FEATURE_NAMES  # noqa: E402

T = ["sbp", "dbp"]
UNACC, SEED = 10.0, 2026
EXPECT = {"sbp": {"ptt_rs1": -1, "ptt_rm1": -1, "ptt_rp1": -1, "pulse_rate_ppg": +1}, "dbp": {"pulse_rate_ppg": +1}}
READABLE = {"ptt_rs1": ("pulse arrival time (R → PPG foot)", 1000, "ms"), "ptt_rm1": ("pulse arrival time (R → max. upslope)", 1000, "ms"),
            "ptt_rp1": ("pulse arrival time (R → PPG peak)", 1000, "ms"), "pulse_rate_ppg": ("heart rate", 1, "bpm"),
            "asc_time_ppg": ("PPG rise time", 1000, "ms"), "desc_time_ppg": ("PPG fall time", 1000, "ms"),
            "lasi_ppg": ("stiffness index (peak → notch)", 1000, "ms"), "pulse_width_ppg": ("pulse width", 1000, "ms"),
            "k_value_ppg": ("pulse K value", 1, "")}


def readable(k):
    """Plain-language feature name (figure labels and explanation sentences)."""
    if k in READABLE:
        return READABLE[k][0]
    base, sig = k.rsplit("_", 1)
    names = {"asc_time": "rise time", "desc_time": "fall time", "asc_slope": "rise slope", "desc_slope": "fall slope",
             "asc_area": "rise area", "desc_area": "fall area", "aid": "rise amplitude", "did": "fall amplitude"}
    return f"{names.get(base, base)} ({ {'ppg': 'PPG', 'vppg': 'PPG 1st deriv.', 'appg': 'PPG 2nd deriv.'}.get(sig, sig) })"


def readings():
    c = pd.read_csv(CODE_DIR / CACHE)
    g = c.groupby(["fold", "split", "sample_id"])
    r = g.agg(pid=("pid", "first"), sbp=("sbp", "first"), dbp=("dbp", "first"),
              sbp_cal=("sbp_cal", "mean"), dbp_cal=("dbp_cal", "mean"),
              sbp_est=("sbp_pred_full", "mean"), dbp_est=("dbp_pred_full", "mean"),
              sbp_nf=("sbp_pred_nofused", "mean"), dbp_nf=("dbp_pred_nofused", "mean"),
              sbp_np=("sbp_pred_noppg", "mean"), dbp_np=("dbp_pred_noppg", "mean"),
              cal_ids=("cal_sample_id", lambda s: tuple(s))).reset_index()
    for t in T:
        r[f"{t}_dhat"] = r[f"{t}_est"] - r[f"{t}_cal"]                    # model's predicted change
        r[f"{t}_dtrue"] = r[t] - r[f"{t}_cal"]                            # true change
        r[f"{t}_abs"] = (r[f"{t}_est"] - r[t]).abs()
        r[f"{t}_c_fused"] = r[f"{t}_est"] - r[f"{t}_nf"]
        r[f"{t}_c_ppg"] = r[f"{t}_est"] - r[f"{t}_np"]
    F = pd.read_csv(CODE_DIR / FEATS).set_index("sample_id")[FEATURE_NAMES]
    tgt = F.reindex(r.sample_id).to_numpy()
    cal = np.stack([F.reindex(list(ids)).mean(axis=0, skipna=True).to_numpy() for ids in r.cal_ids])
    dF = pd.DataFrame(tgt - cal, columns=FEATURE_NAMES, index=r.index)
    ph = pd.concat([pd.read_csv(CODE_DIR / MODEL_RUN / f"fold_{f}" / f"predictions_{s}_k3.csv", usecols=["sample_id", "phase", "condition"])
                    .assign(fold=f, split=s) for f in sorted(r.fold.unique()) for s in ("val", "test")])
    r = r.merge(ph, on=["fold", "split", "sample_id"], how="left")
    r["setting"] = np.where(r.phase == "ambulatory", "home", "lab")
    return r, dF


def main():
    from src.viz.style import NEUTRAL, SETTING, TARGET, figure, panel_labels, save, setup
    r, dF = readings()
    out, L = {}, ["# Interpretability C — \"Why did the BP estimate change?\" (Aurora-BP, test subjects; no-time model)", ""]
    coefs = {t: [] for t in T}
    for t in T:
        r[f"{t}_sur"], r[f"{t}_sur_true"], r[f"{t}_flag"] = np.nan, np.nan, False
        for k in FEATURE_NAMES:
            r[f"{t}_contrib_{k}"] = np.nan
        for f in sorted(r.fold.unique()):
            v = (r.fold == f) & (r.split == "val")
            te = (r.fold == f) & (r.split == "test")
            med = dF[v].median()
            Xv, Xt = dF[v].fillna(med), dF[te].fillna(med)
            mu, sd = Xv.mean(), Xv.std().replace(0, 1.0)
            Zv, Zt = ((Xv - mu) / sd).to_numpy(), ((Xt - mu) / sd).to_numpy()
            cv = list(GroupKFold(5).split(Zv, groups=r.pid[v]))
            for target, col in [(f"{t}_dhat", f"{t}_sur"), (f"{t}_dtrue", f"{t}_sur_true")]:
                m = LassoCV(cv=cv, random_state=SEED, max_iter=20000).fit(Zv, r.loc[v, target])
                r.loc[te, col] = m.predict(Zt)
                if target.endswith("dhat"):
                    coefs[t].append(pd.Series(m.coef_, index=FEATURE_NAMES))
                    resid_v = np.abs(r.loc[v, target].to_numpy() - m.predict(Zv))
                    r.loc[te, f"{t}_flag"] = np.abs(r.loc[te, target].to_numpy() - r.loc[te, col].to_numpy()) > np.quantile(resid_v, 0.9)
                    r.loc[te, [f"{t}_contrib_{k}" for k in FEATURE_NAMES]] = (Zt * m.coef_)     # mmHg per feature
    test = r[r.split == "test"].copy()
    for t in T:
        y, yt = test[f"{t}_dhat"], test[f"{t}_dtrue"]
        r2 = 1 - ((y - test[f"{t}_sur"]) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2t = 1 - ((yt - test[f"{t}_sur_true"]) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        r2m = 1 - ((yt - y) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        cf = pd.concat(coefs[t], axis=1)
        mean_c, nz = cf.mean(axis=1), (cf != 0).mean(axis=1)
        top = mean_c.abs().sort_values(ascending=False).index[:12]
        def verdict(k, e):
            if nz[k] * 5 < 3:                                              # sign judged only if selected in >= 3 of 5 folds
                return "not consistently selected"
            return "agrees" if np.sign(mean_c[k]) == e else "OPPOSITE"
        signs = {k: ("selected in %d/5 folds; mean coef %+.2f; expected %s → %s" %
                     (int(round(nz[k] * 5)), mean_c[k], "−" if e < 0 else "+", verdict(k, e)))
                 for k, e in EXPECT[t].items()}
        flag = test[f"{t}_flag"].astype(bool)
        a = test[f"{t}_abs"]
        out[t] = {"r2_surrogate_model": r2, "r2_surrogate_true": r2t, "r2_model_true_change": r2m, "signs": signs,
                  "coef_mean": mean_c.to_dict(), "selected_share": nz.to_dict(),
                  "flag_rate": float(flag.mean()), "mae_flagged": float(a[flag].mean()), "mae_not_flagged": float(a[~flag].mean()),
                  "unacc_flagged": float((a[flag] > UNACC).mean()), "unacc_not_flagged": float((a[~flag] > UNACC).mean())}
        L += [f"## {t.upper()}", "",
              f"- Fidelity: the surrogate explains **R² = {r2:.2f}** of the model's predicted change on test subjects "
              f"(the same feature set explains R² = {r2t:.2f} of the true change; the model itself explains R² = {r2m:.2f} of the true change).",
              "- Pre-registered sign expectations: " + "; ".join(f"{k}: {v}" for k, v in signs.items()),
              f"- Plausibility flag (|Δ̂ − surrogate| > validation 90th percentile): {flag.mean():.1%} of readings flagged; "
              f"MAE flagged {a[flag].mean():.2f} vs not flagged {a[~flag].mean():.2f} mmHg; unacceptable {(a[flag] > UNACC).mean():.1%} vs "
              f"{(a[~flag] > UNACC).mean():.1%}", "",
              "| Feature (top 12 by mean |standardised coefficient|) | Mean coefficient (mmHg per SD of ΔF) | Selected in folds |", "|---|---|---|"]
        L += [f"| {k} | {mean_c[k]:+.2f} | {int(nz[k] * 5)}/5 |" for k in top]
        L += ["", "**C2 pathway contributions** (test; mean |contribution| in mmHg; fused share = |c_fused| / (|c_fused| + |c_PPG|))", "",
              "| Setting | Readings | mean |c_fused| | mean |c_PPG| | median fused share |", "|---|---|---|---|---|"]
        for st, g in [("all", test)] + list(test.groupby("setting")):
            cf_, cp_ = g[f"{t}_c_fused"].abs(), g[f"{t}_c_ppg"].abs()
            share = (cf_ / (cf_ + cp_).replace(0, np.nan)).median()
            out[t][f"c2_{st}"] = {"fused": cf_.mean(), "ppg": cp_.mean(), "fused_share_median": share}
            L.append(f"| {st} | {len(g):,} | {cf_.mean():.2f} | {cp_.mean():.2f} | {share:.0%} |")
        L.append("")
    tab = CODE_DIR / "paper/tables/interpretability"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "C_explain.md").write_text("\n".join(L))
    json.dump(out, open(tab / "C_explain.json", "w"), indent=2, default=float)
    print("\n".join(L))

    # example explanations (pre-specified selection)
    rng = np.random.default_rng(SEED)
    E = ["# C — example explanations (4 test readings with |Δ̂ SBP| ≥ 10 mmHg, drawn at random, seed 2026: 2 in-lab, 2 home)", ""]
    for st in ["lab", "home"]:
        pool = test[(test.sbp_dhat.abs() >= 10) & (test.setting == st)]
        for i in rng.choice(len(pool), size=min(2, len(pool)), replace=False):
            row = pool.iloc[i]
            contrib = pd.Series({k: row[f"sbp_contrib_{k}"] for k in FEATURE_NAMES})
            top3 = [k for k in contrib.abs().sort_values(ascending=False).index[:3] if abs(contrib[k]) >= 0.5]   # >= 0.5 mmHg only
            parts = []
            for k in top3:
                name, scale, unit = READABLE.get(k, (readable(k), 1, "(signal units)"))
                parts.append(f"{name} {dF.loc[row.name, k] * scale:+.0f} {unit} → {contrib[k]:+.1f} mmHg".replace("  ", " "))
            E.append(f"- **{st}, {row.condition}**: SBP estimated {row.sbp_dhat:+.0f} mmHg vs calibration (true change {row.sbp_dtrue:+.0f}); "
                     f"surrogate {row.sbp_sur:+.0f}: " + ("; ".join(parts) if parts else "no feature change of ≥ 0.5 mmHg")
                     + (" — **plausibility flag**" if row.sbp_flag else ""))
    (tab / "C_examples.md").write_text("\n".join(E))
    print("\n".join(E))

    # Figure I3
    setup()
    fig, axes = figure("double", height=2.6, ncols=3)
    ax = axes[0]
    keys = pd.concat([pd.Series(out[t]["coef_mean"]).abs() for t in T], axis=1).max(axis=1).sort_values(ascending=False).index[:10][::-1]
    y = np.arange(len(keys))
    for j, t in enumerate(T):
        ax.barh(y + (j - 0.5) * 0.38, [out[t]["coef_mean"][k] for k in keys], height=0.36, color=TARGET[t]["color"], label=TARGET[t]["label"])
    ax.set(yticks=y, yticklabels=[readable(k) for k in keys], xlabel="Surrogate coefficient (mmHg per SD of change)")
    ax.axvline(0, color=NEUTRAL["line"], lw=0.6)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(loc="lower right")
    ax = axes[1]
    x = np.arange(2)
    for j, st in enumerate(["lab", "home"]):
        ax.bar(x + (j - 0.5) * 0.38, [out[t][f"c2_{st}"]["fused_share_median"] * 100 for t in T], width=0.36,
               color=SETTING[st]["color"], label=SETTING[st]["label"])
    ax.axhline(50, color=NEUTRAL["line"], ls="--", lw=0.6)
    ax.set(xticks=x, xticklabels=[TARGET[t]["label"] for t in T], ylabel="Fused (ECG + PPG) share of the change (%)", ylim=(0, 100))
    ax.legend(loc="upper right")
    ax = axes[2]
    for j, t in enumerate(T):
        ax.bar(np.arange(2) + (j - 0.5) * 0.38, [out[t]["mae_not_flagged"], out[t]["mae_flagged"]], width=0.36,
               color=TARGET[t]["color"], label=TARGET[t]["label"])
    ax.set(xticks=[0, 1], xticklabels=["Plausible", "Implausible (flagged)"], ylabel="MAE (mmHg)",
           ylim=(0, max(out[t][k] for t in T for k in ("mae_not_flagged", "mae_flagged")) * 1.3))
    ax.legend(loc="upper center", ncol=2)
    panel_labels(axes)
    save(fig, "interp_C_explain", source=__file__)


if __name__ == "__main__":
    main()
