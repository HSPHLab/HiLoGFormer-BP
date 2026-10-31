#!/usr/bin/env python
"""
Calibration-free methods on Aurora-BP, scored on exactly the calibrated test targets (read-only on runs/).

  python scripts/report_calfree_aurora.py

Targets = the 21,280 readings of runs/hicaf_bp_cal_notime (k = 3 file), so every row is directly comparable with
the calibrated table (paper/tables/calibrated). Sanity rows: HiCAF-BP-Cal (k = 3) and the calibration value only
must reproduce 8.53 / 6.19 and 11.43 / 8.54 mmHg.
Output: paper/tables/calibration_free_aurora/CALIBRATION_FREE_AURORA.md (+ .csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

R = Path("runs")
T = ["sbp", "dbp"]
OUT = Path("paper/tables/calibration_free_aurora")


def load(run, name):
    fs = [R / run / f"fold_{i}" / name for i in range(5)]
    missing = [str(f) for f in fs if not f.exists()]
    if missing:
        raise SystemExit(f"missing prediction files: {missing}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


ref = load("hicaf_bp_cal_notime", "predictions_test_k3.csv")
assert not ref.sample_id.duplicated().any(), "reference: a reading is predicted twice"
meta = pd.read_csv("data_processed/v1/metadata.csv", usecols=["sample_id", "phase"])
print("phase values in metadata:", meta.phase.value_counts().to_dict())
if "phase" in ref.columns:   # prediction files already carry phase: check it against the metadata
    chk = ref[["sample_id", "phase"]].merge(meta, on="sample_id", how="left", suffixes=("", "_meta"))
    print("phase agrees with metadata:", bool((chk.phase == chk.phase_meta).all()))
else:
    ref = ref.merge(meta, on="sample_id", how="left")
assert ref.phase.notna().all(), "reference readings without phase"
home = ref.phase.str.contains("ambul|home", case=False).to_numpy()
print(f"targets {len(ref):,} | subjects {ref.pid.nunique():,} | in-lab {(~home).sum():,} | home {home.sum():,}")

est = {"HiCAF-BP-Cal (ours, k = 3)": {t: ref[f"{t}_pred"].to_numpy() for t in T},
       "Calibration value only (k = 3)": {t: ref[f"{t}_cal_mean"].to_numpy() for t in T}}
for name, run in [("HiCAF-BP, uncalibrated (ours)", "hicaf_bp_main"), ("ResNet1D", "resnet1d_ecgppg"),
                  ("rU-Net", "pub_runet"), ("UTransBPNet", "pub_utransbpnet")]:
    p = load(run, "predictions_test.csv")
    cols = [f"{t}_pred" for t in T]
    if not set(cols + ["sample_id"] + T) <= set(p.columns):
        raise SystemExit(f"{run}: unexpected columns {list(p.columns)}")
    if p.sample_id.duplicated().any():
        raise SystemExit(f"{run}: a reading is predicted twice")
    p = p.set_index("sample_id")
    miss = set(ref.sample_id) - set(p.index)
    if miss:
        raise SystemExit(f"{run}: {len(miss)} target readings missing")
    q = p.loc[ref.sample_id]
    dref = max(float(np.abs(q[t].to_numpy() - ref[t].to_numpy()).max()) for t in T)
    print(f"{run}: {len(p):,} predictions, {len(ref):,} targets selected, max |reference difference| {dref:.3f} mmHg")
    est[name] = {t: q[f"{t}_pred"].to_numpy() for t in T}

ours = "HiCAF-BP-Cal (ours, k = 3)"
subsets = {"all": np.ones(len(ref), bool), "in-lab": ~home, "home": home}
rows, L = [], ["# Calibration-free methods on Aurora-BP (same 21,280 test targets as the calibrated table)", "",
               "Error = estimate − reference. Δ = subject-level mean MAE of the method − ours (k = 3); "
               "p = Wilcoxon signed-rank on per-subject MAE.", ""]
for sname, m in subsets.items():
    L += [f"## {sname} ({m.sum():,} readings, {ref.pid[m].nunique():,} subjects)", "",
          "| Method | SBP MAE | SBP ME ± SD | SBP r | DBP MAE | DBP ME ± SD | DBP r | vs ours: Δ SBP, p; Δ DBP, p |",
          "|---|---|---|---|---|---|---|---|"]
    pid = ref.pid[m].to_numpy()
    per_ours = {t: pd.Series(np.abs(est[ours][t][m] - ref[t].to_numpy()[m])).groupby(pid).mean() for t in T}
    for name, e in est.items():
        cells, r = [], {}
        for t in T:
            err = e[t][m] - ref[t].to_numpy()[m]
            mae, me, sd = np.abs(err).mean(), err.mean(), err.std(ddof=1)
            rr = np.corrcoef(e[t][m], ref[t].to_numpy()[m])[0, 1]
            cells += [f"{mae:.2f}", f"{me:+.2f} ± {sd:.2f}", f"{rr:.2f}"]
            r.update({f"{t}_mae": mae, f"{t}_me": me, f"{t}_sd": sd, f"{t}_r": rr})
        cmp = []
        if name != ours:
            for t in T:
                per = pd.Series(np.abs(e[t][m] - ref[t].to_numpy()[m])).groupby(pid).mean()
                pv = wilcoxon(per, per_ours[t]).pvalue
                cmp.append(f"{per.mean() - per_ours[t].mean():+.2f}, {'p < 0.001' if pv < 0.001 else f'p {pv:.3f}'}")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {'; '.join(cmp) if cmp else '–'} |")
        rows.append({"subset": sname, "method": name, **r})
    L.append("")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "CALIBRATION_FREE_AURORA.md").write_text("\n".join(L))
pd.DataFrame(rows).to_csv(OUT / "calibration_free_aurora.csv", index=False)
print("\n".join(L))
print(f"Saved {OUT}/")
