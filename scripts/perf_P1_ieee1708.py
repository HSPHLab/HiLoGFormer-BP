#!/usr/bin/env python
"""
Measurement performance evaluation, P1 - IEEE Std 1708-2025 accuracy report and grading (docs/PERFORMANCE_EVALUATION_PLAN.md,
sections 1-4). CPU only; frozen predictions; no model is run.

  python scripts/perf_P1_ieee1708.py

Device class: snapshot BP, personalized BP calibration (IEEE 1708-2025 Table 1) -> static test, test with BP change, test after a
certain period (Table 4). Primary set: auscultatory protocol of the Aurora-BP test subjects (5 folds pooled):
  calibration   initial visit, the model's k = 3 calibration readings, taken from the frozen cache (verified: "Calibration start
                1-3"; "Seated calibration" for subjects without them)
  static        initial visit, "Static seated challenge 1-3"                                   (§4.4.2)
  BP change     initial visit, challenge readings "Static challenge start 1-3" and "Exercise challenge start 1-2" (all of them =
                "All BP change measurements"), binned by the reference change from the calibration BP (-20..-10, -10..-5,
                5..15, 15..25 mmHg graded; -5..5 and beyond the bins reported)                    (§4.4.3, Table 5)
  after period  return visit, "Temporal seated challenge 1-3" (> 24 h after calibration)         (§4.4.4)
  BP levels     entry BP = mean of the calibration readings, 2017 ACC/AHA classes               (§4.5.3.4)
Metrics (Table 9): MAD, MAPD, MD, SD, CP5/10/15; 95 % CI of MAD by subject bootstrap (1,000; seed 2026) and one-sided 90 % upper
bound (§4.6.2, Annex E). Grade (Table 6): A MAD <= 5; B 5-6 and |MD| <= 5; C 6-7 and |MD| <= 5; otherwise D. Final grade = worst
over SBP/DBP and over overall accuracy, static test, all BP-change measurements, each graded BP-change level and the test after a
certain period (§4.5.3.1-4.5.3.3; plan change log 2026-10-14).
Outputs (paper/performance_evaluation/): tables STD_Table_P1_accuracy_report, STD_Table_P2_grades,
STD_Table_P3_induced_change_report, STD_Table_P10_error_correlation_distribution (.md + .json); figures
STD_Fig_P1_graphical_report_SBP / _DBP and STD_Fig_P2_accuracy_by_test (.pdf + png/).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
SEC = CODE_DIR / "paper/performance_evaluation"
T = ["sbp", "dbp"]
N_BOOT, SEED = 1000, 2026
PULSE_CACHE = "runs/interpretability/pulsedb_v1notime.npz"   # zero-shot PulseDB, every fold model on every record (no-time model)
CACHES = ["runs/interpretability/pairs_notime.csv.gz", "runs/interpretability/pairs_v1.csv.gz"]   # frozen caches: calibration readings
META = "data_processed/v1/metadata.csv"
CALIBRATED = [("ours", "runs/hicaf_bp_cal_notime"), ("ours_time", "runs/hicaf_bp_cal_t_scratch"), ("hgctnet", "runs/pub_hgctnet"),
              ("bpbls", "runs/pub_bpbls"), ("ragcbpnet", "runs/pub_ragcbpnet")]
CAL_FREE = [("ours_uncal", "runs/hicaf_bp_main"), ("resnet1d", "runs/resnet1d_ecgppg"), ("runet", "runs/pub_runet"),
            ("utransbpnet", "runs/pub_utransbpnet")]
BINS = [(-20, -10), (-10, -5), (-5, 5), (5, 15), (15, 25)]
BIN_LAB = ["−20…−10", "−10…−5", "−5…5", "5…15", "15…25"]
STD_LEVELS = ["−20…−10", "−10…−5", "5…15", "15…25"]                    # graded BP-change levels (Table 9 rows)
REQ = {"−20…−10": (0.10, 9), "−10…−5": (0.20, 17), "−5…5": (0.0, 0), "5…15": (0.20, 17), "15…25": (0.25, 22)}   # Table 5
GRADE_FILL = {"A": "#DCEFD9", "B": "#EEF4D3", "C": "#FBEBCB", "D": "#F7D4D0"}


def grade(mad, md):
    if mad <= 5:
        return "A"
    if abs(md) <= 5 and mad <= 6:
        return "B"
    if abs(md) <= 5 and mad <= 7:
        return "C"
    return "D"


def worst(gs):
    return max(gs, key="ABCD".index) if gs else "–"


def bp_class(s, d):
    if s >= 140 or d >= 90:
        return "Stage 2 hypertension"
    if s >= 130 or d >= 80:
        return "Stage 1 hypertension"
    if s >= 120:
        return "Elevated"
    return "Normal"


def metrics(err, ref, pid, rng_seed=SEED, n_boot=N_BOOT):
    """IEEE 1708-2025 Table 9 metrics + subject-bootstrap CI and one-sided 90 % upper bound of MAD."""
    err, ref, pid = np.asarray(err, float), np.asarray(ref, float), np.asarray(pid)
    a = np.abs(err)
    out = {"n": int(len(err)), "subjects": int(len(np.unique(pid))), "MAD": float(a.mean()),
           "MAPD": float(np.mean(100 * a / ref)), "MD": float(err.mean()), "SD": float(err.std(ddof=1)) if len(err) > 1 else float("nan"),
           "CP5": float(np.mean(a <= 5) * 100), "CP10": float(np.mean(a <= 10) * 100), "CP15": float(np.mean(a <= 15) * 100)}
    codes, uniq = pd.factorize(pid)
    S, N = np.bincount(codes, weights=a, minlength=len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(rng_seed)
    boots = np.array([(w @ S) / (w @ N) for w in (rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq))) for _ in range(n_boot))])
    out["MAD_CI"] = np.percentile(boots, [2.5, 97.5]).tolist()
    out["MAD_U90"] = float(np.percentile(boots, 90))
    out["grade"] = grade(out["MAD"], out["MD"])
    return out


def icc1(err, pid):
    """One-way random-effects ICC(1) of the errors (within-subject error correlation)."""
    d = pd.DataFrame({"e": err, "g": pid})
    g = d.groupby("g").e
    k = g.size()
    d = d[d.g.isin(k[k >= 2].index)]
    g = d.groupby("g").e
    n, N = g.ngroups, len(d)
    k0 = (N - (g.size() ** 2).sum() / N) / (n - 1)
    msb = (g.size() * (g.mean() - d.e.mean()) ** 2).sum() / (n - 1)
    msw = ((d.e - g.transform("mean")) ** 2).sum() / (N - n)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def read_preds(fp):
    """Frozen prediction file -> sample_id, sbp_pred, dbp_pred (tolerates the column spellings used across our runs)."""
    p = pd.read_csv(fp)
    alias = {"sbp_pred": ["sbp_pred", "pred_sbp", "sbp_hat", "sbp_est"], "dbp_pred": ["dbp_pred", "pred_dbp", "dbp_hat", "dbp_est"]}
    out = {"sample_id": p["sample_id"]}
    for k, cands in alias.items():
        c = next((c for c in cands if c in p.columns), None)
        if c is None:
            raise KeyError(f"{fp}: no {k} column; columns are {list(p.columns)}")
        out[k] = p[c]
    return pd.DataFrame(out)


def _boot_w(n_sub, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_sub, np.full(n_sub, 1 / n_sub), size=n_boot).astype(float)


def ba_repeated(err, pid):
    """Bland & Altman (2007), multiple readings per subject, true value varying: total SD from between- and within-subject
    variance components of the differences; returns bias, SD, LoA with subject-bootstrap 95 % CIs."""
    codes, uniq = pd.factorize(pid)
    n = np.bincount(codes).astype(float)
    S = np.bincount(codes, weights=err)
    Q = np.bincount(codes, weights=err ** 2)

    def stat(w):
        N, G = w @ n, w.sum()
        mu = (w @ S) / N
        ssw = w @ (Q - S ** 2 / n)
        ssb = w @ (n * (S / n - mu) ** 2)
        msw, msb = ssw / (N - G), ssb / (G - 1)
        k0 = (N - (w @ n ** 2) / N) / (G - 1)
        sd = math.sqrt(max(msb - msw, 0) / k0 + msw)
        return mu, sd, mu - 1.96 * sd, mu + 1.96 * sd
    est = stat(np.ones(len(uniq)))
    boots = np.array([stat(w) for w in _boot_w(len(uniq))])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    return {k: (float(est[i]), float(ci[0, i]), float(ci[1, i])) for i, k in enumerate(["bias", "sd", "loa_low", "loa_high"])}


def slope_cluster(y, x, pid):
    """OLS slope of y on x with subject-cluster bootstrap 95 % CI (proportional bias)."""
    codes, uniq = pd.factorize(pid)

    def fit(wr):
        xm, ym = np.average(x, weights=wr), np.average(y, weights=wr)
        return float(np.sum(wr * (x - xm) * (y - ym)) / np.sum(wr * (x - xm) ** 2))
    b = fit(np.ones(len(x)))
    boots = [fit(w[codes]) for w in _boot_w(len(uniq))]
    return b, *np.percentile(boots, [2.5, 97.5]).tolist()


def ratio_ci(hit, pid):
    """Proportion with subject-bootstrap 95 % CI."""
    codes, uniq = pd.factorize(pid)
    H, N = np.bincount(codes, weights=hit.astype(float), minlength=len(uniq)), np.bincount(codes, minlength=len(uniq)).astype(float)
    boots = [(w @ H) / (w @ N) for w in _boot_w(len(uniq))]
    return float(hit.mean()), *np.percentile(boots, [2.5, 97.5]).tolist()


def within_sd(v, pid):
    """Pooled within-subject SD (sqrt of the within-subject mean square)."""
    d = pd.DataFrame({"v": v, "g": pid})
    g = d.groupby("g").v
    d = d[g.transform("size") >= 2]
    g = d.groupby("g").v
    return float(math.sqrt(((d.v - g.transform("mean")) ** 2).sum() / (len(d) - g.ngroups)))


def model_instances(p6):
    """Reproducibility between model instances: the 5 fold models (no-time variant) applied to the same zero-shot PulseDB records
    with the same calibration (frozen interpretability cache). Per-model MAD, their range, and the between-model SD per reading."""
    fp = CODE_DIR / PULSE_CACHE
    L = ["", "## Reproducibility between model instances (zero-shot PulseDB)", ""]
    if not fp.exists():
        return L + [f"{PULSE_CACHE} not found."]
    d = np.load(fp, allow_pickle=True)
    est, ref, pid, prot = d["est"].astype(float), np.stack([d["sbp"], d["dbp"]], 1).astype(float), d["pid"], d["protocol"].astype(str)
    L += ["The five fold models of HiCAF-BP-Cal (trained on different Aurora-BP subjects) applied to the same PulseDB "
          "records with the same calibration as in the frozen zero-shot evaluation (cal_mean of the interpretability cache). Between-model SD = SD of the five estimates of a reading (ddof = 1).", "",
          "| Set | Target | Records (subjects) | MAD per model, min–max | Range | MAD of the 5-model mean | Between-model SD, median (95th pct.) |",
          "|---|---|---|---|---|---|---|"]
    sets = [("All", np.ones(len(ref), bool))] + [(str(v), prot == v) for v in sorted(set(prot))]
    for sl, mk in sets:
        for j, t in enumerate(T):
            e = est[:, mk, j] - ref[None, mk, j]
            mads = np.abs(e).mean(1)
            ens = np.abs(est[:, mk, j].mean(0) - ref[mk, j]).mean()
            sd = est[:, mk, j].std(0, ddof=1)
            p6[f"instances/{sl}/{t}"] = {"mad_models": mads.tolist(), "mad_ensemble": float(ens), "sd_median": float(np.median(sd)),
                                         "sd_p95": float(np.percentile(sd, 95)), "n": int(mk.sum())}
            L.append(f"| {sl} | {t.upper()} | {mk.sum():,} ({len(set(pid[mk])):,}) | {mads.min():.2f}–{mads.max():.2f} | {np.ptp(mads):.2f} | "
                     f"{ens:.2f} | {np.median(sd):.2f} ({np.percentile(sd, 95):.2f}) |")
    return L


def load():
    """Targets = test readings of the frozen cache; calibration identity, calibration BP and hours since calibration are taken
    from the cache (the readings the model actually conditioned on), never from measurement names."""
    cache = next(CODE_DIR / f for f in CACHES if (CODE_DIR / f).exists())
    c = pd.read_csv(cache)
    c = c[(c.split == "test") & (c.cal_rank < 3)]
    cal = c.groupby("pid").agg(sbp_cal=("sbp_cal", "mean"), dbp_cal=("dbp_cal", "mean"), n_cal=("cal_sample_id", "nunique")).reset_index()
    cal_ids = set(c.cal_sample_id)
    tg = c.groupby("sample_id").agg(pid=("pid", "first"), fold=("fold", "first"), sbp=("sbp", "first"), dbp=("dbp", "first"),
                                    hours=("hours_since_cal", "mean")).reset_index()
    meta = pd.read_csv(CODE_DIR / META, low_memory=False, usecols=lambda k: k in {"sample_id", "phase", "measurement", "protocol"})
    tg = tg.merge(meta, on="sample_id", how="left").merge(cal, on="pid", how="left")
    cal_meas = c.drop_duplicates("cal_sample_id")[["pid", "cal_sample_id"]].merge(meta[["sample_id", "measurement"]], left_on="cal_sample_id",
                                                                                   right_on="sample_id", how="left")
    for t in T:
        tg[f"{t}_change"] = tg[t] - tg[f"{t}_cal"]
    tg["entry_class"] = [bp_class(s, d) for s, d in zip(tg.sbp_cal, tg.dbp_cal)]
    ms = tg.measurement.astype(str)
    au = (tg.protocol == "auscultatory") & ~tg.sample_id.isin(cal_ids)
    tg["test"] = np.select([au & (tg.phase == "initial") & ms.str.startswith("Static seated challenge"),
                            au & (tg.phase == "initial") & ms.str.match(r"^(Static challenge start|Exercise challenge start)"),
                            au & (tg.phase == "return") & ms.str.startswith("Temporal seated challenge") & (tg.hours > 24)],
                           ["static", "change", "after"], default="other")
    tg.loc[au & (tg.phase == "initial") & ms.str.startswith("Seated calibration"), "test"] = "seated_extra"   # sensitivity only
    preds = {}
    for key, run in CALIBRATED + CAL_FREE:
        name = "predictions_test_k3.csv" if (key, run) in CALIBRATED else "predictions_test.csv"
        fps = [CODE_DIR / run / f"fold_{f}" / name for f in range(5)]
        if not all(fp.exists() for fp in fps):
            print(f"skip {key}: {run}/fold_*/{name} not found")
            continue
        preds[key] = pd.concat([read_preds(fp) for fp in fps]).drop_duplicates("sample_id").set_index("sample_id")
    preds["cal_only"] = tg.set_index("sample_id")[["sbp_cal", "dbp_cal"]].rename(columns={"sbp_cal": "sbp_pred", "dbp_cal": "dbp_pred"})
    other_initial = tg[(tg.protocol == "auscultatory") & (tg.phase == "initial") & tg.test.isin(["other", "seated_extra"])].measurement.astype(str)
    return tg, preds, cal_meas, other_initial, cache.name


def main():
    from src.viz.style import METHOD, TARGET, figure, panel_labels, save, setup
    t0 = time.time()
    (SEC / "tables").mkdir(parents=True, exist_ok=True)
    (SEC / "figures").mkdir(parents=True, exist_ok=True)
    tg, preds, cal_meas, other_initial, cache_name = load()
    prim = tg[tg.test.isin(["static", "change", "after"])].copy()
    seated_extra = tg[tg.test == "seated_extra"]
    strip = lambda s: s.str.replace(r"\s*\d+$", "", regex=True).value_counts().to_dict()
    after_all = tg[(tg.protocol == "auscultatory") & (tg.phase == "return") & tg.measurement.astype(str).str.startswith("Temporal seated challenge")]
    chk = {"cache": cache_name, "calibration_measurements": strip(cal_meas[cal_meas.pid.isin(prim.pid)].measurement.astype(str)),
           "calibration_measurements_other_subjects": strip(cal_meas[~cal_meas.pid.isin(prim.pid)].measurement.astype(str)),
           "after_period_readings_excluded_le_24h": int((after_all.hours <= 24).sum()),
           "calibration_readings_per_subject": prim.drop_duplicates("pid").n_cal.value_counts().to_dict(),
           "initial_auscultatory_targets_not_used": strip(other_initial),
           "subjects_primary": int(prim.pid.nunique()), "readings_primary": int(len(prim)),
           "readings_per_test": prim.test.value_counts().to_dict(),
           "after_period_hours_median": float(prim.loc[prim.test == "after", "hours"].median()),
           "after_period_hours_min": float(prim.loc[prim.test == "after", "hours"].min()),
           "methods": sorted(preds)}
    print("checks:", json.dumps(chk, indent=1, default=str), flush=True)
    for k, p in preds.items():
        missing = set(prim.sample_id) - set(p.index)
        assert not missing, f"{k}: {len(missing)} primary readings without prediction"

    def subset(name, t):
        if name == "static":
            return prim[prim.test == "static"]
        if name == "change":                                         # challenge readings = "All BP change measurements"
            return prim[prim.test == "change"]
        if name == "static_plus_seated":                             # sensitivity: + unused "Seated calibration" readings
            return pd.concat([prim[prim.test == "static"], seated_extra])
        if name == "after":
            return prim[prim.test == "after"]
        if name == "overall":
            return prim
        raise KeyError(name)

    res = {}
    for key, p in preds.items():
        calfree = key in dict(CAL_FREE)
        r = {}
        for t in T:
            for sname in (["static"] if calfree else ["overall", "static", "static_plus_seated", "change", "after"]):
                d = subset(sname, t)
                e = p.loc[d.sample_id, f"{t}_pred"].to_numpy() - d[t].to_numpy()
                r[f"{t}/{sname}"] = metrics(e, d[t], d.pid)
                if sname == "change":
                    ch = d[f"{t}_change"].to_numpy()
                    for (lo, hi), lab in zip(BINS, BIN_LAB):
                        mk = (ch >= lo) & ((ch < hi) if hi != 25 else (ch <= hi))
                        if mk.sum() >= 2:
                            r[f"{t}/change/{lab}"] = metrics(e[mk], d[t].to_numpy()[mk], d.pid.to_numpy()[mk])
                    for lab, mk in [("< −20", ch < -20), ("> 25", ch > 25)]:
                        if mk.sum() >= 2:
                            r[f"{t}/change/{lab}"] = metrics(e[mk], d[t].to_numpy()[mk], d.pid.to_numpy()[mk])
                    nb = ~((ch >= -5) & (ch < 5))
                    r[f"{t}/change/|Δ| ≥ 5 only"] = metrics(e[nb], d[t].to_numpy()[nb], d.pid.to_numpy()[nb])
            if not calfree:
                d = subset("overall", t)
                e = p.loc[d.sample_id, f"{t}_pred"].to_numpy() - d[t].to_numpy()
                for cl in ["Normal", "Elevated", "Stage 1 hypertension", "Stage 2 hypertension"]:
                    mk = (d.entry_class == cl).to_numpy()
                    if mk.sum() >= 2:
                        r[f"{t}/level/{cl}"] = metrics(e[mk], d[t].to_numpy()[mk], d.pid.to_numpy()[mk])
        g = lambda keys: worst([r[k]["grade"] for k in keys if k in r])
        r["g_static"] = g([f"{t}/static" for t in T])
        if calfree:
            r["grade_final"] = r["g_static"]
        else:
            r["g_overall"] = g([f"{t}/overall" for t in T])
            r["g_change"] = g([f"{t}/change" for t in T])
            lv = [f"{t}/change/{b}" for t in T for b in STD_LEVELS if f"{t}/change/{b}" in r]
            r["g_levels"] = g(lv)
            r["worst_level"] = max(lv, key=lambda k: r[k]["MAD"])
            r["g_after"] = g([f"{t}/after" for t in T])
            r["grade_final"] = worst([r["g_overall"], r["g_static"], r["g_change"], r["g_levels"], r["g_after"]])
            r["levels_pass"] = all(r[f"{t}/level/{cl}"]["MAD"] <= 6 for t in T for cl in ["Normal", "Elevated", "Stage 1 hypertension"]
                                   if f"{t}/level/{cl}" in r)
        res[key] = r
        print(f"{key:12s} final grade {r['grade_final']}  (" + ", ".join(f"{k[2:]} {r[k]}" for k in
              ["g_overall", "g_static", "g_change", "g_levels", "g_after"] if k in r) + ")", flush=True)

    # ---------------- STD_Table_P1: accuracy report (Table 9 layout), ours v1 (and no-time)
    def rows(r, t, keys):
        out = []
        for lab, k in keys:
            if k in r:
                x = r[k]
                out.append(f"| {t.upper()} | {lab} | {x['n']:,} ({x['subjects']}) | {x['MAD']:.2f} ({x['MAD_CI'][0]:.2f}–{x['MAD_CI'][1]:.2f}) | "
                           f"{x['MAD_U90']:.2f} | {x['MAPD']:.1f} | {x['MD']:+.2f} | {x['SD']:.2f} | {x['CP5']:.0f} | {x['CP10']:.0f} | {x['CP15']:.0f} | {x['grade']} |")
        return out
    head = ["| Target | Test / subset | Readings (subjects) | MAD (95 % CI) | MAD 90 % upper bound | MAPD (%) | MD | SD | CP5 (%) | CP10 (%) | CP15 (%) | Grade |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    L = ["# STD_Table_P1 — Accuracy report following IEEE Std 1708-2025 (Table 9 layout)", "",
         "Aurora-BP test subjects, auscultatory protocol (reference by trained observers), 5-fold CV pooled; k = 3 calibration readings. "
         "Grades per IEEE Std 1708-2025 Table 6 (MAD with |MD| ≤ 5 mmHg for grades B and C). BP change = reference − calibration BP "
         "(mean of the calibration readings). 95 % CI and 90 % upper bound of MAD: subject-level bootstrap (1,000). Evaluated following "
         "the procedures and grading criteria of the standard; not a formal validation study (STD_Table_P9).", ""]
    for key in ["ours", "ours_time"]:
        if key not in res:
            continue
        r = res[key]
        L += [f"## {METHOD[key]['label']} — final grade {r['grade_final']}", "",
              f"Grades: overall accuracy {r['g_overall']}; static test {r['g_static']}; all BP-change measurements {r['g_change']}; "
              f"worst BP-change level {r['g_levels']} ({r['worst_level'].replace('/change/', ' ').upper().replace('…', '…')} mmHg); "
              f"after > 24 h {r['g_after']}. BP-level criterion (MAD ≤ 6 except stage 2): {'met' if r['levels_pass'] else 'not met'}.", ""] + head
        for t in T:
            keys = [("Overall accuracy (§4.5.3.1)", f"{t}/overall"), ("Static test (§4.4.2)", f"{t}/static"),
                    ("Static test + unused seated calibration readings (sensitivity)", f"{t}/static_plus_seated")]
            keys += [(f"BP change {lab} mmHg (§4.5.3.2)" + ("" if lab in STD_LEVELS else ", not graded"), f"{t}/change/{lab}")
                     for lab in ["< −20"] + BIN_LAB + ["> 25"]]
            keys += [("All BP change measurements (§4.5.3.2)", f"{t}/change"), ("BP change, |Δ| ≥ 5 only (supplementary)", f"{t}/change/|Δ| ≥ 5 only"),
                     ("After a certain period, > 24 h (§4.5.3.3)", f"{t}/after")]
            keys += [(f"BP level: {cl} (§4.5.3.4)", f"{t}/level/{cl}") for cl in ["Normal", "Elevated", "Stage 1 hypertension", "Stage 2 hypertension"]]
            L += rows(r, t, keys)
        L.append("")
    (SEC / "tables/STD_Table_P1_accuracy_report.md").write_text("\n".join(L))

    # ---------------- STD_Table_P2: grades of all methods
    L2 = ["# STD_Table_P2 — Grades following IEEE Std 1708-2025 (Table 6), all methods", "",
          "Calibrated methods (personalized calibration): static test, test with BP change, test after a certain period (Table 4, row 1). "
          "Calibration-free methods: static test only (Table 4, row 6). Cells: MAD SBP / DBP in mmHg (grade = worse of the two). Final grade "
          "= worst over all columns (§4.5.3.1: the grade depends on the maximum MAD; §4.5.3.2–4.5.3.3: the criteria apply at every BP-change "
          "level and after the period).", "",
          "| Method | Group | Overall | Static | All BP change | Worst BP-change level | After > 24 h | Final grade | BP-level criterion |",
          "|---|---|---|---|---|---|---|---|---|"]
    cell = lambda r, s: f"{r['sbp/' + s]['MAD']:.2f} / {r['dbp/' + s]['MAD']:.2f} ({worst([r['sbp/' + s]['grade'], r['dbp/' + s]['grade']])})"
    for key, r in res.items():
        cf = key in dict(CAL_FREE)
        grp = "calibration-free" if cf else ("reference" if key == "cal_only" else "calibrated")
        if cf:
            L2.append(f"| {METHOD[key]['label']} | {grp} | – | {cell(r, 'static')} | – | – | – | **{r['grade_final']}** | – |")
        else:
            wl = r["worst_level"]
            L2.append(f"| {METHOD[key]['label']} | {grp} | {cell(r, 'overall')} | {cell(r, 'static')} | {cell(r, 'change')} | "
                      f"{wl.split('/')[0].upper()} {wl.split('/')[-1]}: {r[wl]['MAD']:.2f} ({r['g_levels']}) | {cell(r, 'after')} | "
                      f"**{r['grade_final']}** | {'met' if r['levels_pass'] else 'not met'} |")
    (SEC / "tables/STD_Table_P2_grades.md").write_text("\n".join(L2))

    # ---------------- STD_Table_P3: induced-change report (Table 13) vs requirements (Table 5)
    d = subset("change", None)
    L3 = ["# STD_Table_P3 — Induced BP change report (IEEE Std 1708-2025 Tables 13 and 5)", "",
          "BP-change test readings (initial visit, auscultatory challenge measurements: Static challenge start 1–3, Exercise challenge "
          "start 1–2); change = reference − calibration BP. Requirement: minimum share of readings and subjects per bin from Table 5.", "",
          "| Target | Bin (mmHg) | Readings | Share | Subjects | Requirement (share, subjects) | Met |", "|---|---|---|---|---|---|---|"]
    p3 = {}
    for t in T:
        ch = d[f"{t}_change"].to_numpy()
        for (lo, hi), lab in zip(BINS, BIN_LAB):
            mk = (ch >= lo) & ((ch < hi) if hi != 25 else (ch <= hi))
            share, ns = mk.mean(), d.pid[mk].nunique()
            rs, rn = REQ[lab]
            met = "–" if rs == 0 else ("yes" if share >= rs and ns >= rn else "no")
            req_txt = "–" if rs == 0 else f"≥ {rs:.0%}, ≥ {rn}"
            p3[f"{t}/{lab}"] = {"n": int(mk.sum()), "share": float(share), "subjects": int(ns), "met": met}
            L3.append(f"| {t.upper()} | {lab} | {mk.sum():,} | {share:.1%} | {ns} | {req_txt} | {met} |")
        for lab, mk in [("< −20", ch < -20), ("> 25", ch > 25)]:
            L3.append(f"| {t.upper()} | {lab} (outside the bins) | {mk.sum():,} | {mk.mean():.1%} | {d.pid[mk].nunique()} | – | – |")
    (SEC / "tables/STD_Table_P3_induced_change_report.md").write_text("\n".join(L3))

    # ---------------- STD_Table_P10: within-subject error correlation, required subjects, error distribution (ours v1)
    L10 = ["# STD_Table_P10 — Within-subject error correlation, required number of subjects and error distribution (HiCAF-BP-Cal)", "",
           "ρ: one-way random-effects ICC of the errors over the static-test triplets; required subjects n = max(85, ceil(42.5(1 + 2ρ))) "
           "(IEEE Std 1708-2025 §4.5.3, Annex E.4). Error distribution (Annex A): Kolmogorov–Smirnov distance D to a normal and to a t4 "
           "distribution with the observed MD and SD (smaller = closer; p-values not reported because parameters are estimated and n is "
           "large); SD/MAD expected ≈ 1.25 (normal) or ≈ 1.41 (t4).", "",
           "| Target | Set | ρ | Required subjects | Available subjects | KS D normal | KS D t4 | Closer to | SD/MAD |", "|---|---|---|---|---|---|---|---|---|"]
    p10 = {}
    for t in T:
        for sname in ["static", "overall"]:
            dd = subset(sname, t)
            e = preds["ours"].loc[dd.sample_id, f"{t}_pred"].to_numpy() - dd[t].to_numpy()
            rho = icc1(e, dd.pid.to_numpy()) if sname == "static" else float("nan")
            nreq = max(85, math.ceil(42.5 * (1 + 2 * rho))) if sname == "static" else None
            md, sd = e.mean(), e.std(ddof=1)
            ks_n = stats.kstest(e, stats.norm(md, sd).cdf).statistic
            ks_t = stats.kstest(e, stats.t(4, md, sd / math.sqrt(2)).cdf).statistic
            ratio = sd / np.abs(e).mean()
            p10[f"{t}/{sname}"] = {"rho": rho, "n_required": nreq, "subjects": int(dd.pid.nunique()), "ks_normal": float(ks_n),
                                   "ks_t4": float(ks_t), "sd_mad": float(ratio)}
            L10.append(f"| {t.upper()} | {sname} | {'' if np.isnan(rho) else f'{rho:.2f}'} | {nreq or ''} | {dd.pid.nunique()} | {ks_n:.3f} | "
                       f"{ks_t:.3f} | {'t4' if ks_t < ks_n else 'normal'} | {ratio:.2f} |")
    (SEC / "tables/STD_Table_P10_error_correlation_distribution.md").write_text("\n".join(L10))
    # ---------------- EXT_Table_P4: repeated-measures Bland-Altman, proportional bias, four-quadrant concordance (P2, P3)
    calm = [k for k in ["ours", "ours_time", "hgctnet", "bpbls", "ragcbpnet"] if k in preds]
    p4 = {}
    L4 = ["# EXT_Table_P4 — Agreement (Bland–Altman for repeated measurements) and tracking of BP change (concordance)", "",
          "Primary set (auscultatory). Bland–Altman: Bland & Altman (2007), multiple readings per subject with varying true value; "
          "SD from between- and within-subject variance components; 95 % CIs by subject bootstrap (1,000). Proportional bias: slope of "
          "the difference on the mean of estimate and reference (subject-cluster bootstrap CI). Concordance: share of readings in which "
          "the estimated change (estimate − calibration BP) has the sign of the reference change, readings with |reference change| "
          "< 5 mmHg excluded (the standard's −5…5 bin).", "",
          "## Agreement", "",
          "| Method | Target | Set | Readings (subjects) | Bias (95 % CI) | SD | 95 % limits of agreement (CI of each limit) | Proportional bias, slope (95 % CI) |",
          "|---|---|---|---|---|---|---|---|"]
    sets = [("static", "Static"), ("change", "BP change"), ("after", "After > 24 h"), ("overall", "All primary")]
    for k in calm:
        for t in T:
            for sn, sl in sets:
                if k != "ours" and sn != "overall":
                    continue
                d = subset(sn, t)
                est = preds[k].loc[d.sample_id, f"{t}_pred"].to_numpy()
                ref = d[t].to_numpy()
                e = est - ref
                ba = ba_repeated(e, d.pid.to_numpy())
                sl_ = slope_cluster(e, (est + ref) / 2, d.pid.to_numpy())
                p4[f"{k}/{t}/{sn}/ba"] = {**ba, "slope": sl_}
                L4.append(f"| {METHOD[k]['label']} | {t.upper()} | {sl} | {len(d):,} ({d.pid.nunique()}) | {ba['bias'][0]:+.2f} ({ba['bias'][1]:+.2f} to "
                          f"{ba['bias'][2]:+.2f}) | {ba['sd'][0]:.2f} | {ba['loa_low'][0]:+.1f} ({ba['loa_low'][1]:+.1f} to {ba['loa_low'][2]:+.1f}) to "
                          f"{ba['loa_high'][0]:+.1f} ({ba['loa_high'][1]:+.1f} to {ba['loa_high'][2]:+.1f}) | {sl_[0]:+.3f} ({sl_[1]:+.3f} to {sl_[2]:+.3f}) |")
    L4 += ["", "## Concordance of BP change (all primary readings with |reference change| ≥ 5 mmHg)", "",
           "| Method | Target | 5–10 mmHg | 10–15 mmHg | ≥ 15 mmHg | All ≥ 5 mmHg (95 % CI) |", "|---|---|---|---|---|---|"]
    for k in calm:
        for t in T:
            d = prim
            rc = d[f"{t}_change"].to_numpy()
            ec = preds[k].loc[d.sample_id, f"{t}_pred"].to_numpy() - d[f"{t}_cal"].to_numpy()
            hit = np.sign(ec) == np.sign(rc)
            cells = []
            for lo_, hi_ in [(5, 10), (10, 15), (15, np.inf)]:
                mk = (np.abs(rc) >= lo_) & (np.abs(rc) < hi_)
                cells.append(f"{hit[mk].mean():.0%} (n = {mk.sum():,})" if mk.any() else "–")
            mk = np.abs(rc) >= 5
            r_ = ratio_ci(hit[mk], d.pid.to_numpy()[mk])
            p4[f"{k}/{t}/concordance"] = {"all": r_, "n": int(mk.sum())}
            L4.append(f"| {METHOD[k]['label']} | {t.upper()} | " + " | ".join(cells) + f" | {r_[0]:.1%} ({r_[1]:.1%}–{r_[2]:.1%}) |")
    (SEC / "tables/EXT_Table_P4_agreement_concordance.md").write_text("\n".join(L4))

    # ---------------- EXT_Table_P6: repeatability and reproducibility (P6)
    p6 = {}
    L6 = ["# EXT_Table_P6 — Repeatability and reproducibility", "",
          "Repeatability: pooled within-subject SD over the static-test triplets (initial visit, seated rest) of the estimates, the "
          "reference readings and the errors. Between sessions: per-subject MAD in the static test (initial visit) vs the after-period "
          "test (return visit, > 24 h), subjects with both; Wilcoxon signed-rank test. Between reference protocols: in-lab readings with "
          "auscultatory vs oscillometric reference (different subjects; descriptive).", "",
          "## Repeatability (static-test triplets)", "",
          "| Method | Target | Within-subject SD of estimates | of reference | of errors | Subjects |", "|---|---|---|---|---|---|"]
    d = subset("static", None)
    for k in calm:
        for t in T:
            est = preds[k].loc[d.sample_id, f"{t}_pred"].to_numpy()
            r_ = (within_sd(est, d.pid.to_numpy()), within_sd(d[t].to_numpy(), d.pid.to_numpy()), within_sd(est - d[t].to_numpy(), d.pid.to_numpy()))
            p6[f"{k}/{t}/repeatability"] = r_
            L6.append(f"| {METHOD[k]['label']} | {t.upper()} | {r_[0]:.2f} | {r_[1]:.2f} | {r_[2]:.2f} | {d.pid.nunique()} |")
    L6 += ["", "## Reproducibility between sessions (per-subject MAD)", "",
           "| Method | Target | Subjects | Median MAD initial (static) | Median MAD return (> 24 h) | Median paired difference | Wilcoxon p | Spearman ρ of subject MADs |",
           "|---|---|---|---|---|---|---|---|"]
    for k in calm:
        for t in T:
            per = {}
            for sn in ["static", "after"]:
                dd = subset(sn, t)
                per[sn] = pd.Series(np.abs(preds[k].loc[dd.sample_id, f"{t}_pred"].to_numpy() - dd[t].to_numpy())).groupby(dd.pid.to_numpy()).mean()
            j = pd.concat(per, axis=1).dropna()
            w = stats.wilcoxon(j.static, j.after)
            rho = stats.spearmanr(j.static, j.after)[0]
            p6[f"{k}/{t}/sessions"] = {"n": len(j), "median_initial": float(j.static.median()), "median_return": float(j.after.median()),
                                      "median_diff": float((j.after - j.static).median()), "p": float(w.pvalue), "rho": float(rho)}
            L6.append(f"| {METHOD[k]['label']} | {t.upper()} | {len(j)} | {j.static.median():.2f} | {j.after.median():.2f} | "
                      f"{(j.after - j.static).median():+.2f} | {w.pvalue:.2g} | {rho:.2f} |")
    osc_lab = tg[(tg.protocol == "oscillometric") & tg.phase.isin(["initial", "return"])]
    L6 += ["", "## Between reference protocols (in-lab readings)", "",
           "| Method | Target | Auscultatory MAD (readings, subjects) | Oscillometric MAD (readings, subjects) |", "|---|---|---|---|"]
    for k in calm:
        for t in T:
            a_ = np.abs(preds[k].loc[prim.sample_id, f"{t}_pred"].to_numpy() - prim[t].to_numpy()).mean()
            if len(osc_lab):
                o_ = np.abs(preds[k].loc[osc_lab.sample_id, f"{t}_pred"].to_numpy() - osc_lab[t].to_numpy()).mean()
                ocell = f"{o_:.2f} ({len(osc_lab):,}, {osc_lab.pid.nunique()})"
            else:
                o_, ocell = float("nan"), "–"
            p6[f"{k}/{t}/protocols"] = (float(a_), float(o_))
            L6.append(f"| {METHOD[k]['label']} | {t.upper()} | {a_:.2f} ({len(prim):,}, {prim.pid.nunique()}) | {ocell} |")
    L6 += model_instances(p6)
    (SEC / "tables/EXT_Table_P6_repeatability_reproducibility.md").write_text("\n".join(L6))

    # ---------------- EXT_Table_P8: free-living / oscillometric extension (not graded)
    p8 = {}
    L8 = ["# EXT_Table_P8 — Oscillometric protocol and free-living readings (extension, not graded)", "",
          "Same metrics as STD_Table_P1; not graded, because IEEE Std 1708-2025 recognises no reference method for dynamic, "
          "unsupervised conditions (Annex D). Ambulatory = home readings with an oscillometric cuff; in-lab = oscillometric readings of "
          "the initial and return visits.", "",
          "| Method | Target | Set | Readings (subjects) | MAD (95 % CI) | MAPD (%) | MD | SD | CP5 (%) | CP10 (%) | CP15 (%) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    osc = tg[(tg.protocol == "oscillometric") & ~tg.sample_id.isin(set(tg.sample_id[tg.test != "other"]))]
    osc_sets = [("Ambulatory (home)", osc[osc.phase == "ambulatory"]), ("In-lab, initial", osc[osc.phase == "initial"]),
                ("In-lab, return", osc[osc.phase == "return"])]
    for k in calm + ["cal_only"]:
        for t in T:
            for sl, dd in osc_sets:
                if len(dd) < 2:
                    continue
                e = preds[k].loc[dd.sample_id, f"{t}_pred"].to_numpy() - dd[t].to_numpy()
                x = metrics(e, dd[t], dd.pid)
                p8[f"{k}/{t}/{sl}"] = x
                L8.append(f"| {METHOD[k]['label']} | {t.upper()} | {sl} | {x['n']:,} ({x['subjects']}) | {x['MAD']:.2f} ({x['MAD_CI'][0]:.2f}–"
                          f"{x['MAD_CI'][1]:.2f}) | {x['MAPD']:.1f} | {x['MD']:+.2f} | {x['SD']:.2f} | {x['CP5']:.0f} | {x['CP10']:.0f} | {x['CP15']:.0f} |")
    (SEC / "tables/EXT_Table_P8_free_living.md").write_text("\n".join(L8))

    # ---------------- STD_Table_P9: deviations from the procedure of the standard
    unmet = [f"{t.upper()} {b}" for t in T for b in STD_LEVELS if p3[f"{t}/{b}"]["met"] == "no"]
    after_h = prim.loc[prim.test == "after", "hours"]
    L9 = ["# STD_Table_P9 — Deviations from the procedure of IEEE Std 1708-2025", "",
          "This is a retrospective evaluation of an estimation method on an existing data set (Aurora-BP), following the procedures and "
          "grading criteria of the standard; it is not a formal validation of a device.", "",
          "| Item | Standard | This evaluation | Expected effect |", "|---|---|---|---|",
          "| Study design | Prospective, device-specific protocol | Retrospective, existing data set, 5-fold subject-wise cross-validation | None on the metrics; independence of test subjects preserved |",
          f"| Subjects | ≥ 85, age 21–50, BP-class quotas | {prim.pid.nunique()} auscultatory test subjects; wider age range; no BP-class quotas | Larger, more heterogeneous sample |",
          "| Reference | Auscultation by two trained observers; mean of adjacent reference readings (Eq. 1) | One consensus auscultatory reading per measurement as provided by Aurora-BP | Reference uncertainty quantified in GUM_Table_P5 |",
          f"| Calibration | Device-specific | Mean of the model's calibration readings: "
          + ", ".join(f"{k} ({v:,} readings)" for k, v in chk["calibration_measurements"].items())
          + f"; {sum(v for k, v in chk['calibration_readings_per_subject'].items() if int(k) < 3)} subjects with fewer than 3 | Entry BP class and BP change refer to this calibration |",
          f"| Unused readings | – | {len(seated_extra):,} initial-visit \"Seated calibration\" readings that were not calibration readings belong to "
          "no test of the standard; reported as a sensitivity row of the static test (STD_Table_P1) | None on the grades |",
          "| Grading | §4.5.3.1–4.5.3.3 | Final grade = worst over SBP/DBP and over overall accuracy, static test, all BP-change measurements, "
          "each BP-change level of Table 9 and the test after the period | Strict reading; levels with few readings can decide the grade |",
          f"| BP-change induction | Protocols of Annex C | Posture and exercise challenges of Aurora-BP; Table 5 requirement not met for: {', '.join(unmet) or 'none'} (STD_Table_P3) | Few large decreases; grades of the BP-change test rest on the available distribution |",
          f"| Test after a certain period | > 24 h, ≥ 45 subjects | Return visit, median {after_h.median():.0f} h (min {after_h.min():.0f} h), {prim.loc[prim.test == 'after'].pid.nunique()} subjects | – |",
          "| Hardware clauses | Wearable sensor, communication, safety, stability (§§5, 7–9) | Not applicable to an estimation method | Out of scope |",
          "| Noise robustness | §6.2 | Simulated noise on the test recording (STD_Table_P7, STD_Fig_P3) | – |"]
    (SEC / "tables/STD_Table_P9_deviations.md").write_text("\n".join(L9))

    json.dump({"checks": chk, "results": res, "induced_change": p3, "error_corr_dist": p10, "agreement": p4, "repeatability": p6,
               "free_living": p8}, open(SEC / "tables/STD_P1_results.json", "w"), indent=1, default=float)
    for f in ["STD_Table_P1_accuracy_report", "STD_Table_P2_grades", "STD_Table_P3_induced_change_report", "STD_Table_P10_error_correlation_distribution",
              "EXT_Table_P4_agreement_concordance", "EXT_Table_P6_repeatability_reproducibility", "EXT_Table_P8_free_living", "STD_Table_P9_deviations"]:
        print("\n" + (SEC / f"tables/{f}.md").read_text())

    # ---------------- STD_Fig_P1: graphical report (§4.6.3), one figure per target
    setup()
    col = {"static": "#0072B2", "change": "#E69F00", "after": "#009E73"}
    lab = {"static": "Static test", "change": "BP-change test", "after": "After > 24 h"}
    for t in T:
        dd = prim
        est = preds["ours"].loc[dd.sample_id, f"{t}_pred"].to_numpy()
        ref = dd[t].to_numpy()
        e, ch = est - ref, dd[f"{t}_change"].to_numpy()
        fig, ax = figure("double", height=4.6, nrows=2, ncols=2)
        ax = ax.ravel()
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(dd))
        cs = np.array([col[x] for x in dd.test.to_numpy()])[order]
        ax[0].scatter(((est + ref) / 2)[order], e[order], s=2, lw=0, c=cs, alpha=0.5, rasterized=True)
        md, sd = e.mean(), e.std(ddof=1)
        for v, ls in [(md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")]:
            ax[0].axhline(v, color="black", lw=0.7, ls=ls)
        ax[0].text(0.99, 0.97, f"MD {md:+.1f}, LoA {md - 1.96 * sd:+.1f} to {md + 1.96 * sd:+.1f} mmHg".replace("-", "−"), transform=ax[0].transAxes,
                   ha="right", va="top", fontsize=6, bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax[0].set(xlabel=f"Mean of estimate and reference {TARGET[t]['label']} (mmHg)", ylabel="Estimate − reference (mmHg)")
        ax[1].scatter(ch[order], e[order], s=2, lw=0, c=cs, alpha=0.5, rasterized=True)
        for b in [-20, -10, -5, 5, 15, 25]:
            ax[1].axvline(b, color="#B0B0B0", lw=0.5, ls=":")
        ax[1].axhline(0, color="black", lw=0.7)
        ax[1].set(xlabel=f"Reference {TARGET[t]['label']} change from calibration (mmHg)", ylabel="Estimate − reference (mmHg)")
        chc = ch[(dd.test == "change").to_numpy()]                       # standard's Fig. 6: BP-change test readings only (= STD_Table_P3)
        lo, hi = np.floor(np.percentile(chc, 0.5)), np.ceil(np.percentile(chc, 99.5))
        ax[2].hist(chc, bins=np.arange(lo, hi + 2.5, 2.5), color=col["change"], alpha=0.85)
        for (a_, b_), l_ in zip(BINS, BIN_LAB):
            if l_ != "−5…5":
                ax[2].axvspan(a_, b_, color="#E8E8E8", zorder=0)
        ax[2].set(xlabel=f"Reference {TARGET[t]['label']} change from calibration (mmHg)", ylabel="Readings (BP-change test)")
        ax[2].text(0.99, 0.97, "shaded: graded BP-change levels", transform=ax[2].transAxes, ha="right", va="top", fontsize=6,
                   bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))
        lim = [min(est.min(), ref.min()) - 5, max(est.max(), ref.max()) + 5]
        ax[3].scatter(ref[order], est[order], s=2, lw=0, c=cs, alpha=0.5, rasterized=True)
        ax[3].plot(lim, lim, color="black", lw=0.7)
        ax[3].set(xlim=lim, ylim=lim, xlabel=f"Reference {TARGET[t]['label']} (mmHg)", ylabel=f"Estimated {TARGET[t]['label']} (mmHg)")
        for k_ in ["static", "change", "after"]:
            ax[3].scatter([], [], s=10, color=col[k_], label=lab[k_])
        ax[3].legend(loc="upper left", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none")
        panel_labels(ax)
        save(fig, f"STD_Fig_P1_graphical_report_{t.upper()}", source=__file__, outdir=SEC / "figures", png_subdir="png")

    # ---------------- STD_Fig_P2: MAD per IEEE test with grade bands, calibrated methods
    cats = [("Overall", "overall"), ("Static", "static"), *[(f"Δ {l_}", f"change/{l_}") for l_ in STD_LEVELS], ("All Δ", "change"),
            ("After > 24 h", "after"),
            ("Normal", "level/Normal"), ("Elevated", "level/Elevated"), ("Stage 1", "level/Stage 1 hypertension"), ("Stage 2", "level/Stage 2 hypertension")]
    meths = [k for k in ["ours", "hgctnet", "bpbls", "ragcbpnet", "cal_only"] if k in res]
    CAP = 12.0                                                            # y axis cap (mmHg); grade bands stay readable
    fig, axes = figure("double", height=2.8, ncols=2, sharey=True)
    x = np.arange(len(cats))
    for ax_, t in zip(axes, T):
        top = min(CAP, max(res[k][f"{t}/{c}"]["MAD"] for k in meths for _, c in cats if f"{t}/{c}" in res[k]) * 1.08)
        for lo_, hi_, g in [(0, 5, "A"), (5, 6, "B"), (6, 7, "C"), (7, max(top, 8), "D")]:
            ax_.axhspan(lo_, hi_, color=GRADE_FILL[g], zorder=0, lw=0)
            ax_.text(len(cats) - 0.45, (lo_ + min(hi_, top)) / 2, g, fontsize=7, fontweight="bold", va="center", color="#606060")
        off = np.linspace(-0.3, 0.3, len(meths))
        for o, k in zip(off, meths):
            ys = np.array([res[k][f"{t}/{c}"]["MAD"] if f"{t}/{c}" in res[k] else np.nan for _, c in cats])
            st = METHOD[k]
            for xi, y in zip(x + o, ys):                                  # off-scale values: arrow at the top with the value
                if y > top:
                    ax_.annotate("", (xi, top - 0.05), xytext=(xi, top - 0.9),
                                 arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=0.9, mutation_scale=6))
                    if k == "ours":                                       # our value printed; comparators in STD_Table_P2
                        ax_.text(xi, top - 1.0, f"{y:.1f}", ha="center", va="top", fontsize=5.5, color=st["color"], fontweight="bold")
            ys = np.where(ys > top, np.nan, ys)
            if k == "ours":
                lo_ci = [y - res[k][f"{t}/{c}"]["MAD_CI"][0] if f"{t}/{c}" in res[k] else 0 for y, (_, c) in zip(ys, cats)]
                hi_ci = [res[k][f"{t}/{c}"]["MAD_CI"][1] - y if f"{t}/{c}" in res[k] else 0 for y, (_, c) in zip(ys, cats)]
                ax_.errorbar(x + o, ys, yerr=[np.nan_to_num(lo_ci), np.nan_to_num(hi_ci)], fmt=st["marker"], color=st["color"], ms=4, lw=0.8, label=st["label"], zorder=3)
            else:
                ax_.plot(x + o, ys, st["marker"] or "_", color=st["color"], ms=4 if st["marker"] else 8, ls="", label=st["label"], zorder=3)
        for xs in (1.5, 2 + len(STD_LEVELS) + 0.5, 2 + len(STD_LEVELS) + 1.5):   # | overall, static | BP change | after | levels
            ax_.axvline(xs, color="#B0B0B0", lw=0.5)
        ax_.set(xticks=x, xlim=(-0.6, len(cats) - 0.2), ylim=(0, top), ylabel=f"{TARGET[t]['label']} MAD (mmHg)")
        ax_.set_xticklabels([c for c, _ in cats], rotation=45, ha="right", fontsize=6)
        ax_.grid(axis="x", visible=False)
        ax_.text(0.015, 0.02, f"↑ MAD > {CAP:.0f} mmHg (off scale; ours labelled, all values in STD_Table_P2)", transform=ax_.transAxes, ha="left", va="bottom", fontsize=5.5,
                 color="#404040")
    h, l_ = axes[0].get_legend_handles_labels()
    order = sorted(range(len(l_)), key=lambda i: [METHOD[k]["label"] for k in meths].index(l_[i]))
    h, l_ = [h[i] for i in order], [l_[i] for i in order]
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(meths), fontsize=6)
    panel_labels(axes, x=-0.12)
    save(fig, "STD_Fig_P2_accuracy_by_test", source=__file__, outdir=SEC / "figures", png_subdir="png")

    # ---------------- README status
    rd = SEC / "README.md"
    if rd.exists():
        s = rd.read_text()
        for name in ["STD_Fig_P1_graphical_report", "STD_Fig_P2_accuracy_by_test", "STD_Table_P1_accuracy_report", "STD_Table_P2_grades",
                     "STD_Table_P3_induced_change_report", "STD_Table_P10_error_correlation_distribution", "EXT_Table_P4_agreement_concordance",
                     "EXT_Table_P6_repeatability_reproducibility", "EXT_Table_P8_free_living", "STD_Table_P9_deviations"]:
            s = "\n".join(l.replace("| pending |", f"| done {time.strftime('%Y-%m-%d')} |") if l.startswith(f"| {name} ") else l for l in s.split("\n"))
        rd.write_text(s)
    print(f"\nruntime {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()