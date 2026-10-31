#!/usr/bin/env python
"""
Measurement performance evaluation, P4 - measurement uncertainty following the GUM (JCGM 100:2008)
(docs/PERFORMANCE_EVALUATION_PLAN.md, section 7, change log v1.2).

  python scripts/perf_P4_uncertainty.py --stage infer  --gpu 2     # GPU, ~minutes: calibration sensitivity on the frozen v1 models
  python scripts/perf_P4_uncertainty.py --stage budget --raw_tsv <Aurora measurements_auscultatory.tsv>   # CPU: budget, table, figure

Measurement model (one reading, k = 3 calibration readings c_1..c_3 of the same subject):
    y = (1/3) sum_i [ x_cal,i + g(S_t, S_c,i, x_cal,i, ctx) ]          (HiCAF-BP-Cal: calibration BP + learned change)
Sensitivity coefficient of the calibration BP, c_cal = dy/dx_cal: all calibration BPs of the reading shifted together by
+/- 5 mmHg (central difference; recordings unchanged), SBP->SBP, DBP->DBP and the cross terms. c_cal = 1 means a calibration
error passes 1:1 into the estimate; < 1 means the model partly absorbs it.
Budget per IEEE test (static, BP change, after > 24 h, all) and target, Type A evaluations:
    u(ref)    reference reading: inter-observer differences of the two auscultatory observers (Aurora raw file), one-observer
              or two-observer-mean value chosen from the data; upper bound: within-subject SD of the static reference triplets
    u1        calibration term = |c| * u(x_cal), u(x_cal) = pooled within-subject SD of the calibration readings, propagated
              through the mean of n_cal estimates: sqrt(sum c_i^2) / n_cal * s_cal
    u_c       combined standard uncertainty of the estimate = sqrt(SD(err)^2 - u(ref)^2)    (err = estimate - reference)
    u2        remaining (signal / model) term = sqrt(u_c^2 - u1^2)
    U         expanded uncertainty, k = 2; k for 95 % under a t4 error distribution = t_0.975,4 / sqrt(2) = 1.96; empirical
              coverage of +/- k * sqrt(u_c^2 + u(ref)^2) on the observed errors reported alongside
Bias (MD) is reported, not corrected. The fold-model (epistemic) spread under domain shift is taken from EXT_Table_P6 (PulseDB)
and reported as a note; it is not part of the Aurora budget (each Aurora reading is estimated by one model only).
Outputs (paper/performance_evaluation/): tables/GUM_Table_P5_uncertainty_budget.md (+ GUM_P4_results.json),
figures/GUM_Fig_P4_uncertainty.pdf (+ png/). Intermediate: runs/performance/calsens_v1.csv.gz (refuses to overwrite).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
SEC = CODE_DIR / "paper/performance_evaluation"
INTER = CODE_DIR / "runs/performance"
T = ["sbp", "dbp"]
DELTA = 5.0                                                              # mmHg, central difference step


# ----------------------------------------------------------------------------------------------------------- GPU stage
def infer(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import BPCropDataset
    from src.data.pairs import PairDataset, calibration_split, eval_pairs, row_times
    from src.data.splits import assign_roles, load_metadata, target_stats
    from src.models import build_model

    INTER.mkdir(parents=True, exist_ok=True)
    fp = INTER / f"calsens_{args.tag}.csv.gz"
    if fp.exists():
        sys.exit(f"{fp} exists - frozen once built; delete it deliberately to rebuild")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = CODE_DIR / args.run
    rows, verify, t0 = [], [], time.time()
    for fold in range(5):
        ck = torch.load(run / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        cal_cfg = cfg["calibration"]
        meta, fs, channels = load_metadata(cfg)
        df = assign_roles(meta, cfg, fold)
        tr, d = df[df.role == "train"].reset_index(drop=True), df[df.role == "test"].reset_index(drop=True)
        targets = list(cfg["data"]["targets"])
        assert targets == ["sbp", "dbp"], targets
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        ym_chk, ys_chk = target_stats(tr, targets)
        assert np.allclose(ym_chk, y_mean, atol=1e-3) and np.allclose(ys_chk, y_std, atol=1e-3), "training statistics differ"
        signals = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r")
        ch_idx = [channels.index(c) for c in cfg["data"]["inputs"]]
        crop = int(round(cfg["data"]["crop_s"] * fs))
        model = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        assert model.demo_embed is None, "demographics branch not handled here"
        use_amp = bool(cfg["train"].get("amp_bf16", True)) and dev.type == "cuda"
        use_ctx = bool(cal_cfg.get("time_context", False))
        ysd, ymn = torch.tensor(y_std, device=dev), torch.tensor(y_mean, device=dev)

        def head(zt, zc, pt, pc, bp_cal, ctx):                          # identical to HiCAFBPCal.forward after encode
            feats = model._compare(zt, zc)
            if model.use_ppg:
                feats += model._compare(pt, pc)
            feats.append(model.bp_embed(bp_cal.to(zt.dtype)))
            if model.ctx_embed is not None:
                feats.append(model.ctx_embed(ctx.to(zt.dtype)))
            return bp_cal.float() + model.head(torch.cat(feats, dim=-1)).float()

        cal, tg = calibration_split(d, cal_cfg["phase"], cal_cfg["conditions"], int(cal_cfg["k_max"]))
        pairs = eval_pairs(d, cal, tg, args.k)
        times = row_times(d) if use_ctx else None
        base = BPCropDataset(d, signals, ch_idx, crop, y_mean, y_std, targets, [], None, None, train=False,
                             augment_cfg=cfg["data"].get("augment"))
        dl = DataLoader(PairDataset(base, pairs=pairs, times=times), batch_size=args.batch, shuffle=False, num_workers=8,
                        pin_memory=dev.type == "cuda")
        keys = ["base"] + [f"{t}{s}" for t in T for s in ("+", "-")]
        res = {k: [] for k in keys}
        with torch.no_grad():
            for b in dl:
                x, xc, bp_cal = b["x"].to(dev), b["x_cal"].to(dev), b["bp_cal"].to(dev).float()
                ctx = b["ctx"].to(dev) if "ctx" in b else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    z, zp = model.encoder.encode(torch.cat([x, xc], dim=0))
                    n = x.shape[0]
                    zt, zc = z[:n], z[n:]
                    pt, pc = (zp[:n], zp[n:]) if zp is not None else (None, None)
                    outs = {"base": head(zt, zc, pt, pc, bp_cal, ctx)}
                    for j, t in enumerate(T):
                        for s, sign in (("+", 1.0), ("-", -1.0)):
                            bp = bp_cal.clone()
                            bp[:, j] += sign * DELTA / y_std[j]          # +/- 5 mmHg on the standardised calibration BP
                            outs[f"{t}{s}"] = head(zt, zc, pt, pc, bp, ctx)
                for k_, v in outs.items():
                    res[k_].append((v.float() * ysd + ymn).cpu().numpy())
        res = {k_: np.concatenate(v) for k_, v in res.items()}
        pids = d["pid"].to_numpy()
        rank = np.array([int(np.where(cal[pids[t_]] == c_)[0][0]) for t_, c_ in pairs])
        fr = pd.DataFrame({"fold": fold, "sample_id": d["sample_id"].to_numpy()[pairs[:, 0]], "pid": pids[pairs[:, 0]],
                           "cal_sample_id": d["sample_id"].to_numpy()[pairs[:, 1]], "cal_rank": rank,
                           "sbp_cal": d["sbp"].to_numpy()[pairs[:, 1]], "dbp_cal": d["dbp"].to_numpy()[pairs[:, 1]],
                           "sbp_pred": res["base"][:, 0], "dbp_pred": res["base"][:, 1]})
        for j_in, t_in in enumerate(T):                                   # c[out <- in] = d est_out / d cal_in
            for j_out, t_out in enumerate(T):
                fr[f"c_{t_out}_{t_in}"] = (res[f"{t_in}+"][:, j_out] - res[f"{t_in}-"][:, j_out]) / (2 * DELTA)
                fr[f"curv_{t_out}_{t_in}"] = (res[f"{t_in}+"][:, j_out] + res[f"{t_in}-"][:, j_out] - 2 * res["base"][:, j_out])
        rows.append(fr)
        frozen_fp = run / f"fold_{fold}" / f"predictions_test_k{args.k}.csv"
        if frozen_fp.exists():
            frozen = pd.read_csv(frozen_fp).set_index("sample_id")
            mine = fr[fr.cal_rank < args.k].groupby("sample_id")[["sbp_pred", "dbp_pred"]].mean()
            common = mine.index.intersection(frozen.index)
            dif = np.abs(mine.loc[common].to_numpy() - frozen.loc[common, ["sbp_pred", "dbp_pred"]].to_numpy())
            verify.append({"fold": fold, "matched": len(common), "n_frozen": len(frozen), "max_abs_diff": float(dif.max()),
                           "mean_abs_diff": float(dif.mean())})
        print(f"fold {fold}: {len(pairs):,} pairs | {(time.time() - t0) / 60:.1f} min", flush=True)
    allp = pd.concat(rows, ignore_index=True)
    allp.to_csv(fp, index=False, compression="gzip")
    v = pd.DataFrame(verify)
    txt = ["# Calibration-sensitivity cache - verification against the frozen k = 3 predictions", "",
           v.round(4).to_string(index=False), "", f"pairs {len(allp):,}; runtime {(time.time() - t0) / 60:.1f} min"]
    (INTER / f"VERIFY_calsens_{args.tag}.md").write_text("\n".join(txt))
    print("\n".join(txt))


# ----------------------------------------------------------------------------------------------------------- CPU stage
def load_p1():
    spec = importlib.util.spec_from_file_location("perf_p1", CODE_DIR / "scripts/perf_P1_ieee1708.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pooled_within_sd(v, g):
    d = pd.DataFrame({"v": v, "g": g}).dropna()
    gg = d.groupby("g").v
    d = d[gg.transform("size") >= 2]
    gg = d.groupby("g").v
    return float(math.sqrt(((d.v - gg.transform("mean")) ** 2).sum() / (len(d) - gg.ngroups))), int(gg.ngroups)


def reference_uncertainty(raw_tsv, meta_fp):
    """Type A u(ref) from the two observers of each auscultatory reading (Aurora raw file)."""
    if not raw_tsv or not Path(raw_tsv).exists():
        return None
    r = pd.read_csv(raw_tsv, sep="\t" if str(raw_tsv).endswith(".tsv") else ",", low_memory=False)   # raw TSV or metadata CSV
    need = ["primary_systolic", "secondary_systolic", "primary_diastolic", "secondary_diastolic"]
    if not all(c in r.columns for c in need):
        return None
    out = {}
    for t, (p, s) in {"sbp": ("primary_systolic", "secondary_systolic"), "dbp": ("primary_diastolic", "secondary_diastolic")}.items():
        dd = (r[p] - r[s]).dropna()
        mean2 = (r[p] + r[s]) / 2
        is_mean = float(np.mean(np.abs(r[t] - mean2).dropna() <= 0.5)) if t in r.columns else float("nan")
        sd_d = float(dd.std(ddof=1))
        two = is_mean >= 0.9                                           # reference = mean of the two observers
        out[t] = {"n": int(len(dd)), "sd_diff": sd_d, "mean_diff": float(dd.mean()), "share_ref_equals_mean": is_mean,
                  "u_one_observer": sd_d / math.sqrt(2), "u_mean_two": sd_d / 2,
                  "u_ref": sd_d / 2 if two else sd_d / math.sqrt(2), "basis": "mean of two observers" if two else "one observer"}
    return out


def budget(args):
    from src.viz.style import TARGET, figure, panel_labels, save, setup
    p1 = load_p1()
    tg, preds, _, _, _ = p1.load()
    prim = tg[tg.test.isin(["static", "change", "after"])].copy()
    cs = pd.read_csv(INTER / f"calsens_{args.tag}.csv.gz")
    cs = cs[cs.cal_rank < args.k]
    # per-reading sensitivity of the k = 3 estimate (all calibration BPs shifted together) and calibration propagation factor
    agg = {f"c_{o}_{i}": "mean" for o in T for i in T}
    per = cs.groupby("sample_id").agg(n_cal=("cal_rank", "size"), **{k: (k, v) for k, v in agg.items()})
    for t in T:
        per[f"prop_{t}"] = cs.assign(c2=cs[f"c_{t}_{t}"] ** 2).groupby("sample_id").c2.sum().pow(0.5) / per.n_cal
    # u(x_cal): pooled within-subject SD of the calibration readings (one set per subject)
    calr = cs[cs.pid.isin(prim.pid)].drop_duplicates("cal_sample_id")        # auscultatory subjects of the primary set
    s_cal = {t: pooled_within_sd(calr[f"{t}_cal"].to_numpy(), calr.pid.to_numpy()) for t in T}
    # u(ref)
    uref_obs = reference_uncertainty(args.raw_tsv, None)
    st = prim[prim.test == "static"]
    uref_trip = {t: pooled_within_sd(st[t].to_numpy(), st.pid.to_numpy())[0] for t in T}
    uref = {t: (uref_obs[t]["u_ref"] if uref_obs else uref_trip[t]) for t in T}
    print("u(ref) inter-observer:", json.dumps(uref_obs, indent=1) if uref_obs else "not available (no --raw_tsv)", flush=True)

    sets = [("static", "Static test"), ("change", "BP-change test"), ("after", "After > 24 h"), ("overall", "All primary")]
    out, L = {}, []
    k_t4 = stats.t.ppf(0.975, 4) / math.sqrt(2)
    L += ["# GUM_Table_P5 — Measurement uncertainty budget of HiCAF-BP-Cal (GUM, JCGM 100:2008)", "",
          "Type A evaluations on the Aurora-BP auscultatory test readings (5-fold CV, k = 3 calibration readings). Measurement model: "
          "estimate = mean over the calibration readings of (calibration BP + learned change). c_cal: sensitivity of the estimate to "
          f"the calibration BP (all calibration BPs shifted together by ± {DELTA:.0f} mmHg on the frozen models; median, IQR). "
          "s(x_cal): pooled within-subject SD of one calibration reading; u1 = c_cal · s(x_cal) propagated through the mean of the "
          "n_cal estimates (≈ c_cal · s(x_cal)/√3). u(ref): "
          + (f"inter-observer differences of the two auscultatory observers ({uref_obs['sbp']['basis']})" if uref_obs else
             "within-subject SD of the static reference triplets (upper bound: includes physiological variation)")
          + ". u_c = √(SD(error)² − u(ref)²) = √(u1² + u2²); U = 2·u_c. Coverage: share of readings with |error − MD| ≤ "
          "2·√(u_c² + u(ref)²). Units mmHg.", "",
          "| Target | Test | Readings (subjects) | c_cal median (IQR) | s(x_cal) | u1 = calibration term | u2 = signal/model term | "
          "u(ref) | u_c | U (k = 2) | Share of u_c² from calibration | MD (not corrected) | Coverage of ± 2σ | k for 95 % (observed) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in T:
        for sn, sl in sets:
            d = prim if sn == "overall" else prim[prim.test == sn]
            d = d[d.sample_id.isin(per.index)]
            e = preds["ours"].loc[d.sample_id, f"{t}_pred"].to_numpy() - d[t].to_numpy()
            c = per.loc[d.sample_id, f"c_{t}_{t}"].to_numpy()
            prop = per.loc[d.sample_id, f"prop_{t}"].to_numpy()
            u1 = float(np.sqrt(np.mean((prop * s_cal[t][0]) ** 2)))
            sd, md = float(e.std(ddof=1)), float(e.mean())
            uc = math.sqrt(max(sd ** 2 - uref[t] ** 2, 0.0))
            u2 = math.sqrt(max(uc ** 2 - u1 ** 2, 0.0))
            sig = math.sqrt(uc ** 2 + uref[t] ** 2)
            cov = float(np.mean(np.abs(e - md) <= 2 * sig))
            k95 = float(np.quantile(np.abs(e - md), 0.95) / sig)
            q = np.percentile(c, [25, 50, 75])
            out[f"{t}/{sn}"] = {"n": int(len(d)), "subjects": int(d.pid.nunique()), "c_q": q.tolist(), "s_xcal": s_cal[t][0],
                                "u1": u1, "u2": u2, "u_ref": uref[t], "u_c": uc, "U": 2 * uc, "share_cal": u1 ** 2 / uc ** 2 if uc else float("nan"),
                                "md": md, "sd": sd, "coverage_2sigma": cov, "k95_observed": k95}
            L.append(f"| {t.upper()} | {sl} | {len(d):,} ({d.pid.nunique()}) | {q[1]:.2f} ({q[0]:.2f}–{q[2]:.2f}) | {s_cal[t][0]:.2f} | "
                     f"{u1:.2f} | {u2:.2f} | {uref[t]:.2f} | {uc:.2f} | {2 * uc:.1f} | {u1 ** 2 / uc ** 2:.0%} | {md:+.2f} | {cov:.1%} | {k95:.2f} |")
    cross = {f"{o}<-{i}": float(np.median(per[f"c_{o}_{i}"])) for o in T for i in T if o != i}
    curv = {t: float(np.percentile(np.abs(cs[f"curv_{t}_{t}"]), 95)) for t in T}
    L += ["", "Notes:",
          f"- Cross-sensitivities (median): SBP estimate to DBP calibration {cross['sbp<-dbp']:+.2f}; DBP estimate to SBP calibration "
          f"{cross['dbp<-sbp']:+.2f} mmHg/mmHg.",
          f"- Linearity: 95th percentile of |f(+{DELTA:.0f}) + f(−{DELTA:.0f}) − 2 f(0)| = {curv['sbp']:.2f} (SBP) / {curv['dbp']:.2f} (DBP) mmHg.",
          f"- Calibration readings: pooled within-subject SD {s_cal['sbp'][0]:.2f} / {s_cal['dbp'][0]:.2f} mmHg ({s_cal['sbp'][1]} subjects).",
          f"- u(ref) alternatives: static-triplet within-subject SD {uref_trip['sbp']:.2f} / {uref_trip['dbp']:.2f} (upper bound)"
          + (f"; one observer {uref_obs['sbp']['u_one_observer']:.2f} / {uref_obs['dbp']['u_one_observer']:.2f}; mean of two "
             f"{uref_obs['sbp']['u_mean_two']:.2f} / {uref_obs['dbp']['u_mean_two']:.2f} (reference = mean of the two observers in "
             f"{uref_obs['sbp']['share_ref_equals_mean']:.0%} of readings)" if uref_obs else "") + ".",
          f"- Coverage factor for 95 % under a t4 error distribution: {k_t4:.2f} (normal: 1.96).",
          "- Model (epistemic) uncertainty under domain shift: between-fold-model SD on zero-shot PulseDB, EXT_Table_P6 — not part of "
          "this budget (each Aurora reading is estimated by one fold model only)."]
    (SEC / "tables").mkdir(parents=True, exist_ok=True)
    (SEC / "tables/GUM_Table_P5_uncertainty_budget.md").write_text("\n".join(L))
    json.dump({"budget": out, "s_cal": s_cal, "u_ref_observers": uref_obs, "u_ref_triplets": uref_trip, "cross": cross,
               "curvature_p95": curv, "k_t4": k_t4}, open(SEC / "tables/GUM_P4_results.json", "w"), indent=1, default=float)
    print("\n".join(L))

    # ---------------- GUM_Fig_P4: (a) sensitivity coefficient, (b) budget per test (variance shares)
    setup()
    fig, ax = figure("double", height=2.6, ncols=2)
    dd = per.loc[per.index.intersection(prim.sample_id)]
    for t in T:
        v = dd[f"c_{t}_{t}"].to_numpy()
        lo, hi = np.percentile(v, [0.5, 99.5])
        ax[0].hist(v, bins=np.linspace(lo, hi, 60), color=TARGET[t]["color"], alpha=0.6, label=f"{TARGET[t]['label']} (median {np.median(v):.2f})")
    ax[0].axvline(1.0, color="black", lw=0.8, ls="--")
    ax[0].text(1.0, ax[0].get_ylim()[1] * 0.97, " 1 = calibration error\n passes 1:1", fontsize=5.5, va="top")
    ax[0].set(xlabel=r"Sensitivity to the calibration BP, $c_\mathrm{cal}$ (mmHg/mmHg)", ylabel="Readings")
    ax[0].legend(loc="upper left", fontsize=6)
    labels, comps = [], []
    for t in T:
        for sn, sl in sets:
            r = out[f"{t}/{sn}"]
            labels.append(f"{TARGET[t]['label']} {sl.replace(' test', '')}")
            comps.append([r["u1"] ** 2, r["u2"] ** 2, r["u_ref"] ** 2])
    comps = np.array(comps)
    y = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    names = [r"Calibration ($u_1^2$)", r"Signal / model ($u_2^2$)", r"Reference ($u_\mathrm{ref}^2$)"]
    cols = ["#E69F00", "#0072B2", "#B0B0B0"]
    for j in range(3):
        ax[1].barh(y, comps[:, j], left=left, color=cols[j], label=names[j], height=0.7)
        left += comps[:, j]
    for yi, tot, r in zip(y, left, [out[f"{t}/{sn}"] for t in T for sn, _ in sets]):
        ax[1].text(tot, yi, f"  $U$ = {r['U']:.1f}", va="center", fontsize=5.5)
    ax[1].set(yticks=y, xlabel="Variance of the error (mmHg²)", xlim=(0, left.max() * 1.25))
    ax[1].set_yticklabels(labels, fontsize=6)
    ax[1].legend(loc="lower right", fontsize=5.5)
    ax[1].grid(axis="y", visible=False)
    panel_labels(ax, x=-0.12)
    save(fig, "GUM_Fig_P4_uncertainty", source=__file__, outdir=SEC / "figures", png_subdir="png")

    rd = SEC / "README.md"
    if rd.exists():
        s = rd.read_text()
        for name in ["GUM_Fig_P4_uncertainty", "GUM_Table_P5_uncertainty_budget"]:
            s = "\n".join(l.replace("| pending |", f"| done {time.strftime('%Y-%m-%d')} |") if l.startswith(f"| {name} ") else l
                          for l in s.split("\n"))
        rd.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["infer", "budget", "all"], default="all")
    ap.add_argument("--run", default="runs/hicaf_bp_cal_notime")
    ap.add_argument("--tag", default="notime", help="cache name: runs/performance/calsens_<tag>.csv.gz")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--raw_tsv", default=None, help="Aurora-BP measurements_auscultatory.tsv (observer columns) for u(ref)")
    args = ap.parse_args()
    if args.stage in ("infer", "all"):
        infer(args)
    if args.stage in ("budget", "all"):
        budget(args)


if __name__ == "__main__":
    main()