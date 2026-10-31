#!/usr/bin/env python
"""
ZERO-SHOT PulseDB evaluation of a re-implemented published method (R7 in docs/REIMPLEMENTATION_PLAN.md).
Exactly the protocol used for our model (scripts/eval_external.py):
  - the 5 Aurora fold models of the method (runs/pub_<method>/fold_k/best.pt); PulseDB never used for training
  - calibration = the first k (1, 3) segments of each subject's recording; targets = all later segments
  - baseline BP = mean of the first k segments' labels (the method's calibration input)
  - handcrafted features: data_processed/features_pwa/pulsedb_v1.csv (same extractor as for Aurora)
  - demographics: age and sex from PulseDB; BMI from the VitalDB subject files (MIMIC has none); history of
    hypertension unavailable; every missing value filled with the TRAINING statistics stored in the model file
  - results per source (MIMIC, VitalDB) and combined; the target segments are checked to be identical to our
    model's zero-shot test, and our model's ensemble is shown in the same table for a like-for-like comparison

  python scripts/eval_published_external.py --method hgctnet --gpu 2 [--folds 0] [--max_subjects 20]

Output: runs/pub_<method>/external_pulsedb/  EXTERNAL_RESULTS.md, metrics.json, predictions_ensemble_k{1,3}.csv,
        predictions_fold{f}_k{k}.npz
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
T = ["sbp", "dbp"]
SEG = 1250


def load_train_helpers():
    spec = importlib.util.spec_from_file_location("train_published", CODE_DIR / "scripts" / "train_published.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vital_bmi(root, pids, cache):
    """BMI of each VitalDB subject from its PulseDB subject file (constant per subject); cached as CSV."""
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["pid", "bmi"])
    todo = sorted(set(pids) - set(have.pid))
    if not todo:
        return have.set_index("pid")["bmi"]
    import h5py
    rows = []
    for pid in todo:
        fp = Path(root) / "PulseDB_v2_0" / "Segment_Files" / "PulseDB_Vital" / f"{pid.split('_', 1)[1]}.mat"
        val = np.nan
        try:
            with h5py.File(fp, "r") as f:
                ds = f["Subj_Wins"]["BMI"]
                x = f[ds[0, 0]][()] if (ds.dtype == h5py.ref_dtype or ds.dtype.kind == "O") else ds[()]
                val = float(np.asarray(x, float).ravel()[0])
        except Exception:
            pass
        rows.append({"pid": pid, "bmi": val if np.isfinite(val) and 10 < val < 80 else np.nan})
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(cache, index=False)                              # cache grows to all subjects ever requested
    return out.set_index("pid")["bmi"]


def vital_demo(root, pids, cache):
    """Height [cm], weight [kg], BMI of each VitalDB subject from its PulseDB subject file; cached as CSV (rU-Net)."""
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["pid", "height", "weight", "bmi"])
    todo = sorted(set(pids) - set(have.pid))
    if not todo:
        return have.set_index("pid")
    import h5py
    rows = []
    for pid in todo:
        fp = Path(root) / "PulseDB_v2_0" / "Segment_Files" / "PulseDB_Vital" / f"{pid.split('_', 1)[1]}.mat"
        r = {"pid": pid}
        for name, lo, hi in [("Height", 100, 230), ("Weight", 25, 250), ("BMI", 10, 80)]:
            val = np.nan
            try:
                with h5py.File(fp, "r") as f:
                    ds = f["Subj_Wins"][name]
                    x = f[ds[0, 0]][()] if (ds.dtype == h5py.ref_dtype or ds.dtype.kind == "O") else ds[()]
                    val = float(np.asarray(x, float).ravel()[0])
            except Exception:
                pass
            r[name.lower()] = val if np.isfinite(val) and lo < val < hi else np.nan
        rows.append(r)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(cache, index=False)
    return out.set_index("pid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="hgctnet")
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--features", default="data_processed/features_pwa/pulsedb_v1.csv")
    ap.add_argument("--pulsedb_root", default="/data0/PulseDB")
    ap.add_argument("--ours", default="runs/hicaf_bp_cal_notime/external_pulsedb")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3])
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--max_subjects", type=int, default=0, help=">0: quick test on the first N subjects per source")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from scipy.stats import wilcoxon

    from src.data.pairs import calibration_split
    from src.evaluation.metrics import regression_metrics

    tp = load_train_helpers()
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = CODE_DIR / "runs" / f"pub_{args.method}"
    out = run_dir / ("external_pulsedb" if not args.max_subjects else "external_pulsedb_test")
    out.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ data
    data = Path(args.data)
    meta = pd.read_csv(data / "metadata.csv", low_memory=False)
    if args.max_subjects:
        keep = meta.groupby("protocol").pid.apply(lambda s: list(pd.unique(s))[:args.max_subjects]).sum()
        meta = meta[meta.pid.isin(keep)]
    meta = meta.sort_values("sample_id").reset_index(drop=True)
    feats = pd.read_csv(args.features)
    meta = meta.merge(feats, on="sample_id", how="left", validate="one_to_one", suffixes=("", "_feat"))
    signals = np.load(data / "signals.npy", mmap_mode="r")
    ch = json.load(open(data / "config.json")).get("channels", ["ecg", "ppg"])
    ch_idx = [ch.index("ecg"), ch.index("ppg")]
    vit = sorted(meta.loc[meta.protocol == "VitalDB", "pid"].unique())
    bmi = vital_bmi(args.pulsedb_root, vit, data / "vitaldb_bmi.csv") if vit else pd.Series(dtype=float)
    g = meta["gender"].astype(str).str.upper().map({"F": 0.0, "M": 1.0})
    if args.method in ("runet", "ragcbpnet"):                    # U3 / G6: sex, age, height [cm], weight [kg], BMI
        vd = vital_demo(args.pulsedb_root, vit, data / "vitaldb_demo.csv") if vit else pd.DataFrame(columns=["height", "weight", "bmi"])
        demo = pd.DataFrame({"gender": g, "age": meta["age"].astype(float), "height": meta.pid.map(vd["height"]).astype(float),
                             "weight": meta.pid.map(vd["weight"]).astype(float), "bmi": meta.pid.map(vd["bmi"]).astype(float)},
                            index=meta.index)
        if args.method == "ragcbpnet":                               # no acquisition time in PulseDB -> training mean (R7)
            demo["hour_cos"], demo["hour_sin"] = np.nan, np.nan
    else:
        demo = pd.DataFrame({"age": meta["age"].astype(float), "gender": g, "bmi": meta.pid.map(bmi).astype(float),
                             "hhp": np.nan}, index=meta.index)
    print(f"PulseDB: {len(meta):,} segments, {meta.pid.nunique():,} subjects | demographics: {list(demo.columns)} | BMI available: "
          f"{demo.bmi.notna().mean():.1%} of segments (VitalDB only) | sex available: {demo.gender.notna().mean():.1%}", flush=True)

    cal, targets = calibration_split(meta, "initial", ["calibration"], max(args.ks))
    tm = meta.iloc[targets].reset_index(drop=True)
    base = {k: tp.baselines(meta, cal, k) for k in args.ks}
    demo_t = demo.iloc[targets].reset_index(drop=True)          # aligned with tm (index 0..n-1)
    print(f"calibration protocol: {len(cal):,} subjects, {len(targets):,} target segments", flush=True)

    # same target segments as our model's zero-shot test (like-for-like)
    ours = {}
    for k in args.ks:
        fp = Path(args.ours) / f"predictions_ensemble_k{k}.csv"
        if fp.exists() and not args.max_subjects:
            o = pd.read_csv(fp)
            if not np.array_equal(o.sample_id.to_numpy(), tm.sample_id.to_numpy()):
                sys.exit(f"target segments differ from our model's zero-shot test ({fp}) - stop")
            ours[k] = o
    if ours:
        print("target segments identical to our model's zero-shot test: OK", flush=True)

    # ------------------------------------------------------------------ predictions per fold model
    from src.published.bpbls import BpBLS
    from src.published.hgctnet import HGCTNet
    from src.published.runet import rUNetBP
    from src.published.utransbpnet import UTransBPNet
    from src.published.ragcbpnet import RAGCBPNet
    builders = {"hgctnet": HGCTNet, "bpbls": BpBLS, "runet": rUNetBP, "utransbpnet": UTransBPNet, "ragcbpnet": RAGCBPNet}
    side_fns = {"hgctnet": tp.side_inputs, "bpbls": tp.side_inputs_bpbls, "runet": tp.side_inputs_runet,
                "utransbpnet": tp.side_inputs_none, "ragcbpnet": tp.side_inputs_ragcbpnet}
    preds = {k: [] for k in args.ks}
    for fold in args.folds:
        ck = torch.load(run_dir / f"fold_{fold}" / "best.pt", map_location="cpu")
        stats = {k: np.asarray(v, float) for k, v in ck["stats"].items()}
        fnames = ck["feature_names"]
        model = builders[ck["method"]](**ck["model_args"]).to(dev)
        model.load_state_dict(ck["model_state"])
        model.eval()
        t0 = time.time()
        fold_pred = {k: np.zeros((len(tm), 2), np.float32) for k in args.ks}
        bs = {"runet": 256, "utransbpnet": 512}.get(ck["method"], 1024)  # large models need smaller inference batches
        for a in range(0, len(tm), args.chunk):
            part = tm.iloc[a:a + args.chunk]
            x = torch.from_numpy(tp.windows(part, signals, ch_idx))
            for k in args.ks:
                f = torch.from_numpy(side_fns[ck["method"]](part, base[k], demo_t, fnames, stats))
                with torch.no_grad():
                    p = [model(xb.to(dev), fb.to(dev)).float().cpu().numpy()
                         for xb, fb in zip(torch.split(x, bs), torch.split(f, bs))]
                fold_pred[k][a:a + len(part)] = np.concatenate(p)
        for k in args.ks:
            preds[k].append(fold_pred[k])
            np.savez_compressed(out / f"predictions_fold{fold}_k{k}.npz", sample_id=tm.sample_id.to_numpy(), pred=fold_pred[k])
        print(f"fold {fold}: {len(tm):,} targets x {len(args.ks)} k in {(time.time() - t0) / 60:.1f} min | "
              f"alpha {ck['model_state']['balance_alpha'].numpy().round(3) if 'balance_alpha' in ck['model_state'] else '-'}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ evaluation (same tables as our zero-shot)
    first_t = {pid: meta.time_s.iloc[pos[0]] for pid, pos in cal.items()}
    tm["hours_since_cal"] = (tm.time_s - tm.pid.map(first_t)) / 3600.0
    sources = [("combined", tm.index)] + [(s, tm.index[tm.protocol == s]) for s in sorted(tm.protocol.unique())]
    name = {"hgctnet": "HGCTNet", "bpbls": "BpBLS", "runet": "rU-Net", "utransbpnet": "UTransBPNet",
            "ragcbpnet": "RAGCBPNet"}.get(args.method, args.method)
    L = [f"# Zero-shot PulseDB — {name} (re-implemented; trained on Aurora-BP only)", "",
         f"{len(tm):,} target segments from {tm.pid.nunique():,} subjects; calibration = first k segments; identical target "
         f"segments to our model's zero-shot test{' (checked)' if ours else ''}. Ensemble = mean of the 5 Aurora fold models "
         f"(none has seen PulseDB). Missing inputs filled with training statistics: history of hypertension (all), BMI (MIMIC), "
         f"PTT features where the ECG-PPG timing is implausible.", ""]
    metrics = {}
    for k in args.ks:
        ens = np.mean(preds[k], axis=0)
        d = tm[["sample_id", "pid", "protocol", "sbp", "dbp", "hours_since_cal", "ecg_sqi"]].copy()
        cal_mean = np.stack([base[k][p] for p in d.pid])
        for j, t in enumerate(T):
            d[f"{t}_pred"] = ens[:, j]
            d[f"{t}_cal_mean"] = cal_mean[:, j]
        d.to_csv(out / f"predictions_ensemble_k{k}.csv", index=False)
        L += [f"## k = {k}", "",
              "| Data | Method | SBP MAE (ens) | SBP MAE 5 models mean ± SD | SBP ME ± SD (ens) | DBP MAE (ens) | DBP MAE 5 models mean ± SD | DBP ME ± SD (ens) | subj. mean MAE SBP / DBP (ens) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for sname, ix in sources:
            q = d.loc[ix]
            rows = [(f"**{name}**", q, np.array([[np.abs(pf[ix.to_numpy(), j] - q[t].to_numpy()).mean() for j, t in enumerate(T)] for pf in preds[k]]))]
            if k in ours:
                oq = ours[k].loc[ix]
                rows.append(("HiCAF-BP-Cal, no time (ours)", oq, None))
            cq = q.copy()
            for t in T:
                cq[f"{t}_pred"] = q[f"{t}_cal_mean"]
            rows.append(("Calibration value only", cq, None))
            for label, qq, fm in rows:
                m = regression_metrics(qq, T, 5)
                subj = [(qq[f"{t}_pred"] - qq[t]).abs().groupby(qq.pid).mean().mean() for t in T]
                spread = (lambda j: f"{fm[:, j].mean():.2f} ± {fm[:, j].std(ddof=1):.2f}" if fm is not None and len(fm) > 1 else "–")
                L.append(f"| {sname} | {label} | {m['sbp']['MAE']:.2f} | {spread(0)} | {m['sbp']['ME']:+.2f} ± {m['sbp']['SDE']:.2f} | "
                         f"{m['dbp']['MAE']:.2f} | {spread(1)} | {m['dbp']['ME']:+.2f} ± {m['dbp']['SDE']:.2f} | {subj[0]:.2f} / {subj[1]:.2f} |")
                metrics[f"k{k}/{sname}/{label}"] = m
            if k in ours:
                a = {t: (q[f"{t}_pred"] - q[t]).abs().groupby(q.pid).mean() for t in T}
                b = {t: (oq[f"{t}_pred"] - oq[t]).abs().groupby(oq.pid).mean() for t in T}
                cells = []
                for t in T:
                    p = wilcoxon(a[t], b[t]).pvalue if len(a[t]) > 10 else float("nan")
                    cells.append(f"{t.upper()} {a[t].mean() - b[t].mean():+.2f} mmHg ({name} better for {(a[t] < b[t]).mean():.0%} of subjects, "
                                 f"p {'< 0.001' if p < 0.001 else f'{p:.3f}'})")
                L.append(f"|  | *{name} − ours, subject level* | " + "; ".join(cells) + " | | | | | | |")
        L.append("")
    L.append(f"Runtime {(time.time() - t_start) / 60:.1f} min.")
    (out / "EXTERNAL_RESULTS.md").write_text("\n".join(L))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
