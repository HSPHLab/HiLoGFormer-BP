#!/usr/bin/env python
"""
Calibration-free zero-shot comparison on PulseDB (plan: docs/REIMPLEMENTATION_PLAN.md, R10). Additional results.

  python scripts/eval_calfree_external.py --gpu 2            # predictions for our frozen uncalibrated models + report
  python scripts/eval_calfree_external.py --report_only      # rebuild the report (e.g. after a new published method)

1. Zero-shot predictions (inference only, frozen fold models, no training) for our calibration-free models:
     hicaf_bp_main   (HiCAF-BP, uncalibrated)      resnet1d_ecgppg   (ResNet1D baseline)
   on exactly the PulseDB target segments used for every other zero-shot test (first 3 segments of each subject =
   calibration readings, excluded; checked against our calibrated model's zero-shot targets). Existing outputs are kept.
2. One report with every calibration-free method available: the two above + published calibration-free methods whose
   zero-shot evaluation exists (rU-Net, UTransBPNet: runs/pub_<method>/external_pulsedb). Calibration-value-only is shown
   as the reference that any calibrated method starts from.
Output: runs/zero_shot_calfree/<experiment>/  predictions_fold{f}.npz, predictions_ensemble.csv
        runs/zero_shot_calfree/CALIBRATION_FREE_ZERO_SHOT.md (+ metrics.json)
"""
import argparse
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
OURS = [("hicaf_bp_main", "HiCAF-BP, uncalibrated (ours)"), ("resnet1d_ecgppg", "ResNet1D (baseline)")]
PUBLISHED = [("runet", "rU-Net"), ("utransbpnet", "UTransBPNet")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--reference", default="runs/hicaf_bp_cal_notime/external_pulsedb/predictions_ensemble_k1.csv")
    ap.add_argument("--out", default="runs/zero_shot_calfree")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from scipy.stats import wilcoxon

    from src.data.pairs import calibration_split
    from src.evaluation.metrics import regression_metrics

    t0 = time.time()
    out = CODE_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    meta = pd.read_csv(data / "metadata.csv", low_memory=False).sort_values("sample_id").reset_index(drop=True)
    cal, targets = calibration_split(meta, "initial", ["calibration"], 3)
    tm = meta.iloc[targets].reset_index(drop=True)
    ref = pd.read_csv(CODE_DIR / args.reference)
    if not np.array_equal(ref.sample_id.to_numpy(), tm.sample_id.to_numpy()):
        sys.exit("target segments differ from the reference zero-shot test - stop")
    print(f"PulseDB targets: {len(tm):,} segments from {tm.pid.nunique():,} subjects (identical to the reference: OK)", flush=True)
    lab = meta[T].to_numpy(float)
    cal_mean = {k: np.stack([lab[cal[p][:k]].mean(axis=0) for p in tm.pid]) for k in (1, 3)}

    # ------------------------------------------------------------------ 1. predictions for our calibration-free models
    if not args.report_only:
        import torch
        from torch.utils.data import DataLoader

        from src.data.dataset import BPCropDataset
        from src.models import build_model
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        signals = np.load(data / "signals.npy", mmap_mode="r")
        prep = json.load(open(data / "config.json"))
        channels = prep.get("channels", ["ecg", "ppg"])
        for exp, label in OURS:
            d = out / exp
            if (d / "predictions_ensemble.csv").exists():
                print(f"{exp}: predictions exist - kept", flush=True)
                continue
            d.mkdir(exist_ok=True)
            preds = []
            for fold in args.folds:
                ck = torch.load(CODE_DIR / "runs" / exp / f"fold_{fold}" / "best.pt", map_location="cpu")
                cfg, st = ck["config"], ck["stats"]
                if "cal" in str(cfg["model"].get("name", "")):
                    sys.exit(f"{exp} is a calibrated model - not for this comparison")
                n_aux = len(cfg["data"].get("aux_targets", [])) if float(cfg["loss"].get("aux_weight", 0)) > 0 else 0
                model = build_model(cfg["model"], input_names=cfg["data"]["inputs"], n_targets=2, n_aux=n_aux).to(dev)
                model.load_state_dict(ck["model_state"], strict=True)
                model.eval()
                y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
                ch_idx = [channels.index(c) for c in cfg["data"]["inputs"]]
                crop = int(round(cfg["data"]["crop_s"] * int(prep["config"]["fs_out"])))
                ds = BPCropDataset(tm, signals, ch_idx, crop, y_mean, y_std, T, [], None, None, train=False, augment_cfg=None)
                dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=16, pin_memory=dev.type == "cuda")
                p = []
                with torch.no_grad():
                    for b in dl:
                        p.append(model(b["x"].to(dev, non_blocking=True))["bp"].float().cpu().numpy())
                p = np.concatenate(p) * y_std + y_mean
                np.savez_compressed(d / f"predictions_fold{fold}.npz", sample_id=tm.sample_id.to_numpy(), pred=p.astype(np.float32))
                preds.append(p)
                print(f"{exp} fold {fold}: {len(p):,} segments ({(time.time() - t0) / 60:.1f} min)", flush=True)
            ens = np.mean(preds, axis=0)
            e = tm[["sample_id", "pid", "protocol", "sbp", "dbp"]].copy()
            for j, t in enumerate(T):
                e[f"{t}_pred"] = ens[:, j]
                e[f"{t}_cal_mean"] = cal_mean[1][:, j]
            e.to_csv(d / "predictions_ensemble.csv", index=False)

    # ------------------------------------------------------------------ 2. report
    methods = []
    for exp, label in OURS:
        d = out / exp
        if (d / "predictions_ensemble.csv").exists():
            folds = [np.load(d / f"predictions_fold{f}.npz")["pred"] for f in args.folds if (d / f"predictions_fold{f}.npz").exists()]
            methods.append((label, pd.read_csv(d / "predictions_ensemble.csv"), folds))
    for m, label in PUBLISHED:
        d = CODE_DIR / "runs" / f"pub_{m}" / "external_pulsedb"
        if (d / "predictions_ensemble_k1.csv").exists():
            folds = [np.load(d / f"predictions_fold{f}_k1.npz")["pred"] for f in args.folds if (d / f"predictions_fold{f}_k1.npz").exists()]
            methods.append((label, pd.read_csv(d / "predictions_ensemble_k1.csv"), folds))
    for label, e, _ in methods:
        if not np.array_equal(e.sample_id.to_numpy(), tm.sample_id.to_numpy()):
            sys.exit(f"{label}: target segments differ - stop")
    ours = methods[0] if methods and methods[0][0].endswith("(ours)") else None
    sources = [("combined", tm.index)] + [(s, tm.index[tm.protocol == s]) for s in sorted(tm.protocol.unique())]
    L = ["# Calibration-free methods — zero-shot PulseDB (additional results)", "",
         f"All models trained on Aurora-BP only (5 CV folds each); none has seen PulseDB. {len(tm):,} target segments from "
         f"{tm.pid.nunique():,} subjects, identical for every method (checked). No method uses a calibration reading; "
         "'calibration value only' is shown as the reference a calibrated method starts from. Ensemble = mean of the 5 fold models.", "",
         "| Data | Method | SBP MAE (ens) | SBP MAE 5 models mean ± SD | SBP ME ± SD (ens) | DBP MAE (ens) | DBP MAE 5 models mean ± SD | DBP ME ± SD (ens) | SBP r / DBP r (ens) | subj. mean MAE SBP / DBP | vs ours: subject-level Δ MAE (% subjects better), p |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    metrics = {}
    for sname, ix in sources:
        rows = [(label, e.loc[ix], np.array([[np.abs(pf[ix.to_numpy(), j] - tm.loc[ix, t].to_numpy()).mean() for j, t in enumerate(T)] for pf in folds]))
                for label, e, folds in methods]
        for k in (1, 3):
            c = tm.loc[ix, ["pid", *T]].copy()
            for j, t in enumerate(T):
                c[f"{t}_pred"] = cal_mean[k][ix.to_numpy(), j]
            rows.append((f"Calibration value only (k = {k}, reference)", c, None))
        per_ours = ({t: (ours[1].loc[ix, f"{t}_pred"] - ours[1].loc[ix, t]).abs().groupby(ours[1].loc[ix, "pid"]).mean() for t in T}
                    if ours else None)
        for label, q, fm in rows:
            m = regression_metrics(q, T, 5)
            subj = {t: (q[f"{t}_pred"] - q[t]).abs().groupby(q.pid).mean() for t in T}
            spread = lambda j: f"{fm[:, j].mean():.2f} ± {fm[:, j].std(ddof=1):.2f}" if fm is not None and len(fm) > 1 else "–"
            vs = "–"
            if per_ours is not None and label != ours[0] and fm is not None:
                cells = []
                for t in T:
                    pv = wilcoxon(subj[t], per_ours[t]).pvalue
                    cells.append(f"{t.upper()} {subj[t].mean() - per_ours[t].mean():+.2f} ({(subj[t] < per_ours[t]).mean():.0%}), "
                                 f"p {'< 0.001' if pv < 0.001 else f'{pv:.3f}'}")
                vs = "; ".join(cells)
            L.append(f"| {sname} | {label} | {m['sbp']['MAE']:.2f} | {spread(0)} | {m['sbp']['ME']:+.2f} ± {m['sbp']['SDE']:.2f} | "
                     f"{m['dbp']['MAE']:.2f} | {spread(1)} | {m['dbp']['ME']:+.2f} ± {m['dbp']['SDE']:.2f} | "
                     f"{m['sbp']['r']:.2f} / {m['dbp']['r']:.2f} | {subj['sbp'].mean():.2f} / {subj['dbp'].mean():.2f} | {vs} |")
            metrics[f"{sname}/{label}"] = m
    L += ["", f"Methods included: {', '.join(lbl for lbl, _, _ in methods)}. Runtime {(time.time() - t0) / 60:.1f} min."]
    (out / "CALIBRATION_FREE_ZERO_SHOT.md").write_text("\n".join(L))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\nSaved {out}/CALIBRATION_FREE_ZERO_SHOT.md")


if __name__ == "__main__":
    main()
