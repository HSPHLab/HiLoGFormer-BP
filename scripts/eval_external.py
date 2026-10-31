#!/usr/bin/env python
"""
ZERO-SHOT external evaluation of a calibrated experiment on preprocessed PulseDB (no training on PulseDB).

  python scripts/eval_external.py --experiment hicaf_bp_cal_notime --gpu 2

For each of the 5 CV fold models (none has seen any PulseDB data) and the 5-model ensemble:
  calibration = the first k (1 or 3) segments of each subject's recording, targets = all later segments,
  prediction for a target = average over its k calibration segments (same protocol as on Aurora).
Fast exact inference: every segment is encoded ONCE; the calibration head is then applied to (target,
calibration) embedding pairs. A self-check per model compares this against the full model forward.

Reported per source (MIMIC, VitalDB) and combined:
  segment level  : MAE, ME ± SD, r, AAMI / BHS / IEEE (pooled), calibration-value-only baseline
  subject level  : mean / median of per-subject MAE (every subject counts equally), % subjects where the model
                   beats calibration-only, Wilcoxon signed-rank on per-subject MAE
  model spread   : mean ± SD of MAE over the 5 fold models
  secondary      : MAE by time since calibration, MAE by ECG quality (ecg_sqi)
Output: runs/<experiment>/external_pulsedb/  EXTERNAL_RESULTS.md, metrics.json,
        predictions_ensemble_k{1,3}.csv, predictions_fold{f}_k{k}.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
T = ["sbp", "dbp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="hicaf_bp_cal_notime")
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 3])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_subjects", type=int, default=0, help=">0: quick test on the first N subjects per source")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.data.dataset import BPCropDataset
    from src.data.pairs import calibration_split, eval_pairs
    from src.evaluation.metrics import regression_metrics
    from src.models import build_model

    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    exp_dir = Path(load_config("base.yaml")["paths"]["runs_dir"]) / args.experiment
    out = exp_dir / ("external_pulsedb" if not args.max_subjects else "external_pulsedb_test")
    out.mkdir(exist_ok=True)

    data = Path(args.data)
    meta = pd.read_csv(data / "metadata.csv", low_memory=False)
    prep = json.load(open(data / "config.json"))
    channels = prep.get("channels", ["ecg", "ppg"])
    if args.max_subjects:
        keep = meta.groupby("protocol").pid.apply(lambda s: list(pd.unique(s))[:args.max_subjects]).sum()
        meta = meta[meta.pid.isin(keep)]
    meta = meta.sort_values("sample_id").reset_index(drop=True)
    signals = np.load(data / "signals.npy", mmap_mode="r")
    print(f"PulseDB: {len(meta):,} segments, {meta.pid.nunique():,} subjects", flush=True)

    @torch.no_grad()
    def encode_all(model, ds):
        zs, zps = [], []
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=dev.type == "cuda")
        for b in loader:
            z, zp = model.encoder.encode(b["x"].to(dev, non_blocking=True))
            zs.append(z.float().cpu())
            zps.append(zp.float().cpu() if zp is not None else torch.zeros(len(z), 0))
        return torch.cat(zs), torch.cat(zps)

    @torch.no_grad()
    def head_predict(model, z, zp, ti, ci, bp_cal, bs=65536):
        res = []
        for s in range(0, len(ti), bs):
            a, c = torch.from_numpy(ti[s:s + bs]), torch.from_numpy(ci[s:s + bs])
            feats = model._compare(z[a].to(dev), z[c].to(dev))
            if model.use_ppg:
                feats += model._compare(zp[a].to(dev), zp[c].to(dev))
            bc = torch.from_numpy(bp_cal[s:s + bs]).to(dev)
            feats.append(model.bp_embed(bc))
            res.append((bc + model.head(torch.cat(feats, dim=-1)).float()).cpu().numpy())
        return np.concatenate(res)

    cal, targets = calibration_split(meta, "initial", ["calibration"], max(args.ks))
    print(f"calibration protocol: {len(cal):,} subjects, {len(targets):,} target segments", flush=True)
    preds = {k: [] for k in args.ks}            # per fold: (n_targets, 2) in mmHg
    base = None
    for fold in args.folds:
        ck = torch.load(exp_dir / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        if int(cfg["model"].get("ctx_dim", 0)) > 0 or int(cfg["model"].get("demo_dim", 0)) > 0:
            sys.exit("this model needs time context / demographics, which PulseDB does not provide - use a no-time model")
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        model = build_model(cfg["model"], input_names=cfg["data"]["inputs"], n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"])
        model.eval()
        ch_idx = [channels.index(c) for c in cfg["data"]["inputs"]]
        ds = BPCropDataset(meta, signals, ch_idx, int(round(cfg["data"]["crop_s"] * int(prep["config"]["fs_out"]))),
                           y_mean, y_std, T, [], None, None, train=False, augment_cfg=None)
        t0 = time.time()
        z, zp = encode_all(model, ds)
        y_std_lab = ds.y                                                   # standardised labels (Aurora statistics)
        if base is None:
            base = {}
        # self-check: fast path == full forward on 256 pairs
        p1 = eval_pairs(meta, cal, targets, 1)[:256]
        xb = torch.stack([ds[int(i)]["x"] for i in p1[:, 0]]).to(dev)
        xc = torch.stack([ds[int(i)]["x"] for i in p1[:, 1]]).to(dev)
        with torch.no_grad():
            full = model(xb, x_cal=xc, bp_cal=torch.from_numpy(y_std_lab[p1[:, 1]]).to(dev))["bp"].float().cpu().numpy()
        fast = head_predict(model, z, zp, p1[:, 0], p1[:, 1], y_std_lab[p1[:, 1]])
        diff = float(np.abs((full - fast) * y_std).max())
        print(f"fold {fold}: encoded {len(z):,} segments in {(time.time() - t0) / 60:.1f} min | "
              f"self-check max |full - fast| = {diff:.4f} mmHg", flush=True)
        if diff > 0.05:
            sys.exit("self-check failed: fast inference does not match the model - stop")
        for k in args.ks:
            pairs = eval_pairs(meta, cal, targets, k)
            bp = head_predict(model, z, zp, pairs[:, 0], pairs[:, 1], y_std_lab[pairs[:, 1]]) * y_std + y_mean
            agg = pd.DataFrame(bp, columns=T).assign(t=pairs[:, 0]).groupby("t")[T].mean()
            preds[k].append(agg.reindex(targets).to_numpy(np.float32))
            np.savez_compressed(out / f"predictions_fold{fold}_k{k}.npz", sample_id=meta.sample_id.to_numpy()[targets],
                                pred=preds[k][-1])
            if k not in base:
                lab = meta[T].to_numpy(np.float32)
                cm = pd.DataFrame(lab[pairs[:, 1]], columns=T).assign(t=pairs[:, 0]).groupby("t")[T].mean()
                base[k] = cm.reindex(targets).to_numpy(np.float32)
        del model, z, zp
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ evaluation
    tm = meta.iloc[targets].reset_index(drop=True)
    first_cal_time = {pid: meta.time_s.iloc[pos[0]] for pid, pos in cal.items()}
    tm["hours_since_cal"] = (tm.time_s - tm.pid.map(first_cal_time)) / 3600.0
    sources = [("combined", tm.index)] + [(s, tm.index[tm.protocol == s]) for s in sorted(tm.protocol.unique())]

    def frame(pred, k):
        d = tm[["sample_id", "pid", "protocol", "sbp", "dbp", "hours_since_cal", "ecg_sqi"]].copy()
        for j, t in enumerate(T):
            d[f"{t}_pred"] = pred[:, j]
            d[f"{t}_cal_mean"] = base[k][:, j]
        return d

    def subject_level(d, col):
        per = {t: (d[f"{t}_{col}"] - d[t]).abs().groupby(d.pid).mean() for t in T}
        return {t: (float(per[t].mean()), float(per[t].median())) for t in T}, per

    from scipy.stats import wilcoxon
    L = [f"# Zero-shot external test on PulseDB — {args.experiment}", "",
         f"Models trained on Aurora-BP only (5 CV folds); PulseDB never used for training or model selection. "
         f"{len(tm):,} target segments from {tm.pid.nunique():,} subjects "
         f"({', '.join(f'{s}: {g.pid.nunique():,}' for s, g in tm.groupby('protocol'))}). "
         f"Calibration = first k segments of each subject's recording.", ""]
    metrics = {}
    for k in args.ks:
        ens = np.mean(preds[k], axis=0)
        d = frame(ens, k)
        d.to_csv(out / f"predictions_ensemble_k{k}.csv", index=False)
        L += [f"## k = {k} calibration segment{'s' if k > 1 else ''}", "",
              "### Segment level (pooled)", "",
              "| Data | Method | Segments / subjects | SBP MAE | SBP ME ± SD | SBP r | SBP BHS / IEEE / AAMI | DBP MAE | DBP ME ± SD | DBP r | DBP BHS / IEEE / AAMI |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        S = ["", "### Subject level (each subject counts equally)", "",
             "| Data | Subjects | SBP: mean / median per-subject MAE (model) | (cal. only) | subjects model better | p | "
             "DBP: mean / median per-subject MAE (model) | (cal. only) | subjects model better | p |", "|---|---|---|---|---|---|---|---|---|---|"]
        F = ["", "### Spread over the 5 fold models (pooled segment MAE, mean ± SD)", "",
             "| Data | SBP MAE | DBP MAE |", "|---|---|---|"]
        for name, ix in sources:
            q = d.loc[ix]
            for meth, col in [("**ensemble of 5 models**", "pred"), ("calibration value only", "cal_mean")]:
                qq = q.copy()
                for t in T:
                    qq[f"{t}_pred"] = q[f"{t}_{col}"]
                m = regression_metrics(qq, T, 5)
                s_, d_ = m["sbp"], m["dbp"]
                aami = lambda x: "pass" if x["AAMI_pass"] else "fail"
                L.append(f"| {name} | {meth} | {len(q):,} / {q.pid.nunique():,} | {s_['MAE']:.2f} | {s_['ME']:+.2f} ± {s_['SDE']:.2f} | "
                         f"{s_['r']:.2f} | {s_['BHS_grade']} / {s_['IEEE1708_grade']} / {aami(s_)} | {d_['MAE']:.2f} | "
                         f"{d_['ME']:+.2f} ± {d_['SDE']:.2f} | {d_['r']:.2f} | {d_['BHS_grade']} / {d_['IEEE1708_grade']} / {aami(d_)} |")
                metrics[f"k{k}/{name}/{col}"] = m
            sm, per_m = subject_level(q, "pred")
            sc, per_c = subject_level(q, "cal_mean")
            row = f"| {name} | {q.pid.nunique():,} |"
            for t in T:
                p = wilcoxon(per_m[t], per_c[t]).pvalue if len(per_m[t]) > 10 else float("nan")
                row += (f" {sm[t][0]:.2f} / {sm[t][1]:.2f} | {sc[t][0]:.2f} / {sc[t][1]:.2f} | "
                        f"{(per_m[t] < per_c[t]).mean():.0%} | {'< 0.001' if p < 0.001 else f'{p:.3f}'} |")
            S.append(row)
            fm = np.array([[np.abs(pf[ix.to_numpy(), j] - q[t].to_numpy()).mean() for j, t in enumerate(T)] for pf in preds[k]])
            F.append(f"| {name} | {fm[:, 0].mean():.2f} ± {fm[:, 0].std(ddof=1) if len(fm) > 1 else 0:.2f} | "
                     f"{fm[:, 1].mean():.2f} ± {fm[:, 1].std(ddof=1) if len(fm) > 1 else 0:.2f} |")
        L += S + F
        if k == min(args.ks):
            bins = [-np.inf, 1, 3, 6, 12, np.inf]
            lab = ["< 1 h", "1–3 h", "3–6 h", "6–12 h", "> 12 h"]
            d["tbin"] = pd.cut(d.hours_since_cal, bins, labels=lab, right=False)
            d["qbin"] = pd.cut(d.ecg_sqi, [-np.inf, 0.6, 0.8, 0.9, np.inf], labels=["< 0.6", "0.6–0.8", "0.8–0.9", "≥ 0.9"], right=False)
            for title, col in [("time since calibration", "tbin"), ("ECG quality (ecg_sqi)", "qbin")]:
                L += ["", f"### MAE by {title} (k = {k}, ensemble)", "",
                      "| Data | Bin | Segments | SBP MAE (model / cal. only) | DBP MAE (model / cal. only) |", "|---|---|---|---|---|"]
                for name, ix in sources:
                    for b, g in d.loc[ix].groupby(col, observed=True):
                        L.append(f"| {name} | {b} | {len(g):,} | {(g.sbp_pred - g.sbp).abs().mean():.2f} / {(g.sbp_cal_mean - g.sbp).abs().mean():.2f} | "
                                 f"{(g.dbp_pred - g.dbp).abs().mean():.2f} / {(g.dbp_cal_mean - g.dbp).abs().mean():.2f} |")
        L.append("")
    L += [f"Runtime {(time.time() - t_start) / 60:.1f} min. Reference BP: invasive arterial line (Aurora: cuff). "
          "Signals: PulseDB ECG_Record_F / PPG_Record_F processed exactly like the Aurora model input."]
    (out / "EXTERNAL_RESULTS.md").write_text("\n".join(L))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2, default=float)
    print("\n".join(L))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
