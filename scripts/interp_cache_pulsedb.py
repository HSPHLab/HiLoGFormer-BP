#!/usr/bin/env python
"""
PulseDB part of the interpretability cache (docs/INTERPRETABILITY_PLAN.md, sections 3-4: generalisation). Inference only.

  python scripts/interp_cache_pulsedb.py --gpu 2

Model: the frozen no-time model (runs/hicaf_bp_cal_notime, 5 Aurora folds; none has seen PulseDB). Exactly the zero-shot
protocol of scripts/eval_external.py (same targets: first 3 segments of each subject = calibration; same fast inference:
every segment encoded once, the head applied to the (target, calibration) pairs).
Per target segment and fold model: k = 3 estimate (mean of the 3 single-calibration estimates), their SD (A1), drift
||z_t - z_c|| (fused, PPG-only; mean over the 3 calibrations; B). Across models: the 5-model ensemble of every pair
(-> ensemble consistency) and the SD of the 5 models' k = 3 estimates (A2).
Verification: each fold's k = 3 estimate must reproduce the frozen predictions_fold{f}_k3.npz.
Output: runs/interpretability/pulsedb_v1notime.npz (+ VERIFY_PULSEDB.md). Refuses to overwrite.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="hicaf_bp_cal_notime")
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import BPCropDataset
    from src.data.pairs import calibration_split, eval_pairs
    from src.models import build_model

    out_dir = CODE_DIR / "runs/interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "pulsedb_v1notime.npz"
    if fp.exists():
        sys.exit(f"{fp} exists - frozen once built")
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False                    # as eval_external.py
    torch.backends.cudnn.allow_tf32 = False
    exp_dir = CODE_DIR / "runs" / args.experiment
    data = CODE_DIR / args.data
    meta = pd.read_csv(data / "metadata.csv", low_memory=False).sort_values("sample_id").reset_index(drop=True)
    prep = json.load(open(data / "config.json"))
    channels = prep.get("channels", ["ecg", "ppg"])
    signals = np.load(data / "signals.npy", mmap_mode="r")
    cal, targets = calibration_split(meta, "initial", ["calibration"], 3)
    pairs = eval_pairs(meta, cal, targets, 3)
    n_t = len(targets)
    pos_of = {int(t): i for i, t in enumerate(targets)}
    row = np.array([pos_of[int(t)] for t in pairs[:, 0]])                # target index of every pair
    n_per = np.bincount(row, minlength=n_t)
    print(f"PulseDB: {n_t:,} targets, {len(pairs):,} pairs (calibrations per target: {np.unique(n_per).tolist()})", flush=True)

    est = np.zeros((len(args.folds), n_t, 2), np.float32)
    cc = np.zeros((len(args.folds), n_t, 2), np.float32)
    drift_f = np.zeros((len(args.folds), n_t), np.float32)
    drift_p = np.zeros((len(args.folds), n_t), np.float32)
    ens_pair = np.zeros((len(pairs), 2), np.float64)
    verify = []
    for fi, fold in enumerate(args.folds):
        ck = torch.load(exp_dir / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        model = build_model(cfg["model"], input_names=cfg["data"]["inputs"], n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"])
        model.eval()
        ch_idx = [channels.index(c) for c in cfg["data"]["inputs"]]
        ds = BPCropDataset(meta, signals, ch_idx, int(round(cfg["data"]["crop_s"] * int(prep["config"]["fs_out"]))),
                           y_mean, y_std, T, [], None, None, train=False, augment_cfg=None)
        zs, zps = [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=dev.type == "cuda"):
                z, zp = model.encoder.encode(b["x"].to(dev, non_blocking=True))
                zs.append(z.float().cpu())
                zps.append(zp.float().cpu() if zp is not None else torch.zeros(len(z), 0))
        z, zp = torch.cat(zs), torch.cat(zps)
        lab = ds.y                                                       # standardised labels (Aurora statistics)
        bp, df_, dp_ = [], [], []
        with torch.no_grad():
            for s in range(0, len(pairs), 65536):
                a = torch.from_numpy(pairs[s:s + 65536, 0]); c = torch.from_numpy(pairs[s:s + 65536, 1])
                za, zc = z[a].to(dev), z[c].to(dev)
                feats = model._compare(za, zc)
                if model.use_ppg:
                    pa, pc = zp[a].to(dev), zp[c].to(dev)
                    feats += model._compare(pa, pc)
                    dp_.append((pa - pc).norm(dim=-1).cpu().numpy())
                bc = torch.from_numpy(lab[pairs[s:s + 65536, 1]]).to(dev)
                feats.append(model.bp_embed(bc))
                bp.append((bc + model.head(torch.cat(feats, dim=-1)).float()).cpu().numpy())
                df_.append((za - zc).norm(dim=-1).cpu().numpy())
        bp = np.concatenate(bp) * y_std + y_mean                            # mmHg, per pair
        df_ = np.concatenate(df_)
        dp_ = np.concatenate(dp_) if dp_ else np.full(len(pairs), np.nan)
        ens_pair += bp
        g = pd.DataFrame({"r": row, "s": bp[:, 0], "d": bp[:, 1], "df": df_, "dp": dp_}).groupby("r")
        m, sd = g[["s", "d"]].mean(), g[["s", "d"]].std()
        est[fi], cc[fi] = m.to_numpy(), sd.to_numpy()
        drift_f[fi], drift_p[fi] = g["df"].mean().to_numpy(), g["dp"].mean().to_numpy()
        frozen = np.load(exp_dir / "external_pulsedb" / f"predictions_fold{fold}_k3.npz")
        assert np.array_equal(frozen["sample_id"], meta.sample_id.to_numpy()[targets]), "target order differs from the frozen file"
        dif = np.abs(est[fi] - frozen["pred"])
        verify.append({"fold": fold, "targets": n_t, "max_abs_diff": float(dif.max()), "mean_abs_diff": float(dif.mean())})
        print(f"fold {fold}: done ({(time.time() - t_start) / 60:.1f} min) | vs frozen k = 3: max {dif.max():.4f}, mean {dif.mean():.5f} mmHg",
              flush=True)
        del model, z, zp
        torch.cuda.empty_cache()

    ens_pair /= len(args.folds)
    g = pd.DataFrame({"r": row, "s": ens_pair[:, 0], "d": ens_pair[:, 1]}).groupby("r")
    ens_est, ens_cc = g[["s", "d"]].mean().to_numpy(), g[["s", "d"]].std().to_numpy()
    tm = meta.iloc[targets]
    first_t = {pid: meta.time_s.iloc[pos[0]] for pid, pos in cal.items()}
    lab_mm = meta[T].to_numpy(np.float32)
    cal_mean = pd.DataFrame(lab_mm[pairs[:, 1]], columns=T).assign(r=row).groupby("r")[T].mean().to_numpy()
    np.savez_compressed(fp, sample_id=tm.sample_id.to_numpy(), pid=tm.pid.to_numpy().astype(str), protocol=tm.protocol.to_numpy().astype(str),
                        sbp=tm.sbp.to_numpy(np.float32), dbp=tm.dbp.to_numpy(np.float32), cal_mean=cal_mean.astype(np.float32),
                        hours_since_cal=((tm.time_s - tm.pid.map(first_t)) / 3600.0).to_numpy(np.float32),
                        est=est, cc=cc, drift_fused=drift_f, drift_ppg=drift_p,
                        ens_est=ens_est.astype(np.float32), ens_cc=ens_cc.astype(np.float32),
                        ens_spread=est.std(axis=0, ddof=1).astype(np.float32), folds=np.array(args.folds))
    v = pd.DataFrame(verify)
    L = ["# PulseDB interpretability cache — verification against the frozen zero-shot predictions (k = 3)", "",
         v.round(5).to_string(index=False), "",
         f"{n_t:,} targets; runtime {(time.time() - t_start) / 60:.1f} min."]
    (out_dir / "VERIFY_PULSEDB.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
