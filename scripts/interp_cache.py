#!/usr/bin/env python
"""
Shared inference cache for the interpretability analyses (docs/INTERPRETABILITY_PLAN.md, section 8). Inference only.

  python scripts/interp_cache.py --gpu 2                 # Aurora, v1, 5 folds, validation + test subjects

For every (target, calibration) pair that the frozen evaluation uses with k = 3 (exactly train.py's eval_pairs), from the
frozen v1 fold model:
  pred_full      the single-calibration estimate (the k = 3 estimate is the mean of the 3, as in train.py)      -> A1
  drift_fused    ||z_t - z_c|| of the shared encoder's fused ECG+PPG embedding; drift_ppg: PPG-only embedding    -> B
  pred_nofused   estimate with the fused comparison set to "no change" (z_t := z_c)                               -> C2
  pred_noppg     estimate with the PPG-only comparison set to "no change"
  pred_notime    estimate with the time context set to "same time as calibration"
Verification: the mean of the 3 single-calibration estimates must reproduce the frozen predictions_<split>_k3.csv and the
first one the frozen k = 1 predictions (differences only from mixed-precision rounding); the script reports both.
Output: runs/interpretability/pairs_v1.csv.gz (+ VERIFY.md, provenance.json). Refuses to overwrite.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/hicaf_bp_cal_notime")
    ap.add_argument("--tag", default="notime", help="output name: pairs_<tag>.csv.gz")
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="runs/interpretability")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import BPCropDataset
    from src.data.pairs import PairDataset, calibration_split, eval_pairs, row_times
    from src.data.splits import assign_roles, load_metadata, target_stats
    from src.models import build_model

    out = CODE_DIR / args.out
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"pairs_{args.tag}.csv.gz"
    if fp.exists():
        sys.exit(f"{fp} exists - the cache is frozen once built; delete it deliberately to rebuild")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = CODE_DIR / args.run
    rows, verify, ck_hash = [], [], {}
    t_start = time.time()

    for fold in args.folds:
        ck_path = run / f"fold_{fold}" / "best.pt"
        ck_hash[fold] = hashlib.sha256(ck_path.read_bytes()).hexdigest()[:16]
        ck = torch.load(ck_path, map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        cal_cfg = cfg["calibration"]
        meta, fs, channels = load_metadata(cfg)
        df = assign_roles(meta, cfg, fold)
        parts = {r: df[df.role == r].reset_index(drop=True) for r in ["train", *args.splits]}
        targets = list(cfg["data"]["targets"])
        y_mean, y_std = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        ym_chk, ys_chk = target_stats(parts["train"], targets)
        assert np.allclose(ym_chk, y_mean, atol=1e-3) and np.allclose(ys_chk, y_std, atol=1e-3), "training statistics differ"
        signals = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r")
        ch_idx = [channels.index(c) for c in cfg["data"]["inputs"]]
        crop = int(round(cfg["data"]["crop_s"] * fs))
        model = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=len(targets), n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        use_amp = bool(cfg["train"].get("amp_bf16", True)) and dev.type == "cuda"
        use_ctx = bool(cal_cfg.get("time_context", False))
        ysd, ymn = torch.tensor(y_std, device=dev), torch.tensor(y_mean, device=dev)

        def head(zt, zc, pt, pc, bp_cal, ctx):
            """HiCAF-BP-Cal head on given embeddings (identical to HiCAFBPCal.forward after encode); standardised BP."""
            feats = model._compare(zt, zc)
            if model.use_ppg:
                feats += model._compare(pt, pc)
            feats.append(model.bp_embed(bp_cal.to(zt.dtype)))
            if model.ctx_embed is not None:
                feats.append(model.ctx_embed(ctx.to(zt.dtype)))
            return bp_cal.float() + model.head(torch.cat(feats, dim=-1)).float()

        for r in args.splits:
            d = parts[r]
            cal, tg = calibration_split(d, cal_cfg["phase"], cal_cfg["conditions"], int(cal_cfg["k_max"]))
            pairs = eval_pairs(d, cal, tg, args.k)
            times_all = row_times(d)                                     # for the analyses (time since calibration)
            times = times_all if use_ctx else None                          # model input only if the model uses it
            base = BPCropDataset(d, signals, ch_idx, crop, y_mean, y_std, targets, [], None, None, train=False,
                                 augment_cfg=cfg["data"].get("augment"))
            dl = DataLoader(PairDataset(base, pairs=pairs, times=times), batch_size=args.batch, shuffle=False,
                            num_workers=8, pin_memory=dev.type == "cuda")
            res = {k: [] for k in ["full", "nofused", "noppg", "notime", "drift_fused", "drift_ppg"]}
            with torch.no_grad():
                for b in dl:
                    x, xc = b["x"].to(dev), b["x_cal"].to(dev)
                    bp_cal = b["bp_cal"].to(dev)
                    ctx = b["ctx"].to(dev) if "ctx" in b else None
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                        z, zp = model.encoder.encode(torch.cat([x, xc], dim=0))
                        n = x.shape[0]
                        zt, zc = z[:n], z[n:]
                        pt, pc = (zp[:n], zp[n:]) if zp is not None else (None, None)
                        ctx_same = None
                        if ctx is not None:                                    # target time := calibration time
                            ctx_same = ctx.clone()
                            ctx_same[:, 0:2] = ctx[:, 2:4]
                            ctx_same[:, 4] = 0.0
                        outs = {"full": head(zt, zc, pt, pc, bp_cal, ctx),
                                "nofused": head(zc, zc, pt, pc, bp_cal, ctx),
                                "noppg": head(zt, zc, pc, pc, bp_cal, ctx) if pt is not None else head(zt, zc, pt, pc, bp_cal, ctx),
                                "notime": head(zt, zc, pt, pc, bp_cal, ctx_same) if ctx is not None else head(zt, zc, pt, pc, bp_cal, ctx)}
                    for k_, v in outs.items():
                        res[k_].append((v.float() * ysd + ymn).cpu().numpy())
                    res["drift_fused"].append((zt.float() - zc.float()).norm(dim=-1).cpu().numpy())
                    res["drift_ppg"].append(((pt.float() - pc.float()).norm(dim=-1) if pt is not None
                                             else torch.full((n,), float("nan"), device=dev)).cpu().numpy())
            res = {k_: np.concatenate(v) for k_, v in res.items()}
            t_pos, c_pos = pairs[:, 0], pairs[:, 1]
            rank = np.zeros(len(pairs), int)
            pids = d["pid"].to_numpy()
            for i, (t, c) in enumerate(pairs):
                rank[i] = int(np.where(cal[pids[t]] == c)[0][0])
            frame = pd.DataFrame({
                "fold": fold, "split": r, "sample_id": d["sample_id"].to_numpy()[t_pos], "pid": pids[t_pos],
                "cal_sample_id": d["sample_id"].to_numpy()[c_pos], "cal_rank": rank,
                "hours_since_cal": times_all[t_pos] - times_all[c_pos],
                "sbp": d["sbp"].to_numpy()[t_pos], "dbp": d["dbp"].to_numpy()[t_pos],
                "sbp_cal": d["sbp"].to_numpy()[c_pos], "dbp_cal": d["dbp"].to_numpy()[c_pos],
                "drift_fused": res["drift_fused"], "drift_ppg": res["drift_ppg"]})
            for k_ in ["full", "nofused", "noppg", "notime"]:
                frame[f"sbp_pred_{k_}"], frame[f"dbp_pred_{k_}"] = res[k_][:, 0], res[k_][:, 1]
            rows.append(frame)
            # verification against the frozen predictions
            for kk, name in [(args.k, f"predictions_{r}_k{args.k}.csv"), (1, f"predictions_{r}.csv")]:
                frozen_fp = run / f"fold_{fold}" / name
                if not frozen_fp.exists():
                    continue
                frozen = pd.read_csv(frozen_fp).set_index("sample_id")
                mine = frame[frame.cal_rank < kk].groupby("sample_id")[["sbp_pred_full", "dbp_pred_full"]].mean()
                common = mine.index.intersection(frozen.index)
                dif = np.abs(mine.loc[common].to_numpy() - frozen.loc[common, ["sbp_pred", "dbp_pred"]].to_numpy())
                verify.append({"fold": fold, "split": r, "k": kk, "n_targets": len(mine), "n_frozen": len(frozen),
                               "matched": len(common), "max_abs_diff": float(dif.max()), "mean_abs_diff": float(dif.mean())})
            print(f"fold {fold} {r}: {len(pairs):,} pairs ({len(tg):,} targets) | {(time.time() - t_start) / 60:.1f} min", flush=True)

    allp = pd.concat(rows, ignore_index=True)
    allp.to_csv(fp, index=False, compression="gzip")
    v = pd.DataFrame(verify)
    L = ["# Interpretability cache — verification against the frozen v1 predictions", "",
         "The k = 3 estimate is the mean of the three single-calibration estimates (train.py: groupby target, mean); k = 1 uses the",
         "first calibration reading. Differences come only from bf16 mixed precision (same setting as the frozen evaluation).", "",
         v.round(4).to_string(index=False), "",
         f"Pairs cached: {len(allp):,} ({allp.groupby('split').size().to_dict()}); runtime {(time.time() - t_start) / 60:.1f} min."]
    (out / f"VERIFY_{args.tag}.md").write_text("\n".join(L))
    json.dump({"run": args.run, "checkpoints_sha256_16": ck_hash, "k": args.k, "splits": args.splits,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(out / f"provenance_{args.tag}.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
