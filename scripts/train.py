#!/usr/bin/env python
"""
Train one model on one fold.

  python scripts/train.py --config experiments/hicaf_bp.yaml     --fold dev --gpu 2
  python scripts/train.py --config experiments/hicaf_bp_cal.yaml --fold 0   --gpu 2
  python scripts/train.py --config experiments/resnet1d.yaml     --fold dev --gpu 2 --smoke
  python scripts/train.py --config experiments/hicaf_bp.yaml --fold dev --set train.lr=1e-4 --name hicaf_lr1e-4

Two modes:
  standard     one recording in -> absolute SBP/DBP
  calibrated   (config calibration.enabled) target + same-subject calibration recording -> BP change;
               evaluated with the fixed calibration protocol in src/data/pairs.py for every k in
               calibration.eval_k, with baselines "calibration value only" and "uncalibrated model + shift".

Output: runs/<experiment name>/<fold_dev | fold_k>/
  config.yaml, provenance.json, train.log, history.csv, best.pt
  predictions_val.csv / predictions_test.csv     (calibrated: primary k; other k as predictions_test_k<k>.csv)
  metrics.json, RESULTS.md
and the registry runs/EXPERIMENTS.csv / .md is rebuilt from all run folders.
"""
import argparse
import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="experiment YAML (relative to configs/ or a path)")
    ap.add_argument("--fold", default="dev", help="'dev' or 0..4")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="override experiment name (folder under runs/)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=1e-4 model.depth=6")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs: checks the pipeline")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing run folder")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)        # must happen before importing torch

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from src.config import load_config, save_config
    from src.data.dataset import BPCropDataset
    from src.data.pairs import PairDataset, calibration_split, candidate_mask, eval_pairs, row_times
    from src.data.splits import assign_roles, load_metadata, smoke_subset, target_stats
    from src.evaluation.metrics import evaluate_predictions, regression_metrics
    from src.evaluation.report import regression_section, run_markdown
    from src.models import build_model
    from src.training.trainer import Trainer
    from src.utils.logging_utils import get_logger
    from src.utils.registry import rebuild_registry
    from src.utils.reproducibility import provenance, set_seed

    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["experiment"]["name"] = args.name
    if args.smoke:
        cfg["experiment"]["name"] += "_smoke"
        cfg["train"]["epochs"] = cfg["smoke"]["epochs"]
    cfg["experiment"]["fold"] = str(args.fold)
    cfg["experiment"]["overrides"] = args.set
    cal_cfg = cfg.get("calibration") or {}
    calibrated = bool(cal_cfg.get("enabled", False))

    fold_tag = "fold_dev" if str(args.fold) == "dev" else f"fold_{int(args.fold)}"
    out = Path(cfg["paths"]["runs_dir"]) / cfg["experiment"]["name"] / fold_tag
    if (out / "metrics.json").exists() and not args.overwrite and not args.smoke:
        sys.exit(f"{out} already has results. Use a new --name, or --overwrite if you really mean it.")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("train", out / "train.log")
    save_config(cfg, out / "config.yaml")
    prov = provenance(cfg, CODE_DIR, out)
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"].get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"experiment {cfg['experiment']['name']} | {fold_tag} | {'calibrated' if calibrated else 'standard'} | "
             f"device {device} ({prov['environment'].get('gpu')}) | git {prov['git'].get('commit')} "
             f"{'(uncommitted changes saved)' if prov['git'].get('dirty') else ''}")

    # ------------------------------------------------------------------ data
    meta, fs, channel_names = load_metadata(cfg)
    df = assign_roles(meta, cfg, args.fold)
    if args.smoke:
        df = smoke_subset(df, cfg, seed=int(cfg["experiment"]["seed"]))
    parts = {r: df[df.role == r].reset_index(drop=True) for r in ["train", "val", "test"]}
    for r, d in parts.items():
        log.info(f"{r:5s}: {len(d):6,} recordings | {d.pid.nunique():4d} subjects | "
                 f"{d.protocol.value_counts().to_dict()}")
    t0 = pd.Timestamp.now()
    signals = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy")     # ~0.7 GB into RAM
    log.info(f"signals {signals.shape} loaded in {(pd.Timestamp.now() - t0).total_seconds():.1f}s")

    inputs = list(cfg["data"]["inputs"])
    ch_idx = [channel_names.index(c) for c in inputs]
    targets = list(cfg["data"]["targets"])
    aux_cols = list(cfg["data"]["aux_targets"]) if cfg["loss"].get("aux_weight", 0) > 0 else []
    y_mean, y_std = target_stats(parts["train"], targets)
    a_mean, a_std = target_stats(parts["train"], aux_cols) if aux_cols else (np.zeros(0), np.ones(0))
    stats = {"y_mean": y_mean, "y_std": y_std, "aux_mean": a_mean, "aux_std": a_std}
    crop = int(round(cfg["data"]["crop_s"] * fs))
    min_rec = cfg["eval"]["min_recordings_within_subject"]

    def make(d, train):
        return BPCropDataset(d, signals, ch_idx, crop, y_mean, y_std, targets, aux_cols, a_mean, a_std,
                             train=train, augment_cfg=cfg["data"]["augment"])

    lk = dict(batch_size=cfg["loader"]["batch_size"], num_workers=cfg["loader"]["num_workers"],
              pin_memory=device.type == "cuda", persistent_workers=cfg["loader"]["num_workers"] > 0)

    if calibrated:
        protocol = {}
        for r in ["val", "test"]:
            cal, tg = calibration_split(parts[r], cal_cfg["phase"], cal_cfg["conditions"], int(cal_cfg["k_max"]))
            protocol[r] = (cal, tg)
            log.info(f"calibration protocol {r}: {len(cal)} subjects with calibration readings, "
                     f"{len(tg):,} target recordings")
        base_eval = {r: make(parts[r], False) for r in ["val", "test"]}
        use_ctx = bool(cal_cfg.get("time_context", False))
        times = {r: (row_times(parts[r]) if use_ctx else None) for r in ["train", "val", "test"]}
        demo_cols = list(cal_cfg.get("demographics") or [])
        demo = {r: None for r in ["train", "val", "test"]}
        if demo_cols:
            def demo_matrix(d):
                return pd.DataFrame({c: (d[c].map({"M": 1.0, "F": 0.0}) if c == "gender"
                                         else pd.to_numeric(d[c], errors="coerce")) for c in demo_cols})
            dm_tr = demo_matrix(parts["train"])
            mu, sd = dm_tr.mean(), dm_tr.std().replace(0, 1.0)
            for r in demo:
                dm = demo_matrix(parts[r])
                demo[r] = ((dm - mu) / sd).fillna(0.0).to_numpy(np.float32)   # missing -> training mean
            log.info(f"demographics {demo_cols}: missing values (train) "
                     f"{dm_tr.isna().mean().round(3).to_dict()}")
        if use_ctx:
            log.info("time context: " + ", ".join(f"{r} {np.isfinite(t).mean():.1%} parsable" for r, t in times.items()))

        def eval_loader(r, k):
            cal, tg = protocol[r]
            pairs = eval_pairs(parts[r], cal, tg, k)
            return DataLoader(PairDataset(base_eval[r], pairs=pairs, times=times[r], demo=demo[r]), shuffle=False, **lk), pairs

        p_protocol = float(cal_cfg.get("train_partner_protocol_prob", 0.0))
        ds_train = PairDataset(make(parts["train"], True), pids=parts["train"]["pid"].to_numpy(),
                               times=times["train"], p_protocol=p_protocol, demo=demo["train"],
                               cand=candidate_mask(parts["train"], cal_cfg["phase"], cal_cfg["conditions"]))
        log.info(f"training partners: protocol-matched with probability {p_protocol}")
        dl_val, val_pairs = eval_loader("val", int(cal_cfg["eval_k"][0]))
        y_val = parts["val"][targets].to_numpy()[val_pairs[:, 0]]
        log.info(f"training targets with a same-subject partner: {len(ds_train):,}")
    else:
        ds_train = make(parts["train"], True)
        dl_val = DataLoader(make(parts["val"], False), shuffle=False, **lk)
        y_val = parts["val"][targets].to_numpy()
    dl_train = DataLoader(ds_train, shuffle=True, drop_last=True, **lk)

    # ------------------------------------------------------------------ model
    model = build_model(cfg["model"], input_names=inputs, n_targets=len(targets), n_aux=len(aux_cols)).to(device)
    uncal_dir = None
    init_cfg = cfg.get("init") or {}
    if init_cfg:
        key = "encoder_checkpoint_dev" if str(args.fold) == "dev" else "encoder_checkpoint"
        ck = Path(init_cfg[key].format(runs_dir=cfg["paths"]["runs_dir"], fold=args.fold))
        state = torch.load(ck, map_location="cpu")["model_state"]
        model.encoder.load_state_dict(state, strict=True)        # strict: encoder config must match the checkpoint
        uncal_dir = ck.parent
        log.info(f"encoder initialised from {ck}")
    if calibrated and cal_cfg.get("uncal_baseline_dir"):          # baseline source independent of init
        key = "uncal_baseline_dir_dev" if str(args.fold) == "dev" else "uncal_baseline_dir"
        uncal_dir = Path(cal_cfg[key].format(runs_dir=cfg["paths"]["runs_dir"], fold=args.fold))
        log.info(f"uncalibrated baseline predictions from {uncal_dir}")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model {cfg['model']['name']}: {n_params / 1e6:.2f} M parameters | input {len(inputs)} x {crop}")

    trainer = Trainer(cfg, model, dl_train, dl_val, y_val, stats, out, log, device)
    best_epoch, best_score = trainer.fit()

    # ------------------------------------------------------------------ final predictions (test touched once)
    keep = ["sample_id", "pid", "role", "protocol", "phase", "measurement", "condition", "is_exercise"] + targets
    results = {}
    for r in ["val", "test"]:
        d = parts[r]
        if len(d) == 0:
            continue
        if not calibrated:
            loader = dl_val if r == "val" else DataLoader(make(d, False), shuffle=False, **lk)
            p = trainer.predict(loader)
            pred = d[keep].copy()
            for j, t in enumerate(targets):
                pred[f"{t}_pred"] = p["bp"][:, j]
                if "gates" in p:
                    pred[f"gate_{t}"] = p["gates"][:, j]
            if "aux" in p:
                for j, c in enumerate(aux_cols):
                    pred[f"{c}_true"] = d[c].to_numpy()
                    pred[f"{c}_pred"] = p["aux"][:, j]
            pred.to_csv(out / f"predictions_{r}.csv", index=False)
            results[r] = evaluate_predictions(pred, parts["train"], min_rec=min_rec)
            continue

        # ---- calibrated evaluation: one result per k, same target recordings for every k
        cal, tg = protocol[r]
        uncal_file = uncal_dir / f"predictions_{r}.csv" if uncal_dir is not None else None
        uncal = pd.read_csv(uncal_file).set_index("sample_id") if uncal_file and uncal_file.exists() else None
        sid = d["sample_id"].to_numpy()
        for k in [int(v) for v in cal_cfg["eval_k"]]:
            loader, pairs = eval_loader(r, k)
            bp = trainer.predict(loader)["bp"]
            agg = pd.DataFrame(bp, columns=targets)
            agg["t"] = pairs[:, 0]
            agg = agg.groupby("t")[targets].mean()                      # average over the k calibration readings
            pred = d.iloc[agg.index.to_numpy()][keep].reset_index(drop=True)
            pids = pred["pid"].to_numpy()
            for t in targets:
                pred[f"{t}_pred"] = agg[t].to_numpy()
                tv = d[t].to_numpy()
                cal_mean = {pid: tv[pos[:k]].mean() for pid, pos in cal.items()}
                pred[f"{t}_cal_mean"] = [cal_mean[x] for x in pids]
                if uncal is not None:
                    up = uncal[f"{t}_pred"]
                    shift = {pid: float((tv[pos[:k]] - up.loc[sid[pos[:k]]].to_numpy()).mean()) for pid, pos in cal.items()}
                    pred[f"{t}_uncal_shift_pred"] = up.loc[pred["sample_id"]].to_numpy() + np.array([shift[x] for x in pids])
            pred["n_cal"] = [min(k, len(cal[x])) for x in pids]
            res = evaluate_predictions(pred, parts["train"], min_rec=min_rec)
            for key, col in [("baseline_calibration_only", "cal_mean"), ("baseline_uncalibrated_shift", "uncal_shift_pred")]:
                if all(f"{t}_{col}" in pred for t in targets):
                    b = pred.copy()
                    for t in targets:
                        b[f"{t}_pred"] = pred[f"{t}_{col}"]
                    res[key] = regression_metrics(b, targets, min_rec)
            tag = r if k == int(cal_cfg["eval_k"][0]) else f"{r}_k{k}"
            pred.to_csv(out / f"predictions_{tag}.csv", index=False)
            results[tag] = res

    info = {"experiment": cfg["experiment"]["name"], "fold": str(args.fold), "best_epoch": best_epoch,
            "best_val_score": best_score, "n_params": n_params, "model": cfg["model"],
            "data": {k: cfg["data"][k] for k in ["dataset", "inputs", "crop_s", "train_without_exercise"]},
            "calibration": cal_cfg if calibrated else None, "init": init_cfg or None,
            "n": {r: int(len(d)) for r, d in parts.items()}, "git": prov["git"].get("commit")}
    json.dump({"info": info, **results}, open(out / "metrics.json", "w"), indent=2, default=float)
    md = run_markdown(f"{cfg['experiment']['name']} / {fold_tag}", info, results["val"], results.get("test"))
    for key in sorted(k for k in results if k not in ("val", "test")):
        md += "\n\n" + "\n".join(regression_section(results[key], key))
    (out / "RESULTS.md").write_text(md)
    log.info("\n" + md)

    # ------------------------------------------------------------------ registry (rebuilt from all run folders)
    rebuild_registry(cfg["paths"]["runs_dir"])
    log.info(f"done -> {out}")


if __name__ == "__main__":
    main()
