#!/usr/bin/env python
"""
One PulseDB adaptation run (plan: docs/PULSEDB_FINETUNING_PLAN.md, decisions D1-D16).

  python scripts/finetune_pulsedb.py --mode finetune --dataset MIMIC --fraction 10 --draw 0 --gpu 2
  python scripts/finetune_pulsedb.py --mode scratch  --dataset VitalDB --fraction 5 --draw 3 --gpu 3
  python scripts/finetune_pulsedb.py --all --mode finetune --gpu 2       # all 50 runs of one mode (skips finished)

mode finetune : starts from Aurora no-time fold model `draw` (full model trainable), lr 3e-5, patience 8,
                Aurora BP scaling (D8, D13, D16)
mode scratch  : random init (seed = 2026 + draw), Aurora no-time recipe (lr 3e-4, patience 10, DropPath
                warm-up 5), BP scaling from its own training subjects (D11, D13)
Subjects come ONLY from the frozen split files (role_d<draw>_f<fraction>): train / val for learning and early
stopping, test for the final numbers (never seen before). Epoch = max(training pairs, 20,000) samples (D16-B).
Test evaluation: first k segments of each test subject = calibration, k = 1 and 3 (D12), plus the
calibration-value-only baseline.
Output: runs/pulsedb_<mode>/<dataset>/f<fraction>/d<draw>/  best.pt, history.csv, train.log, config.yaml,
        provenance.json (incl. split hashes), predictions_test_k1.csv, predictions_test_k3.csv, metrics.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

DATASETS, FRACTIONS, DRAWS = ["MIMIC", "VitalDB"], [5, 10, 20, 50, 100], [0, 1, 2, 3, 4]
MIN_EPOCH_SAMPLES = 20_000
SETTINGS = {
    "finetune": {"train.lr": 3e-5, "train.epochs": 30, "train.patience": 8, "train.drop_path_warmup_epochs": 0},
    "scratch": {"train.lr": 3e-4, "train.epochs": 30, "train.patience": 10, "train.drop_path_warmup_epochs": 5},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["finetune", "scratch"], required=True)
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--fraction", type=int, choices=FRACTIONS)
    ap.add_argument("--draw", type=int, choices=DRAWS)
    ap.add_argument("--all", action="store_true", help="run every dataset x fraction x draw of this mode (skips finished runs)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--aurora", default="hicaf_bp_cal_notime", help="Aurora experiment with the 5 fold models / recipe")
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs, separate output folder")
    return ap.parse_args()


def run_one(args, dataset, fraction, draw, signals, meta_all, log_root):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, RandomSampler

    from src.config import load_config, save_config
    from src.data.dataset import BPCropDataset
    from src.data.pairs import PairDataset, calibration_split, eval_pairs
    from src.data.splits import target_stats
    from src.evaluation.metrics import evaluate_predictions, regression_metrics
    from src.models import build_model
    from src.training.trainer import Trainer
    from src.utils.logging_utils import get_logger
    from src.utils.reproducibility import provenance, set_seed

    tag = f"pulsedb_{args.mode}{'_smoke' if args.smoke else ''}"
    base_cfg = load_config(f"experiments/{args.aurora}.yaml",
                           [f"{k}={v}" for k, v in SETTINGS[args.mode].items()])
    runs_dir = Path(base_cfg["paths"]["runs_dir"])
    out = runs_dir / tag / dataset / f"f{fraction}" / f"d{draw}"
    if (out / "metrics.json").exists() and not args.smoke:
        print(f"skip (done): {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"{tag}_{dataset}_{fraction}_{draw}", out / "train.log")

    # ---------------------------------------------------------------- config
    if args.mode == "finetune":
        ck = torch.load(runs_dir / args.aurora / f"fold_{draw}" / "best.pt", map_location="cpu")
        cfg = ck["config"]
        for k, v in SETTINGS[args.mode].items():                     # same overrides on the checkpoint's config
            sec, key = k.split(".")
            cfg[sec][key] = v
    else:
        ck, cfg = None, base_cfg
    if int(cfg["model"].get("ctx_dim", 0)) > 0 or int(cfg["model"].get("demo_dim", 0)) > 0:
        sys.exit("model needs time context / demographics - not available in PulseDB")
    cfg["paths"]["data_dir"] = str(Path(args.data).resolve())
    cfg["experiment"].update({"name": f"{tag}/{dataset}/f{fraction}/d{draw}", "seed": 2026 + draw, "fold": f"d{draw}",
                              "pulsedb": {"mode": args.mode, "dataset": dataset, "fraction_pct_of_pool": fraction,
                                          "draw": draw, "aurora_model": f"{args.aurora}/fold_{draw}" if ck else None,
                                          "min_epoch_samples": MIN_EPOCH_SAMPLES}})
    if args.smoke:
        cfg["train"]["epochs"] = 2
    set_seed(2026 + draw, bool(cfg["experiment"].get("deterministic", False)))
    save_config(cfg, out / "config.yaml")
    prov = provenance(cfg, CODE_DIR, out)
    split_file = Path(args.data) / "splits" / f"splits_{dataset}.csv"
    manifest = json.load(open(Path(args.data) / "splits" / "splits_manifest.json"))
    sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
    if sha != manifest["files"][split_file.name]:
        sys.exit(f"{split_file} does not match the frozen manifest hash - stop")
    prov["pulsedb_split"] = {"file": str(split_file), "sha256": sha, "column": f"role_d{draw}_f{fraction}"}
    json.dump(prov, open(out / "provenance.json", "w"), indent=2, default=str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- data (roles from the frozen split only)
    roles = pd.read_csv(split_file).set_index("pid")[f"role_d{draw}_f{fraction}"]
    meta = meta_all[meta_all.protocol == dataset].copy()
    meta["role"] = meta.pid.map(roles)
    if meta.role.isna().any():
        sys.exit("subjects without a role in the split file - stop")
    if args.smoke:
        keep = pd.concat([meta[meta.role == r].pid.drop_duplicates().head(n) for r, n in [("train", 20), ("val", 6), ("test", 6)]])
        meta = meta[meta.pid.isin(keep)]
    parts = {r: meta[meta.role == r].sort_values("sample_id").reset_index(drop=True) for r in ["train", "val", "test"]}
    overlap = (set(parts["train"].pid) & set(parts["val"].pid)) | (set(parts["train"].pid) & set(parts["test"].pid)) \
        | (set(parts["val"].pid) & set(parts["test"].pid))
    if overlap:
        sys.exit(f"LEAKAGE: {len(overlap)} subjects in more than one role - stop")
    for r, d in parts.items():
        log.info(f"{r:5s}: {d.pid.nunique():5,} subjects | {len(d):8,} segments")

    targets = list(cfg["data"]["targets"])
    if ck is not None:
        y_mean, y_std = np.asarray(ck["stats"]["y_mean"], np.float32), np.asarray(ck["stats"]["y_std"], np.float32)
        log.info(f"BP scaling: Aurora statistics (mean {y_mean.round(1)}, sd {y_std.round(1)})")
    else:
        y_mean, y_std = target_stats(parts["train"], targets)
        log.info(f"BP scaling: own training subjects (mean {y_mean.round(1)}, sd {y_std.round(1)})")
    stats = {"y_mean": y_mean, "y_std": y_std, "aux_mean": np.zeros(0), "aux_std": np.ones(0)}
    prep = json.load(open(Path(args.data) / "config.json"))
    ch_idx = [prep.get("channels", ["ecg", "ppg"]).index(c) for c in cfg["data"]["inputs"]]
    crop = int(round(cfg["data"]["crop_s"] * int(prep["config"]["fs_out"])))

    def make(d, train):
        return BPCropDataset(d, signals, ch_idx, crop, y_mean, y_std, targets, [], None, None,
                             train=train, augment_cfg=cfg["data"]["augment"])

    lk = dict(batch_size=cfg["loader"]["batch_size"], num_workers=cfg["loader"]["num_workers"],
              pin_memory=device.type == "cuda", persistent_workers=cfg["loader"]["num_workers"] > 0)
    protocol = {r: calibration_split(parts[r], "initial", ["calibration"], 3) for r in ["val", "test"]}
    base_eval = {r: make(parts[r], False) for r in ["val", "test"]}

    def eval_loader(r, k):
        cal, tg = protocol[r]
        pairs = eval_pairs(parts[r], cal, tg, k)
        return DataLoader(PairDataset(base_eval[r], pairs=pairs), shuffle=False, **lk), pairs

    ds_train = PairDataset(make(parts["train"], True), pids=parts["train"].pid.to_numpy())
    n_epoch = max(len(ds_train), MIN_EPOCH_SAMPLES)
    sampler = RandomSampler(ds_train, replacement=len(ds_train) < MIN_EPOCH_SAMPLES, num_samples=n_epoch)
    dl_train = DataLoader(ds_train, sampler=sampler, drop_last=True, **lk)
    dl_val, val_pairs = eval_loader("val", 1)
    y_val = parts["val"][targets].to_numpy()[val_pairs[:, 0]]
    log.info(f"training pairs: {len(ds_train):,} targets, {n_epoch:,} samples per epoch | val pairs {len(val_pairs):,}")

    # ---------------------------------------------------------------- model
    model = build_model(cfg["model"], input_names=cfg["data"]["inputs"], n_targets=len(targets), n_aux=0).to(device)
    if ck is not None:
        model.load_state_dict(ck["model_state"], strict=True)
        log.info(f"initialised from Aurora {args.aurora}/fold_{draw} (all layers trainable)")
    else:
        log.info(f"random initialisation (seed {2026 + draw})")
    trainer = Trainer(cfg, model, dl_train, dl_val, y_val, stats, out, log, device)
    best_epoch, best_score = trainer.fit()
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model_state"])   # evaluate the best epoch

    # ---------------------------------------------------------------- test (touched once)
    d = parts["test"]
    cal, tg = protocol["test"]
    keep = ["sample_id", "pid", "protocol", "phase", "condition", "measurement", "is_exercise", "time_s", "ecg_sqi"] + targets
    results = {}
    for k in (1, 3):
        loader, pairs = eval_loader("test", k)
        bp = trainer.predict(loader, model=model)["bp"]
        agg = pd.DataFrame(bp, columns=targets).assign(t=pairs[:, 0]).groupby("t")[targets].mean()
        pred = d.iloc[agg.index.to_numpy()][[c for c in keep if c in d]].reset_index(drop=True)
        for t in targets:
            pred[f"{t}_pred"] = agg[t].to_numpy()
            tv = d[t].to_numpy()
            cm = {pid: tv[pos[:k]].mean() for pid, pos in cal.items()}
            pred[f"{t}_cal_mean"] = pred.pid.map(cm).to_numpy()
        pred.to_csv(out / f"predictions_test_k{k}.csv", index=False)
        res = evaluate_predictions(pred, parts["train"], min_rec=5)
        b = pred.copy()
        for t in targets:
            b[f"{t}_pred"] = pred[f"{t}_cal_mean"]
        res["baseline_calibration_only"] = regression_metrics(b, targets, 5)
        results[f"test_k{k}"] = res
        m = res["overall"] if "overall" in res else regression_metrics(pred, targets, 5)
        log.info(f"TEST k={k}: SBP MAE {m['sbp']['MAE']:.2f} | DBP MAE {m['dbp']['MAE']:.2f} "
                 f"({d.pid.nunique()} subjects, {len(pred):,} target segments)")
    info = {"mode": args.mode, "dataset": dataset, "fraction_pct_of_pool": fraction, "draw": draw,
            "best_epoch": best_epoch, "best_val_score": best_score,
            "n_subjects": {r: int(p.pid.nunique()) for r, p in parts.items()},
            "n_segments": {r: int(len(p)) for r, p in parts.items()}, "split_sha256": sha, "git": prov["git"].get("commit")}
    json.dump({"info": info, **results}, open(out / "metrics.json", "w"), indent=2, default=float)
    log.info(f"done -> {out}")
    del model, trainer
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import pandas as pd
    jobs = ([(ds, f, r) for ds in DATASETS for f in FRACTIONS for r in DRAWS] if args.all
            else [(args.dataset, args.fraction, args.draw)])
    if any(v is None for j in jobs for v in j):
        sys.exit("give --dataset, --fraction and --draw, or --all")
    meta = pd.read_csv(Path(args.data) / "metadata.csv", low_memory=False)
    print(f"loading {args.data}/signals.npy ...", flush=True)
    signals = np.load(Path(args.data) / "signals.npy")
    for i, (ds, f, r) in enumerate(jobs, 1):
        print(f"\n=== [{i}/{len(jobs)}] {args.mode} {ds} {f}% draw {r}", flush=True)
        run_one(args, ds, f, r, signals, meta, None)


if __name__ == "__main__":
    main()
