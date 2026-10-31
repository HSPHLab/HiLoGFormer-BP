#!/usr/bin/env python
"""
Train and evaluate a re-implemented published method on our Aurora-BP folds (docs/REIMPLEMENTATION_PLAN.md, R1-R9).

  python scripts/train_published.py --method hgctnet --fold 0 --gpu 2 [--smoke]

Same data, folds, subjects and test recordings as our model (R1); ECG + PPG, the same fixed 10 s window (R2, R3);
baseline BP from our calibration readings (R4); handcrafted features from data_processed/features_pwa (R5);
published training settings with early stopping on our validation subjects (R6).
Outputs (runs/pub_<method>/fold_<k>/), in the same format as our model so the same report scripts apply:
  best.pt (weights + every statistic needed at test time), history.csv, train.log, config.json, provenance.json,
  predictions_test.csv (k = 1), predictions_test_k3.csv (k = 3), metrics.json
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
T = ["sbp", "dbp"]

METHODS = {
    "hgctnet": {  # docs/reimplementation/HGCTNet.md
        "features": "aurora_v1", "k_train": 3, "loss": "huber", "huber_delta": 4.0,
        "lr": 1e-3, "betas": (0.9, 0.999), "weight_decay": 1e-4, "batch_size": 64, "max_epochs": 200, "patience": 20,
        "model": {"num_hid": 8, "depth": 2, "num_heads": 2},
    },
    "bpbls": {  # docs/reimplementation/BpBLS.md (official code values, rule B0)
        "features": "aurora_v1", "k_train": 3, "epochs": 50, "lr": 1.0, "weight_decay": 2 ** -12,
        "step_size": 50, "step_gamma": 0.9, "ppr_gamma": 2 ** -7, "ridge_lambda": 2 ** -8, "beta_cal": 0.29,
        "model": {"feature_nodes": 10, "feature_windows": 46, "enhancement_nodes": 28},
    },
    "runet": {  # docs/reimplementation/rU-Net.md (paper only; decisions U1-U8)
        "features": "aurora_v1", "calibration_free": True, "k_train": 3, "lr": 1e-5, "batch_size": 32, "max_epochs": 512,
        "patience": 30, "loss": "mse_standardised", "model": {},
    },
    "utransbpnet": {  # docs/reimplementation/UTransBPNet.md (paper only; decisions T1-T8)
        "features": "aurora_v1", "calibration_free": True, "k_train": 3, "lr": 9e-4, "weight_decay": 1e-3,
        "batch_size": 32, "max_epochs": 200, "patience": 20, "loss": "mse_minmax",
        "model": {"depth": 12, "heads": 8, "ff": 1024, "dropout": 0.1},
    },
    "ragcbpnet": {  # docs/reimplementation/RAGCBPNet.md (preprint only; decisions G1-G9)
        "features": "aurora_v1", "k_train": 1, "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 128, "max_epochs": 200,
        "patience": 20, "loss": "mae_standardised", "model": {"heads": 4},
    },
}


# ---------------------------------------------------------------------------------------------- data preparation
def demographics(df):
    """age, gender (F=0, M=1), BMI, history of hypertension (managed/unmanaged=1, norm=0, else NaN)."""
    h, w = df["height"].astype(float), df["weight"].astype(float)
    imperial = np.nanmedian(h) < 100                      # inches / pounds
    bmi = 703.0 * w / h ** 2 if imperial else w / (h / 100.0) ** 2
    htn = df["self_report_htn"].astype(str).str.lower().map({"managed": 1.0, "unmanaged": 1.0, "norm": 0.0})
    g = df["gender"].astype(str).str.upper().map({"F": 0.0, "M": 1.0})
    return pd.DataFrame({"age": df["age"].astype(float).to_numpy(), "gender": g.to_numpy(), "bmi": bmi.to_numpy(),
                         "hhp": htn.to_numpy()}, index=df.index), ("imperial" if imperial else "metric")


def demographics_runet(df):
    """rU-Net: sex (F=0, M=1), age, height [cm], weight [kg], BMI; Aurora's imperial units converted (U3)."""
    h, w = df["height"].astype(float), df["weight"].astype(float)
    imperial = np.nanmedian(h) < 100
    h_cm, w_kg = (h * 2.54, w * 0.45359237) if imperial else (h, w)
    g = df["gender"].astype(str).str.upper().map({"F": 0.0, "M": 1.0})
    return pd.DataFrame({"gender": g.to_numpy(), "age": df["age"].astype(float).to_numpy(), "height": h_cm.to_numpy(),
                         "weight": w_kg.to_numpy(), "bmi": (w_kg / (h_cm / 100.0) ** 2).to_numpy()},
                        index=df.index), ("imperial->metric" if imperial else "metric")


def side_inputs_runet(part, base, demo, feat_names, stats):
    """rU-Net side input: the 5 demographics, standardised with training statistics (missing -> training mean)."""
    d = demo.loc[part.index].to_numpy(float)
    return ((np.where(np.isnan(d), stats["demo_fill"], d) - stats["demo_mean"]) / stats["demo_std"]).astype(np.float32)


def demographics_ragcbpnet(df):
    """RAGCBPNet side information: the 5 rU-Net demographics (metric) + circadian encoding of the acquisition hour (G6)."""
    demo, units = demographics_runet(df)
    if "date_time" in df:
        t = pd.to_datetime(df["date_time"], errors="coerce")
        h = (t.dt.hour + t.dt.minute / 60.0).to_numpy(float)
    else:
        h = np.full(len(df), np.nan)
    demo["hour_cos"], demo["hour_sin"] = np.cos(2 * np.pi * h / 24), np.sin(2 * np.pi * h / 24)
    return demo, units


def side_inputs_ragcbpnet(part, base, demo, feat_names, stats):
    """[7 standardised demographics + hour, standardised baseline SBP/DBP, standardised handcrafted features] (NaN-free)."""
    d = demo.loc[part.index].to_numpy(float)
    d = (np.where(np.isnan(d), stats["demo_fill"], d) - stats["demo_mean"]) / stats["demo_std"]
    b = (np.stack([base[p] for p in part["pid"]]) - stats["base_mean"]) / stats["base_std"]
    f = part[feat_names].to_numpy(float)
    f = (np.where(np.isnan(f), stats["feat_fill"], f) - stats["feat_mean"]) / stats["feat_std"]
    return np.concatenate([d, b, f], axis=1).astype(np.float32)


def side_inputs_none(part, base, demo, feat_names, stats):
    """Signal-only methods (UTransBPNet): a constant placeholder column (ignored by the model)."""
    return np.zeros((len(part), 1), np.float32)


def diagnose_nonfinite(model, X, dev, log):
    """Locate where non-finite validation outputs arise (eval mode): input, buffers (BatchNorm running statistics),
    first module with a non-finite output, and whether the transformer's eval fast path is responsible."""
    import torch
    model.eval()
    xb = fb = None
    with torch.no_grad():
        for i in range(0, len(X[0]), 256):
            out = model(X[0][i:i + 256].to(dev), X[1][i:i + 256].to(dev))
            if not torch.isfinite(out).all():
                xb, fb = X[0][i:i + 256].to(dev), X[1][i:i + 256].to(dev)
                log.info(f"DIAG first non-finite validation batch starts at row {i}; non-finite rows in it: "
                         f"{(~torch.isfinite(out).all(1)).sum().item()} / {len(out)}")
                break
    if xb is None:
        log.info("DIAG no non-finite outputs reproduced")
        return
    log.info(f"DIAG input finite: {bool(torch.isfinite(xb).all())}; |input| max {xb.abs().max().item():.2f}")
    bad = [n for n, b in model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
    log.info(f"DIAG non-finite buffers: {bad[:10] if bad else 'none'}")
    seen = []
    hooks = [m.register_forward_hook(lambda mod, i, o, name=n: seen.append((name, o))) for n, m in model.named_modules() if n]
    with torch.no_grad():
        model(xb, fb)
    for h in hooks:
        h.remove()
    for name, o in seen:
        t = o if torch.is_tensor(o) else (o[0] if isinstance(o, (tuple, list)) and len(o) and torch.is_tensor(o[0]) else None)
        if t is not None and t.is_floating_point():
            if not torch.isfinite(t).all():
                log.info(f"DIAG first module with a non-finite output (execution order): {name}")
                break
            if name.endswith("transformer") or name.startswith("e4"):
                log.info(f"DIAG {name}: |output| max {t.abs().max().item():.3g}")
    tr = getattr(model, "transformer", None)
    if tr is not None:
        drop = [m for m in tr.modules() if isinstance(m, torch.nn.Dropout)]
        ps = [m.p for m in drop]
        for m in drop:
            m.p = 0.0
        for l in tr.layers:
            l.train()                                                  # bypasses the eval fast path; dropout off
        with torch.no_grad():
            out2 = model(xb, fb)
        for m, p in zip(drop, ps):
            m.p = p
        model.eval()
        log.info(f"DIAG same batch with the transformer outside its eval fast path: finite = {bool(torch.isfinite(out2).all())}")


def baselines(part, cal, k):
    """Baseline SBP/DBP per subject = mean of the first k calibration readings (fewer if fewer exist)."""
    lab = part[T].to_numpy(float)
    return {pid: lab[pos[:k]].mean(axis=0) for pid, pos in cal.items() if len(pos)}


def side_inputs(part, base, demo, feat_names, stats):
    """[baseline SBP, baseline DBP, standardised demographics, standardised features] per row (NaN-free)."""
    b = np.stack([base[p] for p in part["pid"]])
    d = demo.loc[part.index].to_numpy(float)
    d = np.where(np.isnan(d), stats["demo_fill"], d)
    d = (d - stats["demo_mean"]) / stats["demo_std"]
    f = part[feat_names].to_numpy(float)
    f = np.where(np.isnan(f), stats["feat_fill"], f)
    f = (f - stats["feat_mean"]) / stats["feat_std"]
    return np.concatenate([b, d, f], axis=1).astype(np.float32)


def fit_stats(train, demo, feat_names):
    """Imputation and scaling statistics from TRAINING subjects only (D10)."""
    d = demo.loc[train.index].to_numpy(float)
    f = train[feat_names].to_numpy(float)
    demo_fill = np.nanmean(d, axis=0)                       # demographics: training mean (R7)
    feat_fill = np.nanmedian(f, axis=0)                     # features: training median (section 8)
    d_imp = np.where(np.isnan(d), demo_fill, d)
    f_imp = np.where(np.isnan(f), feat_fill, f)
    sd = lambda x: np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    return {"demo_fill": demo_fill, "demo_mean": d_imp.mean(axis=0), "demo_std": sd(d_imp),
            "feat_fill": feat_fill, "feat_mean": f_imp.mean(axis=0), "feat_std": sd(f_imp),
            "demo_min": d_imp.min(axis=0), "demo_max": d_imp.max(axis=0),
            "feat_min": f_imp.min(axis=0), "feat_max": f_imp.max(axis=0)}


def side_inputs_bpbls(part, base, demo, feat_names, stats):
    """BpBLS input: [demographics, features] min-max scaled with training statistics (official normalize_data),
    then baseline [DBP, SBP] in mmHg appended (official torch.cat((x, cal)))."""
    rng = lambda lo, hi: np.where(hi - lo > 1e-8, hi - lo, 1.0)
    d = demo.loc[part.index].to_numpy(float)
    d = (np.where(np.isnan(d), stats["demo_fill"], d) - stats["demo_min"]) / rng(stats["demo_min"], stats["demo_max"])
    f = part[feat_names].to_numpy(float)
    f = (np.where(np.isnan(f), stats["feat_fill"], f) - stats["feat_min"]) / rng(stats["feat_min"], stats["feat_max"])
    b = np.stack([base[p] for p in part["pid"]])[:, [1, 0]]          # [SBP, DBP] -> [DBP, SBP]
    return np.concatenate([d, f, b], axis=1).astype(np.float32)


def windows(part, signals, ch_idx, seg=1250):
    s0 = (part["offset"].astype(int) + part["crop_start_eval"].astype(int)).to_numpy()
    return np.stack([np.asarray(signals[a:a + seg, ch_idx], np.float32).T for a in s0])


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.config import load_config
    from src.data.pairs import calibration_split
    from src.data.splits import assign_roles, load_metadata
    from src.evaluation.metrics import evaluate_predictions
    from src.features.pwa_features import FEATURE_NAMES
    from src.utils.logging_utils import get_logger
    from src.utils.reproducibility import provenance, set_seed

    M = METHODS[args.method]
    cfg = load_config("experiments/hicaf_bp_cal_t_scratch.yaml")          # data / folds / calibration settings of v1
    cfg["experiment"]["name"] = f"pub_{args.method}{'_smoke' if args.smoke else ''}"
    cfg["published"] = {"method": args.method, **{k: v for k, v in M.items() if k != "model"}, "model": M["model"]}
    out = Path(cfg["paths"]["runs_dir"]) / cfg["experiment"]["name"] / f"fold_{args.fold}"
    if (out / "metrics.json").exists() and not args.smoke:
        sys.exit(f"{out} already finished")
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"{args.method}_{args.fold}", out / "train.log")
    seed = int(cfg["experiment"]["seed"]) + (0 if str(args.fold) == "dev" else int(args.fold))
    set_seed(seed, False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data: v1 metadata + roles of this fold + handcrafted features
    meta, fs, ch = load_metadata(cfg)
    df = assign_roles(meta, cfg, args.fold)
    feat_file = CODE_DIR / "data_processed" / "features_pwa" / f"{M['features']}.csv"
    feats = pd.read_csv(feat_file)
    df = df.merge(feats[["sample_id", *FEATURE_NAMES]], on="sample_id", how="left", validate="one_to_one")
    demo, units = {"runet": demographics_runet, "ragcbpnet": demographics_ragcbpnet}.get(args.method, demographics)(df)
    if args.smoke:
        keep = pd.concat([df[df.role == r].pid.drop_duplicates().head(n) for r, n in [("train", 40), ("val", 10), ("test", 10)]])
        df = df[df.pid.isin(keep)]
    cal_cfg = cfg["calibration"]
    parts, cal, tg = {}, {}, {}
    for r in ("train", "val", "test"):
        parts[r] = df[df.role == r].sort_values("sample_id").copy()
        cal[r], tg[r] = calibration_split(parts[r].reset_index(drop=True), cal_cfg["phase"], cal_cfg["conditions"], 3)
    # training rows: every recording of subjects that have calibration readings (H6)
    tr = parts["train"]
    base_tr = baselines(tr.reset_index(drop=True), cal["train"], M["k_train"])
    if not M.get("calibration_free"):
        tr = tr[tr.pid.isin(base_tr)]                    # calibrated methods need a baseline (H6)
    stats = fit_stats(tr, demo, FEATURE_NAMES)
    if args.method == "ragcbpnet":                                     # baseline scaling from training rows only
        b_tr = np.stack([base_tr[p] for p in tr["pid"]])
        stats["base_mean"], stats["base_std"] = b_tr.mean(0), b_tr.std(0)
        log.info(f"hour of day available for {np.isfinite(demo.loc[tr.index, 'hour_cos']).mean():.1%} of training rows")
    log.info(f"{args.method} fold {args.fold}: train {tr.pid.nunique()} subjects / {len(tr):,} recordings "
             f"(excluded without calibration: {parts['train'].pid.nunique() - tr.pid.nunique()}); "
             f"val {parts['val'].pid.nunique()} / test {parts['test'].pid.nunique()} subjects; demographics units: {units}; "
             f"feature NaN in training rows: {tr[FEATURE_NAMES].isna().any(axis=1).mean():.1%}")
    signals = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r")
    ch_idx = [ch.index(c) for c in ("ecg", "ppg")]

    side_fn = {"hgctnet": side_inputs, "bpbls": side_inputs_bpbls, "runet": side_inputs_runet,
               "utransbpnet": side_inputs_none, "ragcbpnet": side_inputs_ragcbpnet}[args.method]

    def tensors(part_sorted, positions, base):
        p = part_sorted.iloc[positions]
        return (torch.from_numpy(windows(p, signals, ch_idx)), torch.from_numpy(side_fn(p, base, demo, FEATURE_NAMES, stats)),
                torch.from_numpy(p[T].to_numpy(np.float32)))

    Xtr = tensors(tr, np.arange(len(tr)), base_tr)
    base_val = baselines(parts["val"].reset_index(drop=True), cal["val"], 3)
    Xva = tensors(parts["val"], tg["val"], base_val)

    @torch.no_grad()
    def predict_any(mdl, X):
        mdl.eval()
        return np.concatenate([mdl(x.to(dev), f.to(dev)).float().cpu().numpy()
                               for x, f in zip(torch.split(X[0], 512), torch.split(X[1], 512))])

    if args.method == "bpbls":
        from src.published.bpbls import BpBLS, ridge_regression_pseudo_inverse_reg
        X = Xtr[1].to(dev)                                                # [demo, features, base DBP, base SBP]
        Y = torch.from_numpy(tr[["dbp", "sbp"]].to_numpy(np.float32)).to(dev)
        pp_subj = (tr.sbp - tr.dbp).groupby(tr.pid).mean()
        PP = torch.from_numpy(tr.pid.map(pp_subj).to_numpy(np.float32)[:, None]).to(dev)
        CAL, b = X[:, -2:], M["beta_cal"]
        model = BpBLS(in_dim=X.shape[1], beta_cal=b, **M["model"]).to(dev)
        net = model.bls
        n_params = sum(p.numel() for p in net.parameters())
        log.info(f"BpBLS: {n_params:,} parameters ({X.shape[1]} inputs); full-batch training on {len(tr):,} recordings")
        opt = torch.optim.Adam(net.parameters(), lr=M["lr"], weight_decay=M["weight_decay"])
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=M["step_size"], gamma=M["step_gamma"])
        mse = torch.nn.MSELoss()
        epochs = 2 if args.smoke else M["epochs"]
        hist, t0 = [], time.time()
        for ep in range(1, epochs + 1):                                   # stage 1: node weights (official SBPA loop)
            net.train()
            outputs, _ = net(X)
            outputs = outputs * b + CAL * (1 - b)
            loss1 = mse(outputs, Y)
            loss2 = mse(outputs[:, 1:] - outputs[:, 0:1], PP)
            loss = loss1 + M["ppr_gamma"] * loss2
            net.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            hist.append({"epoch": ep, "loss": loss.item(), "loss_bp": loss1.item(), "loss_ppr": loss2.item()})
            if ep in (1, 2, 10, 25, epochs):
                log.info(f"epoch {ep:3d} | loss {loss.item():.2f} (BP {loss1.item():.2f}, PPR {loss2.item():.2f}) | {time.time() - t0:.1f}s")
        net.eval()                                                        # stage 2: closed-form output weights
        with torch.no_grad():
            _, A = net(X)
            _, W = ridge_regression_pseudo_inverse_reg(A, Y, M["ridge_lambda"], M["ppr_gamma"], PP)
            model.W.copy_(W)
        pv = predict_any(model, Xva)
        mae = np.abs(pv - Xva[2].numpy()).mean(axis=0)
        best_ep, best = epochs, float(mae.mean())
        log.info(f"closed-form output weights done | validation (information only, no selection): MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f}")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        torch.save({"model_state": model.state_dict(), "stats": {k: v.tolist() for k, v in stats.items()},
                    "feature_names": FEATURE_NAMES, "model_args": {"in_dim": int(X.shape[1]), "beta_cal": b, **M["model"]},
                    "method": args.method, "k_train": M["k_train"], "demographic_units": units, "epoch": epochs}, out / "best.pt")

    if args.method == "runet":
        from src.published.runet import rUNetBP
        model = rUNetBP(n_demo=Xtr[1].shape[1]).to(dev)
        y_mean, y_std = Xtr[2].numpy().mean(0), Xtr[2].numpy().std(0)            # training labels only (U5)
        model.y_mean.copy_(torch.tensor(y_mean)); model.y_std.copy_(torch.tensor(y_std))
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"rU-Net: {n_params / 1e6:.2f} M parameters; label scaling mean {y_mean.round(1)}, sd {y_std.round(1)}")
        opt = torch.optim.Adam(model.parameters(), lr=M["lr"])
        dl = DataLoader(TensorDataset(*Xtr), batch_size=M["batch_size"], shuffle=True, drop_last=True)
        ym, ys = model.y_mean, model.y_std
        max_epochs = 2 if args.smoke else M["max_epochs"]
        best, best_ep, wait, hist, t0 = np.inf, -1, 0, [], time.time()
        for ep in range(1, max_epochs + 1):
            model.train()
            tl = []
            for x, f, y in dl:
                opt.zero_grad(set_to_none=True)
                y = y.to(dev)
                loss = torch.nn.functional.mse_loss(model.forward_std(x.to(dev), f.to(dev)), (y - ym) / ys)
                loss.backward()
                opt.step()
                tl.append(loss.item())
            pv = predict_any(model, Xva)
            mae = np.abs(pv - Xva[2].numpy()).mean(axis=0)
            score = float(mae.mean())
            hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_sbp_mae": mae[0], "val_dbp_mae": mae[1], "val_score": score})
            star = score < best
            if star:
                best, best_ep, wait = score, ep, 0
                torch.save({"model_state": model.state_dict(), "stats": {k: v.tolist() for k, v in stats.items()},
                            "feature_names": FEATURE_NAMES, "model_args": {"n_demo": int(Xtr[1].shape[1])}, "method": args.method,
                            "calibration_free": True, "demographic_units": units, "epoch": ep}, out / "best.pt")
            else:
                wait += 1
            log.info(f"epoch {ep:3d} | train {np.mean(tl):.4f} | val MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f} | "
                     f"{time.time() - t0:.0f}s{'  *best*' if star else ''}")
            if wait >= M["patience"]:
                log.info(f"early stop after {ep} epochs (patience {M['patience']})")
                break
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model_state"])

    if args.method == "utransbpnet":
        from src.published.utransbpnet import UTransBPNet
        model = UTransBPNet(seq_len=Xtr[0].shape[-1], **M["model"]).to(dev)
        y_lab = Xtr[2].numpy()
        y_min, y_rng = y_lab.min(0), y_lab.max(0) - y_lab.min(0)                    # training labels only (T5)
        model.y_min.copy_(torch.tensor(y_min)); model.y_range.copy_(torch.tensor(y_rng))
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"UTransBPNet: {n_params / 1e6:.2f} M parameters (paper 32.55 M); label min-max {y_min.round(1)} + {y_rng.round(1)}")
        opt = torch.optim.Adam(model.parameters(), lr=M["lr"], weight_decay=M["weight_decay"])
        dl = DataLoader(TensorDataset(*Xtr), batch_size=M["batch_size"], shuffle=True, drop_last=True)
        ymn, yrg = model.y_min, model.y_range
        max_epochs = 2 if args.smoke else M["max_epochs"]
        best, best_ep, wait, hist, t0 = np.inf, -1, 0, [], time.time()
        for ep in range(1, max_epochs + 1):
            model.train()
            tl = []
            for x, f, y in dl:
                opt.zero_grad(set_to_none=True)
                y = y.to(dev)
                loss = torch.nn.functional.mse_loss(model.forward_scaled(x.to(dev)), (y - ymn) / yrg)
                loss.backward()
                opt.step()
                tl.append(loss.item())
            pv = predict_any(model, Xva)
            mae = np.abs(pv - Xva[2].numpy()).mean(axis=0)
            score = float(mae.mean())
            hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_sbp_mae": mae[0], "val_dbp_mae": mae[1], "val_score": score})
            star = score < best
            if star:
                best, best_ep, wait = score, ep, 0
                torch.save({"model_state": model.state_dict(), "stats": {k: v.tolist() for k, v in stats.items()},
                            "feature_names": FEATURE_NAMES, "model_args": {"seq_len": int(Xtr[0].shape[-1]), **M["model"]},
                            "method": args.method, "calibration_free": True, "demographic_units": units, "epoch": ep}, out / "best.pt")
            else:
                wait += 1
            if not np.isfinite(pv).all():                                         # never continue on non-finite outputs
                log.info(f"epoch {ep:3d} | train {np.mean(tl):.5f} | NON-FINITE validation outputs - diagnosing and stopping")
                diagnose_nonfinite(model, Xva, dev, log)
                sys.exit(2)
            log.info(f"epoch {ep:3d} | train {np.mean(tl):.5f} | val MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f} | "
                     f"{time.time() - t0:.0f}s{'  *best*' if star else ''}")
            if wait >= M["patience"]:
                log.info(f"early stop after {ep} epochs (patience {M['patience']})")
                break
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model_state"])

    if args.method == "ragcbpnet":
        from src.published.ragcbpnet import RAGCBPNet
        model = RAGCBPNet(n_feats=len(FEATURE_NAMES), **M["model"]).to(dev)
        y_mean, y_std = Xtr[2].numpy().mean(0), Xtr[2].numpy().std(0)            # training labels only (G8)
        model.y_mean.copy_(torch.tensor(y_mean)); model.y_std.copy_(torch.tensor(y_std))
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"RAGCBPNet: {n_params / 1e6:.2f} M parameters; label scaling mean {y_mean.round(1)}, sd {y_std.round(1)}")
        opt = torch.optim.Adam(model.parameters(), lr=M["lr"], weight_decay=M["weight_decay"])
        dl = DataLoader(TensorDataset(*Xtr), batch_size=M["batch_size"], shuffle=True, drop_last=True)
        ym, ys = model.y_mean, model.y_std
        max_epochs = 2 if args.smoke else M["max_epochs"]
        best, best_ep, wait, hist, t0 = np.inf, -1, 0, [], time.time()
        for ep in range(1, max_epochs + 1):
            model.train()
            tl = []
            for x, f, y in dl:
                opt.zero_grad(set_to_none=True)
                y = y.to(dev)
                loss = torch.nn.functional.l1_loss(model.forward_std(x.to(dev), f.to(dev)), (y - ym) / ys)   # MAE loss
                loss.backward()
                opt.step()
                tl.append(loss.item())
            pv = predict_any(model, Xva)
            if not np.isfinite(pv).all():
                log.info(f"epoch {ep:3d} | NON-FINITE validation outputs - stopping")
                sys.exit(2)
            mae = np.abs(pv - Xva[2].numpy()).mean(axis=0)
            score = float(mae.mean())
            hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_sbp_mae": mae[0], "val_dbp_mae": mae[1], "val_score": score})
            star = score < best
            if star:
                best, best_ep, wait = score, ep, 0
                torch.save({"model_state": model.state_dict(), "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
                            "feature_names": FEATURE_NAMES, "model_args": {"n_feats": len(FEATURE_NAMES), **M["model"]},
                            "method": args.method, "k_train": M["k_train"], "demographic_units": units, "epoch": ep}, out / "best.pt")
            else:
                wait += 1
            log.info(f"epoch {ep:3d} | train {np.mean(tl):.4f} | val MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f} | "
                     f"{time.time() - t0:.0f}s{'  *best*' if star else ''}")
            if wait >= M["patience"]:
                log.info(f"early stop after {ep} epochs (patience {M['patience']})")
                break
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model_state"])

    if args.method == "hgctnet":
        from src.published.hgctnet import HGCTNet
        model = HGCTNet(in_channels=2, seq_len=Xtr[0].shape[-1], n_feats=len(FEATURE_NAMES), n_demo=4, **M["model"]).to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"HGCTNet: {n_params / 1e6:.2f} M parameters, {model.key_size} time steps after the CNN")
        opt = torch.optim.Adam(model.parameters(), lr=M["lr"], betas=M["betas"], weight_decay=M["weight_decay"])
        loss_fn = torch.nn.HuberLoss(delta=M["huber_delta"])
        dl = DataLoader(TensorDataset(*Xtr), batch_size=M["batch_size"], shuffle=True, drop_last=True)

        @torch.no_grad()
        def predict(X):
            model.eval()
            outp = [model(x.to(dev), f.to(dev)).float().cpu().numpy()
                    for x, f in zip(torch.split(X[0], 512), torch.split(X[1], 512))]
            return np.concatenate(outp)

        max_epochs = 2 if args.smoke else M["max_epochs"]
        best, best_ep, wait, hist, t0 = np.inf, -1, 0, [], time.time()
        for ep in range(1, max_epochs + 1):
            model.train()
            tl = []
            for x, f, y in dl:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x.to(dev), f.to(dev)), y.to(dev))
                loss.backward()
                opt.step()
                tl.append(loss.item())
            pv = predict(Xva)
            mae = np.abs(pv - Xva[2].numpy()).mean(axis=0)
            score = float(mae.mean())
            hist.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_sbp_mae": mae[0], "val_dbp_mae": mae[1], "val_score": score})
            star = score < best
            if star:
                best, best_ep, wait = score, ep, 0
                torch.save({"model_state": model.state_dict(), "stats": {k: v.tolist() for k, v in stats.items()},
                            "feature_names": FEATURE_NAMES, "model_args": {"in_channels": 2, "seq_len": int(Xtr[0].shape[-1]),
                            "n_feats": len(FEATURE_NAMES), "n_demo": 4, **M["model"]}, "method": args.method,
                            "k_train": M["k_train"], "demographic_units": units, "epoch": ep}, out / "best.pt")
            else:
                wait += 1
            log.info(f"epoch {ep:3d} | train {np.mean(tl):.3f} | val MAE SBP {mae[0]:.2f} DBP {mae[1]:.2f} | "
                     f"alpha {model.balance_alpha.detach().cpu().numpy().round(3)} | {time.time() - t0:.0f}s{'  *best*' if star else ''}")
            if wait >= M["patience"]:
                log.info(f"early stop after {ep} epochs (patience {M['patience']})")
                break
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        model.load_state_dict(torch.load(out / "best.pt", map_location="cpu")["model_state"])

    # test (touched once): our model's test recordings, baseline from the first k calibration readings
    te = parts["test"]
    keep = ["sample_id", "pid", "protocol", "phase", "condition", "measurement", "is_exercise", "date_time", "age_group",
            "gender", "bmi_group"] + T
    results = {}
    for k in (1, 3):
        base_te = baselines(te.reset_index(drop=True), cal["test"], k)
        Xte = tensors(te, tg["test"], base_te)
        bp = predict_any(model, Xte)
        pred = te.iloc[tg["test"]][[c for c in keep if c in te]].reset_index(drop=True)
        cal_mean = np.stack([base_te[p] for p in pred["pid"]])            # [SBP, DBP], method-independent
        for j, t in enumerate(T):
            pred[f"{t}_pred"] = bp[:, j]
            pred[f"{t}_cal_mean"] = cal_mean[:, j]
        pred.to_csv(out / ("predictions_test.csv" if k == 1 else f"predictions_test_k{k}.csv"), index=False)
        res = evaluate_predictions(pred, tr, min_rec=5)
        results[f"test_k{k}"] = res
        m = res["overall"]
        log.info(f"TEST k={k}: SBP MAE {m['sbp']['MAE']:.2f} | DBP MAE {m['dbp']['MAE']:.2f} ({te.pid.nunique()} subjects, {len(pred):,} recordings)")
    prov = provenance(cfg, CODE_DIR, out)
    prov["features_sha256"] = hashlib.sha256(feat_file.read_bytes()).hexdigest()
    json.dump(prov, open(out / "provenance.json", "w"), indent=2, default=str)
    json.dump(cfg, open(out / "config.json", "w"), indent=2, default=str)
    info = {"method": args.method, "fold": args.fold, "n_params": int(n_params), "best_epoch": best_ep, "best_val_score": best,
            "n_train_subjects": int(tr.pid.nunique()), "n_train_recordings": int(len(tr)),
            "calibration_free": bool(M.get("calibration_free", False))}
    json.dump({"info": info, **results}, open(out / "metrics.json", "w"), indent=2, default=float)
    log.info(f"done -> {out}")


if __name__ == "__main__":
    main()
