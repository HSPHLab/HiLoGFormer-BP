#!/usr/bin/env python
"""
Measurement performance evaluation, P5 - robustness to noise (IEEE Std 1708-2025 §6.2; docs/PERFORMANCE_EVALUATION_PLAN.md,
section 8).

  python scripts/perf_P5_noise.py --stage infer --gpu 2      # GPU, ~10-20 min: all methods x 26 noise conditions
  python scripts/perf_P5_noise.py --stage report             # CPU: observed SNR of real recordings, tables, figures

Readings: a fixed random sample of 3,000 primary-set test readings (auscultatory; static, BP-change and after-period tests;
seed 2026), identical for every method and condition. Each reading is estimated by the fold model that did not see its subject.
Noise is written into the TEST reading's samples of a private copy-on-write map of signals.npy (the file is never modified;
calibration recordings stay clean), so every method reads exactly the same noisy samples; feature-based methods re-extract
their 31 features (src/features/pwa_features.py) from the noisy window.
Noise (per channel, scaled to the channel's power in the window): white Gaussian; baseline wander (5 sinusoids, 0.05-0.5 Hz,
random phases); motion artefact (band-limited 0.5-5 Hz noise in 1-3 bursts covering 20-50 % of the window, Tukey edges).
Applied to both channels; motion also to ECG only and PPG only. SNR: clean, 30, 20, 10, 5, 0 dB.
Verification (clean condition): re-extracted features vs data_processed/features_pwa/aurora_v1.csv, and every method's
predictions vs its frozen predictions; a method that does not reproduce is excluded and reported.
Methods: HiCAF-BP-Cal v1, HGCTNet, BpBLS, RAGCBPNet (calibrated; k = 3), rU-Net, UTransBPNet (calibration-free); calibration
value only (noise-independent reference line).
Outputs (paper/performance_evaluation/): tables/STD_Table_P7_noise_robustness.md (+ STD_P5_results.json),
figures/STD_Fig_P3_noise_robustness_SBP / _DBP (.pdf + png/). Intermediate: runs/performance/noise_preds.csv.gz.
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
SEC = CODE_DIR / "paper/performance_evaluation"
INTER = CODE_DIR / "runs/performance"
T = ["sbp", "dbp"]
SEED, N_READ, SEG = 2026, 3000, 1250
SNRS = [30, 20, 10, 5, 0]
KINDS = [("white", "both"), ("wander", "both"), ("motion", "both"), ("motion", "ecg"), ("motion", "ppg")]
KIND_LAB = {("white", "both"): "White noise", ("wander", "both"): "Baseline wander", ("motion", "both"): "Motion, ECG + PPG",
            ("motion", "ecg"): "Motion, ECG only", ("motion", "ppg"): "Motion, PPG only"}
CONDS = [("clean", "both", np.inf)] + [(k, c, s) for k, c in KINDS for s in SNRS]
OURS_RUN = "runs/hicaf_bp_cal_notime"
PUBLISHED = {"hgctnet": 3, "bpbls": 3, "ragcbpnet": 3, "runet": 1, "utransbpnet": 1}   # k of the frozen prediction file
FEAT_CSV = "data_processed/features_pwa/aurora_v1.csv"


def cond_name(c):
    return "clean" if c[0] == "clean" else f"{c[0]}_{c[1]}_{c[2]}dB"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------------------- noise
def unit_noise(kind, n, fs, rng):
    """Zero-mean, unit-variance noise of one channel."""
    if kind == "white":
        z = rng.standard_normal(n)
    elif kind == "wander":
        t = np.arange(n) / fs
        z = sum(np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 2 * np.pi)) for _ in range(5))
    elif kind == "motion":
        from scipy.signal import butter, sosfiltfilt
        from scipy.signal.windows import tukey
        sos = butter(4, [0.5, 5.0], btype="band", fs=fs, output="sos")
        bp = sosfiltfilt(sos, rng.standard_normal(n + 2 * int(fs)))[int(fs):int(fs) + n]
        mask = np.zeros(n)
        n_b = int(rng.integers(1, 4))
        total = rng.uniform(0.2, 0.5) * n
        for L in np.maximum((rng.dirichlet(np.ones(n_b)) * total).astype(int), int(0.5 * fs)):
            a = int(rng.integers(0, max(n - L, 1)))
            mask[a:a + L] = np.maximum(mask[a:a + L], tukey(len(mask[a:a + L]), 0.3))
        z = bp * mask
    else:
        raise ValueError(kind)
    z = z - z.mean()
    sd = z.std()
    return z / sd if sd > 0 else z


class NoisySignals:
    """Copy-on-write map of signals.npy; writes the noise of one condition into the selected readings, restores after."""

    def __init__(self, path, regions, ch_ecg, ch_ppg, fs):
        self.arr = np.load(path, mmap_mode="c")
        self.regions = regions                                   # sample_id -> (start, stop)
        self.ch = {"ecg": ch_ecg, "ppg": ch_ppg}
        self.fs = fs
        self.orig = {sid: np.array(self.arr[a:b], copy=True) for sid, (a, b) in regions.items()}

    def apply(self, ci, cond):
        kind, chans, snr = cond
        self.restore()
        if kind == "clean":
            return
        for sid, (a, b) in self.regions.items():
            rng = np.random.default_rng([SEED, ci, int(sid)])
            for name in (["ecg", "ppg"] if chans == "both" else [chans]):
                j = self.ch[name]
                x = self.orig[sid][:, j].astype(np.float64)
                p_s = x.var()
                self.arr[a:b, j] = (x + unit_noise(kind, b - a, self.fs, rng) * np.sqrt(p_s / 10 ** (snr / 10))).astype(self.arr.dtype)

    def restore(self):
        for sid, (a, b) in self.regions.items():
            self.arr[a:b] = self.orig[sid]


def features_of(signals, rows, ch_ecg, ch_ppg, fs, pwa, norm="window"):
    """31 features from the evaluation window. rows: (window start, recording offset, recording length).
    norm: 'window' = z-score over the 10 s window; 'raw' = no scaling; 'recording' = z-score over the whole recording."""
    out = np.full((len(rows), 31), np.nan)
    for i, (a, off, ln) in enumerate(rows):
        w = np.asarray(signals[a:a + SEG][:, [ch_ecg, ch_ppg]], np.float64)
        if norm == "window":
            w = (w - w.mean(0)) / (w.std(0) + 1e-6)
        elif norm == "recording":
            r = np.asarray(signals[off:off + ln][:, [ch_ecg, ch_ppg]], np.float64)
            w = (w - r.mean(0)) / (r.std(0) + 1e-6)
        with np.errstate(all="ignore"):
            out[i] = pwa.extract_features(w[:, 0], w[:, 1], fs)[0]
    return out


def match_share(F, ref):
    both = np.isfinite(F) & np.isfinite(ref)
    return float(np.isclose(F[both], ref[both], rtol=1e-4, atol=1e-6).mean()) if both.any() else 0.0


# ------------------------------------------------------------------------------------------------------------- GPU stage
def infer(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.data.dataset import BPCropDataset
    from src.data.pairs import PairDataset, calibration_split, eval_pairs, row_times
    from src.data.splits import assign_roles, load_metadata
    from src.features import pwa_features as pwa
    from src.models import build_model
    from src.published.bpbls import BpBLS
    from src.published.hgctnet import HGCTNet
    from src.published.ragcbpnet import RAGCBPNet
    from src.published.runet import rUNetBP
    from src.published.utransbpnet import UTransBPNet

    INTER.mkdir(parents=True, exist_ok=True)
    fp_out = INTER / "noise_preds.csv.gz"
    if fp_out.exists() and not args.force:
        sys.exit(f"{fp_out} exists - use --force to rebuild")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tp = load_module("train_published", "scripts/train_published.py")
    p1 = load_module("perf_p1", "scripts/perf_P1_ieee1708.py")
    tg, frozen, _, _, _ = p1.load()
    prim = tg[tg.test.isin(["static", "change", "after"])]
    sample = prim.sample(n=min(N_READ, len(prim)), random_state=SEED).sort_values("sample_id")
    sids = set(sample.sample_id)
    print(f"{len(sample):,} readings, {sample.pid.nunique()} subjects; tests {sample.test.value_counts().to_dict()}", flush=True)

    cfg_v1 = load_config("experiments/hicaf_bp_cal_t_scratch.yaml")
    meta, fs, channels = load_metadata(cfg_v1)
    ch_ecg, ch_ppg = channels.index("ecg"), channels.index("ppg")
    ck0 = torch.load(CODE_DIR / OURS_RUN / "fold_0" / "best.pt", map_location="cpu")
    crop = int(round(ck0["config"]["data"]["crop_s"] * fs))
    L = max(crop, SEG)
    m = meta.set_index("sample_id").loc[sorted(sids)]
    assert ((m.crop_start_eval + L) <= m.length).all(), "noise region exceeds a recording"
    regions = {int(s): (int(o + c), int(o + c + L)) for s, o, c in zip(m.index, m.offset, m.crop_start_eval)}
    sig = NoisySignals(Path(cfg_v1["paths"]["data_dir"]) / "signals.npy", regions, ch_ecg, ch_ppg, fs)
    print(f"fs {fs} Hz; model crop {crop} samples; noise region {L} samples per reading", flush=True)

    # ---------------- fold-wise preparation (done once; only the signals change between conditions)
    feats_csv = pd.read_csv(CODE_DIR / FEAT_CSV)
    prep = {}
    for fold in range(5):
        fs_ids = set(sample.loc[sample.fold == fold, "sample_id"])
        if not fs_ids:
            continue
        P = {}
        # ours
        ck = torch.load(CODE_DIR / OURS_RUN / f"fold_{fold}" / "best.pt", map_location="cpu")
        cfg, st = ck["config"], ck["stats"]
        df = assign_roles(meta, cfg, fold)
        d = df[df.role == "test"].reset_index(drop=True)
        cal, tgi = calibration_split(d, cfg["calibration"]["phase"], cfg["calibration"]["conditions"], int(cfg["calibration"]["k_max"]))
        pairs = eval_pairs(d, cal, tgi, 3)
        keep = np.isin(d.sample_id.to_numpy()[pairs[:, 0]], list(fs_ids))
        model = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=2, n_aux=0).to(dev)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        P["ours"] = dict(model=model, d=d, pairs=pairs[keep], cfg=cfg, y_mean=np.asarray(st["y_mean"], np.float32),
                         y_std=np.asarray(st["y_std"], np.float32), times=row_times(d) if cfg["calibration"].get("time_context") else None,
                         ch_idx=[channels.index(c) for c in cfg["data"]["inputs"]])
        # published
        dfp = assign_roles(meta, cfg_v1, fold).merge(feats_csv[["sample_id", *pwa.FEATURE_NAMES]], on="sample_id", how="left",
                                                     validate="one_to_one")
        te = dfp[dfp.role == "test"].sort_values("sample_id").copy()
        cal_te, tg_te = calibration_split(te.reset_index(drop=True), cfg_v1["calibration"]["phase"], cfg_v1["calibration"]["conditions"], 3)
        part0 = te.iloc[tg_te]
        part0 = part0[part0.sample_id.isin(fs_ids)]
        for meth, k in PUBLISHED.items():
            fpk = CODE_DIR / f"runs/pub_{meth}/fold_{fold}/best.pt"
            if not fpk.exists():
                continue
            ckp = torch.load(fpk, map_location="cpu")
            demo, _ = {"runet": tp.demographics_runet, "ragcbpnet": tp.demographics_ragcbpnet}.get(meth, tp.demographics)(dfp)
            builder = {"hgctnet": HGCTNet, "bpbls": BpBLS, "runet": rUNetBP, "utransbpnet": UTransBPNet, "ragcbpnet": RAGCBPNet}[meth]
            mdl = builder(**ckp["model_args"]).to(dev)
            mdl.load_state_dict(ckp["model_state"])
            mdl.eval()
            P[meth] = dict(model=mdl, part=part0, base=tp.baselines(te.reset_index(drop=True), cal_te, k), demo=demo,
                           stats={kk: np.asarray(v, float) for kk, v in ckp["stats"].items()}, fnames=ckp["feature_names"],
                           side={"hgctnet": tp.side_inputs, "bpbls": tp.side_inputs_bpbls, "runet": tp.side_inputs_runet,
                                 "utransbpnet": tp.side_inputs_none, "ragcbpnet": tp.side_inputs_ragcbpnet}[meth])
        prep[fold] = P
    print(f"prepared in {(time.time() - t0) / 60:.1f} min", flush=True)

    @torch.no_grad()
    def run_ours(P):
        cfg = P["cfg"]
        base = BPCropDataset(P["d"], sig.arr, P["ch_idx"], crop, P["y_mean"], P["y_std"], list(cfg["data"]["targets"]), [], None, None,
                             train=False, augment_cfg=cfg["data"].get("augment"))
        dl = DataLoader(PairDataset(base, pairs=P["pairs"], times=P["times"]), batch_size=256, shuffle=False, num_workers=0)
        model, use_amp = P["model"], bool(cfg["train"].get("amp_bf16", True)) and dev.type == "cuda"
        out = []
        for b in dl:
            kw = {"ctx": b["ctx"].to(dev)} if "ctx" in b else {}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                o = model(b["x"].to(dev), x_cal=b["x_cal"].to(dev), bp_cal=b["bp_cal"].to(dev), **kw)["bp"]
            out.append(o.float().cpu().numpy() * P["y_std"] + P["y_mean"])
        pr = pd.DataFrame(np.concatenate(out), columns=["sbp_pred", "dbp_pred"])
        pr["sample_id"] = P["d"].sample_id.to_numpy()[P["pairs"][:, 0]]
        return pr.groupby("sample_id")[["sbp_pred", "dbp_pred"]].mean()

    @torch.no_grad()
    def run_pub(P, feats):
        p = P["part"].copy()
        if P["fnames"]:
            p[list(P["fnames"])] = feats.loc[p.sample_id, list(P["fnames"])].to_numpy()
        x = torch.from_numpy(tp.windows(p, sig.arr, [ch_ecg, ch_ppg]))
        f = torch.from_numpy(P["side"](p, P["base"], P["demo"], P["fnames"], P["stats"]))
        o = np.concatenate([P["model"](xb.to(dev), fb.to(dev)).float().cpu().numpy()
                            for xb, fb in zip(torch.split(x, 256), torch.split(f, 256))])
        return pd.DataFrame(o, columns=["sbp_pred", "dbp_pred"], index=pd.Index(p.sample_id.to_numpy(), name="sample_id"))

    rows_all, verify = [], []
    rows_ix = [(regions[int(s)][0], int(m.at[s, "offset"]), int(m.at[s, "length"])) for s in sorted(sids)]
    # which signal scaling reproduces the stored feature table? (tested on clean signals, first 300 readings)
    ref300 = feats_csv.set_index("sample_id").loc[sorted(sids)[:300], pwa.FEATURE_NAMES].to_numpy(float)
    shares = {nm: match_share(features_of(sig.arr, rows_ix[:300], ch_ecg, ch_ppg, fs, pwa, nm), ref300) for nm in ["window", "raw", "recording"]}
    feat_norm = max(shares, key=shares.get)
    print(f"feature scaling check (share of stored feature values reproduced): {shares} -> using '{feat_norm}'", flush=True)
    if shares[feat_norm] < 0.99:
        sys.exit("no signal scaling reproduces the stored features (>= 99 %) - send scripts that built features_pwa/aurora_v1.csv")
    for ci, cond in enumerate(CONDS):
        sig.apply(ci, cond)
        F = pd.DataFrame(features_of(sig.arr, rows_ix, ch_ecg, ch_ppg, fs, pwa, feat_norm), columns=pwa.FEATURE_NAMES,
                         index=pd.Index(sorted(sids), name="sample_id"))
        if cond[0] == "clean":                                      # verification of the feature extraction
            ref = feats_csv.set_index("sample_id").loc[F.index, pwa.FEATURE_NAMES]
            both = F.notna() & ref.notna()
            dif = (F - ref).abs()[both]
            verify.append({"item": "features (re-extracted vs stored)", "n": int(both.values.sum()),
                           "nan_agree": float((F.isna() == ref.isna()).values.mean()),
                           "max_abs_diff": float(np.nanmax(dif.values)), "share_exact_1e-6": float((dif <= 1e-6).values.sum() / both.values.sum())})
        res = {}
        for fold, P in prep.items():
            res.setdefault("ours", []).append(run_ours(P["ours"]))
            for meth in PUBLISHED:
                if meth in P:
                    res.setdefault(meth, []).append(run_pub(P[meth], F))
        for meth, parts in res.items():
            pr = pd.concat(parts)
            pr["method"], pr["condition"] = meth, cond_name(cond)
            rows_all.append(pr.reset_index())
            if cond[0] == "clean":
                fz = frozen[meth].loc[pr.index]
                dif = np.abs(pr[["sbp_pred", "dbp_pred"]].to_numpy() - fz[["sbp_pred", "dbp_pred"]].to_numpy())
                verify.append({"item": f"predictions {meth} (clean vs frozen)", "n": int(len(pr)), "nan_agree": float("nan"),
                               "max_abs_diff": float(dif.max()), "share_exact_1e-6": float((dif <= 0.05).mean())})
        print(f"[{ci + 1}/{len(CONDS)}] {cond_name(cond)} | {(time.time() - t0) / 60:.1f} min", flush=True)
    sig.restore()
    allp = pd.concat(rows_all, ignore_index=True)
    allp.to_csv(fp_out, index=False, compression="gzip")
    v = pd.DataFrame(verify)
    txt = ["# P5 noise robustness - verification (clean condition)", "",
           "Features: re-extracted from the clean evaluation window vs the stored feature table. Predictions: clean-condition "
           "predictions vs each method's frozen predictions; last column = share within 0.05 mmHg for predictions.", "",
           v.round(4).to_string(index=False), "", f"runtime {(time.time() - t0) / 60:.1f} min"]
    (INTER / "VERIFY_noise.md").write_text("\n".join(txt))
    json.dump({"sample_ids": sorted(int(s) for s in sids), "conditions": [cond_name(c) for c in CONDS], "seed": SEED,
               "crop": crop, "region": L, "fs": fs, "feature_scaling": feat_norm, "feature_scaling_check": shares}, open(INTER / "noise_design.json", "w"), indent=1)
    print("\n".join(txt))


# ------------------------------------------------------------------------------------------------------------- CPU stage
def template_snr(x, idx, pre, post, n_res=None):
    """Beat-template SNR proxy (dB): beats aligned at idx (or resampled cycles), template = median beat."""
    beats = []
    for i, (a, b) in enumerate(idx):
        seg = x[a:b]
        if n_res:
            seg = np.interp(np.linspace(0, len(seg) - 1, n_res), np.arange(len(seg)), seg)
        beats.append(seg - seg.mean())
    if len(beats) < 3:
        return np.nan
    B = np.array(beats)
    tpl = np.median(B, axis=0)
    if not n_res:                                                     # ECG: refine alignment by +/- 2 samples (sampling jitter)
        k = 2
        x_al = []
        for (a, b), _ in zip(idx, beats):
            best = max(range(-k, k + 1), key=lambda s: np.dot(x[max(a + s, 0):b + s][:len(tpl)] - x[max(a + s, 0):b + s][:len(tpl)].mean(), tpl)
                       if a + s >= 0 and b + s <= len(x) else -np.inf)
            seg = x[a + best:b + best]
            x_al.append(seg - seg.mean())
        B = np.array(x_al)
        tpl = np.median(B, axis=0)
    res = B - tpl
    return float(10 * np.log10(tpl.var() / max(res.var(), 1e-12)))


def observed_snr(tg, args):
    """SNR proxy of real recordings: in-lab auscultatory (primary set) and ambulatory (home) windows, clean signals."""
    from src.config import load_config
    from src.data.splits import load_metadata
    from src.features import pwa_features as pwa
    cfg = load_config("experiments/hicaf_bp_cal_t_scratch.yaml")
    meta, fs, channels = load_metadata(cfg)
    ce, cp = channels.index("ecg"), channels.index("ppg")
    sigs = np.load(Path(cfg["paths"]["data_dir"]) / "signals.npy", mmap_mode="r")
    m = meta.set_index("sample_id")
    sets = {"In-lab (auscultatory test readings)": tg[tg.test.isin(["static", "change", "after"])].sample_id,
            "Home (ambulatory test readings)": tg[(tg.protocol == "oscillometric") & (tg.phase == "ambulatory")].sample_id}
    out = {}
    for name, ids in sets.items():
        ids = pd.Series(ids).sample(n=min(1500, len(ids)), random_state=SEED)
        e_snr, p_snr = [], []
        for sid in ids:
            a = int(m.at[sid, "offset"] + m.at[sid, "crop_start_eval"])
            w = np.asarray(sigs[a:a + SEG][:, [ce, cp]], np.float64)
            w = (w - w.mean(0)) / (w.std(0) + 1e-6)
            cycles, rpk, _, _ = pwa.detect_points(w[:, 0], w[:, 1], fs)
            pre, post = int(0.2 * fs), int(0.4 * fs)
            r_ix = [(r - pre, r + post) for r in rpk if r - pre >= 0 and r + post <= SEG]
            e_snr.append(template_snr(w[:, 0], r_ix, pre, post))
            p_snr.append(template_snr(w[:, 1], [(c["s"], c["v"] + 1) for c in cycles], 0, 0, n_res=64))
        q = lambda v: {k: float(np.nanpercentile(v, p)) for k, p in [("p5", 5), ("p25", 25), ("median", 50)]} | {"valid": float(np.mean(np.isfinite(v)))}
        out[name] = {"ECG": q(np.array(e_snr)), "PPG": q(np.array(p_snr)), "n": int(len(ids))}
    return out


def report(args):
    from src.viz.style import METHOD, TARGET, figure, panel_labels, save, setup
    p1 = load_module("perf_p1", "scripts/perf_P1_ieee1708.py")
    tg, frozen, _, _, _ = p1.load()
    pr = pd.read_csv(INTER / "noise_preds.csv.gz")
    design = json.load(open(INTER / "noise_design.json"))
    ref = tg.set_index("sample_id").loc[design["sample_ids"]]
    cal_only = pd.DataFrame({"sample_id": ref.index, "sbp_pred": ref.sbp_cal.to_numpy(), "dbp_pred": ref.dbp_cal.to_numpy()})
    pr = pd.concat([pr] + [cal_only.assign(method="cal_only", condition=c) for c in design["conditions"]], ignore_index=True)
    pr = pr.join(ref[["pid", "sbp", "dbp"]], on="sample_id")
    excluded = []
    vt = (INTER / "VERIFY_noise.md").read_text() if (INTER / "VERIFY_noise.md").exists() else ""
    res = {}
    for (meth, cond), g in pr.groupby(["method", "condition"]):
        r = {}
        for t in T:
            e = g[f"{t}_pred"] - g[t]
            r[t] = {"MAD": float(e.abs().mean()), "MD": float(e.mean()), "n": int(e.notna().sum())}
        r["grade"] = p1.worst([p1.grade(r[t]["MAD"], r[t]["MD"]) for t in T])
        res[(meth, cond)] = r
    meths = [m for m in ["ours", "hgctnet", "bpbls", "ragcbpnet", "cal_only", "runet", "utransbpnet"] if (m, "clean") in res]
    snr_obs = observed_snr(tg, args)
    lab = lambda c: "∞" if c == "clean" else c.split("_")[-1].replace("dB", "")

    L = ["# STD_Table_P7 — Robustness to noise (IEEE Std 1708-2025 §6.2)", "",
         f"{len(design['sample_ids']):,} primary-set test readings (seed {SEED}), identical for every method and condition; noise added "
         "to the test reading only (calibration recordings clean); feature-based methods re-extract their features from the noisy "
         "window. Cells: MAD SBP / DBP (mmHg). Grade per Table 6 on these readings (worse of SBP and DBP).", ""]
    for kind, chans in KINDS:
        conds = ["clean"] + [f"{kind}_{chans}_{s}dB" for s in SNRS]
        L += [f"## {KIND_LAB[(kind, chans)]}", "", "| Method | " + " | ".join(f"SNR {lab(c)} dB" for c in conds)
              + " | Lowest SNR with grade ≤ C | First SNR with MAD + 1 mmHg over clean |", "|---|" + "---|" * (len(conds) + 2)]
        for mth in meths:
            cells, ok = [], "none"
            for c in conds:
                r = res.get((mth, c))
                cells.append("–" if r is None else f"{r['sbp']['MAD']:.2f} / {r['dbp']['MAD']:.2f} ({r['grade']})")
            for c in conds[1:]:                                       # descending SNR; stop at the first failure
                r = res.get((mth, c))
                if r is None or r["grade"] == "D":
                    break
                ok = f"{lab(c)} dB"
            if res.get((mth, "clean"), {}).get("grade") == "D":
                ok = "fails when clean"
            deg = "none down to 0 dB"
            if mth != "cal_only" and (mth, "clean") in res:
                for c in conds[1:]:
                    r = res.get((mth, c))
                    if r is not None and max(r[t]["MAD"] - res[(mth, "clean")][t]["MAD"] for t in T) >= 1.0:
                        deg = f"{lab(c)} dB"
                        break
            else:
                deg = "–"
            L.append(f"| {METHOD[mth]['label']} | " + " | ".join(cells) + f" | {ok} | {deg} |")
        L.append("")
    L += ["## SNR of the real recordings (beat-template proxy)", "",
          "SNR = 10·log10(var(median beat) / var(beat − median beat)) per 10 s window (ECG: beats aligned at the R peak, −0.2 to "
          "+0.4 s; PPG: foot-to-foot cycles resampled to 64 points). Beat-to-beat physiological variation counts as noise, so the "
          "proxy is conservative.", "",
          "| Recordings | Channel | Windows (share valid) | 5th percentile (worst case) | 25th percentile | Median |", "|---|---|---|---|---|---|"]
    for name, o in snr_obs.items():
        for chn in ["ECG", "PPG"]:
            q = o[chn]
            L.append(f"| {name} | {chn} | {o['n']:,} ({q['valid']:.0%}) | {q['p5']:.1f} dB | {q['p25']:.1f} dB | {q['median']:.1f} dB |")
    if vt:
        L += ["", "## Verification (clean condition)", "", "```", *vt.split("\n")[4:-2], "```"]
    (SEC / "tables").mkdir(parents=True, exist_ok=True)
    (SEC / "tables/STD_Table_P7_noise_robustness.md").write_text("\n".join(L))
    json.dump({"results": {f"{m}|{c}": v for (m, c), v in res.items()}, "observed_snr": snr_obs}, open(SEC / "tables/STD_P5_results.json", "w"),
              indent=1, default=float)
    print("\n".join(L))

    # ---------------- STD_Fig_P3: MAD vs SNR per noise type, calibrated methods, grade bands
    setup()
    GRADE_FILL = p1.GRADE_FILL
    xs = [35] + SNRS                                                  # 35 = clean (labelled ∞)
    for t in T:
        fig, axes = figure("single", height=5.0, nrows=3, ncols=2, sharey=True)
        axes = axes.ravel()
        cal_m = [m for m in ["ours", "hgctnet", "bpbls", "ragcbpnet", "cal_only"] if m in meths]
        top = {"sbp": 20.0, "dbp": 15.0}[t]                          # fixed cap; off-scale values in STD_Table_P7
        for ax, (kind, chans) in zip(axes, KINDS):
            for lo, hi, g in [(0, 5, "A"), (5, 6, "B"), (6, 7, "C"), (7, max(top, 8), "D")]:
                ax.axhspan(lo, hi, color=GRADE_FILL[g], zorder=0, lw=0)
            ref_q = snr_obs["Home (ambulatory test readings)"]["ECG" if chans == "ecg" else "PPG"]
            for v_, ls in [(ref_q["median"], ":"), (ref_q["p5"], "--")]:
                if np.isfinite(v_):
                    ax.axvline(min(max(v_, -1), 33), color="#606060", lw=0.6, ls=ls)
            for m_ in cal_m:
                ys = [res[(m_, "clean" if s == 35 else f"{kind}_{chans}_{s}dB")][t]["MAD"] for s in xs]
                st = METHOD[m_]
                ax.plot(xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"] or None, ms=3, lw=1.0, label=st["label"])
                for x_, y_ in zip(xs, ys):                                # off-scale values: marker on the upper edge
                    if y_ > top:
                        ax.plot(x_ + {"hgctnet": 0.9, "ragcbpnet": -0.9}.get(m_, 0.0), top, marker=st["marker"] or "o", color=st["color"], ms=3, clip_on=False, zorder=4)
            ax.set(title=KIND_LAB[(kind, chans)], xticks=xs, xlim=(37, -2), ylim=(0, top))
            ax.set_xticklabels(["∞", "30", "20", "10", "5", "0"])
            ax.title.set_fontsize(7)
            ax.grid(axis="x", visible=False)
        for ax in axes[3:5]:
            ax.set_xlabel("SNR of the test recording (dB)")
        for ax in (axes[0], axes[2], axes[4]):
            ax.set_ylabel(f"{TARGET[t]['label']} MAD (mmHg)")
        axes[5].axis("off")
        h, l_ = axes[0].get_legend_handles_labels()
        axes[5].legend(h, l_, loc="center", fontsize=6.5, frameon=False,
                       title=f"dotted / dashed: median / 5th pct.\nof the SNR at home (PPG; ECG in d)\nmarkers on the upper edge:\nMAD > {top:.0f} mmHg", title_fontsize=6)
        panel_labels(axes[:5], x=-0.26)
        save(fig, f"STD_Fig_P3_noise_robustness_{t.upper()}", source=__file__, outdir=SEC / "figures", png_subdir="png")

    rd = SEC / "README.md"
    if rd.exists():
        s = rd.read_text()
        for name in ["STD_Fig_P3_noise_robustness", "STD_Table_P7_noise_robustness"]:
            s = "\n".join(l.replace("| pending |", f"| done {time.strftime('%Y-%m-%d')} |") if l.startswith(f"| {name} ") else l
                          for l in s.split("\n"))
        rd.write_text(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["infer", "report", "all"], default="all")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("infer", "all"):
        infer(args)
    if args.stage in ("report", "all"):
        report(args)


if __name__ == "__main__":
    main()