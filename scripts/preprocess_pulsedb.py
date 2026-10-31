#!/usr/bin/env python
"""
PulseDB v2.0 -> our data format, for ZERO-SHOT external testing (nothing here is used for training).

  python scripts/preprocess_pulsedb.py --workers 32

Selection (from eda/segment_index.parquet, no signal reading needed):
  1. one recording per subject: the case_id with the most segments
  2. physiological labels: SBP 70-220, DBP 30-130, SBP - DBP >= 10 mmHg
  3. ECG quality: |R-peaks - PPG peaks| <= 2, ECG HR 40-150 bpm, RR irregularity (rr_cv) <= 0.2
  4. subjects with >= 4 remaining segments (1-3 calibration + targets)
  5. cap per subject (default 200): the first k_cal segments in time are always kept (calibration),
     the rest are spread evenly over the remaining recording
Signals (identical treatment to the Aurora model input):
  ECG_Record_F / PPG_Record_F (filtered on the whole recording by PulseDB, 125 Hz, 10 s = 1250 samples)
  -> our Aurora filters (ECG Butterworth 0.5-40 Hz order 4; PPG Chebyshev II 0.5-10 Hz order 4, 20 dB),
     zero-phase with odd-extension padding of the full segment length (no edge transients)
  -> ECG polarity: flipped if skewness < 0 (R-peaks upward, as in the Aurora input)
  -> PPG: NOT flipped (same orientation as Aurora: derivative skew +1.1 vs +1.0, rising 30-34% of the time)
  -> z-score per segment and channel
Output: data_processed/pulsedb_v1/  signals.npy (N*1250 x 2, float32), metadata.csv, config.json,
        PREPROCESSING.md, figures/examples.png
Every segment is its own "recording": phase = initial, condition = calibration for all, sample_id in time
order within each subject, so the existing calibration protocol takes the FIRST k segments as calibration.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FS, SEG = 125, 1250
RULES = {"sbp": (70, 220), "dbp": (30, 130), "min_pp": 10, "max_peak_mismatch": 2, "hr": (40, 150),
         "max_rr_cv": 0.2, "min_segments": 4}


# ------------------------------------------------------------------ selection
def select_segments(idx, cap=200, k_cal=3, log=print):
    for c in ["source", "sid", "case_id", "subject"]:
        if c in idx:
            idx[c] = idx[c].astype(str)
    steps = []

    def step(name, d):
        steps.append({"step": name, **{f"{s}_{x}": v for s, g in d.groupby("source")
                                       for x, v in [("subjects", g.sid.nunique()), ("segments", len(g))]}})
        log(f"  {name:55s} " + " | ".join(f"{s}: {g.sid.nunique():,} subj, {len(g):,} seg"
                                          for s, g in d.groupby("source")))

    step("all segments", idx)
    size = idx.groupby(["sid", "case_id"]).size().rename("n").reset_index()
    main = size.sort_values(["sid", "n", "case_id"], ascending=[True, False, True]).drop_duplicates("sid")
    d = idx.merge(main[["sid", "case_id"]], on=["sid", "case_id"])
    step("1. largest recording per subject", d)
    d = d[d.sbp.between(*RULES["sbp"]) & d.dbp.between(*RULES["dbp"]) & ((d.sbp - d.dbp) >= RULES["min_pp"])]
    step("2. physiological SBP/DBP", d)
    d = d[((d.n_rpeaks - d.n_ppg_peaks).abs() <= RULES["max_peak_mismatch"]) & d.hr_ecg.between(*RULES["hr"])
          & (d.rr_cv <= RULES["max_rr_cv"])]
    step("3. ECG quality", d)
    n = d.groupby("sid").size()
    d = d[d.sid.isin(n[n >= RULES["min_segments"]].index)]
    step(f"4. subjects with >= {RULES['min_segments']} segments", d)

    keep = []
    for sid, g in d.sort_values(["sid", "segment_id"]).groupby("sid", sort=False):
        if len(g) <= cap:
            keep.append(g)
            continue
        first = g.iloc[:k_cal]
        rest = g.iloc[k_cal:]
        pos = np.unique(np.round(np.linspace(0, len(rest) - 1, cap - k_cal)).astype(int))
        keep.append(pd.concat([first, rest.iloc[pos]]))
    d = pd.concat(keep, ignore_index=True)
    step(f"5. cap {cap} per subject (first {k_cal} always kept)", d)
    return d.sort_values(["source", "sid", "segment_id"]).reset_index(drop=True), pd.DataFrame(steps)


# ------------------------------------------------------------------ signal processing
def make_filters(fs=FS):
    from scipy.signal import butter, cheby2
    ecg = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    ppg = cheby2(4, 20, [0.5, 10.0], btype="bandpass", fs=fs, output="sos")
    return ecg, ppg


def process_segment(ecg, ppg, sos_ecg, sos_ppg, refilter=True):
    from scipy.signal import sosfiltfilt
    from scipy.stats import skew
    ecg = np.asarray(ecg, float).ravel()
    ppg = np.asarray(ppg, float).ravel()
    if len(ecg) != SEG or len(ppg) != SEG or not (np.isfinite(ecg).all() and np.isfinite(ppg).all()):
        return None
    if refilter:
        ecg = sosfiltfilt(sos_ecg, ecg, padtype="odd", padlen=SEG - 1)
        ppg = sosfiltfilt(sos_ppg, ppg, padtype="odd", padlen=SEG - 1)
    if skew(ecg) < 0:
        ecg = -ecg
    out = np.stack([ecg, ppg], axis=1)
    sd = out.std(axis=0)
    if (sd < 1e-8).any():
        return None
    return ((out - out.mean(axis=0)) / sd).astype(np.float32)


def ecg_sqi(ecg, fs=FS):
    """Beat-template ECG quality (0-1): mean correlation of each detected beat with the median beat.
    Stored as a column for analysis only - NOT used to select segments."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ecg, height=1.5, distance=int(0.3 * fs))
    h = int(0.25 * fs)
    pk = pk[(pk >= h) & (pk < len(ecg) - h)]
    if len(pk) < 3:
        return np.nan
    beats = np.stack([ecg[k - h:k + h] for k in pk])
    tpl = np.median(beats, axis=0)
    r = [np.corrcoef(b, tpl)[0, 1] for b in beats]
    return float(np.nanmean(r))


def read_field(f, gw, name, i):
    import h5py
    ds = gw[name]
    if ds.dtype == h5py.ref_dtype or ds.dtype.kind == "O":
        return np.asarray(f[ds[0, int(i)]][()], float).ravel()
    return np.asarray(ds[()], float).ravel()


def read_subject(job):
    """job = (file path, list of seg_index, refilter). Returns (positions kept, array (n, 1250, 2))."""
    path, seg_idx, refilter = job
    import h5py
    sos_ecg, sos_ppg = make_filters()
    keep, arrs, sqi = [], [], []
    with h5py.File(path, "r") as f:
        gw = f["Subj_Wins"]
        for j, i in enumerate(seg_idx):
            x = process_segment(read_field(f, gw, "ECG_Record_F", i), read_field(f, gw, "PPG_Record_F", i),
                                sos_ecg, sos_ppg, refilter)
            if x is not None:
                keep.append(j)
                arrs.append(x)
                sqi.append(ecg_sqi(x[:, 0]))
    return keep, (np.stack(arrs) if arrs else np.zeros((0, SEG, 2), np.float32)), sqi


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data0/PulseDB")
    ap.add_argument("--out", default="data_processed/pulsedb_v1")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--k_cal", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no_refilter", action="store_true", help="use PulseDB's Record_F filtering only")
    ap.add_argument("--max_subjects", type=int, default=0, help=">0: quick test on the first N subjects per source")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    if (out / "signals.npy").exists():
        sys.exit(f"{out}/signals.npy already exists - choose another --out or delete it deliberately")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s=""):
        log_lines.append(str(s))
        print(s, flush=True)

    log("# PulseDB preprocessing\n\n## Segment selection")
    idx = pd.read_parquet(Path(args.root) / "eda" / "segment_index.parquet")
    if args.max_subjects:
        for c in ["source", "sid"]:
            idx[c] = idx[c].astype(str)
        keep_sids = idx.groupby("source").sid.apply(lambda s: list(pd.unique(s))[:args.max_subjects]).sum()
        idx = idx[idx.sid.isin(keep_sids)]
    sel, steps = select_segments(idx, cap=args.cap, k_cal=args.k_cal, log=log)

    log(f"\n## Reading and processing {len(sel):,} segments from {sel.sid.nunique():,} subject files "
        f"({args.workers} workers, refilter={not args.no_refilter})")
    seg_root = Path(args.root) / "PulseDB_v2_0" / "Segment_Files"
    jobs, groups = [], []
    for sid, g in sel.groupby("sid", sort=False):
        src, subj = sid.split("_", 1)
        jobs.append((str(seg_root / f"PulseDB_{src}" / f"{subj}.mat"), g.seg_index.astype(int).tolist(), not args.no_refilter))
        groups.append(g)
    signals = np.lib.format.open_memmap(out / "signals_tmp.npy", mode="w+", dtype=np.float32, shape=(len(sel) * SEG, 2))
    rows, sqis, pos, done = [], [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for (keep, arr, sqi), g in zip(ex.map(read_subject, jobs, chunksize=4), groups):
            if len(keep):
                sqis.extend(sqi)
                signals[pos * SEG:(pos + len(keep)) * SEG] = arr.reshape(-1, 2)
                rows.append(g.iloc[keep])
                pos += len(keep)
            done += 1
            if done % 250 == 0:
                log(f"  {done:,}/{len(jobs):,} subjects, {pos:,} segments ({(time.time() - t0) / 60:.1f} min)")
    signals.flush()
    del signals
    tmp = np.load(out / "signals_tmp.npy", mmap_mode="r")
    final = np.lib.format.open_memmap(out / "signals.npy", mode="w+", dtype=np.float32, shape=(pos * SEG, 2))
    final[:] = tmp[:pos * SEG]
    final.flush()
    del final, tmp
    (out / "signals_tmp.npy").unlink()

    m = pd.concat(rows, ignore_index=True)
    # row order = signal order (subjects in selection order, segments in time order) - never re-sort here
    gender = m["sex"].astype(str) if "sex" in m else ""
    meta = pd.DataFrame({
        "sample_id": np.arange(len(m)), "pid": m.sid.values, "protocol": m.source.map({"Vital": "VitalDB"}).fillna(m.source).values,
        "phase": "initial", "condition": "calibration", "measurement": "segment", "is_exercise": False,
        "sbp": m.sbp.values, "dbp": m.dbp.values, "offset": np.arange(len(m)) * SEG, "length": SEG,
        "crop_start_eval": 0, "include_main": 1, "date_time": pd.NaT, "time_s": m.t0.values,
        "case_id": m.case_id.values, "segment_id": m.segment_id.values, "seg_index": m.seg_index.values,
        "age": m.age.values, "gender": gender.values if hasattr(gender, "values") else gender,
        "hr_ecg": m.hr_ecg.values, "ppg_abp_corr": m.ppg_abp_corr.values, "ecg_sqi": np.asarray(sqis, float)})
    meta.to_csv(out / "metadata.csv", index=False)
    json.dump({"config": {"fs_out": FS, "segment_samples": SEG, "cap": args.cap, "k_cal": args.k_cal,
                          "refilter": not args.no_refilter, "rules": RULES, "source": str(args.root),
                          "signals": "ECG_Record_F, PPG_Record_F", "ppg_flipped": False},
               "channels": ["ecg", "ppg"]}, open(out / "config.json", "w"), indent=2)
    steps.to_csv(out / "selection_steps.csv", index=False)

    log(f"\n## Result: {len(meta):,} segments, {meta.pid.nunique():,} subjects "
        f"({(time.time() - t0) / 60:.1f} min)")
    for src, g in meta.groupby("protocol"):
        n = g.groupby("pid").size()
        log(f"  {src}: {g.pid.nunique():,} subjects, {len(g):,} segments (per subject median {n.median():.0f}, "
            f"min {n.min()}, max {n.max()}) | SBP {g.sbp.mean():.1f} ± {g.sbp.std():.1f} | DBP {g.dbp.mean():.1f} ± {g.dbp.std():.1f}"
            f" | ECG SQI median {g.ecg_sqi.median():.2f}, < 0.8: {(g.ecg_sqi < 0.8).mean():.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = np.load(out / "signals.npy", mmap_mode="r")
    ex_rows = meta.groupby("protocol").head(1).index.tolist() + meta.groupby("protocol").tail(1).index.tolist()
    fig, ax = plt.subplots(len(ex_rows), 1, figsize=(14, 2.2 * len(ex_rows)), sharex=True)
    t = np.arange(SEG) / FS
    for a, r in zip(np.atleast_1d(ax), ex_rows):
        x = sig[r * SEG:(r + 1) * SEG]
        a.plot(t, x[:, 0], lw=0.7, label="ECG")
        a.plot(t, x[:, 1] - 4, lw=0.7, label="PPG (offset)")
        a.set_title(f"{meta.protocol[r]} {meta.pid[r]} seg {meta.segment_id[r]} | SBP {meta.sbp[r]:.0f} DBP {meta.dbp[r]:.0f}", fontsize=9)
        a.legend(fontsize=7, loc="upper right")
    np.atleast_1d(ax)[-1].set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig(out / "figures" / "examples.png", dpi=110)
    (out / "PREPROCESSING.md").write_text("\n".join(log_lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
