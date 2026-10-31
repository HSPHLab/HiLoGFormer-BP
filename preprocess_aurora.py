#!/usr/bin/env python
"""
Aurora-BP preprocessing (v1) — ECG + PPG for cuffless BP regression.

What it does
  1. Builds one metadata row per measurement (labels, flags, demographics, split).
  2. Reads every waveform once, filters / resamples / normalises ECG + PPG.
  3. Saves all signals in ONE memory-mappable array + a metadata CSV that links
     each recording (sample_id) to its subject, demographics and quality flags.
  4. Writes a decision log (PREPROCESSING_METHODS.md) for the paper's Methods section.

Nothing is deleted: excluded recordings stay in metadata.csv with the reason.

Usage (from the repository root):
  python preprocess_aurora.py --limit 300      # quick test run (writes to .../v1_test)
  python preprocess_aurora.py                  # full run       (writes to .../v1)
"""
import argparse, json, logging, platform, sys, time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy
from scipy import signal as sps
from scipy.stats import skew, kurtosis

# =============================================================================
# CONFIG — every number used in preprocessing lives here and is saved to config.json
# =============================================================================
CFG = dict(
    version="v1",
    data_dir="data/AuroraBP",
    project_dir=".",
    eda_index="eda_outputs/measurement_index.csv",  # frozen split
    fs_in=500,                  # Hz, all Aurora streams resampled to 500 Hz (verified in EDA)
    fs_out=125,                 # Hz, output sampling rate
    ecg_band=[0.5, 40.0],       # Hz, Butterworth band-pass
    ecg_order=4,
    ppg_band=[0.5, 10.0],       # Hz, Chebyshev II band-pass (Liang et al., 2018)
    ppg_order=4,
    ppg_stopband_db=20,         # Chebyshev II minimum stop-band attenuation
    trim_s=0.5,                 # seconds removed at each end after filtering (edge transients)
    crop_s=10.0,                # model input length (seconds)
    max_nan_frac=0.05,          # recordings with more missing samples are not stored
    max_flat_s=1.0,             # flag flat-line segments longer than this
    quality_threshold=0.65,     # dataset optical_quality threshold (Aurora-BP paper)
    sbp_range=[70, 250],        # mmHg, plausible SBP
    dbp_range=[35, 150],        # mmHg, plausible DBP
    ecg_polarity_chunk_s=2.0,
    rpeak_min_distance_s=0.33,  # max ~180 bpm
    ppg_ensemble_window_s=0.6,
    pat_search_s=[0.04, 0.50],  # PPG foot is searched only in this window after the R-peak
    pat_valid_s=[0.08, 0.40],   # physiologically plausible pulse arrival time at the wrist
    min_rpeaks=5,               # minimum detected beats in the stored recording for the SQI check
    hr_agreement_bpm=10,        # used only for choosing recordings in the PPG-orientation vote
    random_seed=42,
)
SIGNAL_CHANNELS = ["ecg", "ppg"]   # column order in signals.npy
DS_FACTOR = CFG["fs_in"] // CFG["fs_out"]

# =============================================================================
# DECISION LOG — written to PREPROCESSING_METHODS.md (for the paper's Methods)
# =============================================================================
DECISIONS = [
    ("D01", "Unit of analysis",
     "One recording = one sample = one BP label (the cuff reading paired with it).",
     "EDA: waveform files are already cut around each cuff reading (ambulatory 15 s, "
     "oscillometric in-lab 30 s, auscultatory ~12-87 s; t always starts at 0).", ""),
    ("D02", "Input channels",
     "ECG ('ekg') and green-light PPG ('optical'). Tonometric pressure and accelerometer are not used in v1.",
     "Project scope: ECG+PPG are available in consumer wearables; ECG-PPG timing (pulse arrival time) "
     "is the main physiological BP marker.", ""),
    ("D03", "Missing samples",
     "Linear interpolation when <= {max_nan_frac:.0%} of a channel is missing; otherwise the recording is not stored.",
     "EDA found no missing samples in scanned files; rule kept as a safeguard.", ""),
    ("D04", "ECG filtering",
     "Butterworth band-pass {ecg_band[0]}-{ecg_band[1]} Hz, design order {ecg_order}, zero-phase (forward-backward, SOS).",
     "Removes the strong baseline wander seen in EDA and high-frequency noise while keeping QRS morphology; "
     "zero-phase filtering avoids shifting ECG/PPG timing, which matters for pulse arrival time.",
     "Common ECG monitoring bandwidth; PulseDB-based work used 0.5-30 Hz [3]."),
    ("D05", "PPG filtering",
     "Chebyshev type II band-pass {ppg_band[0]}-{ppg_band[1]} Hz, design order {ppg_order}, "
     "{ppg_stopband_db} dB stop-band attenuation, zero-phase (SOS).",
     "4th-order Chebyshev II gave the best PPG quality (skewness SQI) among 9 filter types and 10 orders.",
     "Liang et al., 2018 [1]."),
    ("D06", "Filter order of operations",
     "Filter the full recording at 500 Hz first, then resample, then trim {trim_s} s from each end.",
     "Filtering before cropping keeps filter edge transients out of the model input.", ""),
    ("D07", "Resampling",
     "500 Hz -> {fs_out} Hz by polyphase resampling with anti-aliasing FIR (scipy resample_poly, factor 4).",
     "All retained content is < 40 Hz (Nyquist at 125 Hz = 62.5 Hz); 4x shorter sequences make CNN/"
     "transformer training cheaper. 125 Hz matches MIMIC/PulseDB.", "PulseDB [2]."),
    ("D08", "ECG polarity",
     "Per recording: split into {ecg_polarity_chunk_s} s chunks; if the median of abs(chunk minima) exceeds the "
     "median of chunk maxima, the ECG is multiplied by -1.",
     "EDA showed upright QRS in some subjects (a000) and inverted QRS in others (o000), likely from wrist "
     "electrode configuration. Standardising polarity lets the model learn one beat shape.", ""),
    ("D09", "PPG orientation",
     "ECG-gated ensemble average of PPG over {ppg_ensemble_window_s} s after each R-peak; a pulse is 'upright' "
     "if its minimum (foot) precedes its maximum (peak). Orientation is decided by majority vote per "
     "protocol x phase group (recordings with ECG/PPG heart-rate agreement <= {hr_agreement_bpm} bpm) and the "
     "flip is applied to the whole group. The recording-level result is stored for QC.",
     "The same device/processing is used within a group, so a group-level decision is more robust than "
     "flipping noisy single recordings.", ""),
    ("D10", "Normalisation",
     "Per recording, per channel z-score (after filtering, resampling, trimming). The pre-normalisation "
     "standard deviation is stored (ecg_amp, ppg_amp).",
     "EDA: raw offsets differ by orders of magnitude between subjects and phases (PPG ~-581,000 vs ~0), and "
     "ambulatory signals look pre-processed while in-lab signals are raw. Z-scoring removes these device/phase "
     "scale shortcuts. Limitation: absolute amplitude information is discarded.", ""),
    ("D11", "Fixed input length",
     "Model input = {crop_s:.0f} s window: random crop during training, centre crop for evaluation "
     "(crop_start_eval stored per recording).",
     "Every recording >= 11.8 s except 2; a fixed length removes the 'recording length reveals "
     "protocol/phase' shortcut. 10 s matches PulseDB segments.", "PulseDB [2]."),
    ("D12", "Label validity",
     "Exclude missing SBP/DBP, SBP outside {sbp_range}, DBP outside {dbp_range}, or SBP <= DBP.",
     "EDA: 2,397 failed ambulatory cuff readings; DBP = 0 or < 35 mmHg in auscultatory exercise challenges "
     "(Korotkoff sounds audible to zero after exercise).", ""),
    ("D13", "Signal quality",
     "Main set: dataset optical_quality >= {quality_threshold}. Additional SQIs are computed and stored but NOT "
     "used to exclude data in v1: PPG skewness [4], ECG kurtosis [5], ECG vs PPG heart-rate agreement, "
     "R-R variability.",
     "Matches the Aurora-BP inclusion threshold; extra SQIs allow sensitivity analyses later. The PulseDB rule "
     "'remove PPG with negative skewness' [2] is NOT applied: in the test run, clean, upright Aurora wrist PPG "
     "(fast upstroke, slow decay) was mostly negatively skewed, so that fingertip-derived rule does not transfer.",
     "Elgendi 2016 [4]; Li et al. 2008 / Clifford et al. 2012 [5]."),
    ("D18", "Pulse arrival time and SQI-strict set",
     "PAT = time from R-peak to PPG foot (minimum of the ECG-gated ensemble PPG, searched {pat_search_s[0]}-"
     "{pat_search_s[1]} s after the R-peak); valid if {pat_valid_s[0]}-{pat_valid_s[1]} s. sqi_pass = ECG/PPG "
     "heart rates within {hr_agreement_bpm} bpm AND valid PAT AND >= {min_rpeaks} R-peaks. "
     "include_strict = include_main AND sqi_pass (sensitivity analysis only).",
     "Test run showed recordings with good ECG but no visible PPG pulses (e.g. o251 return) and PAT outliers at "
     "0 s and >0.45 s from unrestricted foot search; the restricted window and HR agreement catch these. "
     "PAT is stored as an interpretable physiological feature, not used as model input.", ""),
    ("D14", "Exercise recordings",
     "Kept and flagged (is_exercise) so results can be reported with and without them.",
     "They provide large within-person BP changes; the original study excluded them from analysis.", ""),
    ("D15", "Subject-level split",
     "Frozen split from the EDA (seed 42, 70/15/15 by subject, done separately per protocol). "
     "Subjects never appear in more than one split.",
     "Prevents identity leakage; identical split for all experiments.", ""),
    ("D16", "Demographic groups for analysis",
     "Stored per recording: age, sex, height, weight, BMI, Fitzpatrick skin type, self-reported hypertension, "
     "CVD history; plus groups age_group (<30, 30-44, 45-59, 60+), bmi_group (WHO), fitz_group (I-II, III, "
     "IV-VI, unknown).",
     "Enables error analysis by demographic group after training. Height/weight assumed in inches/pounds "
     "(values like 68 / 200); BMI = 703 * lb / in^2.", ""),
    ("D17", "No silent deletion",
     "Every measurement keeps a row in metadata.csv with status, exclusion reasons and inclusion flags.",
     "Transparent participant/recording flow for the paper (CONSORT-style counts below).", ""),
]
DECISIONS.sort(key=lambda d: d[0])
REFERENCES = [
    "[1] Liang Y, Elgendi M, Chen Z, Ward R. An optimal filter for short photoplethysmogram signals. "
    "Scientific Data 5:180076 (2018). doi:10.1038/sdata.2018.76",
    "[2] Wang W, Mohseni P, Kilgore KL, Najafizadeh L. PulseDB: A large, cleaned dataset based on MIMIC-III "
    "and VitalDB for benchmarking cuff-less blood pressure estimation methods. Front. Digit. Health (2023). "
    "doi:10.3389/fdgth.2022.1090854",
    "[3] Validation of deep learning models for cuffless blood pressure estimation on a large benchmarking "
    "dataset (PulseDB-based; ECG 0.5-30 Hz, PPG 0.5-15 Hz). doi:10.20517/chatmed.2023.23",
    "[4] Elgendi M. Optimal signal quality index for photoplethysmogram signals. Bioengineering 3(4):21 (2016).",
    "[5] Li Q, Mark RG, Clifford GD. Robust heart rate estimation from multiple asynchronous noisy sources "
    "using signal quality indices and a Kalman filter. Physiol. Meas. 29:15-32 (2008); Clifford GD et al. "
    "Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms. "
    "Physiol. Meas. 33:1419 (2012).",
    "NOTE: verify all reference details before submission.",
]

# =============================================================================
# Filters (module level so every worker process builds them once)
# =============================================================================
ECG_SOS = sps.butter(CFG["ecg_order"], CFG["ecg_band"], btype="bandpass", fs=CFG["fs_in"], output="sos")
PPG_SOS = sps.cheby2(CFG["ppg_order"], CFG["ppg_stopband_db"], CFG["ppg_band"], btype="bandpass",
                     fs=CFG["fs_in"], output="sos")


def longest_flat_run(x, tol=1e-9):
    d = np.abs(np.diff(x)) < tol
    if not d.any():
        return 0
    edges = np.diff(np.r_[0, d.astype(np.int8), 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(runs.max()) + 1


def interp_nans(x):
    bad = np.isnan(x)
    if bad.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


def ecg_is_inverted(x, fs):
    n = int(CFG["ecg_polarity_chunk_s"] * fs)
    k = len(x) // n
    if k == 0:
        return False
    c = x[: k * n].reshape(k, n)
    return float(np.median(-c.min(axis=1))) > float(np.median(c.max(axis=1)))


def detect_rpeaks(e, fs):
    thr = max(1.0, 0.5 * np.percentile(e, 99))
    pk, _ = sps.find_peaks(e, height=thr, distance=int(CFG["rpeak_min_distance_s"] * fs))
    return pk


def ppg_ensemble(p, rpk, fs):
    """ECG-gated ensemble average of PPG.
    Returns dict: t_min / t_max over the full window (orientation test), and the foot time searched only
    inside pat_search_s, for both possible orientations (min if upright, max if flipped)."""
    out = dict(ens_t_min=np.nan, ens_t_max=np.nan, ens_beats=0, foot_up=np.nan, foot_flip=np.nan)
    if len(rpk) < 3:
        return out
    rr = np.median(np.diff(rpk))
    w = int(min(CFG["ppg_ensemble_window_s"] * fs, 0.95 * rr))
    beats = [p[r:r + w] for r in rpk if r + w <= len(p)]
    out["ens_beats"] = len(beats)
    if len(beats) < 3 or w < 10:
        return out
    avg = np.mean(beats, axis=0)
    out["ens_t_min"], out["ens_t_max"] = np.argmin(avg) / fs, np.argmax(avg) / fs
    lo, hi = int(CFG["pat_search_s"][0] * fs), min(int(CFG["pat_search_s"][1] * fs), w)
    if hi - lo >= 5:
        seg = avg[lo:hi]
        out["foot_up"], out["foot_flip"] = (lo + np.argmin(seg)) / fs, (lo + np.argmax(seg)) / fs
    return out


def process_recording(item):
    """Read one waveform file and return (info dict, float32 array [n, 2] or None)."""
    sid, path = item
    fs, fo = CFG["fs_in"], CFG["fs_out"]
    info = {"sample_id": sid}
    try:
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in ("t", "ekg", "optical"))
        raw = {"ecg": df["ekg"].to_numpy(float), "ppg": df["optical"].to_numpy(float)}
        n = len(df)
        info["raw_len_s"] = n / fs
        for ch, x in raw.items():
            info[f"{ch}_nan_frac"] = float(np.isnan(x).mean())
            info[f"{ch}_flat_s"] = longest_flat_run(x[~np.isnan(x)]) / fs if (~np.isnan(x)).any() else np.nan
        if n / fs < CFG["crop_s"] + 2 * CFG["trim_s"]:
            info["status"] = "too_short"
            return info, None
        if max(info["ecg_nan_frac"], info["ppg_nan_frac"]) > CFG["max_nan_frac"]:
            info["status"] = "too_many_nans"
            return info, None

        ecg = sps.sosfiltfilt(ECG_SOS, interp_nans(raw["ecg"]))
        ppg = sps.sosfiltfilt(PPG_SOS, interp_nans(raw["ppg"]))
        ecg = sps.resample_poly(ecg, 1, DS_FACTOR)
        ppg = sps.resample_poly(ppg, 1, DS_FACTOR)
        tr = int(CFG["trim_s"] * fo)
        ecg, ppg = ecg[tr:len(ecg) - tr], ppg[tr:len(ppg) - tr]

        inv = ecg_is_inverted(ecg, fo)
        if inv:
            ecg = -ecg
        info["ecg_inverted"] = bool(inv)

        info["ecg_amp"], info["ppg_amp"] = float(ecg.std()), float(ppg.std())
        if info["ecg_amp"] == 0 or info["ppg_amp"] == 0:
            info["status"] = "constant_signal"
            return info, None
        ecg = (ecg - ecg.mean()) / ecg.std()
        ppg = (ppg - ppg.mean()) / ppg.std()

        # ---- quality indices / physiology summaries (stored, not used for exclusion) ----
        rpk = detect_rpeaks(ecg, fo)
        info["n_rpeaks"] = int(len(rpk))
        if len(rpk) >= 3:
            rr = np.diff(rpk) / fo
            info["hr_ecg"] = float(60 / np.median(rr))
            info["rr_cv"] = float(rr.std() / rr.mean())
        ppk, _ = sps.find_peaks(ppg, distance=int(CFG["rpeak_min_distance_s"] * fo), prominence=0.5)
        if len(ppk) >= 3:
            info["hr_ppg"] = float(60 / np.median(np.diff(ppk) / fo))
        if "hr_ecg" in info and "hr_ppg" in info:
            info["hr_diff"] = abs(info["hr_ecg"] - info["hr_ppg"])
        ens = ppg_ensemble(ppg, rpk, fo)
        info.update(ens)
        t_min, t_max = ens["ens_t_min"], ens["ens_t_max"]
        info["ppg_upright"] = bool(t_min < t_max) if np.isfinite(t_min) else np.nan
        info["ppg_skew_raw"] = float(skew(ppg))           # sign fixed later if the group is flipped
        info["ecg_kurtosis"] = float(kurtosis(ecg, fisher=False))
        info["status"] = "ok"
        return info, np.stack([ecg, ppg], axis=1).astype(np.float32)
    except Exception as e:  # keep going; error is recorded
        info["status"] = "read_error"
        info["error"] = f"{type(e).__name__}: {e}"
        return info, None


# =============================================================================
# Metadata helpers
# =============================================================================
def condition_of(name):
    n = str(name).lower()
    if n.startswith("measurement"):
        return "ambulatory"
    for key, cond in [("exercise", "exercise"), ("running", "exercise"), ("walking", "exercise"),
                      ("cool down", "exercise"), ("calibration", "calibration"), ("static", "static_challenge"),
                      ("temporal", "temporal_challenge"), ("supine", "supine"), ("standing", "standing"),
                      ("sitting", "sitting"), ("signal check", "signal_check")]:
        if key in n:
            return cond
    return "other"


def bp_class4(s, d):
    if pd.isna(s) or pd.isna(d):
        return np.nan
    if s >= 140 or d >= 90:
        return "stage2"
    if s >= 130 or d >= 80:
        return "stage1"
    if s >= 120:
        return "elevated"
    return "normal"


def add_demographic_groups(df):
    if {"height", "weight"} <= set(df.columns):
        df["bmi"] = 703.0 * df["weight"] / df["height"] ** 2
        df["bmi_group"] = pd.cut(df["bmi"], [0, 18.5, 25, 30, 1e9],
                                 labels=["underweight", "normal", "overweight", "obese"], right=False).astype(str)
    if "age" in df:
        df["age_group"] = pd.cut(df["age"], [0, 30, 45, 60, 200],
                                 labels=["<30", "30-44", "45-59", "60+"], right=False).astype(str)
    if "fitzpatrick_scale" in df:
        f = df["fitzpatrick_scale"]
        df["fitz_group"] = np.select([f.isin([1, 2]), f.eq(3), f.isin([4, 5, 6])],
                                     ["I-II", "III", "IV-VI"], default="unknown")
    return df


def build_metadata(data_dir, log):
    parts = pd.read_csv(data_dir / "participants.tsv", sep="\t")
    meas = []
    for proto in ["auscultatory", "oscillometric"]:
        f = data_dir / f"measurements_{proto}.tsv"
        m = pd.read_csv(f, sep="\t")
        m["protocol"] = proto
        meas.append(m)
    meas = pd.concat(meas, ignore_index=True)
    log.info(f"participants: {len(parts)} | measurements: {len(meas)}")

    wave_files = [p for p in data_dir.rglob("*.tsv") if p.parent != data_dir]
    by_name = {p.name: p for p in wave_files}
    log.info(f"waveform files on disk: {len(wave_files)}")

    def resolve(row):
        raw = str(row.get("waveform_file_path", ""))
        if raw and raw != "nan":
            for c in (data_dir / raw, data_dir / raw.lstrip("/\\")):
                if c.is_file():
                    return str(c)
            if Path(raw).name in by_name:
                return str(by_name[Path(raw).name])
        guess = f"{row['pid']}.{row['phase']}.{str(row['measurement']).replace(' ', '_')}.tsv"
        return str(by_name[guess]) if guess in by_name else None

    meas["wave_path"] = meas.apply(resolve, axis=1)
    meas["condition"] = meas["measurement"].map(condition_of)
    meas["is_exercise"] = meas["condition"].eq("exercise")
    meas["date_time_parsed"] = pd.to_datetime(meas["date_time"], errors="coerce")
    meas["hour"] = meas["date_time_parsed"].dt.hour
    meas = meas.drop(columns=["date_time_parsed"])

    # label validity
    s, d = meas["sbp"], meas["dbp"]
    lo_s, hi_s = CFG["sbp_range"]; lo_d, hi_d = CFG["dbp_range"]
    meas["label_missing"] = s.isna() | d.isna()
    meas["label_implausible"] = ~meas["label_missing"] & ((s < lo_s) | (s > hi_s) | (d < lo_d) | (d > hi_d) | (s <= d))
    meas["label_valid"] = ~meas["label_missing"] & ~meas["label_implausible"]
    meas["bp_class4"] = [bp_class4(a, b) for a, b in zip(s, d)]

    meas = meas.merge(parts, on="pid", how="left", suffixes=("", "_participant"))
    meas = add_demographic_groups(meas)
    meas.insert(0, "sample_id", np.arange(len(meas)))
    return meas


def attach_split(meta, eda_index, log):
    p = Path(eda_index)
    if not p.exists():
        raise FileNotFoundError(f"Frozen split not found: {p}. Run the EDA notebook first.")
    e = pd.read_csv(p, usecols=["pid", "split"], low_memory=False).dropna()
    per = e.groupby("pid")["split"].agg(lambda s: s.unique().tolist())
    clash = per[per.map(len) > 1]
    if len(clash):
        raise ValueError(f"Subjects with more than one split in EDA index: {list(clash.index)[:10]}")
    meta["split"] = meta["pid"].map(per.map(lambda v: v[0])).fillna("unassigned")
    log.info("split (subjects): " + meta.groupby("split")["pid"].nunique().to_dict().__repr__())
    return meta


# =============================================================================
# QC figures
# =============================================================================
def qc_figures(meta, signals, out, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qc = out / "qc"; qc.mkdir(exist_ok=True)
    fo, fi = CFG["fs_out"], CFG["fs_in"]
    ok = meta[meta["status"].eq("ok")]
    if ok.empty:
        return
    ex = ok.groupby(["protocol", "phase"]).sample(1, random_state=CFG["random_seed"])
    for _, r in ex.iterrows():
        raw = pd.read_csv(r["wave_path"], sep="\t", usecols=lambda c: c in ("t", "ekg", "optical"))
        sec = 8
        x0 = int(CFG["trim_s"] * fi)
        a, b = int(r["offset"]), int(r["offset"] + r["length"])
        proc = signals[a:b][: sec * fo]
        fig, ax = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        t_raw = np.arange(sec * fi) / fi
        ax[0].plot(t_raw, raw["ekg"].to_numpy()[x0:x0 + sec * fi], lw=0.6); ax[0].set_ylabel("ECG raw")
        ax[1].plot(np.arange(len(proc)) / fo, proc[:, 0], lw=0.8); ax[1].set_ylabel("ECG processed (z)")
        ax[2].plot(t_raw, raw["optical"].to_numpy()[x0:x0 + sec * fi], lw=0.6, color="tab:green")
        ax[2].set_ylabel("PPG raw")
        ax[3].plot(np.arange(len(proc)) / fo, proc[:, 1], lw=0.8, color="tab:green")
        ax[3].set_ylabel("PPG processed (z)"); ax[3].set_xlabel("t (s)")
        fig.suptitle(f"{r['pid']} | {r['phase']} | {r['measurement']} | SBP {r['sbp']} / DBP {r['dbp']} | "
                     f"ECG inverted: {r['ecg_inverted']} | PPG flipped: {r['ppg_flipped']}\n"
                     f"optical_quality {r['optical_quality']:.2f} | hr_diff {r['hr_diff']:.1f} bpm | "
                     f"PAT {r['pat_s']:.3f} s | sqi_pass {r['sqi_pass']}")
        plt.tight_layout(); plt.savefig(qc / f"raw_vs_processed_{r['protocol']}_{r['phase']}.png", dpi=110)
        plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(18, 3.5))
    for proto, g in ok.groupby("protocol"):
        ax[0].hist(g["ppg_skewness"].dropna(), bins=50, alpha=0.6, label=proto)
        ax[1].hist(g["ecg_kurtosis"].dropna().clip(upper=30), bins=50, alpha=0.6, label=proto)
        ax[2].hist(g["hr_diff"].dropna().clip(upper=60), bins=60, alpha=0.6, label=proto)
        ax[3].hist(g["pat_s"].dropna(), bins=50, alpha=0.6, label=proto)
    for a, t in zip(ax, ["PPG skewness", "ECG kurtosis (clipped 30)", "|HR_ECG - HR_PPG| (bpm, clipped 60)",
                         "Approx. pulse arrival time (s)"]):
        a.set_title(t); a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(qc / "sqi_distributions.png", dpi=110); plt.close()

    m = ok[ok["include_main"] & ok["pat_valid"]]
    if len(m) > 10:
        r_all = m["pat_s"].corr(m["sbp"])
        r_sub = m.groupby("pid").filter(lambda g: len(g) >= 5).groupby("pid").apply(
            lambda g: g["pat_s"].corr(g["sbp"])).median()
        plt.figure(figsize=(5.5, 4))
        plt.scatter(m["pat_s"], m["sbp"], s=3, alpha=0.3)
        plt.xlabel("PAT (s)"); plt.ylabel("SBP (mmHg)"); plt.grid(alpha=0.3)
        plt.title(f"PAT vs SBP (main, valid PAT)\nall r = {r_all:.2f} | median within-subject r = {r_sub:.2f}")
        plt.tight_layout(); plt.savefig(qc / "pat_vs_sbp.png", dpi=110); plt.close()
    bad = ok[~ok["sqi_pass"].astype(bool)]
    for _, r in bad.sample(min(3, len(bad)), random_state=CFG["random_seed"]).iterrows():
        a, b = int(r["offset"]), int(r["offset"] + r["length"])
        proc = signals[a:b][: 8 * fo]
        fig, ax = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
        ax[0].plot(np.arange(len(proc)) / fo, proc[:, 0], lw=0.8); ax[0].set_ylabel("ECG (z)")
        ax[1].plot(np.arange(len(proc)) / fo, proc[:, 1], lw=0.8, color="tab:green"); ax[1].set_ylabel("PPG (z)")
        fig.suptitle(f"SQI FAIL | {r['pid']} {r['phase']} {r['measurement']} | hr_diff {r['hr_diff']:.1f} | "
                     f"PAT {r['pat_s']:.2f} | n_rpeaks {r['n_rpeaks']} | optical_quality {r['optical_quality']:.2f}")
        plt.tight_layout(); plt.savefig(qc / f"sqi_fail_{r['pid']}_{r['sample_id']}.png", dpi=110); plt.close()
    log.info(f"QC figures saved to {qc}")


# =============================================================================
# Methods log
# =============================================================================
def write_methods_md(path, meta, flow, orient_tbl, runinfo):
    L = [f"# Aurora-BP preprocessing — {CFG['version']}", "",
         f"Generated: {runinfo['finished']}  |  run time: {runinfo['minutes']:.1f} min  |  "
         f"test run: {runinfo['limit'] is not None}", "",
         f"Data: `{CFG['data_dir']}`  |  Output: `{runinfo['out_dir']}`", "",
         "## Decisions", "",
         "| ID | Step | Decision | Rationale | Reference |", "|---|---|---|---|---|"]
    for i, step, dec, why, ref in DECISIONS:
        L.append(f"| {i} | {step} | {dec.format(**CFG)} | {why.format(**CFG)} | {ref} |")
    L += ["", "## Recording / subject flow", "", "| Step | Recordings | Subjects |", "|---|---|---|"]
    for name, n_rec, n_sub in flow:
        L.append(f"| {name} | {n_rec:,} | {n_sub:,} |")
    L += ["", "## PPG orientation vote per group (D09)", "", orient_tbl.to_markdown() if hasattr(orient_tbl, "to_markdown") and _has_tabulate() else "```\n" + orient_tbl.to_string() + "\n```"]
    ok = meta[meta.status.eq("ok")]
    L += ["", "## ECG polarity (D08)", "", "```",
          ok.assign(ecg_inverted=ok["ecg_inverted"].astype(float))
            .groupby(["protocol", "phase"])["ecg_inverted"].mean().round(3).to_string(), "```"]
    L += ["", "## Main set composition", "", "```",
          meta[meta.include_main].groupby(["split", "protocol"]).agg(
              recordings=("sample_id", "size"), subjects=("pid", "nunique"),
              sbp_mean=("sbp", "mean"), dbp_mean=("dbp", "mean")).round(1).to_string(), "```", "",
          "```", pd.crosstab(meta[meta.include_main]["split"], meta[meta.include_main]["bp_class4"],
                             normalize="index").round(3).to_string(), "```"]
    L += ["", "## Software", "", "```", "\n".join(f"{k}: {v}" for k, v in runinfo["versions"].items()), "```",
          "", "## References", ""] + REFERENCES
    path.write_text("\n".join(L))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N random recordings (test run)")
    ap.add_argument("--workers", type=int, default=16, help="parallel CPU workers (shared server: keep modest)")
    ap.add_argument("--data_dir", default=CFG["data_dir"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--eda_index", default=CFG["eda_index"])
    args = ap.parse_args()

    CFG["data_dir"], CFG["eda_index"] = args.data_dir, args.eda_index
    out = Path(args.out_dir or f"{CFG['project_dir']}/data_processed/{CFG['version']}"
               + ("_test" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("prep"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(out / "preprocess.log", mode="w"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    t0 = time.time()
    log.info(f"Aurora-BP preprocessing {CFG['version']} | out: {out} | limit: {args.limit} | workers: {args.workers}")

    meta = build_metadata(Path(CFG["data_dir"]), log)
    meta = attach_split(meta, CFG["eda_index"], log)

    todo = meta[meta["wave_path"].notna()]
    if args.limit:
        todo = todo.sample(min(args.limit, len(todo)), random_state=CFG["random_seed"])
    items = list(zip(todo["sample_id"], todo["wave_path"]))
    log.info(f"processing {len(items):,} waveform files ...")

    infos, arrays = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, (info, arr) in enumerate(ex.map(process_recording, items, chunksize=16), 1):
            infos.append(info)
            if arr is not None:
                arrays[info["sample_id"]] = arr
            if k % 2000 == 0 or k == len(items):
                log.info(f"  {k:,}/{len(items):,} done ({time.time() - t0:.0f} s)")
    info_df = pd.DataFrame(infos)
    for c in ("ecg_inverted", "ppg_upright"):
        if c in info_df:
            info_df[c] = pd.to_numeric(info_df[c].map({True: 1.0, False: 0.0}), errors="coerce")
    meta = meta.merge(info_df, on="sample_id", how="left")
    meta["status"] = np.where(meta["wave_path"].isna(), "no_waveform_file", meta["status"].fillna("not_processed"))
    log.info("status counts: " + meta["status"].value_counts().to_dict().__repr__())
    errs = meta[meta["status"].eq("read_error")]
    for _, r in errs.head(20).iterrows():
        log.warning(f"read_error sample {r.sample_id} {r.wave_path}: {r.error}")

    # ---- D09: PPG orientation vote per protocol x phase ----
    ok = meta["status"].eq("ok")
    vote = meta[ok & (meta["hr_diff"] <= CFG["hr_agreement_bpm"]) & meta["ppg_upright"].notna()]
    orient = vote.groupby(["protocol", "phase"])["ppg_upright"].agg(
        n_voting="size", share_upright=lambda s: s.astype(float).mean())
    orient["flip_group"] = orient["share_upright"] < 0.5
    log.info("PPG orientation vote:\n" + orient.round(3).to_string())
    flip_map = orient["flip_group"].to_dict()
    meta["ppg_flipped"] = [bool(flip_map.get((p, ph), False)) if s == "ok" else np.nan
                           for p, ph, s in zip(meta.protocol, meta.phase, meta.status)]
    for sid in meta.loc[meta["ppg_flipped"].eq(True), "sample_id"]:
        if sid in arrays:
            arrays[sid][:, 1] *= -1
    flipped = meta["ppg_flipped"].eq(True)
    meta["ppg_skewness"] = np.where(flipped, -meta["ppg_skew_raw"], meta["ppg_skew_raw"])
    meta["pat_s"] = np.where(flipped, meta["foot_flip"], meta["foot_up"])     # PPG foot after R-peak (D18)
    lo, hi = CFG["pat_valid_s"]
    meta["pat_valid"] = meta["pat_s"].between(lo, hi)
    meta["sqi_pass"] = (meta["status"].eq("ok") & (meta["hr_diff"] <= CFG["hr_agreement_bpm"])
                        & meta["pat_valid"] & (meta["n_rpeaks"] >= CFG["min_rpeaks"]))
    up = pd.to_numeric(meta["ppg_upright"], errors="coerce").astype(float)      # 1 / 0 / NaN
    meta["ppg_upright_after_flip"] = np.where(flipped, 1.0 - up, up)

    # ---- write signals (one array) + offsets ----
    order = sorted(arrays)
    lengths = np.array([len(arrays[s]) for s in order], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    sig_path = out / "signals.npy"
    mm = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float32, shape=(int(lengths.sum()), 2))
    for s, o, n in zip(order, offsets, lengths):
        mm[o:o + n] = arrays[s]
    mm.flush(); del mm
    pos = pd.DataFrame({"sample_id": order, "offset": offsets, "length": lengths})
    meta = meta.merge(pos, on="sample_id", how="left")
    meta["offset"] = meta["offset"].fillna(-1).astype(np.int64)
    meta["length"] = meta["length"].fillna(0).astype(np.int64)
    crop = int(CFG["crop_s"] * CFG["fs_out"])
    meta["crop_start_eval"] = np.where(meta["length"] >= crop, (meta["length"] - crop) // 2, -1)
    log.info(f"signals.npy: {lengths.sum():,} samples x 2 channels ({sig_path.stat().st_size / 1e9:.2f} GB)")

    # ---- inclusion flags ----
    has_sig = meta["length"] >= crop
    in_split = meta["split"].isin(["train", "val", "test"])
    q_ok = meta["optical_quality"] >= CFG["quality_threshold"]
    meta["include_any_quality"] = has_sig & meta["label_valid"] & in_split
    meta["include_main"] = meta["include_any_quality"] & q_ok
    meta["include_strict"] = meta["include_main"] & meta["sqi_pass"]
    reasons = []
    for r_has, r_lab_m, r_lab_i, r_split, r_q, st in zip(has_sig, meta.label_missing, meta.label_implausible,
                                                          in_split, q_ok, meta.status):
        rs = []
        if not r_has: rs.append(f"no_signal:{st}")
        if r_lab_m: rs.append("label_missing")
        if r_lab_i: rs.append("label_implausible")
        if not r_split: rs.append("no_split")
        if not r_q: rs.append("low_optical_quality")
        reasons.append(";".join(rs))
    meta["exclusion_reasons"] = reasons

    # ---- flow table ----
    def cnt(mask): return int(mask.sum()), int(meta.loc[mask, "pid"].nunique())
    allm = pd.Series(True, index=meta.index)
    flow = [("All measurements", *cnt(allm)),
            ("Waveform file found", *cnt(meta.wave_path.notna())),
            ("Signal stored (processed OK, >= 10 s)", *cnt(has_sig)),
            ("+ BP label present", *cnt(has_sig & ~meta.label_missing)),
            ("+ BP label plausible", *cnt(has_sig & meta.label_valid)),
            ("+ subject in split (= include_any_quality)", *cnt(meta.include_any_quality)),
            (f"+ optical_quality >= {CFG['quality_threshold']} (= include_main)", *cnt(meta.include_main)),
            ("   of which exercise recordings", *cnt(meta.include_main & meta.is_exercise)),
            ("include_main + ECG/PPG SQI pass (= include_strict, sensitivity set)", *cnt(meta.include_strict))]
    for f in flow:
        log.info(f"FLOW | {f[0]}: {f[1]:,} recordings, {f[2]:,} subjects")

    # ---- save ----
    meta = meta.drop(columns=["ppg_skew_raw"], errors="ignore")
    meta.to_csv(out / "metadata.csv", index=False)
    meta.drop_duplicates("pid")[["pid", "protocol", "split"]].to_csv(out / "splits.csv", index=False)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "pandas": pd.__version__, "platform": platform.platform()}
    runinfo = {"finished": datetime.now().strftime("%Y-%m-%d %H:%M"), "minutes": (time.time() - t0) / 60,
               "limit": args.limit, "out_dir": str(out), "versions": versions}
    json.dump({"config": CFG, "channels": SIGNAL_CHANNELS, "run": runinfo,
               "ecg_sos": ECG_SOS.tolist(), "ppg_sos": PPG_SOS.tolist()},
              open(out / "config.json", "w"), indent=2, default=str)
    write_methods_md(out / "PREPROCESSING_METHODS.md", meta, flow, orient.round(3), runinfo)

    sig = np.load(sig_path, mmap_mode="r")
    qc_figures(meta, sig, out, log)
    log.info(f"DONE in {(time.time() - t0) / 60:.1f} min. Files in {out}")


if __name__ == "__main__":
    main()