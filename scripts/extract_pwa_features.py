#!/usr/bin/env python
"""
Extract the 31 handcrafted ECG/PPG features (src/features/pwa_features.py) for every window of a dataset.
The window is exactly the 10 s window every model sees at test time (offset + crop_start_eval, 1,250 samples).

  python scripts/extract_pwa_features.py --data data_processed/v1         --name aurora_v1
  python scripts/extract_pwa_features.py --data data_processed/pulsedb_v1 --name pulsedb_v1

Output: data_processed/features_pwa/<name>.csv  (sample_id, 31 features, n_cycles, notch_fallback_frac)
        data_processed/features_pwa/<name>_QC.md (coverage, NaN rates, medians)
Windows with fewer than 2 valid cycles get NaN features (imputed later with training statistics, per fold).
Refuses to overwrite an existing output.
"""
import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.pwa_features import FEATURE_NAMES, extract_features  # noqa: E402

FS, SEG = 125, 1250
_G = {}


def _init(sig_path, ie, ip):
    _G["sig"] = np.load(sig_path, mmap_mode="r")
    _G["ie"], _G["ip"] = ie, ip


def _work(chunk):
    out = []
    for sid, s0 in chunk:
        w = np.asarray(_G["sig"][s0:s0 + SEG], float)
        f, n, fb = extract_features(w[:, _G["ie"]], w[:, _G["ip"]], FS)
        out.append([sid, *f, n, fb])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="data_processed/features_pwa")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{args.name}.csv"
    if fp.exists():
        sys.exit(f"{fp} exists - features are frozen once extracted; delete it deliberately to recreate")
    t0 = time.time()
    meta = pd.read_csv(data / "metadata.csv", low_memory=False)
    ch = json.load(open(data / "config.json")).get("channels", ["ecg", "ppg"])
    if "include_main" in meta:
        meta = meta[meta.include_main.astype(bool)]
    start = meta["offset"].astype(int) + (meta["crop_start_eval"].astype(int) if "crop_start_eval" in meta else 0)
    jobs = list(zip(meta.sample_id.astype(int), start))
    chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
    print(f"{len(jobs):,} windows from {data} ({len(chunks)} chunks, {args.workers} workers)", flush=True)
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(str(data / "signals.npy"), ch.index("ecg"), ch.index("ppg"))) as pool:
        for i, r in enumerate(pool.imap(_work, chunks), 1):
            rows += r
            if i % 100 == 0:
                print(f"  {len(rows):,}/{len(jobs):,} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows, columns=["sample_id", *FEATURE_NAMES, "n_cycles", "notch_fallback_frac"])
    df.to_csv(fp, index=False)
    ok = df[FEATURE_NAMES].notna().any(axis=1)
    m = meta.set_index("sample_id")
    L = [f"# PWA features — {args.name}", "",
         f"Source: `{data}` (metadata sha256 {hashlib.sha256((data / 'metadata.csv').read_bytes()).hexdigest()[:16]}…); "
         f"window = offset + crop_start_eval, {SEG} samples at {FS} Hz; {(time.time() - t0) / 60:.1f} min.", "",
         f"- windows: {len(df):,}; with features (>= 2 valid cycles): {ok.mean():.2%}",
         f"- cycles per window: median {df.n_cycles.median():.0f}",
         f"- dicrotic notch from the APPG fallback: {df.notch_fallback_frac.mean():.1%} of cycles", ""]
    if "protocol" in m:
        L += ["| Group | Windows | With features | Median cycles |", "|---|---|---|---|"]
        for g, idx in df.groupby(m.loc[df.sample_id, "protocol"].to_numpy()).groups.items():
            d = df.loc[idx]
            L.append(f"| {g} | {len(d):,} | {d[FEATURE_NAMES].notna().any(axis=1).mean():.2%} | {d.n_cycles.median():.0f} |")
        L.append("")
    L += ["| Feature | NaN rate (windows with features) | Median | IQR |", "|---|---|---|---|"]
    for f in FEATURE_NAMES:
        x = df.loc[ok, f]
        L.append(f"| {f} | {x.isna().mean():.2%} | {x.median():.4g} | {x.quantile(.25):.4g} – {x.quantile(.75):.4g} |")
    (out / f"{args.name}_QC.md").write_text("\n".join(L))
    print("\n".join(L[:8]))
    print(f"\nSaved {fp} and {args.name}_QC.md")


if __name__ == "__main__":
    main()
