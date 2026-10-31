#!/usr/bin/env python
"""
Create the FROZEN 5-fold subject-level cross-validation assignment (run once).

Output: data_processed/v1/folds.csv with one row per subject:
  pid, protocol, subj_mean_sbp, fold, role_f0 ... role_f4   (role = train / val / test)

Design (logged in folds_info.json):
  * Subjects are split separately per protocol, so every fold has both protocols.
  * Within a protocol, subjects are sorted by their mean SBP and assigned to folds in shuffled
    blocks of 5, so every fold covers the full BP range (stratification by BP level).
  * For fold k: test = subjects with fold == k. From the remaining subjects, a fixed ~15% is the
    validation set (early stopping), chosen the same stratified way; the rest is training.
  * Every subject is in the test set exactly once across the 5 folds.

Usage:  python scripts/make_folds.py            (refuses to overwrite an existing folds.csv)
"""
import argparse, hashlib, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

K = 5


def stratified_blocks(df, n_groups, rng, sort_col="subj_mean_sbp"):
    """Assign group ids 0..n_groups-1: sort by sort_col, then shuffle ids inside each block."""
    df = df.sort_values(sort_col, kind="mergesort")
    ids = np.empty(len(df), dtype=int)
    for start in range(0, len(df), n_groups):
        block = rng.permutation(n_groups)[: min(n_groups, len(df) - start)]
        ids[start:start + len(block)] = block
    return pd.Series(ids, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="overwrite (only if you are sure)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    out = Path(cfg["paths"]["folds_file"])
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists. Folds are frozen; use --force only if you really mean it.")

    meta = pd.read_csv(data_dir / "metadata.csv", low_memory=False)
    valid = meta[meta["label_valid"].astype(bool)]
    subj = meta.groupby("pid").agg(protocol=("protocol", "first")).reset_index()
    subj = subj.merge(valid.groupby("pid")["sbp"].mean().rename("subj_mean_sbp"), on="pid", how="left")
    subj["subj_mean_sbp"] = subj["subj_mean_sbp"].fillna(subj["subj_mean_sbp"].median())

    rng = np.random.default_rng(args.seed)
    subj["fold"] = -1
    for proto, g in subj.groupby("protocol"):
        subj.loc[g.index, "fold"] = stratified_blocks(g, K, rng)

    n_val_groups = int(round(1 / args.val_frac))          # e.g. 0.15 -> about 1 in 7
    for k in range(K):
        role = pd.Series("train", index=subj.index)
        role[subj["fold"] == k] = "test"
        rest = subj[subj["fold"] != k]
        for proto, g in rest.groupby("protocol"):
            grp = stratified_blocks(g, n_val_groups, rng)
            role[grp[grp == 0].index] = "val"
        subj[f"role_f{k}"] = role

    # ---- checks ----
    assert (subj[[f"role_f{k}" for k in range(K)]] == "test").sum(axis=1).eq(1).all(), "each subject must be test once"
    subj = subj.sort_values("pid").reset_index(drop=True)
    subj.to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    summary = {}
    for k in range(K):
        r = subj[f"role_f{k}"]
        summary[f"fold_{k}"] = {role: {"subjects": int((r == role).sum()),
                                       "mean_sbp": round(float(subj.loc[r == role, "subj_mean_sbp"].mean()), 2),
                                       "by_protocol": subj[r == role]["protocol"].value_counts().to_dict()}
                                for role in ["train", "val", "test"]}
    info = {"created": datetime.now().isoformat(timespec="seconds"), "seed": args.seed, "k": K,
            "val_frac": args.val_frac, "n_subjects": int(len(subj)), "sha256": sha, "summary": summary}
    json.dump(info, open(out.with_name("folds_info.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print(f"\nSaved {out}  (sha256 {sha[:12]}...)")


if __name__ == "__main__":
    main()
