#!/usr/bin/env python
"""
Subject-level, leakage-free splits for PulseDB fine-tuning (MIMIC and VitalDB handled as SEPARATE datasets).

  python scripts/make_pulsedb_splits.py

Per dataset (seed 2026, unit = subject; every segment of a subject always has the subject's role):
  1. test set: 30% of subjects, stratified by the subject's mean SBP (5 strata) -> fixed for ALL runs
  2. pool: the other 70% of subjects
  3. 5 draws r = 0..4 (draw r <-> Aurora fold model r <-> from-scratch seed r): a stratified random ordering of
     the pool; the fraction f uses the FIRST round(f * |pool|) subjects of that ordering, so
     5% ⊂ 10% ⊂ 20% ⊂ 50% ⊂ 100% (nested); 100% = the whole pool for every draw
  4. validation: every 5th subject of the ordering (positions 4, 9, 14, ...) -> 20% of each subset is
     validation, also nested; the rest is training
Output (data_processed/pulsedb_v1/splits/):
  splits_<dataset>.csv   one row per subject: pid, dataset, mean_sbp, stratum, test (bool), order_d<r>,
                         role_d<r>_f<pct> in {train, val, unused, test}
  SPLITS_AUDIT.md        counts per role / draw / fraction, BP balance, and every leakage check (must all PASS)
  splits_manifest.json   seed, fractions, file hashes
Refuses to overwrite existing split files (they are frozen once created).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = [5, 10, 20, 50, 100]          # % of the pool; 0% = zero-shot (no training data)
N_DRAWS, TEST_FRAC, N_STRATA, VAL_EVERY, SEED = 5, 0.30, 5, 5, 2026


def stratified_order(pids, strata, rng):
    """Random order in which every prefix is (approximately) stratified: shuffle within strata, interleave."""
    groups = [list(rng.permutation(pids[strata == s])) for s in np.unique(strata)]
    groups = [g for g in groups if g]
    out = []
    while any(groups):
        for g in rng.permutation(len(groups)):
            if groups[g]:
                out.append(groups[g].pop())
    return out


def make_dataset_splits(meta, ds):
    subj = meta.groupby("pid").agg(mean_sbp=("sbp", "mean"), mean_dbp=("dbp", "mean"), n_seg=("sbp", "size")).reset_index()
    subj["dataset"] = ds
    subj["stratum"] = pd.qcut(subj.mean_sbp.rank(method="first"), N_STRATA, labels=False)
    rng = np.random.default_rng(SEED)
    test = set()
    for s, g in subj.groupby("stratum"):
        pids = rng.permutation(g.pid.to_numpy())
        test.update(pids[:int(round(TEST_FRAC * len(pids)))])
    subj["test"] = subj.pid.isin(test)
    pool = subj[~subj.test]
    for r in range(N_DRAWS):
        order = stratified_order(pool.pid.to_numpy(), pool.stratum.to_numpy(), np.random.default_rng(SEED + 1 + r))
        pos = {p: i for i, p in enumerate(order)}
        subj[f"order_d{r}"] = subj.pid.map(pos)
        for f in FRACTIONS:
            n = int(round(f / 100 * len(order)))
            o = subj[f"order_d{r}"]
            role = np.where(subj.test, "test",
                            np.where(o.isna() | (o >= n), "unused",
                                     np.where(o % VAL_EVERY == VAL_EVERY - 1, "val", "train")))
            subj[f"role_d{r}_f{f}"] = role
    return subj


def audit(subj, meta, ds, L):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        L.append(f"| {name} | {'PASS' if cond else '**FAIL**'} |")

    seg = meta.groupby("pid").size()
    test_p = set(subj.pid[subj.test])
    pool_p = set(subj.pid[~subj.test])
    L += [f"## {ds}", "",
          f"Subjects: {len(subj):,} | test: {len(test_p):,} ({len(test_p) / len(subj):.1%}, "
          f"{int(seg[list(test_p)].sum()):,} segments) | pool: {len(pool_p):,} ({int(seg[list(pool_p)].sum()):,} segments)", "",
          "### Training / validation subjects per fraction (each of the 5 draws)", "",
          "| Fraction of pool | Draw | Train subj (seg) | Val subj (seg) | Mean SBP train / val / test |", "|---|---|---|---|---|"]
    mt = subj.loc[subj.test, "mean_sbp"].mean()
    for f in FRACTIONS:
        for r in range(N_DRAWS):
            c = subj[f"role_d{r}_f{f}"]
            tr, va = subj.pid[c == "train"], subj.pid[c == "val"]
            L.append(f"| {f}% | {r} | {len(tr):,} ({int(seg[tr].sum()):,}) | {len(va):,} ({int(seg[va].sum()):,}) | "
                     f"{subj.mean_sbp[c == 'train'].mean():.1f} / {subj.mean_sbp[c == 'val'].mean():.1f} / {mt:.1f} |")
    L += ["", "### Leakage and consistency checks", "", "| Check | Result |", "|---|---|"]
    check("every subject has exactly one row", subj.pid.is_unique)
    check("all segments of a subject come from ONE recording (case_id)", (meta.groupby("pid").case_id.nunique() == 1).all())
    check("test and pool are disjoint and cover all subjects", not (test_p & pool_p) and len(test_p | pool_p) == len(subj))
    check("test set identical in every draw and fraction",
          all(set(subj.pid[subj[f"role_d{r}_f{f}"] == "test"]) == test_p for r in range(N_DRAWS) for f in FRACTIONS))
    check("no test subject is ever train / val",
          all(not (set(subj.pid[subj[f"role_d{r}_f{f}"].isin(["train", "val"])]) & test_p) for r in range(N_DRAWS) for f in FRACTIONS))
    check("train and val disjoint in every draw and fraction",
          all(not (set(subj.pid[subj[f"role_d{r}_f{f}"] == "train"]) & set(subj.pid[subj[f"role_d{r}_f{f}"] == "val"]))
              for r in range(N_DRAWS) for f in FRACTIONS))
    nested = True
    for r in range(N_DRAWS):
        for a, b in zip(FRACTIONS[:-1], FRACTIONS[1:]):
            for role in ("train", "val"):
                nested &= set(subj.pid[subj[f"role_d{r}_f{a}"] == role]) <= set(subj.pid[subj[f"role_d{r}_f{b}"] == role])
    check("nested: train(5%) ⊂ train(10%) ⊂ ... ⊂ train(100%), same for val", nested)
    check("100% = whole pool in every draw",
          all(set(subj.pid[subj[f"role_d{r}_f100"].isin(["train", "val"])]) == pool_p for r in range(N_DRAWS)))
    check("validation is about 20% of every subset",
          all(abs((subj[f"role_d{r}_f{f}"] == "val").sum() / max(subj[f"role_d{r}_f{f}"].isin(["train", "val"]).sum(), 1) - 0.2) < 0.03
              for r in range(N_DRAWS) for f in FRACTIONS))
    check(f"subject IDs are dataset-specific (all start with '{'MIMIC' if ds == 'MIMIC' else 'Vital'}_')",
          subj.pid.str.startswith("MIMIC_" if ds == "MIMIC" else "Vital_").all())
    L.append("")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_processed/pulsedb_v1")
    args = ap.parse_args()
    data = Path(args.data)
    out = data / "splits"
    if out.exists() and any(out.glob("splits_*.csv")):
        sys.exit(f"{out} already contains split files - they are frozen. Delete them deliberately to recreate.")
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "metadata.csv", low_memory=False)

    L = ["# PulseDB fine-tuning splits — audit", "",
         f"Unit of splitting = SUBJECT (pid). MIMIC and VitalDB are split separately and never mixed. Seed {SEED}. "
         f"Test = {TEST_FRAC:.0%} of subjects (stratified by mean SBP, {N_STRATA} strata), fixed for all runs. "
         f"Fractions {FRACTIONS} % of the pool (nested), {N_DRAWS} draws (draw r ↔ Aurora fold model r ↔ from-scratch seed r), "
         f"validation = every {VAL_EVERY}th subject of each draw's order (20%, nested).", ""]
    manifest = {"seed": SEED, "test_frac": TEST_FRAC, "fractions_pct_of_pool": FRACTIONS, "n_draws": N_DRAWS,
                "val_every": VAL_EVERY, "strata": N_STRATA, "source_metadata_sha256":
                hashlib.sha256((data / "metadata.csv").read_bytes()).hexdigest(), "files": {}}
    all_ok = True
    for ds in sorted(meta.protocol.unique()):
        m = meta[meta.protocol == ds]
        subj = make_dataset_splits(m, ds)
        all_ok &= audit(subj, m, ds, L)
        fp = out / f"splits_{ds}.csv"
        subj.to_csv(fp, index=False)
        manifest["files"][fp.name] = hashlib.sha256(fp.read_bytes()).hexdigest()
    L += [f"**Overall: {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED - do not use these splits'}**", "",
          "File hashes (recorded in every fine-tuning run's provenance):", ""] + \
         [f"- `{k}`: `{v[:16]}…`" for k, v in manifest["files"].items()]
    (out / "SPLITS_AUDIT.md").write_text("\n".join(L))
    json.dump(manifest, open(out / "splits_manifest.json", "w"), indent=2)
    print("\n".join(L))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
