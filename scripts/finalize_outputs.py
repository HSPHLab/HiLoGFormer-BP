#!/usr/bin/env python
"""
Close-as-we-go: copy FINAL figures and tables into paper/final/ under their paper names, with draft captions, checksums and
provenance, and write-protect them. Entries are added to REGISTRY as each analysis is completed.

  python scripts/finalize_outputs.py                     # finalise every registry entry not yet final
  python scripts/finalize_outputs.py --replace Fig_I2_drift --reason "legend overlapped curve"   # deliberate replacement

Output: paper/final/figures/<name>.pdf (+ preview png, json sidecar), paper/final/tables/<name>.md, paper/final/MANIFEST.md
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "paper" / "final"

# (final name, source file, draft caption / description). Figures: .pdf; tables: .md
REGISTRY = [
    # ---------------- interpretability A
    ("Fig_I1_trust", "paper/figures/interp_A_trust.pdf",
     "Trust indicator from calibration consistency (Aurora-BP test subjects; HiCAF-BP-Cal without time context). (a) MAE of the "
     "retained readings when the least consistent readings are flagged, compared with flagging by time since calibration "
     "(dashed: all readings). (b, c) MAE per trust level (green / amber / red; cut-points from validation subjects) for SBP and "
     "DBP, with the share of unacceptable readings (|error| > 10 mmHg)."),
    ("Table_S_I_A_trust", "paper/tables/interpretability/A_trust.md", "Analysis A on Aurora-BP (full results)."),
    ("Table_S_I_A_trust_v1_sensitivity", "paper/tables/interpretability/v1_sensitivity/A_trust_v1.md",
     "Analysis A on Aurora-BP with the v1 (time-context) model: sensitivity analysis."),
    # ---------------- interpretability B
    ("Fig_I2_drift", "paper/figures/interp_B_drift.pdf",
     "Physiological drift score D = ||z_t − z_c|| (Aurora-BP test subjects). (a) MAE by decile of D and of time since calibration; "
     "(b) ROC curves for an unacceptable SBP reading: D, time since calibration, and both (validation-fitted); (c) zero-shot "
     "PulseDB: MAE by decile of D (MIMIC, VitalDB; SBP and DBP)."),
    ("Table_S_I_B_drift", "paper/tables/interpretability/B_drift.md", "Analysis B (full results, incl. recalibration rules)."),
    ("Table_S_I_B_drift_by_setting", "paper/tables/interpretability/B_drift_by_setting_EXPLORATORY.md",
     "Drift score by setting and condition (exploratory, descriptive)."),
    # ---------------- interpretability C1 / C2
    ("Fig_I3_explain", "paper/figures/interp_C_explain.pdf",
     "Why did the estimate change? (a) Physiological surrogate: mean standardised Lasso coefficients of feature changes (target − "
     "calibration windows) for the model's predicted SBP and DBP change (fitted on validation subjects); (b) share of the predicted "
     "change carried by the fused ECG+PPG comparison vs the PPG-only comparison (counterfactual occlusion), in-lab and at home; "
     "(c) MAE of readings whose predicted change is physiologically plausible vs flagged as implausible."),
    ("Table_S_I_C_explain", "paper/tables/interpretability/C_explain.md", "Analyses C1 and C2 (full results)."),
    ("Table_S_I_C_examples", "paper/tables/interpretability/C_examples.md", "Example per-reading explanations (pre-specified random selection)."),
    # ---------------- interpretability C3
    ("Fig_I4_saliency", "paper/figures/interp_C3_saliency.pdf",
     "Beat-level saliency (integrated gradients) of the predicted SBP and DBP change, averaged over normalised ECG (R peak to R peak) "
     "and PPG (foot to foot) cycles of 2,000 random test readings (grey: mean waveform) - shown because the faithfulness test was "
     "passed; otherwise the panel shows the faithfulness test (deletion of salient vs random samples)."),
    ("Table_S_I_C3_saliency", "paper/tables/interpretability/C3_saliency.md", "Analysis C3: faithfulness test of the saliency maps."),
    # ---------------- interpretability D
    ("Fig_I5_subgroups", "paper/figures/interp_D_subgroups.pdf",
     "Reliability profile by subgroup (Aurora-BP test subjects): MAE (95 % CI) of HiCAF-BP-Cal for (a) SBP and (b) DBP; filled "
     "markers: AAMI criterion met (|ME| <= 5 and SD <= 8 mmHg), open markers: not met; diamonds: RAGCBPNet (best published "
     "calibrated method); dashed: all readings."),
    ("Table_S_I_D_subgroups", "paper/tables/interpretability/D_subgroups.md", "Analysis D (full results by subgroup)."),
    # ---------------- interpretability summary
    ("Table_I1_decision_rules", "paper/tables/interpretability/Table_I1_decision_rules.md",
     "Interpretability decision rules: definition, validation-fixed threshold and test performance."),
    # ---------------- embeddings
    ("Fig_E1_embeddings_aurora", "paper/figures/emb_aurora.pdf",
     "Embedding z of HiCAF-BP-Cal (fold-0 model, Aurora-BP test recordings): PCA (top) and t-SNE (bottom), coloured by SBP, "
     "setting (in-lab vs home) and for the 8 subjects with most recordings (grey: all others)."),
    ("Fig_E2_embeddings_delta", "paper/figures/emb_delta.pdf",
     "Calibration-relative embedding dz = z_target − mean z of the calibration recordings (what the calibration head compares), "
     "PCA and t-SNE, coloured by the true SBP and DBP change from calibration."),
    ("Fig_E3_embeddings_domain", "paper/figures/emb_domain.pdf",
     "Embedding z of Aurora-BP and zero-shot PulseDB recordings (MIMIC, VitalDB; 100 random subjects each), PCA and t-SNE, "
     "coloured by dataset and by SBP."),
    # ---------------- computational profile
    ("Fig_C1_compute_tradeoff", "paper/figures/compute_tradeoff.pdf",
     "Accuracy versus computational cost per 10 s reading (frozen fold-0 models): Aurora-BP SBP MAE against (a) GFLOPs and (b) CPU "
     "latency with one thread, including handcrafted-feature extraction for feature-based methods (log scales). Calibrated "
     "methods (one cuff calibration reading): HiCAF-BP-Cal, HGCTNet, BpBLS, RAGCBPNet; calibration-free: HiCAF-BP uncalibrated, "
     "ResNet1D, rU-Net, UTransBPNet - the two groups solve tasks of different difficulty."),
    ("Table_C1_compute", "paper/tables/compute/COMPUTE.md",
     "Computational profile: parameters, size, GFLOPs, CPU (1 and 8 threads) and GPU latency, GPU throughput and memory, accuracy."),
    ("Table_E1_embeddings", "paper/tables/embeddings/EMBEDDINGS.md",
     "Quantitative embedding properties (5 fold models): subject purity, cross-subject k-NN R² of BP and of BP change, "
     "within-setting robustness, domain separability."),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def protect(p):
    os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--replace-all", action="store_true", help="re-finalise every available entry (logged reason required)")
    args = ap.parse_args()
    if (args.replace or args.replace_all) and not args.reason:
        sys.exit("--replace / --replace-all need --reason (it is logged in the manifest)")
    (FINAL / "figures" / "preview").mkdir(parents=True, exist_ok=True)
    (FINAL / "tables").mkdir(parents=True, exist_ok=True)
    log_fp = FINAL / "manifest.json"
    log = json.loads(log_fp.read_text()) if log_fp.exists() else {}
    for name, src, caption in REGISTRY:
        replace_this = args.replace_all or args.replace == name
        src = ROOT / src
        is_fig = src.suffix == ".pdf"
        dst = FINAL / ("figures" if is_fig else "tables") / f"{name}{src.suffix}"
        if name in log and not replace_this:
            if dst.exists() and sha(dst) != log[name]["sha256"]:
                sys.exit(f"{dst} differs from its manifest checksum - investigate")
            continue
        if not src.exists():
            print(f"  waiting: {name} (source {src.relative_to(ROOT)} not found)")
            continue
        if replace_this and dst.exists():
            os.chmod(dst, 0o644)
            for extra in (FINAL / "figures" / "preview" / f"{name}.png", FINAL / "figures" / f"{name}.json"):
                if extra.exists():
                    os.chmod(extra, 0o644)
        shutil.copy2(src, dst)
        protect(dst)
        entry = {"source": str(src.relative_to(ROOT)), "sha256": sha(dst), "finalised": time.strftime("%Y-%m-%d %H:%M"),
                 "caption": caption}
        if is_fig:
            for extra_src, extra_dst in [(src.parent / "preview" / f"{src.stem}.png", FINAL / "figures" / "preview" / f"{name}.png"),
                                         (src.with_suffix(".json"), FINAL / "figures" / f"{name}.json")]:
                if extra_src.exists():
                    shutil.copy2(extra_src, extra_dst)
                    protect(extra_dst)
            side = src.with_suffix(".json")
            if side.exists():
                entry["made_by"] = json.loads(side.read_text()).get("source")
                entry["font"] = json.loads(side.read_text()).get("font")
        if replace_this and name in log:
            entry["replaced"] = {"previous_sha256": log.get(name, {}).get("sha256"), "reason": args.reason}
        log[name] = entry
        print(f"  FINAL: {name}  <- {entry['source']}  ({entry['sha256']})")
    log_fp.write_text(json.dumps(log, indent=2))
    L = ["# Final paper outputs (write-protected; replacement only with a logged reason)", "",
         "| Name | Source | Made by | SHA-256 (16) | Finalised | Caption / description |", "|---|---|---|---|---|---|"]
    for name, _, _ in REGISTRY:
        if name in log:
            e = log[name]
            rep = f" (replaced: {e['replaced']['reason']})" if "replaced" in e else ""
            L.append(f"| {name} | `{e['source']}` | `{e.get('made_by', '')}` | {e['sha256']} | {e['finalised']}{rep} | {e['caption']} |")
    (FINAL / "MANIFEST.md").write_text("\n".join(L))
    print(f"\n{sum(n in log for n, _, _ in REGISTRY)}/{len(REGISTRY)} registry entries final -> {FINAL / 'MANIFEST.md'}")


if __name__ == "__main__":
    main()
