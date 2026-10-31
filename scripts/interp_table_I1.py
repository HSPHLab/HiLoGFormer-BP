#!/usr/bin/env python
"""
Table I1 - all interpretability decision rules (docs/INTERPRETABILITY_PLAN.md, section 7). Assembly only (no new estimates).

  python scripts/interp_table_I1.py

Reads paper/tables/interpretability/{A_trust, B_drift, C_explain, D_subgroups, C3_saliency}.json (no-time model, Aurora-BP
test subjects; every threshold fixed on validation subjects). For each rule: flag rate, MAE of flagged vs not-flagged
readings, share of unacceptable readings (|error| > 10 mmHg) among flagged vs not flagged, share of all unacceptable readings
captured, and the relative risk of an unacceptable reading (flagged vs not flagged).
Output: paper/tables/interpretability/Table_I1_decision_rules.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "paper/tables/interpretability"
T = ["sbp", "dbp"]


def j(name):
    return json.loads((TAB / f"{name}.json").read_text())


def row(question, rule, threshold, fr, mae_f, mae_n, u_f, u_n, base):
    cap = u_f * fr / base if base > 0 else float("nan")
    rr = u_f / u_n if u_n > 0 else float("nan")
    return (f"| {question} | {rule} | {threshold} | {fr:.0%} | {mae_f:.2f} / {mae_n:.2f} | {u_f:.0%} / {u_n:.0%} | {cap:.0%} | {rr:.2f} |")


def main():
    A, B, C, D = j("A_trust"), j("B_drift"), j("C_explain"), j("D_subgroups")
    C3 = j("C3_saliency") if (TAB / "C3_saliency.json").exists() else None
    L = ["# Table I1 — Interpretability decision rules (Aurora-BP test subjects; HiCAF-BP-Cal without time context)", "",
         "All thresholds fixed on validation subjects of each fold. Unacceptable reading: |error| > 10 mmHg. RR = relative risk of an "
         "unacceptable reading, flagged vs not flagged.", ""]
    for t in T:
        base = 1 - D[f"{t}/all"]["w10"]
        mae_all = D[f"{t}/all"]["mae"]
        L += [f"## {t.upper()} (all readings: MAE {mae_all:.2f} mmHg, unacceptable {base:.0%})", "",
              "| Question | Rule | Threshold | Flagged | MAE flagged / not (mmHg) | Unacceptable flagged / not | Unacceptable captured | RR |",
              "|---|---|---|---|---|---|---|---|"]
        a = A[t]["cov80"]
        fr = 1 - a["achieved"]
        base_a = a["unacc_flagged"] * fr + a["unacc_retained"] * (1 - fr)          # A: readings with 3 calibration readings only
        L.append(row("A. Can I trust this reading?", "red trust level → confirm with cuff (consistency of the 3 single-calibration estimates; "
                     "readings with 3 calibration readings)", "validation 80th percentile", fr, a["mae_flagged"], a["mae_retained"],
                     a["unacc_flagged"], a["unacc_retained"], base_a))
        for lab, name in [("recalibrate if D > T", "B. When to recalibrate?"), ("time rule, same flag rate", "B. (comparator)"),
                          ("recalibrate if D > T or time > N", "B. (combined rule)")]:
            b = B[f"{t}/rule/{lab}"]
            fr = b["flag_rate"]
            u_f = b["ppv"]
            u_n = (base - u_f * fr) / (1 - fr)
            thr = {"recalibrate if D > T": "T = validation 80th percentile of D", "time rule, same flag rate": "longest since calibration, same flag rate",
                   "recalibrate if D > T or time > N": "T as above; N = validation 80th percentile of time"}[lab]
            L.append(row(name, lab.replace("D", "drift score D", 1) if lab.startswith("recal") else lab, thr, fr,
                         b["mae_flagged"], b["mae_not_flagged"], u_f, u_n, base))
        c = C[t]
        L.append(row("C. Why did the estimate change?", "implausible change → confirm with cuff (large difference between the predicted change and the physiological surrogate)",
                     "validation 90th percentile of the residual", c["flag_rate"], c["mae_flagged"], c["mae_not_flagged"],
                     c["unacc_flagged"], c["unacc_not_flagged"], base))
        caution = [k.split("/", 1)[1].replace("/", ": ") for k, v in D.items() if k.startswith(f"{t}/") and k != f"{t}/all" and not v["aami"]]
        L += ["", f"**D. For which patients does it work?** AAMI criterion (|ME| ≤ 5, SD ≤ 8 mmHg) for all readings: "
              f"{'met' if D[f'{t}/all']['aami'] else 'not met'}. Use with caution (criterion not met): "
              + ("; ".join(caution) if caution else "none") + ".", ""]
    if C3 is not None:
        L += [f"**C3 (beat-level saliency):** faithfulness criterion {'passed - saliency maps reported (Fig. I4)' if C3['faithful'] else 'not passed - saliency maps not reported'} "
              f"({C3['n']} readings).", ""]
    (TAB / "Table_I1_decision_rules.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
