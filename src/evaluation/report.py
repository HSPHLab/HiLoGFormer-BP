"""Readable result tables (markdown) and paper figures."""
import json

import numpy as np

HDR = ["| Set | N | Subj | SBP MAE | SBP ME ± SD | SBP r | SBP BHS / IEEE | DBP MAE | DBP ME ± SD | DBP r | DBP BHS / IEEE |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]


def row(name, m):
    s, d = m["sbp"], m["dbp"]
    return (f"| {name} | {m['n']:,} | {m['subjects']:,} | {s['MAE']:.2f} | {s['ME']:+.2f} ± {s['SDE']:.2f} | "
            f"{s['r']:.2f} | {s['BHS_grade']} / {s['IEEE1708_grade']} | {d['MAE']:.2f} | "
            f"{d['ME']:+.2f} ± {d['SDE']:.2f} | {d['r']:.2f} | {d['BHS_grade']} / {d['IEEE1708_grade']} |")


def regression_section(res, title):
    L = [f"## {title}", ""] + HDR + [row("**Model**", res["overall"])]
    for key, name in [("baseline_training_mean", "Baseline: training mean"),
                      ("baseline_calibration_only", "Baseline: calibration value only"),
                      ("baseline_uncalibrated_shift", "Baseline: uncalibrated model + shift"),
                      ("reference_subject_mean_oracle", "Reference: subject-mean oracle")]:
        if key in res:
            L.append(row(name, res[key]))
    for k, v in res.items():
        if k.startswith("by_"):
            for g, m in v.items():
                L.append(row(f"{k[3:]} = {g}", m))
    o = res["overall"]
    L += ["", "| Target | within 5 / 10 / 15 mmHg | 95% LoA | AAMI (\\|ME\\|≤5, SD≤8) | within-subject r, median [IQR] (n subj) |",
          "|---|---|---|---|---|"]
    for t in ["sbp", "dbp"]:
        m = o[t]
        L.append(f"| {t.upper()} | {m['within5']:.1%} / {m['within10']:.1%} / {m['within15']:.1%} | "
                 f"{m['LoA_low']:+.1f} to {m['LoA_high']:+.1f} | {'pass' if m['AAMI_pass'] else 'fail'} | "
                 f"{m['within_subject_r_median']:.2f} [{m['within_subject_r_q1']:.2f}, "
                 f"{m['within_subject_r_q3']:.2f}] ({m['within_subject_n']}) |")
    return L


def classification_section(cls):
    L = ["## Post-hoc classification (thresholds applied to predicted SBP/DBP)", "",
         "| Scheme | Acc | Bal. acc | Macro-F1 | Kappa (quad.) | AUROC | AUPRC |", "|---|---|---|---|---|---|---|"]
    for name, r in cls.items():
        L.append(f"| {name} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['macro_f1']:.3f} | "
                 f"{r['kappa']:.3f} | {r.get('auroc', float('nan')):.3f} | {r.get('auprc', float('nan')):.3f} |")
    for name, r in cls.items():
        L += ["", f"**{name}** per class (sensitivity / specificity / precision / F1, support):", ""]
        for lab, c in r["per_class"].items():
            L.append(f"- {lab}: {c['sensitivity']:.3f} / {c['specificity']:.3f} / {c['precision']:.3f} / "
                     f"{c['f1']:.3f} ({c['support']})")
        L.append(f"- confusion matrix (rows = true, cols = predicted, order {r['labels']}): `{r['confusion_matrix']}`")
    return L


def run_markdown(name, info, res_val, res_test=None):
    L = [f"# {name}", "", "```", json.dumps(info, indent=1, default=str), "```", ""]
    if res_test is not None:
        L += regression_section(res_test, "Test") + [""] + classification_section(res_test["classification"]) + [""]
    L += regression_section(res_val, "Validation")
    return "\n".join(L)


def paper_figures(pred, out_dir, title=""):
    """Bland-Altman, error histogram and predicted-vs-true scatter for SBP and DBP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in ["sbp", "dbp"]:
        y, p = pred[t].to_numpy(float), pred[f"{t}_pred"].to_numpy(float)
        e = p - y; me, sd = e.mean(), e.std(ddof=1)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ax[0].scatter((y + p) / 2, e, s=3, alpha=0.25)
        for v, ls in [(me, "-"), (me + 1.96 * sd, "--"), (me - 1.96 * sd, "--")]:
            ax[0].axhline(v, color="tab:red", ls=ls, lw=1)
        ax[0].set_xlabel(f"mean of reference and predicted {t.upper()} (mmHg)"); ax[0].set_ylabel("predicted - reference (mmHg)")
        ax[0].set_title(f"Bland-Altman: bias {me:+.2f}, LoA {me - 1.96 * sd:+.1f} to {me + 1.96 * sd:+.1f}")
        ax[1].hist(e, bins=80, alpha=0.8)
        for k in (5, 10, 15):
            ax[1].axvline(k, color="grey", ls=":", lw=1); ax[1].axvline(-k, color="grey", ls=":", lw=1)
        ax[1].set_xlabel("error (mmHg)"); ax[1].set_title("Error distribution (±5/10/15 mmHg)")
        lo, hi = np.percentile(np.r_[y, p], [0.5, 99.5])
        ax[2].scatter(y, p, s=3, alpha=0.25); ax[2].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[2].set_xlim(lo, hi); ax[2].set_ylim(lo, hi)
        ax[2].set_xlabel(f"reference {t.upper()}"); ax[2].set_ylabel(f"predicted {t.upper()}")
        ax[2].set_title(f"r = {np.corrcoef(y, p)[0, 1]:.2f}")
        for a in ax:
            a.grid(alpha=0.3)
        fig.suptitle(f"{title} — {t.upper()}")
        plt.tight_layout(); plt.savefig(out_dir / f"{t}_bland_altman_hist_scatter.png", dpi=130); plt.close()


def write_md_table(df, path):
    """Plain markdown table (no extra dependencies)."""
    from pathlib import Path
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.astype(str).values]
    Path(path).write_text("\n".join(lines) + "\n")
