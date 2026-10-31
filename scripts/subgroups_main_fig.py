#!/usr/bin/env python
"""
Single-column subgroup figure for the main paper (plotting only; reads the frozen table, no inference).

  python scripts/subgroups_main_fig.py [--out ~/paper_figures/subgroups_main.pdf]

Input : paper/tables/interpretability/D_subgroups.md (written by scripts/interp_D_subgroups.py)
Output: subgroups_main.pdf / .png (default ~/paper_figures, because paper/figures is write-protected)
(a) SBP, (b) DBP: MAE of HiCAF-BP-Cal (95 % CI; filled = AAMI criterion met) and of RAGCBPNet per subgroup.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]


def read_table(path):
    data, tgt = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            tgt = line[3:].strip()
            data[tgt] = []
        elif tgt and line.startswith("| ") and not line.startswith("| Subgroup") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            m = re.match(r"([\d.]+) \(([\d.]+)–([\d.]+)\)", c[3])
            data[tgt].append(dict(group=c[0], level=c[1], mae=float(m.group(1)), lo=float(m.group(2)),
                                  hi=float(m.group(3)), aami=(c[6] == "yes"), rag=float(c[7])))
    return data


def label(g, lv):
    if g == "all":
        return "All readings"
    if g == "age":
        return "Age " + lv
    if g == "sex":
        return lv.capitalize()
    if g == "BMI":
        return "BMI " + lv
    if g.startswith("hypertension"):
        return {"normotensive": "Normotensive", "managed": "Hypertension, managed",
                "unmanaged": "Hypertension, unmanaged"}[lv]
    if g == "setting":
        return {"in-lab, initial": "In-lab, initial visit", "in-lab, return": "In-lab, return visit",
                "ambulatory (home)": "Home"}[lv]
    if g.startswith("pulse-rate"):
        return "|\u0394PR| " + lv
    if g.startswith("signal"):
        return "Signal quality: " + lv
    if g.startswith("true"):
        return "|\u0394BP| " + lv
    return f"{g} {lv}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(CODE_DIR / "paper/tables/interpretability/D_subgroups.md"))
    ap.add_argument("--out", default=str(Path.home() / "paper_figures" / "subgroups_main.pdf"))
    a = ap.parse_args()
    data = read_table(a.table)
    rows = data["SBP"]
    labels = [label(r["group"], r["level"]) for r in rows]
    assert labels == [label(r["group"], r["level"]) for r in data["DBP"]], "SBP and DBP rows differ"

    font = next((f for f in ("Times New Roman", "Liberation Serif", "STIXGeneral")
                 if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}), "serif")
    plt.rcParams.update({"font.family": "serif", "font.serif": [font], "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    y, pos, prev = [], 0.0, None
    for r in rows:                                                     # small gap between subgroup families
        if prev is not None and r["group"] != prev:
            pos += 0.6
        y.append(pos)
        pos += 1
        prev = r["group"]
    y = -np.array(y)

    blue, purple = "#0072B2", "#CC79A7"
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 4.05), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, t, xl in zip(axes, ["SBP", "DBP"], [(4.5, 23), (3.5, 17)]):
        R = data[t]
        ax.axvline(R[0]["mae"], color="#606060", lw=0.6, ls="--", zorder=1)   # MAE over all readings
        for yy, r in zip(y, R):
            ax.plot([r["lo"], r["hi"]], [yy, yy], color=blue, lw=0.9, zorder=2)
            ax.plot(r["mae"], yy, "o", ms=3.4, mfc=blue if r["aami"] else "white", mec=blue, mew=0.9, zorder=3)
            ax.plot(r["rag"], yy, "D", ms=2.8, color=purple, zorder=2)
        ax.set_xlim(*xl)
        ax.set_xlabel(f"{t} MAE (mmHg)")
        ax.grid(axis="x", lw=0.3, alpha=0.4)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(0.0, 1.01, lab, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    h = [plt.Line2D([], [], marker="o", ls="", ms=3.4, mfc=blue, mec=blue, label="HiCAF-BP-Cal: AAMI criterion met"),
         plt.Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=blue, label="not met"),
         plt.Line2D([], [], marker="D", ls="", ms=2.8, color=purple, label="RAGCBPNet")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6, handletextpad=0.2,
               columnspacing=0.8, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.30, right=0.98, top=0.965, bottom=0.135)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"saved {out} ({len(rows)} subgroups; font {font})")


if __name__ == "__main__":
    main()
