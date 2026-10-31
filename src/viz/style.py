"""
Common style for ALL paper figures (docs/FIGURE_STYLE.md). Every figure script must use this module.

    from src.viz.style import setup, figure, save, panel_labels, METHOD, TARGET, DATASET, SETTING
    setup()                                            # once, before plotting
    fig, axes = figure("double", height=2.6, ncols=2)   # IEEE double column (7.16 in)
    ...
    panel_labels(axes)                                 # (a), (b), ...
    save(fig, "interp_A_risk_coverage", source=__file__)

Rules (fixed for the whole paper):
  - Font: Times New Roman. If it is not installed, a TrueType copy placed in `paper/fonts/` is used; otherwise
    Liberation Serif (metric-compatible with Times New Roman), then STIX. The font actually used is printed and stored.
  - Sizes: 8 pt text, 7 pt ticks and legends, 9 pt bold panel labels; IEEE widths single 3.5 in / double 7.16 in.
  - Output: vector PDF (fonts embedded as TrueType, Type 42), raster parts at 600 dpi, tight bounding box;
    a 300 dpi PNG preview in paper/figures/preview/; a JSON sidecar with the source script and font.
  - Colours: Okabe-Ito colour-blind-safe palette; each method / target / dataset keeps ONE colour across all figures.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib import font_manager                                        # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"
FONT_DIR = ROOT / "paper" / "fonts"

WIDTH = {"single": 3.5, "double": 7.16, "onehalf": 5.3}                    # inches (IEEE JBHI)
DPI = 600

# Okabe-Ito colour-blind-safe palette
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}

# one colour / marker / line per method, used in every figure
METHOD = {
    "ours":        {"label": "HiCAF-BP-Cal (ours)",           "color": OI["blue"],       "marker": "o", "ls": "-"},
    "ours_notime": {"label": "HiCAF-BP-Cal, no time (ours)",  "color": "#5DA5DA",        "marker": "o", "ls": "--"},
    "ours_time":   {"label": "HiCAF-BP-Cal + time context", "color": "#5DA5DA", "marker": "o", "ls": "--"},
    "ours_uncal":  {"label": "HiCAF-BP, uncalibrated (ours)", "color": "#1F3F73",        "marker": "s", "ls": "-."},
    "hgctnet":     {"label": "HGCTNet",                       "color": OI["orange"],     "marker": "^", "ls": "-"},
    "bpbls":       {"label": "BpBLS",                         "color": OI["green"],      "marker": "v", "ls": "-"},
    "ragcbpnet":   {"label": "RAGCBPNet",                     "color": OI["purple"],     "marker": "D", "ls": "-"},
    "runet":       {"label": "rU-Net",                        "color": OI["vermillion"], "marker": "P", "ls": "-"},
    "utransbpnet": {"label": "UTransBPNet",                   "color": OI["sky"],        "marker": "X", "ls": "-"},
    "resnet1d":    {"label": "ResNet1D",                      "color": OI["grey"],       "marker": "s", "ls": ":"},
    "cal_only":    {"label": "Calibration value only",        "color": OI["black"],      "marker": "",  "ls": "--"},
}
TARGET = {"sbp": {"label": "SBP", "color": "#B2182B"}, "dbp": {"label": "DBP", "color": "#2166AC"}}
DATASET = {"aurora": {"label": "Aurora-BP", "color": OI["blue"]}, "mimic": {"label": "MIMIC", "color": OI["vermillion"]},
           "vitaldb": {"label": "VitalDB", "color": OI["green"]}}
SETTING = {"lab": {"label": "In-lab", "color": "#4D4D4D"}, "home": {"label": "Ambulatory (home)", "color": OI["orange"]}}
NEUTRAL = {"fill": "#D9D9D9", "line": "#4D4D4D", "highlight": OI["vermillion"]}
TRAFFIC = {"green": "#4DAF4A", "amber": "#FFB000", "red": "#D62728"}         # trust indicator (interpretability A)

_FONT = {"name": None}


def _resolve_font():
    """Times New Roman if available (system or paper/fonts/*.ttf), else Liberation Serif, else STIX."""
    for f in sorted(FONT_DIR.glob("*.tt[fc]")) if FONT_DIR.exists() else []:
        font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Times New Roman", "Liberation Serif", "STIXGeneral"):
        if cand in names:
            return cand
    return "serif"


def setup():
    """Apply the paper style (call once per script)."""
    name = _resolve_font()
    _FONT["name"] = name
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [name, "Times New Roman", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.8,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.4, "grid.alpha": 0.35, "grid.color": "#B0B0B0",
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "errorbar.capsize": 2,
        "savefig.dpi": DPI, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,                              # embed TrueType fonts (IEEE PDF eXpress)
    })
    if name != "Times New Roman":
        print(f"[style] Times New Roman not found - using '{name}' (put a times.ttf into {FONT_DIR} to use Times New Roman)")
    return name


def figure(width="single", height=2.4, nrows=1, ncols=1, **kw):
    """New figure at IEEE width ('single', 'double', 'onehalf' or inches)."""
    w = WIDTH.get(width, width) if isinstance(width, str) else float(width)
    kw.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), **kw)
    return fig, axes


def panel_labels(axes, x=-0.16, y=1.04, start="a"):
    """Bold (a), (b), ... in the top-left corner of each axis."""
    import numpy as np
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.text(x, y, f"({chr(ord(start) + i)})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name, source=None, outdir=FIG_DIR, preview=True, png_subdir="preview"):
    """Save `name`.pdf (vector, 600 dpi for raster parts, embedded fonts) + preview PNG + provenance sidecar."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    fig.savefig(pdf, format="pdf", dpi=DPI)
    if preview:
        (outdir / png_subdir).mkdir(exist_ok=True)
        fig.savefig(outdir / png_subdir / f"{name}.png", dpi=300)
    src = None
    if source:
        try:
            src = str(Path(source).resolve().relative_to(ROOT))
        except ValueError:
            src = str(source)
    side = {"figure": pdf.name, "source": src,
            "font": _FONT["name"], "dpi": DPI, "size_in": [round(v, 2) for v in fig.get_size_inches()],
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (outdir / f"{name}.json").write_text(json.dumps(side, indent=2))
    plt.close(fig)
    print(f"[style] saved {pdf}")
    return pdf
