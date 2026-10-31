"""
Experiment registry, REBUILT from the run folders instead of appended to.

Every finished run has runs/<experiment>/fold_<k>/metrics.json + config.yaml. The registry
(runs/EXPERIMENTS.csv and .md) is regenerated from those files, under a file lock and with an
atomic file replace, so runs finishing at the same time can never corrupt or lose lines.
Folders starting with "_" (e.g. runs/_failed) are ignored.

Rebuild by hand at any time:   python -m src.utils.registry
"""
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.report import write_md_table


def _row(run_dir):
    m = json.load(open(run_dir / "metrics.json"))
    if "test" not in m or "val" not in m:
        return None
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    info, o, v, c = m["info"], m["test"]["overall"], m["val"]["overall"], m["test"]["classification"]
    exp = info["experiment"]
    return {
        "finished": datetime.fromtimestamp((run_dir / "metrics.json").stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "experiment": exp, "fold": info["fold"], "model": info["model"]["name"],
        "inputs": "+".join(info["data"]["inputs"]), "dataset": info["data"]["dataset"],
        "params_M": round(info["n_params"] / 1e6, 2), "best_epoch": info["best_epoch"],
        "VAL_score": round((v["sbp"]["MAE"] + v["dbp"]["MAE"]) / 2, 3),
        "VAL_sbp_MAE": round(v["sbp"]["MAE"], 2), "VAL_dbp_MAE": round(v["dbp"]["MAE"], 2),
        "sbp_MAE": round(o["sbp"]["MAE"], 2), "dbp_MAE": round(o["dbp"]["MAE"], 2),
        "sbp_ME_SD": f"{o['sbp']['ME']:+.1f}±{o['sbp']['SDE']:.1f}",
        "dbp_ME_SD": f"{o['dbp']['ME']:+.1f}±{o['dbp']['SDE']:.1f}",
        "sbp_r": round(o["sbp"]["r"], 3), "sbp_within_subj_r": round(o["sbp"]["within_subject_r_median"], 3),
        "binary_acc": round(c["binary_130_80"]["accuracy"], 3),
        "binary_auroc": round(c["binary_130_80"].get("auroc", float("nan")), 3),
        "3class_acc": round(c["3class"]["accuracy"], 3),
        "overrides": " ".join(cfg.get("experiment", {}).get("overrides") or []),
        "git": (info.get("git") or "")[:8], "smoke": exp.endswith("_smoke"),
    }


def rebuild_registry(runs_dir):
    runs_dir = Path(runs_dir)
    rows = []
    for mf in sorted(runs_dir.glob("*/fold_*/metrics.json")):
        if mf.parts[-3].startswith("_"):
            continue
        try:
            r = _row(mf.parent)
        except Exception as e:                      # a broken run folder must not block the registry
            print(f"[registry] skipped {mf.parent}: {type(e).__name__}: {e}")
            r = None
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["finished", "experiment", "fold"]).reset_index(drop=True)
    with open(runs_dir / ".registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for name, writer in [("EXPERIMENTS.csv", lambda p: df.to_csv(p, index=False)),
                                 ("EXPERIMENTS.md", lambda p: write_md_table(df, p))]:
                tmp = runs_dir / f".{name}.tmp"
                writer(tmp)
                os.replace(tmp, runs_dir / name)    # atomic: readers never see a half-written file
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.config import load_config
    d = rebuild_registry(load_config("base.yaml")["paths"]["runs_dir"])
    print(f"registry rebuilt: {len(d)} runs")