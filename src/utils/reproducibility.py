"""
Everything needed to trace a result back to its exact code, data, config and environment.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    except ImportError:
        pass


def file_sha256(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def git_state(repo_dir, out_dir=None):
    """Commit hash + whether there are uncommitted changes; saves the diff next to the run."""
    def run(cmd):
        return subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=20)
    try:
        commit = run(["git", "rev-parse", "HEAD"]).stdout.strip() or None
        diff = run(["git", "diff", "HEAD"]).stdout
        status = run(["git", "status", "--porcelain"]).stdout
    except Exception:
        return {"commit": None, "note": "git not available"}
    if commit is None:
        return {"commit": None, "note": "not a git repository (run: git init && git add -A && git commit)"}
    dirty = bool(status.strip())
    if dirty and out_dir is not None:
        Path(out_dir, "uncommitted.diff").write_text(diff)
    return {"commit": commit, "dirty": dirty}


def environment_info():
    info = {"time": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
            "python": platform.python_version(), "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:
        pass
    try:
        import pandas, scipy, sklearn
        info.update({"pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__})
    except ImportError:
        pass
    return info


def provenance(cfg, code_dir, out_dir):
    """Collect and save provenance.json in the run folder."""
    data_dir = Path(cfg["paths"]["data_dir"])
    prov = {"git": git_state(code_dir, out_dir),
            "data_config_sha256": file_sha256(data_dir / "config.json"),
            "metadata_sha256": file_sha256(data_dir / "metadata.csv"),
            "folds_sha256": file_sha256(cfg["paths"]["folds_file"]),
            "environment": environment_info()}
    json.dump(prov, open(Path(out_dir) / "provenance.json", "w"), indent=2)
    return prov
