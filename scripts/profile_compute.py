#!/usr/bin/env python
"""
Computational profile of HiCAF-BP-Cal and every comparison method (docs/COMPUTE_PLAN.md). Frozen fold-0 checkpoints.

  python scripts/profile_compute.py --gpu 2

Per method: parameters, fp32 size, FLOPs per reading (PyTorch profiler; matmul/conv operations), latency for one reading
(batch 1) on CPU with 1 thread (proxy for a low-power processor) and with N threads, and on the GPU; GPU throughput and peak
memory at batch 256. Our calibrated model is profiled as deployed: the calibration recording is encoded ONCE and cached, so
each new reading costs one encoder pass + the head ("per reading"); the cost with re-encoding the calibration recording is
also reported. Feature-based methods (HGCTNet, BpBLS, RAGCBPNet) additionally need the 31 handcrafted features per reading;
their extraction time (CPU, 1 thread, real 10 s windows) is measured separately and added for the end-to-end CPU latency.
Timing: inference mode, warm-up, then repeated runs; median and 95th percentile; GPU timings with CUDA synchronisation.
Outputs: paper/tables/compute/COMPUTE.md (+ .json), paper/figures/compute_tradeoff.pdf
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
FS, L = 125, 1250
# Aurora-BP 5-fold CV, k = 3 for calibrated methods (uncalibrated: no calibration), pooled test MAE SBP / DBP (frozen reports: paper/tables/calibrated, zero_shot_calfree, Aurora comparisons)
MAE = {"ours": (8.53, 6.19), "ours_uncal": (12.66, 8.91), "resnet1d": (12.91, 9.17), "hgctnet": (9.98, 7.10), "bpbls": (10.86, 7.95),
       "ragcbpnet": (9.83, 6.92), "runet": (13.34, 9.43), "utransbpnet": (13.73, 9.56)}
METHODS = [  # key, run, kind, uses handcrafted features
    ("ours", "runs/hicaf_bp_cal_notime", "ours_cal", False),
    ("ours_uncal", "runs/hicaf_bp_main", "ours_uncal", False),
    ("resnet1d", "runs/resnet1d_ecgppg", "ours_uncal", False),
    ("hgctnet", "runs/pub_hgctnet", "published", True),
    ("bpbls", "runs/pub_bpbls", "published", True),
    ("ragcbpnet", "runs/pub_ragcbpnet", "published", True),
    ("runet", "runs/pub_runet", "published", False),
    ("utransbpnet", "runs/pub_utransbpnet", "published", False),
]


def bench(fn, sync, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000.0
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--flops_only", action="store_true", help="recompute FLOPs; keep the saved timings (paper/tables/compute/COMPUTE.json)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from src.models import build_model
    torch.manual_seed(2026)
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    hw = {"cpu": platform.processor() or "", "cpu_logical_cores": os.cpu_count(), "torch": torch.__version__,
          "gpu": torch.cuda.get_device_name(0) if gpu else "none", "threads_multi": args.threads,
          "load_average_1_5_15min": [round(v, 1) for v in os.getloadavg()]}                  # shared server: record the load
    try:
        hw["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    print(json.dumps(hw), flush=True)

    def load(run, kind):
        ck = torch.load(CODE_DIR / run / "fold_0" / "best.pt", map_location="cpu")
        if kind == "published":
            from src.published.hgctnet import HGCTNet
            from src.published.bpbls import BpBLS
            from src.published.runet import rUNetBP
            from src.published.utransbpnet import UTransBPNet
            from src.published.ragcbpnet import RAGCBPNet
            b = {"hgctnet": HGCTNet, "bpbls": BpBLS, "runet": rUNetBP, "utransbpnet": UTransBPNet, "ragcbpnet": RAGCBPNet}
            m = b[ck["method"]](**ck["model_args"])
        else:
            cfg = ck["config"]
            m = build_model(cfg["model"], input_names=list(cfg["data"]["inputs"]), n_targets=2, n_aux=0)
        m.load_state_dict(ck["model_state"], strict=True)
        return m.eval(), ck

    def make_call(key, kind, m, dev, B):
        """Return fn() running one inference of batch B on dev, exactly as the model is used."""
        x = torch.randn(B, 2, L, device=dev)
        if kind == "ours_cal":
            xc = torch.randn(B, 2, L, device=dev)
            bp = torch.zeros(B, 2, device=dev)
            ctx = torch.zeros(B, m.ctx_dim, device=dev) if m.ctx_embed is not None else None
            with torch.inference_mode():
                zc, pc = m.encoder.encode(xc)                             # calibration embedding: computed once, cached

            def per_reading():
                z, zp = m.encoder.encode(x)
                feats = m._compare(z, zc)
                if m.use_ppg:
                    feats += m._compare(zp, pc)
                feats.append(m.bp_embed(bp))
                if m.ctx_embed is not None:
                    feats.append(m.ctx_embed(ctx))
                return m.head(torch.cat(feats, dim=-1))

            def with_cal():
                return m(x, x_cal=xc, bp_cal=bp, ctx=ctx)
            return per_reading, with_cal
        if kind == "ours_uncal":
            return (lambda: m(x)), None
        # published: side-input width found by probing the smallest width the model accepts
        for d in range(0, 200):
            try:
                with torch.inference_mode():
                    m(x[:1], torch.zeros(1, d, device=dev))
                break
            except Exception:
                continue
        f = torch.randn(B, d, device=dev)
        return (lambda: m(x, f)), None

    from torch.utils._python_dispatch import TorchDispatchMode
    aten = torch.ops.aten

    class FlopCounter(TorchDispatchMode):
        """FLOPs (2 x multiply-adds) of matrix products and convolutions (1-D and 2-D), counted at the operator level."""
        def __init__(self):
            super().__init__()
            self.flops = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            f = func.overloadpacket
            if f is aten.mm:
                a, b = args[0], args[1]
                self.flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
            elif f is aten.addmm:
                a, b = args[1], args[2]
                self.flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
            elif f is aten.bmm:
                a, b = args[0], args[1]
                self.flops += 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
            elif f is aten.baddbmm:
                a, b = args[1], args[2]
                self.flops += 2 * a.shape[0] * a.shape[1] * a.shape[2] * b.shape[2]
            elif f in (aten.convolution, aten._convolution):
                w = args[1]
                self.flops += 2 * out.numel() * int(np.prod(w.shape[1:]))      # out elements x (C_in/groups x kernel)
            return out

    def flops(fn):
        with torch.no_grad(), FlopCounter() as fc:
            fn()
        return float(fc.flops)

    if args.flops_only:
        saved = json.load(open(CODE_DIR / "paper/tables/compute/COMPUTE.json"))
        rows, hw, feat_ms = saved["rows"], saved["hardware"], saved["feature_ms"]
        for r in rows:
            run, kind = {k: (ru, ki) for k, ru, ki, _ in METHODS}[r["method"]]
            m, _ = load(run, kind)
            fn1, fn1_cal = make_call(r["method"], kind, m, torch.device("cpu"), 1)
            r["gflops"] = flops(fn1) / 1e9
            if fn1_cal:
                r["gflops_with_cal"] = flops(fn1_cal) / 1e9
            print(f"{r['method']:12s} {r['gflops']:.3f} GFLOPs", flush=True)
        return report(rows, hw, feat_ms, hw["threads_multi"])

    rows = []
    for key, run, kind, feat in METHODS:
        if not (CODE_DIR / run / "fold_0" / "best.pt").exists():
            print(f"skip {key}: {run}/fold_0/best.pt not found", flush=True)
            continue
        m, ck = load(run, kind)
        n_par = sum(p.numel() for p in m.parameters())
        r = {"method": key, "params": n_par, "size_mb": n_par * 4 / 2 ** 20}
        cpu = torch.device("cpu")
        fn1, fn1_cal = make_call(key, kind, m, cpu, 1)
        r["gflops"] = flops(fn1) / 1e9
        if fn1_cal:
            r["gflops_with_cal"] = flops(fn1_cal) / 1e9
        for th, tag in [(1, "cpu1"), (args.threads, "cpuN")]:
            torch.set_num_threads(th)
            with torch.inference_mode():
                r[f"{tag}_ms"], r[f"{tag}_p95"] = bench(fn1, lambda: None, 5, 30 if th == 1 else 100)
                if fn1_cal:
                    r[f"{tag}_ms_with_cal"], _ = bench(fn1_cal, lambda: None, 5, 30 if th == 1 else 100)
        torch.set_num_threads(args.threads)
        if gpu:
            mg = m.to(gpu)
            sync = torch.cuda.synchronize
            g1, g1c = make_call(key, kind, mg, gpu, 1)
            with torch.inference_mode():
                r["gpu_ms"], r["gpu_p95"] = bench(g1, sync, 20, 200)
                if g1c:
                    r["gpu_ms_with_cal"], _ = bench(g1c, sync, 20, 200)
                torch.cuda.reset_peak_memory_stats()
                gB, _ = make_call(key, kind, mg, gpu, 256)
                ms, _ = bench(gB, sync, 5, 30)
                r["gpu_throughput_per_s"] = 256 / (ms / 1000.0)
                r["gpu_peak_mem_mb_b256"] = torch.cuda.max_memory_allocated() / 2 ** 20
            m.to("cpu")
            torch.cuda.empty_cache()
        r["uses_features"] = feat
        rows.append(r)
        print(f"{key:12s} {n_par / 1e6:7.2f} M | {r['gflops']:7.3f} GFLOPs | CPU1 {r['cpu1_ms']:8.2f} ms | CPU{args.threads} {r['cpuN_ms']:7.2f} ms"
              + (f" | GPU {r['gpu_ms']:6.2f} ms, {r['gpu_throughput_per_s']:9.0f}/s" if gpu else ""), flush=True)

    # handcrafted features (31 ECG/PPG features) on real Aurora windows, CPU, 1 thread
    import pandas as pd
    from src.features.pwa_features import extract_features
    meta = pd.read_csv(CODE_DIR / "data_processed/v1/metadata.csv", low_memory=False)
    meta = meta[meta.include_main.astype(bool)].sample(200, random_state=2026)
    ch = json.load(open(CODE_DIR / "data_processed/v1/config.json")).get("channels", ["ecg", "ppg"])
    sig = np.load(CODE_DIR / "data_processed/v1/signals.npy", mmap_mode="r")
    wins = [np.asarray(sig[int(o) + int(c): int(o) + int(c) + L], float) for o, c in zip(meta["offset"], meta["crop_start_eval"])]
    t = []
    for w in wins:
        t0 = time.perf_counter()
        extract_features(w[:, ch.index("ecg")], w[:, ch.index("ppg")], FS)
        t.append((time.perf_counter() - t0) * 1000)
    feat_ms = float(np.median(t))
    for r in rows:
        r["feature_ms"] = feat_ms if r["uses_features"] else 0.0
        r["cpu1_end_to_end_ms"] = r["cpu1_ms"] + r["feature_ms"]
    report(rows, hw, feat_ms, args.threads)


def report(rows, hw, feat_ms, threads):
    names = {"ours": "HiCAF-BP-Cal (ours)", "ours_uncal": "HiCAF-BP, uncalibrated (ours)", "resnet1d": "ResNet1D", "hgctnet": "HGCTNet",
             "bpbls": "BpBLS", "ragcbpnet": "RAGCBPNet", "runet": "rU-Net", "utransbpnet": "UTransBPNet"}
    L_ = ["# Computational profile (one 10 s ECG + PPG reading; frozen fold-0 models)", "",
          f"Hardware: CPU {hw['cpu']} ({hw['cpu_logical_cores']} logical cores; load average at start {hw['load_average_1_5_15min']}); "
          f"GPU {hw['gpu']}; PyTorch {hw['torch']}. "
          f"Latency: median of repeated single-reading runs after warm-up (95th percentiles in COMPUTE.json); CPU with 1 thread and {threads} threads. "
          "FLOPs: 2 x multiply-adds of all matrix products and 1-D/2-D convolutions, counted at the operator level (element-wise "
          "operations, normalisation and FFTs not included). HiCAF-BP-Cal per reading = new reading with the "
          "calibration embedding cached (deployment); in brackets: with re-encoding the calibration recording. Feature-based methods: "
          f"+ {feat_ms:.1f} ms (median) to extract the 31 handcrafted features on the CPU (1 thread, 200 real windows).", "",
          "| Method | Parameters (M) | Size (MB, fp32) | GFLOPs / reading | CPU 1 thread (ms) | CPU end-to-end incl. features (ms) | "
          f"CPU {threads} threads (ms) | GPU batch 1 (ms) | GPU throughput (readings/s, batch 256) | GPU memory (MB, batch 256) | Aurora MAE SBP / DBP |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cal = lambda k: f" ({r[k]:.1f})" if k in r else ""
        L_.append(f"| {names[r['method']]} | {r['params'] / 1e6:.2f} | {r['size_mb']:.1f} | {r['gflops']:.3g}{cal('gflops_with_cal')} | "
                  f"{r['cpu1_ms']:.1f}{cal('cpu1_ms_with_cal')} | {r['cpu1_end_to_end_ms']:.1f} | "
                  f"{r['cpuN_ms']:.1f}{cal('cpuN_ms_with_cal')} | "
                  + (f"{r['gpu_ms']:.2f}{cal('gpu_ms_with_cal')} | {r['gpu_throughput_per_s']:,.0f} | {r['gpu_peak_mem_mb_b256']:.0f} | " if "gpu_ms" in r else "– | – | – | ")
                  + f"{MAE[r['method']][0]:.2f} / {MAE[r['method']][1]:.2f} |")
    tab = CODE_DIR / "paper/tables/compute"
    tab.mkdir(parents=True, exist_ok=True)
    (tab / "COMPUTE.md").write_text("\n".join(L_))
    json.dump({"hardware": hw, "feature_ms": feat_ms, "rows": rows}, open(tab / "COMPUTE.json", "w"), indent=2, default=float)
    print("\n".join(L_))

    # figure: accuracy vs computational cost
    from src.viz.style import METHOD, figure, panel_labels, save, setup
    setup()
    fig, axes = figure("double", height=2.5, ncols=2)
    for ax, xk, xl in [(axes[0], "gflops", "GFLOPs per reading (log scale)"), (axes[1], "cpu1_end_to_end_ms", "CPU latency, 1 thread, incl. features (ms, log scale)")]:
        for r in rows:
            st = METHOD[r["method"]]
            ax.scatter(r[xk], MAE[r["method"]][0], marker=st["marker"] or "o", color=st["color"], s=28, zorder=3, label=st["label"])
            ax.annotate(st["label"].replace(" (ours)", "").replace(", uncalibrated", " uncal."), (r[xk], MAE[r["method"]][0]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.set(xscale="log", xlabel=xl, ylabel="Aurora SBP MAE (mmHg)")
    panel_labels(axes)
    save(fig, "compute_tradeoff", source=__file__)


if __name__ == "__main__":
    main()
