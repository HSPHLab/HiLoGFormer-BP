# HiLoGFormer-BP

Code for the paper

> **HiLoGFormer-BP: A Hierarchical Local–Global Transformer With Bidirectional Cross-Modal Enhancement for Reference-Conditioned Blood Pressure Estimation Using Wearable ECG–PPG Signals**
> Farhad Hassan, Zubair Akbar, Mubashir Ali, Ubaidullah alias Kashif, Jingzhen Li, Wang Zihao, Zedong Nie
> Under review.

HiLoGFormer-BP estimates systolic and diastolic blood pressure (SBP, DBP) from a 10-s recording of the wrist ECG and PPG. It is calibrated with one to three cuff readings of the same person and needs no retraining for a new user. A siamese network encodes the current recording and the calibration recording with the same encoder and estimates the change of BP since calibration from the comparison of the two embeddings.

The encoder works on both signals on a common time axis:

1. a multi-scale convolutional tokenizer per signal (kernels of about 56, 120 and 250 ms, 64-ms tokens);
2. transformer blocks per signal with local and global attention;
3. bidirectional cross-attention between ECG and PPG with rotary position encoding, so that attention scores depend on the time lag between the two signals;
4. a fusion encoder over the concatenated tokens with attention pooling.

The head compares the fused embeddings and, separately, the PPG-only embeddings of the two recordings, and adds the estimated change to the calibration BP.

> **Naming.** In the code, HiLoGFormer-BP is implemented as `hicaf_bp_cal` (final configuration: `configs/experiments/hicaf_bp_cal_notime.yaml`). The uncalibrated version of the same encoder is `hicaf_bp`.

## Repository layout

```
configs/
  base.yaml               paths, data and training defaults (edit the paths first)
  models/                 model hyperparameters
  experiments/            one file per experiment of the paper (see below)
src/
  data/                   dataset, calibration pairs, folds, augmentations
  models/                 hicaf_bp_cal.py (HiLoGFormer-BP), hicaf_bp.py (encoder), resnet1d.py, layers/
  published/              re-implementations of HGCTNet, BpBLS, RAGCBPNet, rU-Net and UTransBPNet
  features/               pulse-wave features (used by the comparison methods and the surrogate model)
  training/               trainer, losses, exponential moving average
  evaluation/             metrics and reports
  interp/                 statistics for the interpretability analyses
  viz/                    figure style
scripts/                  data preparation, training, evaluation and analyses (see below)
preprocess_aurora.py      preprocessing of Aurora-BP
aurora_bp_eda.ipynb       data inventory of Aurora-BP; writes the measurement index used by preprocess_aurora.py
docs/reimplementation/    decisions and deviations for each re-implemented method
```

## Installation

Tested with Python 3.10.20 and PyTorch 2.0.1 (CUDA 11.8).

```bash
conda create -n hilogformer python=3.10
conda activate hilogformer
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

No data are included in this repository. The data sets have to be obtained from their providers under their own terms of use.

- **Aurora-BP** (Mieloszyk et al., *Sci. Rep.*, 2022): wrist ECG and PPG with auscultatory and oscillometric reference BP, recorded in the laboratory and at home. Used for training and cross-validation.
- **MIMIC-III** and **VitalDB**, as segmented by **PulseDB** (Wang et al., *Front. Digit. Health*, 2023). Used for the external tests and the adaptation experiment. MIMIC-III requires credentialed access on PhysioNet.

After download, set the paths in `configs/base.yaml` and in the arguments of `preprocess_aurora.py`. All paths in `configs/base.yaml` are relative to the repository root, so the scripts should be run from there.

## Running the experiments

Every script describes its inputs and outputs in its header and lists its arguments with `--help`. Results are written to `runs/` (models, predictions) and `paper/` (tables, figures); neither folder is part of the repository.

**1. Data**

| Step | Script |
|---|---|
| Inventory of Aurora-BP (writes the measurement index) | `aurora_bp_eda.ipynb` |
| Preprocessing of Aurora-BP | `preprocess_aurora.py` |
| Subject-wise folds | `scripts/make_folds.py` |
| Pulse-wave features | `scripts/extract_pwa_features.py` |
| Preprocessing and splits of MIMIC-III and VitalDB (PulseDB) | `scripts/preprocess_pulsedb.py`, `scripts/make_pulsedb_splits.py` |

**2. Training (five folds per experiment)**

| Model | Command |
|---|---|
| HiLoGFormer-BP and its variants | `scripts/train.py` with a file from `configs/experiments/` |
| Re-implemented comparison methods | `scripts/train_published.py` |
| Cross-validation summary | `scripts/aggregate_cv.py` |

| Experiment in the paper | Configuration |
|---|---|
| HiLoGFormer-BP (final model) | `hicaf_bp_cal_notime` |
| Without calibration | `hicaf_bp` (run name `hicaf_bp_main`) |
| With time context | `hicaf_bp_cal_t_scratch` |
| Ablations | `abl_nt_no_cross`, `abl_nt_sinusoidal`, `abl_nt_no_gate`, `abl_nt_single_scale`, `abl_nt_all_global`, `abl_nt_ppg_only`, `abl_nt_ecg_only` |
| ResNet1D baseline | `resnet1d` |

**3. Evaluation**

| Analysis | Script |
|---|---|
| Accuracy of the calibrated and calibration-free methods on Aurora-BP | `scripts/report_calibrated.py`, `scripts/report_calfree_aurora.py` |
| Ablation study | `scripts/ablation_report.py` |
| External tests on MIMIC-III and VitalDB | `scripts/eval_external.py`, `scripts/eval_published_external.py`, `scripts/eval_calfree_external.py`, `scripts/collect_pulsedb_results.py` |
| Adaptation with part of the external data | `scripts/finetune_pulsedb.py` |
| IEEE Std 1708-2025 accuracy report and grading | `scripts/perf_P1_ieee1708.py` |
| Uncertainty budget (GUM) | `scripts/perf_P4_uncertainty.py` |
| Robustness to noise | `scripts/perf_P5_noise.py` |
| Computational cost | `scripts/profile_compute.py` |
| Interpretability (drift score, plausibility, attributions, subgroups) | `scripts/interp_run_all.sh` or the `scripts/interp_*.py` scripts |
| Embeddings | `scripts/embed_viz.py` |
| Remaining tables and figures | `scripts/paper_analyses.py`, `scripts/final_report.py`, `scripts/finalize_outputs.py`, `scripts/embed_fig_main.py`, `scripts/subgroups_main_fig.py` |

## Comparison methods

HGCTNet, BpBLS, RAGCBPNet, rU-Net and UTransBPNet were re-implemented from their publications and, where available, the code of their authors, and trained and tested with the same data, folds, preprocessing and calibration readings as HiLoGFormer-BP. The decisions taken where a publication left details open, and all deviations from it, are listed in `docs/reimplementation/`. The original code of these methods is not part of this repository and remains under the terms of its authors.

## Citation

If you use this code, please cite the paper (the reference will be updated on publication):

```bibtex
@article{hassan2026hilogformer,
  author  = {Hassan, Farhad and Akbar, Zubair and Ali, Mubashir and {Ubaidullah alias Kashif} and Li, Jingzhen and Wang, Zihao and Nie, Zedong},
  title   = {{HiLoGFormer-BP}: A Hierarchical Local--Global Transformer With Bidirectional Cross-Modal Enhancement for Reference-Conditioned Blood Pressure Estimation Using Wearable {ECG}--{PPG} Signals},
  note    = {Under review},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The license does not cover the data sets, which remain under the terms of their providers.
