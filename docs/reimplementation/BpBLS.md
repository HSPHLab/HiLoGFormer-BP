# BpBLS — re-implementation specification (supplementary material)

Status: **5-fold CV and zero-shot PulseDB complete; frozen (tag v1-pub-bpbls).** Common protocol: `docs/REIMPLEMENTATION_PLAN.md` (R1–R9).

## 1. Sources

| Source | Used for |
|---|---|
| Z.-X. Huang et al., "BpBLS: A Knowledge-Embedded Bi-Incremental Broad Learning System for Wearable Cuffless Blood Pressure Estimation," IEEE JBHI 30(6):5489–5500, 2026. DOI 10.1109/JBHI.2025.3640660 | method (Sec. III), features (Table V), settings (Sec. IV-C), results |
| Official code: https://github.com/Huangzx1023/BpBLS-for-BP-Estimation (`main.py`, `utils.py`; downloaded as zip, sha256 5ac31aafa7ee77d60850941600ff073e0a60073c93422d12df7198f3f7e298a7) | exact network, training procedure, closed-form solution, all hyperparameters |
| Z.-D. Liu et al., IEEE JBHI 27(9), 2023 (ref. [6] of BpBLS) | feature definitions (shared with HGCTNet, `docs/reimplementation/HGCTNet.md` section 4) |

## 2. Rule B0: where the paper text and the official code disagree, we follow the code

The code reproduces the paper's own reported model size (49.5 K parameters, Sec. V-C) exactly with its settings, whereas the
settings stated in the paper text (12 windows, 75 enhancement nodes) give about 19 K. The code is therefore taken as the
configuration that produced the published results. Every disagreement is listed in section 7.

## 3. Model (official `main.py`)

| Part | Implementation (kept) |
|---|---|
| Input | [handcrafted features + demographics, baseline SBP/DBP] (the baseline BP is ALSO an input feature, `torch.cat((train_x, train_cal))`) |
| Input normalisation | BatchNorm1d on the input |
| Feature nodes | Linear(input → 10 × 46 = 460) → BatchNorm1d → sigmoid (the "windows" are realised as one linear layer of 460 units) |
| Enhancement nodes | Linear(460 → 28) → BatchNorm1d → tanh |
| Output | Linear(460 + 28 → 2, no bias); column 0 = DBP, column 1 = SBP |
| Calibration | BP = 0.29 × output + 0.71 × baseline BP (fixed β = 0.29) |

With our 37 inputs (31 features + 4 demographics + 2 baseline values) the model has 32,414 parameters (the code with its 74 inputs: 49,508, matching the paper's 49.5 K; the paper-text settings would give 19,003).

## 4. Training (official `main.py`) — two stages

1. **Node weights by gradient descent ("SBPA")**: 50 epochs, each one **full-batch** step on the whole training set; Adam,
   learning rate **1.0**, weight decay 2⁻¹², StepLR(50, 0.9) (no effect within 50 steps); seed 42 (+ fold).
   Loss = MSE(calibrated output, BP) + 2⁻⁷ × MSE(predicted SBP − DBP, subject's mean pulse pressure), where the
   calibrated output uses the trainable output layer and β = 0.29.
2. **Output weights in closed form** (`ridge_regression_pseudo_inverse_reg`): A = [feature nodes | enhancement nodes] of
   the training data (eval mode), B⁺ = ((1 + γ)AᵀA + λI)⁺ Aᵀ with γ = 2⁻⁷, λ = 2⁻⁸; b = γ/(1+γ), c = 1/(1 − b²);
   W_SBP = c B⁺ (b·DBP + SBP + b·PP), W_DBP = c B⁺ (DBP + b·SBP − b·PP); PP = each training subject's mean pulse pressure.
   Prediction: BP = 0.29 × (A W) + 0.71 × baseline BP.
3. No early stopping and no model selection (fixed 50 epochs). The official code prints test metrics every epoch but never
   selects on them; we evaluate the test set once, after training.

## 5. Inputs (ours)

| Input | Ours | Reason |
|---|---|---|
| Handcrafted features | the frozen 31 ECG/PPG features (`data_processed/features_pwa/aurora_v1.csv`); identical to Table V for ECG + PPG except the PPG intensity ratio | R2, D6 of HGCTNet (same extractor) |
| Demographics | age, sex, BMI (703 × lb / in²), history of hypertension; same mapping as HGCTNet | Table V |
| Feature / demographic scaling | min–max with training-subject minimum and maximum (the official `normalize_data`); missing values first filled with the training median (features) / mean (demographics) | as the code; leakage-free |
| Baseline BP | training: mean of each subject's first 3 calibration readings; test: k = 1 and k = 3 | R4 (as HGCTNet H2) |
| Pulse pressure (PPR target) | mean SBP − DBP of each training subject over its training recordings | paper Eq. (9): "average pulse pressure for each person" |
| Training subjects | our fold's training subjects with calibration readings (validation subjects not used: no model selection) | R1, H6 |

## 6. Evaluation

Identical to HGCTNet (section 6 of its specification): Aurora 5-fold CV on our model's test recordings (k = 1, 3);
zero-shot PulseDB with `scripts/eval_published_external.py` (same target segments, missing inputs from training statistics).
The official test-time averaging over segments of one recording (`compute_mean_values`) does not apply: each of our
test samples is one recording / one segment.

## 7. Deviations and paper–code discrepancies (complete list)

| # | Paper text | Official code (followed) / ours | Reason |
|---|---|---|---|
| P1 | 12 feature windows, 75 enhancement nodes | 46 windows × 10 nodes, 28 enhancement nodes | B0: code reproduces the reported 49.5 K parameters |
| P2 | γ (PPR) = 2⁻¹, λ (ridge) = 2⁻¹⁰ | γ = 2⁻⁷, λ = 2⁻⁸ | B0 |
| P3 | Adam learning rate 10⁻⁴ | 1.0 (full-batch steps) | B0 |
| P4 | β "based on previous work" (value not given) | 0.29 (final model; the code's incremental branch uses 0.30, not used here) | B0 |
| P5 | Algorithm 1: per-sample updates | one full-batch step per epoch | B0 |
| P6 | baseline BP used only in the calibration formula (Eq. 22) | also an input feature | B0 |
| D1 | ECG + PPG + tonometry features | ECG + PPG features | R2 |
| D2 | 47-feature framework incl. tonometry; PPG intensity ratio | 31 ECG/PPG features; intensity ratio dropped | R2; HGCTNet D6 |
| D3 | baseline = mean of the first visit | mean of the first 3 calibration readings (also k = 1) | R4 |
| D4 | single 8:2 subject split | our 5 frozen folds | R1 |
| D5 | bi-incremental updates (data, signals) | trained once on the full training set with both signals | the incremental updates reproduce the full-data solution; accuracy is the paper's "all subsets, all signals" setting |
| D6 | features min–max scaled (implied by the code, values precomputed) | min–max with training statistics per fold, after imputation | leakage-free |

## 8. Implementation notes (filled during implementation)

- `src/published/bpbls.py`: official `BLS` class and `ridge_regression_pseudo_inverse_reg` copied; an inference wrapper applies
  the fixed calibration (0.29) and returns [SBP, DBP]. The closed-form transcription was verified on simulated data with a
  known solution (weights recovered exactly).
- Training (`scripts/train_published.py --method bpbls`): stage 1 = 50 full-batch Adam steps on the official loss; stage 2 =
  closed-form output weights on the training subjects' node activations (eval mode). Validation error logged for
  information only (no selection).

## 9. Provenance

- Official code: zip sha256 above (downloaded 2026-10-12; GitHub download from the server failed, file downloaded manually)
- Our code: `src/published/bpbls.py`, training via `scripts/train_published.py --method bpbls`

## 10. Results (frozen)

Stage 1 (official learning rate 1.0): the loss rises at step 2, then falls steadily to about 115-124 by step 50 in all folds.

**Aurora-BP (5-fold CV, our test recordings, pooled):**

| k | BpBLS SBP / DBP MAE | Calibration only SBP / DBP | Our v1 SBP / DBP | Predictions move from baseline (SD) SBP / DBP | corr(predicted change, true change) SBP / DBP |
|---|---|---|---|---|---|
| 1 | 11.47 / 8.51 | 12.12 / 9.27 | 8.90 / 6.44 | 2.61 / 1.99 | 0.43 / 0.45 |
| 3 | 10.86 / 7.95 | 11.43 / 8.54 | 8.48 / 6.22 | 2.57 / 1.90 | 0.41 / 0.40 |

Paper (Aurora, single 8:2 split, ECG+PPG+tonometry, quality-filtered): SBP MAE 6.14, DBP 4.96.

**Zero-shot PulseDB (ensemble of the 5 fold models; identical target segments to our model), k = 1:**

| Data | BpBLS SBP / DBP | Calibration only | Our no-time model |
|---|---|---|---|
| combined | 16.71 / 9.29 | 17.87 / 9.55 | 15.63 / 10.59 |
| MIMIC | 14.46 / 8.16 | 15.48 / 7.96 | 15.18 / 10.68 |
| VitalDB | 18.47 / 10.17 | 19.73 / 10.79 | 15.98 / 10.53 |

BpBLS improves on the calibration value in both datasets (narrower error spread) and beats our model on MIMIC for SBP and DBP; our model is clearly better for VitalDB SBP (-2.2 mmHg, better for 61% of subjects). Full tables: `runs/pub_bpbls/external_pulsedb/EXTERNAL_RESULTS.md`.
