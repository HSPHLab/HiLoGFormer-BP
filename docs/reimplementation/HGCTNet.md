# HGCTNet — re-implementation specification (supplementary material)

Status: **features extracted (Aurora, PulseDB); model ported; smoke test passed (2.40 M parameters, 12 time steps); 5-fold CV and zero-shot PulseDB complete; frozen (tag v1-pub-hgctnet).** Common protocol: `docs/REIMPLEMENTATION_PLAN.md` (R1–R9).

## 1. Sources

| Source | Used for |
|---|---|
| Z.-D. Liu et al., "HGCTNet: Handcrafted Feature-Guided CNN and Transformer Network for Wearable Cuffless Blood Pressure Measurement," IEEE JBHI 28(7):3882–3894, 2024. DOI 10.1109/JBHI.2024.3395445 | method, hyperparameters (Sec. IV-C, Fig. 6), CNN shapes (Table A1), feature list (Tables II, B2) |
| Official code: https://github.com/zdzdliu/HGCTNet (`HGCTNet.ipynb`; commit recorded in section 9) | exact layer implementation (CNN, additive attention, transformer, fusion, calibration) |
| Z.-D. Liu et al., "Cuffless Blood Pressure Measurement Using Smartwatches: A Large-Scale Validation Study," IEEE JBHI 27(9):4216–4227, 2023 (Table A1, Fig. 3) | exact definitions of the handcrafted features and characteristic points |
| F. Miao et al., IEEE JBHI 21(6):1730–1740, 2017 (ref. [17] of Liu 2023) | pulse K value |

## 2. Model (ported exactly from the official notebook)

| Module | Published implementation (kept) | Our port |
|---|---|---|
| CNN | [conv3-BN-ReLU]×2 (32) → maxpool 2 → [conv3-BN-ReLU]×2 (64) → maxpool 2 → BasicBlock 64→128 (3 convs + 1×1 shortcut) → maxpool k2 s3 → BasicBlock 128→256 → maxpool k2 s3 → BasicBlock 256→256 → maxpool k2 s3 | identical; **input channels 2 (ECG, PPG)** instead of 3 |
| Handcrafted-feature-guided attention (HGA) | additive attention: query = handcrafted features, keys = the 256 CNN channels (each a vector over time), tanh, linear → 1, softmax over channels; channels re-weighted | identical; hidden size **8** (paper Fig. 6; the notebook's example call uses 16); query size = number of features (31) |
| Transformer | pre-LayerNorm encoder, **2 layers, 2 heads**, embedding 256, FFN ×2 with ReLU, dropout 0.1, sinusoidal positional encoding, attention scaled by √256 (sic, full embedding size), average over time | identical (quirks kept) |
| Fusion | (transformer output + time-averaged attended CNN features) ⊙ MLP([demographics, handcrafted features]); MLP 51→128→ReLU→Dropout 0.5→256 | identical; MLP input = 4 + 31 = **35** |
| Regressor | 256→128→ReLU→Dropout 0.5→2 | identical |
| Calibration | BP = α ⊙ prediction + (1 − α) ⊙ baseline BP, α ∈ R² learnable, initialised `torch.randn` | identical (random initialisation kept) |
| Weight init | Kaiming normal (fan_out) for conv and linear, zero bias; BN weight 1, bias 0 | identical |

Sequence length: the paper's Aurora input (20 s at 500 Hz = 10,000 samples) gives 93 time steps after the CNN; our common input (R3: 10 s at 125 Hz = 1,250 samples) gives **12 time steps**. The architecture is length-agnostic and is not modified.

## 3. Inputs

| Input | Paper (Aurora) | Ours | Reason |
|---|---|---|---|
| Signals | ECG + PPG + tonometry, 20 s, 500 Hz, Butterworth ECG 0.5–40 Hz, PPG/PPW 0.5–20 Hz, z-score | **ECG + PPG**, the same fixed **10 s** window at **125 Hz** used by all methods (our preprocessing: ECG Butterworth 0.5–40 Hz, PPG Chebyshev-II 0.5–10 Hz, z-score) | R1–R3: identical input for every method |
| Handcrafted features | top 47 of 67 (RReliefF), incl. tonometry features | **31 ECG/PPG features** (section 4), no selection | R2; the paper's own top 47 contained 31 of the 32 ECG/PPG features; PIR dropped (deviation D6) |
| Demographics | age, gender, BMI, history of hypertension | the same four from Aurora metadata: age; gender (F = 0, M = 1); BMI = 703 × weight [lb] / height [in]² (Aurora stores imperial units; detected automatically, logged per run); history of hypertension = 1 if `self_report_htn` ∈ {managed, unmanaged}, 0 if `norm`, training-set mean if undefined | as published |
| Baseline BP | mean BP of the initial visit | **mean of the subject's first 3 calibration readings** (training); evaluated with k = 3 (as trained) and k = 1 | R4, H2 |
| Normalisation of features / demographics | not specified | z-score with training-subject statistics (per fold) | needed for an MLP; no test information used |

## 4. Handcrafted features (31), computed on the processed 10 s window

Per cardiac cycle, then averaged over the cycles of the window (as in Liu 2023). Characteristic points (Liu 2023, Fig. 3):
raw PPG: s = foot (cycle onset), m = maximum upslope, p = systolic peak, n = dicrotic notch, v = end of cycle (= next foot);
VPPG / APPG (1st / 2nd derivative of PPG): s = minimum before the main peak, p = main peak, v = minimum after the peak.

| # | Feature | Definition (Liu 2023, Table A1) | Signals |
|---|---|---|---|
| 1–3 | PTT_RS1, PTT_RM1, PTT_RP1 | time from ECG R peak to PPG s / m / p | ECG + PPG |
| 4–6 | Ascending time | time s → p | PPG, VPPG, APPG |
| 7–9 | Descending time | time p → v | PPG, VPPG, APPG |
| 10–12 | Ascending slope | slope between s and p | PPG, VPPG, APPG |
| 13–15 | Descending slope | slope between p and v | PPG, VPPG, APPG |
| 16–18 | Ascending area | area below the curve between s and p | PPG, VPPG, APPG |
| 19–21 | Descending area | area below the curve between p and v | PPG, VPPG, APPG |
| 22–24 | Ascending intensity difference (AID) | amplitude p − s | PPG, VPPG, APPG |
| 25–27 | Descending intensity difference (DID) | amplitude p − v | PPG, VPPG, APPG |
| 28 | Large artery stiffness index (LASI) | time p → n | PPG |
| 29 | Pulse width | time m → n | PPG |
| 30 | Pulse rate | 60 / cardiac cycle (s → v) | PPG |
| 31 | Pulse K value | (mean − min) / (max − min) of the cycle (Miao 2017) | PPG |

Areas are integrated relative to the lower endpoint of the segment (s for ascending, v for descending); times in seconds; amplitudes in the units of the processed (z-scored) signal (deviation D7). The characteristic-point detection algorithm (Liu 2023 refers to an unavailable earlier study for details) is our own and is documented in section 8 after visual validation on Aurora and PulseDB beats.

## 5. Training (paper Sec. IV-C, Fig. 6)

| Setting | Paper | Ours |
|---|---|---|
| Loss | Huber, δ = 4 (on mmHg) | same |
| Optimizer | Adam, lr 1e-3, β = (0.9, 0.999), weight decay 1e-4 | same |
| Batch / epochs | 64 / 200 with early stopping (patience not given) | 64 / max 200, **early stopping patience 20** on the validation score (H4) |
| Target scale | mmHg (no label normalisation; required by the calibration formula) | same |
| Augmentation | none | none; the fixed evaluation window is used for training too (H3) |
| Split | single random 6:2:2 subject split | our 5 frozen folds, same train/val/test subjects as our model (R1) |
| Training subjects | all | subjects without calibration readings excluded (no baseline BP; H6) |
| Model selection | best validation model | best validation score = mean of SBP and DBP MAE (k = 3 baseline), as for our model |

## 6. Evaluation

- Aurora-BP: pooled over the 5 folds on exactly our model's test recordings (calibration readings excluded); k = 3 (as trained) and k = 1.
- PulseDB zero-shot (MIMIC, VitalDB separately): the 5 fold models; baseline = mean of the first k segments; features computed on the same 10 s segments with the same extractor; demographics: age and gender from PulseDB; BMI read from the VitalDB subject files (`Subj_Wins/BMI`, cached in `data_processed/pulsedb_v1/vitaldb_bmi.csv`; MIMIC has none: training mean); history of hypertension unavailable (training mean); all fill values are the training statistics stored in each fold's model file (R7). Script: `scripts/eval_published_external.py` (checks that the target segments are identical to our model's zero-shot test and reports both side by side, with a paired subject-level test).

## 7. Deviations from the paper (complete list)

| # | Paper | Ours | Reason |
|---|---|---|---|
| D1 | ECG + PPG + tonometry | ECG + PPG | R2 (wearable signals used by all methods) |
| D2 | 20 s at 500 Hz (93 time steps after CNN) | 10 s at 125 Hz (12 time steps) | R3 (same input for all methods) |
| D3 | PPG band-pass 0.5–20 Hz Butterworth | 0.5–10 Hz Chebyshev-II (our preprocessing) | R1 (shared preprocessed data) |
| D4 | 47 features selected by RReliefF from 67 | 31 ECG/PPG features, no selection | R2; their top 47 already contained 31 of the 32 ECG/PPG features (H1) |
| D5 | baseline = mean of the initial visit | mean of the first 3 calibration readings (also tested with 1) | R4, H2 |
| D6 | PPG intensity ratio (peak / foot intensity of the raw optical signal) included | dropped | undefined on band-passed, zero-mean signals (the foot is near zero); raw optical levels are not part of the shared input |
| D7 | amplitude features on the raw signal scale | on the z-scored processed signal | R1; scale-free, identical for all recordings |
| D8 | early-stopping patience not reported | 20 epochs | H4 |
| D9 | single 6:2:2 split | our 5 frozen folds | R1 |
| D10 | feature/demographic normalisation not reported | z-score, training statistics per fold | required; leakage-free |
| D11 | characteristic-point detection per an earlier study | our own detector (section 8) | detection details not published |

## 8. Implementation notes (filled during implementation)

- Characteristic-point detector (`src/features/pwa_features.py`; our own, D11):
  - ECG R peaks: peaks above half the 99.5th percentile, at least 0.33 s apart.
  - PPG maximum upslope m: peaks of the first derivative above 40% of its 99th percentile, at least 0.33 s apart.
  - PPG foot s: intersecting-tangent point (tangent at m meets the level of the preceding minimum, searched back to
    0.4 s or the previous beat); v = the next beat's foot. Cycles kept if 0.33 s <= s->v <= 2.0 s (30-180 bpm).
  - Peak p: maximum within 0.35 s after m. Dicrotic notch n: first local minimum of the PPG between p + 0.05 s and
    60% of p->v, with prominence >= 2% of the pulse amplitude (so ripples on flat wrist peaks are not taken as a notch); if none exists, the maximum of the second derivative in that stretch (fallback, rate reported).
  - VPPG: p = m; s = minimum from 0.15 s before the foot to p; v = minimum from p to the end of the cycle.
    APPG: p = maximum from 0.1 s before the foot to m (a-wave); s = minimum before it; v = minimum in the
    following 40% of the cycle (b-wave).
  - Pulse arrival: the last R peak 0.05-0.8 s before m; a cycle's three PTTs are discarded if R -> foot is not in (0, 0.6] s.
  - Areas: sum of (signal - lower endpoint value) / fs over the segment.
  - Validated on simulated pulses (pulse rate exact; PTTs track the true value with a constant offset);
    visual check on real Aurora / MIMIC / VitalDB beats: `eda_outputs/pwa_check/` (2,000 random windows each):
    - s, m, p and the derivative points placed as in Liu 2023 Fig. 3 in all three datasets; windows with features
      99.9% / 100% / 100%; median 11 / 13 / 11 cycles per 10 s window (Aurora / MIMIC / VitalDB).
    - Dicrotic notch fallback 84% / 73% / 51% of cycles: the notch is often absent at the wrist and in older or
      critically ill patients (physiology); the second-derivative inflection is the standard substitute.
    - First check placed n on ripples of flat-topped Aurora peaks -> prominence rule added (above).
    - ECG-PPG timing: median R -> PPG foot 0.22 s (Aurora, physiological), 0.50 s (MIMIC, implausibly long),
      and no valid value in 70% of VitalDB windows (the R peak often follows the PPG upstroke). PulseDB's ECG and
      PPG are therefore not physiologically time-aligned; affected PTT features are NaN and imputed with the
      training median. This affects every ECG+PPG method in the zero-shot test equally (incl. ours) and is
      reported as a limitation of PulseDB for timing-based features.
- Windows with fewer than 2 valid cycles: all features NaN; NaN values are imputed with the training-fold median
  (per fold, no test information).

## 9. Provenance

- Official notebook commit: (recorded below)
- Our code: `src/published/hgctnet.py` (model), `src/features/pwa_features.py` (features), training via `scripts/train_published.py --method hgctnet`
- Official notebook commit: 6d5103cac4a6fe77d5033ae96fc4e9e68f2aee30 (cloned 2026-10-10)

## 10. Results (frozen)

**Learned calibration weight.** alpha converged to 0.01-0.04 in all folds (two slightly negative). This is NOT a collapse:
BP = baseline + alpha * (prediction - baseline), and the network compensates with large outputs. On Aurora the
predictions move away from the baseline with SD 7.8 (SBP) / 5.6 (DBP) mmHg and the predicted change correlates with
the true change (r = 0.51 / 0.53). (We first suspected a collapse; the check above refuted it - recorded for transparency.)

**Aurora-BP (5-fold CV, our test recordings, pooled):**

| k | HGCTNet SBP / DBP MAE | Calibration only SBP / DBP | Our v1 SBP / DBP |
|---|---|---|---|
| 1 | 10.55 / 7.59 | 12.12 / 9.27 | 8.90 / 6.44 |
| 3 | 9.98 / 7.10 | 11.43 / 8.54 | 8.48 / 6.22 |

Paper (Aurora, single 6:2:2 split, ECG+PPG+tonometry, 20 s at 500 Hz, quality-filtered, baseline = initial-visit mean):
SBP MAE 6.1, DBP 5.2; its ECG+PPG variant SBP 6.5. Differences in protocol are listed in section 7; our test set also
contains the ambulatory (home) readings.

**Zero-shot PulseDB (ensemble of the 5 fold models; identical target segments to our model):**

| k = 1 | HGCTNet SBP / DBP MAE | Calibration only | Our no-time model |
|---|---|---|---|
| combined | 18.23 / 9.62 | 17.87 / 9.55 | 15.63 / 10.59 |
| MIMIC | 15.59 / 7.91 | 15.48 / 7.96 | 15.18 / 10.68 |
| VitalDB | 20.29 / 10.95 | 19.73 / 10.79 | 15.98 / 10.53 |

HGCTNet's learned changes do not transfer to PulseDB (no better than the calibration value; PTT features largely
unusable there because of the non-physiological ECG-PPG timing). Full tables: `runs/pub_hgctnet/external_pulsedb/EXTERNAL_RESULTS.md`.
