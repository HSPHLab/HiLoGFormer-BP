# rU-Net — re-implementation specification (supplementary material)

Status: **5-fold CV and zero-shot PulseDB complete; frozen (tag v1-pub-runet).** Common protocol: `docs/REIMPLEMENTATION_PLAN.md` (R1–R9).

## 1. Sources

| Source | Used for |
|---|---|
| J. Chen et al., "rU-Net, Multi-Scale Feature Fusion and Transfer Learning: Unlocking the Potential of Cuffless Blood Pressure Monitoring With PPG and ECG," IEEE JBHI 29(1):166–176, 2025. DOI 10.1109/JBHI.2024.3483301 | everything (Sec. II, Figs. 1–2, Sec. III-A) |
| Official code | **none published** (the paper gives no repository) — the architecture is reconstructed from the text and Figs. 1–2; every unspecified detail is our choice, marked "ours" below |

## 2. Method as described in the paper

- Input: ECG and PPG, 10 s at 125 Hz (1250 × 2) — identical to our common input. Demographics (VitalDB: sex, age, height, weight, BMI).
- Three parallel feature extractors (Fig. 1a):
  1. **Time series**: 1-D rU-Net → average pooling → bidirectional GRU (64 units, residual, concatenation merge) → concat → attention; output 250 × 128.
  2. **Time-frequency**: STFT of ECG and PPG (Hann window 256, hop 5) → 250 × 128 × 2 → 2-D rU-Net → fully connected layer with 1 unit (compresses one dimension) → bidirectional GRU → concat → attention; output 250 × 128.
  3. **Demographics**: fully connected layers, reshaped and tiled to the time length of every rU-Net level (1-D: 1250, 625, 125; 2-D: 250 × 128, 125 × 64, 25 × 32) and concatenated with the features there.
- **rU-Net** (Fig. 1b, 2c): U-Net whose convolution layers are replaced by residual blocks (ResNet-18 style): kernel 3, "same" padding, ReLU, 1 × 1 residual connection; filters 32 → 64 → 128; level 2 first convolution stride 2; level 3 stride 5 (1-D) / (5, 2) (2-D); CBAM (channel + spatial attention) at the deepest scale; up-sampling after the level-3 and level-2 decoder blocks; transformer attention (Q = K = V = [features, demographics]) fusing encoder and decoder at each scale.
- **Fusion**: concatenation of [STFT features, time-series features, demographic features] along the time axis → multi-head attention (2 heads).
- **Regressor**: two separate heads (SBP, DBP): flatten → FC → ReLU → FC.
- Training (Sec. III-A): TensorFlow; Adam; MSE loss; learning rate 1e-5; batch 32; fixed 512 epochs; the best model on the validation set retained.
- Calibration-free for new subjects (their "calibration-based" setting = test subjects also in training, which our protocol excludes); the paper's fine-tuning transfer learning is not part of our comparison.

## 3. Decisions (agreed before any result)

| # | Decision | Chosen | Reason |
|---|---|---|---|
| U1 | Group | **calibration-free** (no baseline BP input): compared with our uncalibrated model and ResNet1D (R8); calibration-value-only shown for reference | as published |
| U2 | Input signals | our processed ECG + PPG (z-scored), the same fixed 10 s window as all methods | R1–R3; paper: PPG Chebyshev-II 0.5–8 Hz, ECG "Pan-Tompkins" processing (not specified further) |
| U3 | Demographics | sex, age, height, weight, BMI; Aurora's imperial units converted to cm / kg (to match PulseDB VitalDB); z-scored with training statistics; missing values → training mean | as published; unit match needed for the zero-shot test |
| U4 | STFT | magnitude of a Hann-window STFT (window 256, hop 5) of the processed ECG and PPG, computed inside the model; first 250 frames and first 128 frequency bins kept (paper shape 250 × 128 × 2) | paper gives window, hop and shape; magnitude and bin choice are ours |
| U5 | Label scale | targets standardised with training mean / SD (loss = MSE on the standardised scale); predictions converted back to mmHg | not specified; with lr 1e-5, unscaled mmHg targets cannot be reached in practice |
| U6 | Training length | max 512 epochs (as published), best validation epoch retained (as published), plus early stopping after 30 epochs without improvement | the selection rule is the paper's; patience saves compute (R6); cap revisited only if the smoke test shows > 1 day per fold |
| U7 | Unspecified architecture details | standard choices, each listed in section 4 | no code |
| U8 | Transfer learning | not used (zero-shot only, as for every method) | R7 |

## 4. Architecture as implemented (ours where the paper is silent)

| Part | Implementation | Source |
|---|---|---|
| Residual block (RSB) | conv(k3, stride s) → ReLU → conv(k3) → (+ 1 × 1 conv(stride s) shortcut) → ReLU; no batch normalisation (none shown in Fig. 2c) | paper + ours (block depth 2 convolutions, as ResNet-18) |
| 1-D rU-Net | encoder RSB 2→32 (1250), RSB 32→64 stride 2 (625), RSB 64→128 stride 5 (125) + CBAM; decoder: attention at each scale, up-sample ×5 → concat skip → RSB →64; up-sample ×2 → concat skip → RSB →32; output 32 × 1250 | paper (filters, strides, CBAM position) + ours (decoder block widths) |
| 2-D rU-Net | the same with 2-D layers on 2 × 250 × 128; strides (2, 2) and (5, 2) → 125 × 64 and 25 × 32; output 32 × 250 × 128 | paper |
| CBAM | channel attention (shared MLP, reduction 16, average- and max-pooled) then spatial attention (7-tap / 7 × 7 convolution on [mean, max]) | Woo et al. 2018 defaults (ours) |
| Demographic injection | demographics → FC(16) → ReLU → FC(C) per scale, tiled over the time (and frequency) positions, concatenated as C extra channels | paper ("FC + tiling"); width ours |
| Scale attention ("ATT") | single-head scaled dot-product self-attention over time positions on [features ‖ demographics] with learned Q/K/V/output projections back to C channels; residual connection | paper (Q = K = V = [F, F_demo]); projections ours |
| 2-D attention cost | on 2-D maps, attention over time steps (frequency averaged for the attention weights) | ours — full attention over 250 × 128 positions is infeasible |
| Time-series branch | average pooling ×5 (1250 → 250) → BiGRU (64 per direction, concatenated = 128) + linear residual → concat demographic token → attention → 250 × 128 | paper + ours (residual form) |
| STFT branch | FC with 1 unit over the 32 channels → 250 × 128 → BiGRU → concat → attention → 250 × 128 | paper |
| Fusion | tokens [STFT (250), time (250), demographics (1)] × 128 → multi-head attention (2 heads) | paper |
| Regressor | per target: flatten (501 × 128) → FC(64) → ReLU → FC(1) | paper + ours (hidden width 64) |
| Dropout / weight decay | none | none mentioned |

## 5. Evaluation

- Aurora: 5-fold CV on our model's test recordings (calibration recordings excluded so every method is scored on the
  same recordings); rU-Net predictions do not depend on k.
- Zero-shot PulseDB (MIMIC, VitalDB): the 5 fold models; demographics: sex and age (both sources), height / weight / BMI
  from the VitalDB subject files, training mean for MIMIC; same target segments as every other method.
- Reported next to our uncalibrated model and ResNet1D (calibration-free group, R8) and the calibration-value-only reference.

## 6. Deviations from the paper (complete list)

| # | Paper | Ours | Reason |
|---|---|---|---|
| D1 | PPG Chebyshev-II 0.5–8 Hz; ECG Pan-Tompkins processing | our shared preprocessing (ECG 0.5–40 Hz Butterworth, PPG 0.5–10 Hz Chebyshev-II, z-score) | R1 |
| D2 | VitalDB (PulseDB), 9:1 segment split inside subjects for the main result | Aurora 5 frozen folds (subject-level) | R1 |
| D3 | TensorFlow | PyTorch | environment |
| D4 | label scale not reported | standardised targets (U5) | trainability at lr 1e-5 |
| D5 | fixed 512 epochs | max 512 epochs, best validation epoch, early stop after 30 epochs without improvement (U6) | compute; same selection rule |
| D6 | unspecified architecture details | section 4 choices | no code |
| D7 | attention over 2-D maps not specified | attention over time steps (U7, section 4) | memory |

## 7. Implementation notes (filled during implementation)

- `src/published/runet.py`; `python -m src.published.runet` prints every stage's shape (all match the paper: 1-D 1250 -> 625 -> 125 -> 1250; 2-D 250x128 -> 125x64 -> 25x32 -> 250x128; branch outputs 250x128; fusion 501 tokens).
- Training: all training subjects of the fold (calibration-free, no baseline needed); MSE on standardised targets (training mean / SD stored in the model as buffers); predictions returned in mmHg.
- Smoke test: about 0.1 s per batch of 32 -> about 1 min per epoch on the full fold; worst case 512 epochs = 8-9 h per fold, within the agreed limit (U6 unchanged).
- Zero-shot: VitalDB height, weight, BMI from the subject files (`data_processed/pulsedb_v1/vitaldb_demo.csv`); MIMIC: training mean. Inference batch 256 (memory).

## 8. Provenance

- Paper only (no official code).
- Our code: `src/published/runet.py`, training via `scripts/train_published.py --method runet`

## 9. Results (frozen)

**Training.** Early stopping (patience 30) ended every fold far below the 512-epoch cap (e.g. fold 2: 47 epochs, best
epoch 17). The model overfits: training loss keeps falling while the validation error rises after about 10-20 epochs
(9.5 M parameters, no dropout or weight decay in the paper). Kept as reconstructed (no tuning of a comparison method).

**Aurora-BP (5-fold CV, our 21,280 test recordings, pooled), calibration-free group (R8):**

| Method | SBP MAE | SBP ME ± SD | DBP MAE | DBP ME ± SD | vs ours (subject level): Δ MAE SBP / DBP, % better, p |
|---|---|---|---|---|---|
| rU-Net | 13.34 | -1.55 ± 17.27 | 9.43 | -0.61 ± 12.00 | SBP +0.38 (45%, p 0.007); DBP +0.35 (43%, p < 0.001) |
| HiCAF-BP, uncalibrated (ours) | 12.66 | +0.01 ± 16.41 | 8.91 | +0.57 ± 11.31 | – |
| ResNet1D | 12.91 | +0.50 ± 16.66 | 9.17 | +0.35 ± 11.67 | SBP +0.27 (46%, p 0.012); DBP +0.27 (47%, p 0.003) |
| Calibration value only (k = 1, reference) | 12.12 | +1.39 ± 16.50 | 9.27 | +4.97 ± 11.64 | – |

Our uncalibrated model is the best calibration-free method on Aurora (paired subject-level tests above).

**Zero-shot PulseDB (ensemble of the 5 fold models; identical target segments), calibration-free group (R10):**

| Data | rU-Net SBP / DBP MAE | Our uncalibrated model | ResNet1D | Calibration value only (k = 1) |
|---|---|---|---|---|
| combined | 18.69 / 11.56 | 24.89 / 18.25 | 24.22 / 21.24 | 17.87 / 9.55 |
| MIMIC | 19.93 / 12.75 | 24.14 / 18.09 | 23.36 / 24.14 | 15.48 / 7.96 |
| VitalDB | 17.73 / 10.63 | 25.47 / 18.37 | 24.89 / 18.99 | 19.73 / 10.79 |

rU-Net is the best calibration-free method zero-shot (smaller bias, +6.9 vs +19 mmHg SBP), plausibly through its
demographic inputs (available for VitalDB). No calibration-free method beats the calibration value alone in the new
population. Paired tests: `runs/zero_shot_calfree/CALIBRATION_FREE_ZERO_SHOT.md`; full rU-Net zero-shot tables (which compare
it with our calibrated model, for reference only): `runs/pub_runet/external_pulsedb/EXTERNAL_RESULTS.md`.

## 10. Addendum (2026-10-12, after UTransBPNet)

Section 9 suggested that rU-Net's smaller zero-shot bias came 'plausibly through its demographic inputs'. UTransBPNet uses no demographics and also has a much smaller zero-shot bias than our uncalibrated model (+9.1 vs +19.2 mmHg SBP), so demographics cannot be the whole explanation. All calibration-free methods have near-zero zero-shot correlation (rU-Net SBP r 0.11); their zero-shot MAE differences reflect mainly where their predictions land relative to the new population's average, not tracking. The section 9 numbers are unchanged.
