# UTransBPNet — re-implementation specification (supplementary material)

Status: **5-fold CV and zero-shot PulseDB complete; frozen (tag v1-pub-utransbpnet).** Common protocol: `docs/REIMPLEMENTATION_PLAN.md` (R1–R10).

## 1. Sources

| Source | Used for |
|---|---|
| Y. Zheng et al., "UTransBPNet for cuffless and calibration-free blood pressure estimation under dynamic conditions," Scientific Reports 15:17654, 2025. DOI 10.1038/s41598-025-02963-3 | everything (Methodology, Fig. 2, Supplementary Table S1 not available to us) |
| Official code | **none published** (data "available on request"; no code repository) — reconstructed from the text and Fig. 2; every unspecified detail is our choice, marked "ours" |
| O. Petit et al., "U-Net Transformer: self and cross attention for medical image segmentation," MLMI 2021 (ref. [27] of the paper) | the multi-head cross-attention (MHCA) design the paper adopts |

## 2. Method as described in the paper

- Input: ECG, PPG and the first and second derivatives of the PPG (VPPG, APPG): 5 s at 125 Hz, shape 625 × 4; each signal min–max scaled to (0, 1); ECG band-pass 0.5–30 Hz, PPG 0.5–15 Hz (Butterworth).
- Improved U-Net encoder (Fig. 2): Conv_SE blocks (three Conv_blocks = Conv1D kernel 7 → BatchNorm → ReLU, then a squeeze-and-excitation block = average pooling → Conv1D 1×1 → ReLU → Conv1D 1×1 → sigmoid, element-wise addition), mean-pooling for three down-sampling steps (625 → 125 → 25 → 5), the input copied to every scale (Fig. 2, "Copy"); channel widths in Fig. 2 approximately 128, 256, 384, 512.
- Transformer: 12 encoder layers on the deepest feature map (5 positions × 512), learned positional encoding.
- Decoder: three up-sampling steps back to the input resolution; at each scale a multi-head cross-attention module: the deeper map Y gives queries and keys, the skip map S gives values; the attention output passes a sigmoid to a filter Z that gates S; the gated S is concatenated with the up-sampled Y.
- Output: a 1-channel BP waveform (625 × 1). Variants: Det (peak detection on the waveform), DeepTune (FC on the transformer output), FinalTune (FC on the predicted waveform, network frozen), **AllTune** (FC on the predicted waveform, all parameters trained) — AllTune is the paper's best and reported model.
- Training: phase I reconstructs the BP waveform; phase II trains the FC layer (AllTune: all parameters) on SBP/DBP; Adam, lr 9e-4, batch 32, weight decay 1e-3, early stopping after 20 epochs without improvement; SBP/DBP min–max scaled with the training set's minimum and maximum. Comparison models in the paper were trained end-to-end on SBP/DBP.
- 32.55 M parameters; calibration-free.

## 3. Decisions (agreed before any result)

| # | Decision | Chosen | Reason |
|---|---|---|---|
| T1 | Variant and training | **AllTune architecture trained end-to-end on SBP/DBP** (network → 1-channel output at input resolution → FC → SBP, DBP; all parameters trained) | phase I needs the continuous arterial BP waveform, which Aurora does not have; end-to-end SBP/DBP training is how the paper trained its comparison models |
| T2 | Group | **calibration-free** (R8, R10) | as published |
| T3 | Input signals | ECG, PPG from our shared preprocessing; VPPG, APPG = first / second derivative of our PPG; each channel min–max scaled to (0, 1) per window | R1; paper's channels and scaling; per-window scaling is ours (the paper does not say per window or per dataset) |
| T4 | Window length | the common **10 s / 1,250 samples** (three ×5 poolings: 1250 → 250 → 50 → 10 positions) | R3 |
| T5 | Targets and loss | SBP/DBP min–max scaled with the training minimum / maximum (as the paper); loss MSE (not stated, ours) | paper + ours |
| T6 | Training length | Adam lr 9e-4, batch 32, weight decay 1e-3 (as published); early stopping after 20 epochs without validation improvement (as published), best validation epoch kept; max 200 epochs (ours, not stated) | paper + R6 |
| T7 | Unspecified architecture details | section 4 | no code |
| T8 | Transfer / scenario fine-tuning | not used (zero-shot only) | R7 |
| T9 | Model size | **keep the reconstruction as is: 45,193,671 parameters** (paper 32.55 M); not shrunk to match | every width / setting is from Fig. 2 or a standard default; the 12-layer, 512-wide transformer alone needs about 25 M parameters, so the paper must differ in unreported details — matching the count would require inventing one |

## 4. Architecture as implemented (ours where the paper is silent)

| Part | Implementation | Source |
|---|---|---|
| Conv_block | Conv1D(kernel 7, same padding) → BatchNorm → ReLU | Fig. 2 |
| Conv_SE | 3 Conv_blocks → SE (global average pooling → Conv1D 1×1 to C/16 → ReLU → Conv1D 1×1 → sigmoid; channel re-weighting) → + residual (1 × 1 convolution when the channel number changes) | Fig. 2; reduction 16 and the residual projection ours |
| Encoder | scale 1 (1250): Conv_SE 4 → 128; mean-pool ×5; scale 2 (250): concat input copy (mean-pooled, 4 ch) → Conv_SE → 256; mean-pool ×5; scale 3 (50): concat copy → Conv_SE → 384; mean-pool ×5; scale 4 (10): concat copy → Conv_SE → 512 | Fig. 2 (widths, copy, mean-pooling); exact concatenation points ours |
| Transformer | 12 pre-LayerNorm encoder layers, width 512, 8 heads, feed-forward 1,024 (×2), dropout 0.1, learned positional embedding (10 positions) | paper (12 layers, learned positions); heads, feed-forward width, dropout ours |
| MHCA (each decoder scale) | Petit et al.: queries / keys from the deeper map Y (1 × 1 projections), values from the skip map S mean-pooled to Y's resolution; attention output → 1 × 1 conv → BatchNorm → sigmoid → up-sampled ×5 = Z; S ⊙ Z concatenated with up-sampled (×5, + Conv_block) Y; then Conv_SE | paper + Petit et al.; head number 8 (ours) |
| Decoder widths | 512 → 384 (scale 3) → 256 (scale 2) → 128 (scale 1) → 1 × 1 conv to 1 channel | Fig. 2 |
| AllTune head | FC(1250 → 2) on the 1-channel output, sigmoid-free (targets in min–max units) | paper (FC on the final layer) |
| Parameter count | reported by the self-test and compared with the paper's 32.55 M | — |

## 5. Evaluation

- Aurora: 5-fold CV on our model's test recordings (calibration recordings excluded); predictions do not depend on k.
- Zero-shot PulseDB (MIMIC, VitalDB): the 5 fold models; same target segments as every other method; no demographic input.
- Calibration-free group (R8): next to our uncalibrated model, ResNet1D and rU-Net; on PulseDB through `scripts/eval_calfree_external.py --report_only` (R10).

## 6. Deviations from the paper (complete list)

| # | Paper | Ours | Reason |
|---|---|---|---|
| D1 | two-phase training (BP waveform, then SBP/DBP) | AllTune architecture trained end-to-end on SBP/DBP | no arterial BP waveform in Aurora (T1) |
| D2 | 5 s windows (625 samples) | 10 s (1,250 samples) | R3 (T4) |
| D3 | ECG 0.5–30 Hz, PPG 0.5–15 Hz Butterworth | shared preprocessing (ECG 0.5–40 Hz, PPG 0.5–10 Hz) | R1 |
| D4 | min–max scaling (scope not stated) | per window and channel | T3 |
| D5 | leave-one-subject-out / ten-fold validation on small dynamic datasets | our 5 frozen Aurora folds | R1 |
| D6 | loss and maximum epochs not stated | MSE; max 200 epochs | T5, T6 |
| D7 | unspecified architecture details | section 4 choices | no code |
| D8 | 32.55 M parameters | 45.19 M | unreported details (T9); difference mostly in the transformer |

## 7. Implementation notes (filled during implementation)

- `src/published/utransbpnet.py`; `python -m src.published.utransbpnet` prints every stage's shape (1250 -> 250 -> 50 -> 10; transformer on 10 positions; decoder back to 1250; 1-channel output; 2 targets). The encoder width 384 + 4 = 388 matches the value printed in Fig. 2.
- Input derivation and per-window min-max scaling inside the model (T3); targets min-max scaled with the training range, converted back to mmHg inside the model (T5).
- Numerical issue found and fixed before any result: in the first full run, validation outputs became non-finite from epoch 2 while the training loss stayed normal. A diagnostic (inputs finite; no corrupted BatchNorm statistics; encoder outputs small) located the first non-finite output in `transformer.layers.2`, and the same batch was finite with the transformer outside PyTorch 2.0.1's fused inference fast path. Disabling the layer's fast path was not enough (the next run failed in `transformer.layers.2.self_attn`, i.e. the fast path of nn.MultiheadAttention). Final fix: the pre-LayerNorm encoder layer is written out explicitly (`EncoderLayer`): the same parameters and the same computation as nn.TransformerEncoderLayer in its standard / training path, verified by the self-test (weights copied from nn.TransformerEncoderLayer, maximum output difference 4.77e-07); parameter count unchanged (45,193,671). Evaluation therefore uses exactly the training computation. The aborted run was deleted; training now stops automatically on any non-finite validation output.
- Smoke test: about 0.16 s per batch of 32 -> about 1.5 min per epoch on a full fold.

## 8. Provenance

- Paper only (no official code).
- Our code: `src/published/utransbpnet.py`, training via `scripts/train_published.py --method utransbpnet`

## 9. Results (frozen)

**Training.** All folds ended by early stopping (patience 20; e.g. 44 and 50 epochs). No non-finite outputs after the
fix in section 7.

**Aurora-BP (5-fold CV, our 21,280 test recordings, pooled), calibration-free group (R8):**

| Method | SBP MAE | SBP ME ± SD | DBP MAE | DBP ME ± SD | vs our uncalibrated model (subject level): Δ MAE SBP / DBP (% better) |
|---|---|---|---|---|---|
| UTransBPNet | 13.73 | -1.21 ± 17.87 | 9.56 | -0.52 ± 12.19 | +1.16 (40%) / +0.68 (39%), both p < 0.001 |
| rU-Net | 13.34 | -1.55 ± 17.27 | 9.43 | -0.61 ± 12.00 | +0.38 (45%) / +0.35 (43%) |
| HiCAF-BP, uncalibrated (ours) | 12.66 | +0.01 ± 16.41 | 8.91 | +0.57 ± 11.31 | – |
| ResNet1D | 12.91 | +0.50 ± 16.66 | 9.17 | +0.35 ± 11.67 | +0.27 (46%) / +0.27 (47%) |
| Calibration value only (k = 1, reference) | 12.12 | +1.39 ± 16.50 | 9.27 | +4.97 ± 11.64 | – |

**Zero-shot PulseDB (ensemble of the 5 fold models; identical target segments), calibration-free group (R10), k-independent:**

| Data | UTransBPNet SBP / DBP | rU-Net | Our uncalibrated model | ResNet1D | Calibration value only (k = 1) |
|---|---|---|---|---|---|
| combined | 18.98 / 16.46 | 18.69 / 11.56 | 24.89 / 18.25 | 24.22 / 21.24 | 17.87 / 9.55 |
| MIMIC | 19.48 / 17.93 | 19.93 / 12.75 | 24.14 / 18.09 | 23.36 / 24.14 | 15.48 / 7.96 |
| VitalDB | 18.60 / 15.32 | 17.73 / 10.63 | 25.47 / 18.37 | 24.89 / 18.99 | 19.73 / 10.79 |

All calibration-free methods have near-zero correlation with the reference BP zero-shot (SBP r 0.00-0.11, DBP r 0.05-0.28);
their zero-shot MAE differences reflect mainly the systematic offset (SBP bias +6.9 rU-Net, +9.1 UTransBPNet, +19.2 ours,
+18.5 ResNet1D). No calibration-free method beats the calibration value alone. Full report:
`runs/zero_shot_calfree/CALIBRATION_FREE_ZERO_SHOT.md`.
