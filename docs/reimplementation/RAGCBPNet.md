# RAGCBPNet — re-implementation specification (supplementary material)

Status: **5-fold CV and zero-shot PulseDB complete; frozen (tag v1-pub-ragcbpnet).** Common protocol: `docs/REIMPLEMENTATION_PLAN.md` (R1–R11).

## 1. Sources

| Source | Used for |
|---|---|
| Huang et al., "RAGCBPNet: An Efficient Feature Fusion ...", medRxiv preprint, 2025. doi 10.1101/2025.10.30.25339198 (not peer-reviewed) | everything (Sec. 3.2–3.4, Fig. 3, Eq. 1) |
| Official code | **none published** — reconstructed from the text and Fig. 3; every unspecified detail is ours |
| A. Cisnal et al., IEEE JBHI 28(10):5769–5781, 2024; VitalPy (github.com/SCAI-Lab/VitalPy, commit 1a7ae80dbc225111258cd468959270ba90ad0bf1) | the 476 handcrafted features of the preprint — attempted, not usable (section 7) |

## 2. Method as described in the preprint

- Aurora-BP static (auscultatory) readings only, quality-filtered; PPG only; 30 s at 500 Hz; Butterworth 0.25–10 Hz + airPLS baseline correction; 476 handcrafted features (Cisnal et al. / VitalPy: 404 time-domain, 58 frequency-domain, statistical); signals down-sampled to 125 Hz, min–max scaled; PPG, VPPG, APPG as network input.
- ResNet50 with a CBAM module → deep features; stage 1: self-attention fuses deep and handcrafted features; stage 2: gate (Eq. 1: Gate = FC1(G)·σ(FC2(G)) + G·(1 − σ(FC2(G)))) on the fused representation with age, gender, height, weight, BMI, baseline SBP/DBP and circadian (cos/sin) encodings of the acquisition hour; regression head: [gate output, baseline SBP/DBP] → FC → ReLU → FC.
- Baseline BP: the first BP value of the calibration period. Loss MAE; Adam, learning rate and weight decay 1e-3; batch 128; 200 epochs; early stopping (patience 20); best validation model. 5-fold subject-wise CV.
- A separate hypertension classifier (contrastive pre-training, triplet loss) is not part of the BP regression and is not used.

## 3. Decisions (agreed before any result)

| # | Decision | Chosen | Reason |
|---|---|---|---|
| G1 | Group and baseline | calibrated; training baseline = the **first** calibration reading (k = 1); evaluated with k = 1 (as trained) and k = 3 | preprint ("first BP value") |
| G2 | Signals | **PPG only** (PPG, VPPG, APPG computed from our processed PPG), as published | the method is PPG-only; our PPG-only ablation (A8) is the like-for-like reference |
| G3 | Window | our common 10 s at 125 Hz | R3 |
| G4 | Handcrafted features | **our frozen 31 ECG/PPG features** (`data_processed/features_pwa/aurora_v1.csv`, the HGCTNet/BpBLS extractor) instead of the 476 VitalPy features — see section 7 | VitalPy could not be run on our data in our environment (option A, agreed) |
| G5 | Feature selection | none | preprint |
| G6 | Side inputs | sex, age, height [cm], weight [kg], BMI (Aurora's imperial units converted), cos/sin of the acquisition hour (Aurora `date_time`; unavailable in PulseDB → training mean), baseline SBP/DBP; all standardised with training statistics | preprint; R7 |
| G7 | Unspecified architecture details | section 4 | no code |
| G8 | Training | MAE loss on standardised targets; Adam lr 1e-3, weight decay 1e-3; batch 128; max 200 epochs; early stopping after 20 epochs without validation improvement; best validation epoch | preprint + standardised targets (not stated, as rU-Net) |
| G9 | Training data | our fold's training subjects (with calibration readings), all readings (in-lab and ambulatory) | R1 (same data for every method) |

## 4. Architecture as implemented (ours where the preprint is silent)

| Part | Implementation | Source |
|---|---|---|
| Input | PPG, VPPG, APPG (derivatives ×125), each min–max scaled per window | preprint; per-window scope ours |
| Backbone | 1-D ResNet50: stem conv 7 (stride 2) + max-pool; bottleneck blocks 3-4-6-3, widths 64-128-256-512, expansion 4 → 2048 channels × 40 positions | preprint (ResNet50); 1-D form ours |
| CBAM | channel (reduction 16) then spatial (kernel 7) attention on the backbone output | preprint (Fig. 3: after ResNet50); defaults ours |
| Stage 1 fusion | tokens = 40 deep tokens (2048 → 256) + 1 handcrafted token (31 → 256); 4-head self-attention (written out explicitly, no fused inference path), residual, LayerNorm, mean over tokens → 256 | preprint (self-attention fusion); token design ours |
| Stage 2 gate | G = [fused (256), side (9)] → Eq. 1 with FC1, FC2: 265 → 265 | preprint |
| Regression head | [gate output (265), baseline SBP/DBP (2)] → FC(128) → ReLU → FC(2) | preprint; width ours |

## 5. Evaluation

Aurora 5-fold CV on our model's test recordings (k = 1, 3); zero-shot PulseDB with `scripts/eval_published_external.py`
(same target segments; hour of day and missing demographics → training statistics); calibrated report (R11).

## 6. Deviations from the preprint (complete list)

| # | Preprint | Ours | Reason |
|---|---|---|---|
| D1 | 476 VitalPy features | our 31 ECG/PPG features (only the PPG-derived ones carry PPG information; the 3 PTT features use the ECG R peak) | section 7 (G4) |
| D2 | 30 s at 500 Hz, own filter (0.25–10 Hz) + airPLS | our shared 10 s / 125 Hz preprocessing | R1, R3 |
| D3 | static readings only, quality-filtered | all readings of our folds | R1 |
| D4 | label scale not stated | standardised targets | G8 |
| D5 | unspecified architecture details | section 4 | no code |

Note on D1: the 31-feature set contains three ECG-based pulse arrival times; with G4 these are the only ECG-derived inputs of
the model (the network input is PPG only).

## 7. VitalPy attempt (why G4 changed)

VitalPy (official package of Cisnal et al.) was cloned (commit above) and tried on 20 Aurora, 10 MIMIC and 10 VitalDB windows
of our preprocessed data, in our unchanged environment:
1. VitalPy assumes a fixed 500 Hz (`params.PPG_SAMPLE_RATE`, `PPGSignal.sampling_rate`) and ignores the time column; on our
   125 Hz windows every pulse appears 4× shorter, fails its 0.3 s minimum pulse cycle, and no window yields features
   ("NO TEMPLATES"). Re-sampling the windows to 500 Hz passed this stage.
2. It then failed on library-version incompatibilities with our environment (`scipy.integrate.trapz` removed in newer scipy;
   `.to_numpy()` on numpy arrays with pandas 2.x), and its quality gate (≥ 6 valid pulses, designed for 30 s recordings)
   rejected most remaining 10 s windows (VitalDB: 10 of 10).
Patching the package or changing the environment was rejected; the method's fusion architecture is kept with our validated
31-feature set. Side effect recorded: installing VitalPy's dependencies upgraded numpy (1.26.4 → 2.2.6); numpy was restored to
1.26.4 before any further run. VitalPy's own new dependencies (seaborn, BaselineRemoval, biosppy, opencv-python, pywavelets,
PeakUtils, bidict, mock, shortuuid) remain installed and are not used by our code.

## 8. Implementation notes (filled during implementation)

- `python -m src.published.ragcbpnet`: input 3 x 1250 -> ResNet50 + CBAM 2048 x 40 -> 41 attention tokens x 256 -> gate 265 -> 2 outputs; 17,453,573 parameters.
- Smoke test: hour of day available for 100% of Aurora training rows; k = 1 and k = 3 results differ (calibrated); about 1 s per epoch on 617 recordings -> roughly 30-40 s per epoch on a full fold.

## 9. Provenance

- Preprint only (no official code); VitalPy commit 1a7ae80 (not used, section 7).
- Our code: `src/published/ragcbpnet.py`, training via `scripts/train_published.py --method ragcbpnet`

## 10. Results (frozen)

**Training.** All folds ended by early stopping (patience 20; 32-56 epochs); no non-finite outputs.

**Aurora-BP (5-fold CV, our 21,280 test recordings, pooled), calibrated group (R11), k = 1 (k = 3):**
RAGCBPNet SBP 10.27 (9.83) / DBP 7.17 (6.92) mmHg; our v1 8.90 (8.48) / 6.44 (6.22); paired per subject, ours better by
1.47 / 0.68 mmHg (71 % / 66 % of subjects, p < 0.001); in-lab 9.07 / 6.11 vs ours 7.63 / 5.67; home 11.60 / 8.33 vs 10.31 / 7.29.

**PPG only vs PPG only** (`paper/tables/calibrated/PPG_ONLY_COMPARISON.md`): our model with PPG only (ablation A8)
9.24 / 6.43 mmHg vs RAGCBPNet 10.27 / 7.17; ours better by 1.06 / 0.70 mmHg per subject (66 % of subjects, p ≈ 1e-33;
k = 3: 1.11 / 0.74, 69 %). The advantage is therefore not due to the ECG.

**Zero-shot PulseDB (ensemble, identical target segments), k = 1:** combined 16.45 / 9.18 (ours 15.63 / 10.59);
MIMIC 14.27 / 8.50 (ours 15.18 / 10.68); VitalDB 18.15 / 9.72 (ours 15.98 / 10.53). SBP not different from ours overall
(p = 0.18), ours better in VitalDB (-1.9 mmHg); RAGCBPNet better for DBP and in MIMIC (smaller bias by staying closer to the
baseline), as for the other calibrated methods. Full tables: `runs/report_calibrated/CALIBRATED_COMPARISON.md`.
