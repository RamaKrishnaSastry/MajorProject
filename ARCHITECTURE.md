# DR-ASPP-DRN Architecture

## Overview

The **DR-ASPP-DRN** (Diabetic Retinopathy – Atrous Spatial Pyramid Pooling – Dual-head Retinal Network) is a multi-task deep learning architecture for joint detection of:

- **DR** (Diabetic Retinopathy) severity – continuous regression output (0–4 scale)
- **DME** (Diabetic Macular Edema) grade – ordinal 4-class classification (No DME / Mild / Moderate / Severe)

---

## Architecture Diagram

```
Input (512×512×3)
       │
       ▼
┌─────────────────────────────────────┐
│  ResNet50 Backbone (ImageNet)        │
│  - 23M parameters                   │
│  - Output: 16×16×2048 feature map   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  ASPP Module (Atrous Spatial        │
│  Pyramid Pooling)                   │
│                                     │
│  Branch 0: 1×1 conv                 │
│  Branch 1: 3×3 dilation rate 6      │
│  Branch 2: 3×3 dilation rate 12     │
│  Branch 3: 3×3 dilation rate 18     │
│  Branch 4: Global Average Pool      │
│                                     │
│  → Concatenate → 1×1 projection     │
│  Output: 16×16×256                  │
└──────────┬──────────────────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────────┐
│ DR Head │  │    DME Head      │
│  GAP    │  │     GAP          │
│  Dense  │  │  Dense(256,relu) │
│  256    │  │  Dropout(0.4)    │
│  Drop   │  │  Dense(4,softmax)│
│ sigmoid │  │                  │
│ (0–1)   │  │ [No/Mild/Mod/Sev]│
└─────────┘  └──────────────────┘
```

---

## Key Components

### 1. ResNet50 Backbone

| Property      | Value           |
|---------------|-----------------|
| Architecture  | ResNet50        |
| Pre-training  | ImageNet        |
| Output shape  | 16×16×2048 (at 512×512 input) |
| Parameters    | ~23.5 M         |
| Trainable     | Yes (full fine-tuning after warm-up) |

The backbone extracts rich hierarchical features from fundus images. ImageNet pre-training provides strong low-level edge/texture features that transfer well to medical images.

### 2. ASPP Module

Atrous Spatial Pyramid Pooling captures multi-scale context by applying dilated convolutions at multiple receptive field sizes:

| Branch | Operation           | Dilation Rate | Receptive Field |
|--------|---------------------|---------------|-----------------|
| 0      | 1×1 Conv            | 1             | Local           |
| 1      | 3×3 Dilated Conv    | 6             | Medium          |
| 2      | 3×3 Dilated Conv    | 12            | Large           |
| 3      | 3×3 Dilated Conv    | 18            | Very large      |
| 4      | Global Average Pool | Global        | Whole image     |

This is critical for DR/DME detection because lesion sizes vary enormously (microaneurysms vs. hard exudate clusters).

### 3. DR Head (Regression)

```
GAP → Dense(256, ReLU) → Dropout(0.4) → Dense(1, sigmoid) → [0, 1]
```

- Sigmoid output maps to DR severity 0–4 (multiply by 4 for clinical score)
- Auxiliary task: regularises the shared ASPP representation
- Loss: Mean Squared Error (MSE) with weight 0.3 in multi-task loss

### 4. DME Head (Classification)

```
GAP → Dense(256, ReLU) → Dropout(0.4) → Dense(4, softmax) → [No, Mild, Mod, Severe]
```

- Softmax over 4 ordinal classes
- Loss: Ordinal-weighted Cross-Entropy (default) or standard Categorical Cross-Entropy
- Main evaluation metric: **Quadratic Weighted Kappa (QWK)**

---

## Training Strategy

### Phase 1: Backbone Feature Learning (Stage 1)
- All layers unfrozen, learning rate `1e-4`
- Adam optimiser with QWK-aware LR reduction
- Ordinal-weighted cross-entropy loss
- Early stopping based on `val_qwk` (patience=5)

### Phase 2: Fine-Tuning (Stage 2)
- Continue from stage 1 best weights
- Lower learning rate `5e-5`
- Stricter LR reduction factor (0.3)
- Targets marginal QWK improvement

---

## Loss Function

### Multi-Task Loss

```
L_total = 0.0 × L_DR + 1.0 × L_DME
```

During DME fine-tuning, the DR head contribution is zeroed. During joint pre-training:

```
L_total = 0.3 × MSE(dr_pred, dr_true) + 1.0 × OrdinalCE(dme_pred, dme_true)
```

### Ordinal Cross-Entropy

The DME loss uses an ordinal penalty matrix to penalise distant misclassifications more:

```
L_ordinal = CE(y, ŷ) × w_penalty(true_class, pred_class)

where w_penalty(i, j) = 1 + (i-j)² / (K-1)²
```

This ensures the model learns ordinal structure (Mild→Severe errors penalised more than Mild→Moderate).

---

## Quadratic Weighted Kappa (QWK)

The primary evaluation metric:

```
QWK = 1 - Σ(w_ij × O_ij) / Σ(w_ij × E_ij)

where:
  w_ij = (i-j)² / (K-1)²   (quadratic penalty)
  O_ij = observed confusion matrix (normalised)
  E_ij = expected under random agreement
```

| QWK Range | Clinical Interpretation |
|-----------|------------------------|
| < 0.40    | Poor agreement         |
| 0.40–0.60 | Fair agreement         |
| 0.60–0.75 | Moderate agreement     |
| 0.75–0.80 | Good agreement         |
| ≥ 0.80    | **Excellent (target)** |

---

## Ablation Study Summary

| Model                 | Architecture          | Expected QWK |
|-----------------------|-----------------------|--------------|
| Model A (Baseline)    | ResNet50 only         | ~0.73–0.75   |
| Model B (ASPP only)   | ResNet50 + ASPP       | ~0.76–0.78   |
| Model C (Full RFA)    | DR-ASPP-DRN (full)    | **≥ 0.80**   |

Each component (ASPP multi-scale context + ordinal loss + multi-task learning) contributes approximately +0.02–0.04 QWK improvement.
