# DR-ASPP-DRN Pipeline — README

## Overview

Complete implementation of the **DR-ASPP-DRN** multi-task pipeline for automated
**Diabetic Retinopathy (DR)** and **Diabetic Macular Edema (DME)** detection from
retinal fundus images.

**Primary metric: Quadratic Weighted Kappa (QWK) ≥ 0.80**

---

## File Structure

```
├── preprocess.py          # Medical fundus image preprocessing (CLAHE, border crop)
├── dataset_loader.py      # IDRiD tf.data pipeline with class-weight handling
├── model.py               # DR-ASPP-DRN architecture (DRN backbone + ASPP)
├── train.py               # QWK-optimised training with custom callbacks
├── evaluate.py            # Comprehensive metrics (QWK, F1, confusion matrix)
├── ablation_study.py      # RFA ablation: Baseline vs ASPP vs DRN+ASPP
├── train_dme_model.ipynb  # End-to-end pipeline notebook
├── config.yaml            # Hyperparameter configuration
├── requirements.txt       # Python dependencies
└── outputs/               # Generated during training
    ├── dme_finetuned.weights.h5
    ├── training_history.json
    ├── evaluation_metrics.json
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── per_class_metrics.png
    ├── metrics_report.txt
    └── training_log.txt
```

---

## Dataset Setup (IDRiD)

Download the **IDRiD Disease Grading** sub-challenge from
[IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
and organise as:

```
dataset/
├── images/
│   ├── IDRiD_001.jpg
│   ├── IDRiD_002.jpg
│   └── ...
└── DME_Grades.csv
    # columns: image_id, dme_grade  (values 0, 1, 2)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train DME Head

```bash
python train.py \
    --config config.yaml \
    --dataset_path /path/to/dataset \
    --output_dir outputs/
```

### 2. Evaluate Trained Model

```python
from evaluate import DMEEvaluator
from dataset_loader import IDRiDDataset

val_ds = IDRiDDataset("/path/to/dataset", split="val").get_dataset()
evaluator = DMEEvaluator(model, val_ds, "outputs/")
metrics = evaluator.run()
print(f"QWK: {metrics['qwk']:.4f}")
```

### 3. Run Ablation Study

```bash
python ablation_study.py \
    --dataset_path /path/to/dataset \
    --output_dir outputs/ablation \
    --epochs 15
```

### 4. End-to-End Notebook

Open `train_dme_model.ipynb` in Jupyter or Kaggle.

---

## Architecture

```
Fundus Image (512×512×3)
        ↓
DRN Backbone (ResNet50 + dilated convolutions)
    conv4: dilation_rate=2
    conv5: dilation_rate=4
    Output: (B, 32, 32, 2048)
        ↓
ASPP Module (multi-scale context fusion)
    Branch 1: 1×1 Conv
    Branch 2: 3×3 @ dilation=6
    Branch 3: 3×3 @ dilation=12
    Branch 4: 3×3 @ dilation=18
    Branch 5: Global context
    Output: (B, 32, 32, 256)
        ↓
       ┌──────────────┬──────────────┐
       │  DR Head     │  DME Head    │
       │  (frozen)    │  (trainable) │
       │  Output: (1) │  Output: (3) │
       └──────────────┴──────────────┘
```

**RFA (Receptive Field Augmentation)** maintains spatial resolution
while expanding the model's effective receptive field — critical for
detecting small retinal lesions (microaneurysms, exudates).

---

## Training Strategy

| Stage | Backbone | ASPP | DR Head | DME Head |
|-------|----------|------|---------|----------|
| Pre-training (EyePACS) | ✅ Trainable | ✅ | ✅ | — |
| DME Fine-tuning (IDRiD) | ❄️ Frozen | ❄️ | ❄️ | ✅ Trainable |

Transfer learning prevents catastrophic forgetting of retinal features
while the DME head adapts to the IDRiD domain.

---

## Expected Results

| Model Variant | QWK (Expected) |
|---------------|----------------|
| Baseline (ResNet50, no ASPP) | ~0.75 |
| RFA-Lite (ResNet50 + ASPP) | ~0.78 |
| **Full RFA (DRN + ASPP)** | **≥0.80** |

Literature benchmarks for comparison:
- Li et al. (2019): 0.82
- Ahammed et al. (2023): 0.84
- Al-Antary et al. (2021): 0.87

---

## Key Design Choices

| Decision | Justification |
|----------|---------------|
| CLAHE preprocessing | Standard in ophthalmology; enhances lesion visibility |
| Border cropping | Removes black acquisition artefacts |
| 512×512 resolution | Balance between lesion detail and compute efficiency |
| Class weight balancing | IDRiD is imbalanced (~49% grade 0); prevents bias |
| QWK early stopping | Optimises the actual clinical metric, not proxy accuracy |
| Frozen backbone | Prevents overfitting on small IDRiD (~516 images) |
