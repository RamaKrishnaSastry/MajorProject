# DR-ASPP-DRN Pipeline – IRDID Edition

## Overview

This repository implements a **multi-task deep learning pipeline** for automated
detection of Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) from
retinal fundus images, following the DR-ASPP-DRN architecture.

```
Fundus Image
      │
      ▼
Preprocessing (border crop → CLAHE → resize → normalise)
      │
      ▼
Backbone  (ResNet50-based DRN, frozen)
      │
      ▼
ASPP  (multi-scale atrous pooling, frozen)
      │
 ┌────┴────┐
 ▼         ▼
DR Head   DME Head  ← trainable (fine-tuning stage)
(dummy)   (3-class softmax)
```

---

## File Structure

```
├── preprocess.py          # Medical fundus image preprocessing
├── dataset_loader.py      # IRDID tf.data pipeline
├── model.py               # DR-ASPP-DRN architecture
├── train.py               # DME head fine-tuning script
├── evaluate.py            # Metrics, confusion matrix, visualisations
├── train_dme_model.ipynb  # End-to-end Jupyter notebook
├── requirements.txt       # Python dependencies
├── config.yaml            # Hyperparameter configuration
└── outputs/               # Generated artefacts
    ├── dme_finetuned.weights.h5
    ├── training_history.json
    ├── evaluation_metrics.json
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── metrics_report.txt
    └── training_log.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### Real IRDID / IDRiD Dataset

1. Download the **IDRiD** dataset (Disease Grading sub-task) from the
   [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
   or Kaggle mirror.
2. Arrange files as:

   ```
   dataset/
   ├── images/
   │   ├── IDRiD_001.jpg
   │   └── ...
   └── DME_Grades.csv     # columns: image_id, dme_grade [, dr_grade]
   ```

3. Update `config.yaml` → `dataset.path`.

### Synthetic / Mock Data (offline testing)

If no dataset is present the pipeline auto-generates 100 synthetic samples
the first time it runs, so you can validate the full pipeline immediately.

---

## Quick Start

### 1 – Preprocess a single image

```python
from preprocess import preprocess_fundus_image
img = preprocess_fundus_image("path/to/fundus.jpg", target_size=(512, 512))
print(img.shape, img.min(), img.max())  # (512, 512, 3)  0.0  1.0
```

### 2 – Build the tf.data pipeline

```python
from dataset_loader import IDRiDDataset
train_ds = IDRiDDataset("dataset", split="train").get_dataset()
val_ds   = IDRiDDataset("dataset", split="val").get_dataset()
```

### 3 – Train the DME head

```bash
python train.py \
    --dataset_path dataset \
    --weights_path pretrain_final.weights.h5 \
    --epochs 30 \
    --output_dir outputs
```

### 4 – Evaluate

```bash
python evaluate.py \
    --dataset_path dataset \
    --weights_path outputs/dme_finetuned.weights.h5 \
    --output_dir outputs
```

### 5 – Notebook

Open `train_dme_model.ipynb` in Jupyter or Google Colab and run all cells.

---

## Preprocessing Details

| Step | Description |
|---|---|
| Border crop | Removes dark acquisition borders (threshold=15) |
| CLAHE | Contrast Limited Adaptive Histogram Equalization on green channel (clip=2.0, grid=8×8) |
| Resize | Bilinear resize to 512×512 |
| Normalise | Scale to [0, 1] float32 |

---

## Model Architecture

| Component | Details |
|---|---|
| Backbone | ResNet50, Block 3 output (conv4_block6_out) → 1024 ch, H/16 |
| ASPP | 1×1, 3×3@r6, 3×3@r12, 3×3@r18, GAP → merge 256 ch |
| DR Head | GAP → Dense(256) → Dropout(0.4) → Dense(1, relu) |
| DME Head | GAP → Dense(256) → Dropout(0.4) → Dense(3, softmax) |

---

## Training Strategy

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 30 (EarlyStopping patience=5) |
| Loss (dme_risk) | categorical_crossentropy |
| Loss (dr_output) | MSE weight=0 (dummy, frozen) |
| Frozen layers | Backbone + ASPP |
| Trainable layers | dme_gap, dme_fc1, dme_dropout, dme_risk |

---

## Evaluation Metrics

| Metric | Target |
|---|---|
| Overall Accuracy | > 0.70 |
| Macro F1-score | ≥ 0.70 |
| Quadratic Weighted Kappa | > 0.60 |

---

## Outputs

| File | Description |
|---|---|
| `dme_finetuned.weights.h5` | Best model weights |
| `training_history.json` | Loss and accuracy per epoch |
| `evaluation_metrics.json` | Full metric dictionary |
| `confusion_matrix.png` | Normalised & raw count heatmaps |
| `per_class_metrics.png` | Precision / Recall / F1 bar chart |
| `class_distribution.png` | Pie chart of label distribution |
| `training_curves.png` | Loss and accuracy learning curves |
| `metrics_report.txt` | Human-readable classification report |
| `training_log.txt` | Full training log |
