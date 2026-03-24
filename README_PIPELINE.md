# IRDID DME Pipeline — Documentation

Multi-task Diabetic Retinopathy (DR) + DME Detection System — Stage 1 implementation.

---

## Overview

This pipeline fine-tunes only the **DME head** of a pretrained multi-task model on the
IRDID (Indian Retinal Image Dataset) fundus photographs.  All other components — backbone,
ASPP, and DR head — remain frozen.

```
Raw Images ──► Preprocessing ──► tf.data pipeline
                                        │
                                  Batch creation
                                        │
                               DME head training
                                        │
                             Validation & Evaluation
                                        │
                              Metrics + Visualisations
```

---

## Repository Layout

```
├── preprocess.py          # Medical image preprocessing
├── dataset_loader.py      # IRDID dataset tf.data pipeline
├── model.py               # ResNet50 + ASPP + dual-head architecture
├── train.py               # Training orchestration & callbacks
├── evaluate.py            # Evaluation metrics & confusion matrix
├── train_dme_model.ipynb  # End-to-end training notebook
├── requirements.txt       # Python dependencies
└── README_PIPELINE.md     # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the IRDID dataset in a directory, e.g. `data/IRDID/`, with:

```
data/IRDID/
├── DME_Grades.csv
└── images/
    ├── IDRiD_001.jpg
    ├── IDRiD_002.jpg
    └── ...
```

`DME_Grades.csv` columns (flexible detection):

| Image name  | Risk of macular edema |
|-------------|----------------------|
| IDRiD_001   | 0                    |
| IDRiD_002   | 2                    |

DME label encoding: **0** = No DME · **1** = Mild · **2** = Moderate · **3** = Severe

### 3. Train the DME head

```bash
# With real data
python train.py \
  --csv data/IRDID/DME_Grades.csv \
  --image-dir data/IRDID/images \
  --weights pretrain_final.weights.h5 \
  --epochs 30 \
  --batch-size 8 \
  --output dme_finetuned.weights.h5

# Quick smoke-test with mock data
python train.py --mock --epochs 2
```

### 4. Evaluate the model

```bash
python evaluate.py \
  --weights dme_finetuned.weights.h5 \
  --csv data/IRDID/DME_Grades.csv \
  --image-dir data/IRDID/images \
  --output-dir results/
```

### 5. Use the training notebook

```bash
jupyter notebook train_dme_model.ipynb
```

---

## Module Reference

### `preprocess.py`

| Function | Description |
|---|---|
| `preprocess_image(image)` | Full pipeline: crop → CLAHE → resize → normalize |
| `load_and_preprocess(path)` | Load from disk and preprocess |
| `make_preprocess_fn(...)` | Returns a `tf.data`-compatible preprocessing function |

**Preprocessing steps:**
1. **Border crop** — remove 10% black edges common in fundus photos
2. **CLAHE** — per-channel histogram equalisation (clip=2.0, grid=8×8)
3. **Resize** — bicubic interpolation to (512, 512)
4. **Normalize** — scale to [0, 1]

### `dataset_loader.py`

| Function | Description |
|---|---|
| `load_dme_csv(csv, image_dir)` | Parse CSV and resolve image paths |
| `build_datasets(csv, image_dir, ...)` | Create `(train_ds, val_ds, class_weights)` |
| `create_mock_dataset(dir, n)` | Generate synthetic data for testing |
| `compute_dme_class_weights(labels)` | Compute balanced class weights |

### `model.py`

| Function | Description |
|---|---|
| `build_backbone(shape, weights)` | ResNet50 feature extractor |
| `build_aspp(x, filters)` | Multi-scale dilated convolution block |
| `build_model(...)` | Full multi-task DR + DME model |
| `build_model_dme_tuning(...)` | DME-only trainable model for fine-tuning |

**Architecture:**

```
Input (512×512×3)
    └─► ResNet50 (backbone, frozen during fine-tuning)
            └─► ASPP (256 filters, frozen)
                    ├─► DR Head  → sigmoid output (frozen)
                    └─► DME Head → 4-class softmax (TRAINABLE)
```

### `train.py`

| Function | Description |
|---|---|
| `compile_dme_model(model, lr)` | Compile with Adam + categorical_crossentropy |
| `build_callbacks(...)` | Checkpoint + EarlyStopping + ReduceLROnPlateau |
| `train(train_ds, val_ds, ...)` | Run training loop, save weights |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss (DME) | categorical_crossentropy |
| Batch size | 8 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| Monitor metric | val_dme_risk_accuracy |

### `evaluate.py`

| Function | Description |
|---|---|
| `get_predictions(model, ds)` | Run inference, return true/pred arrays |
| `compute_accuracy(y, ŷ)` | Overall accuracy |
| `compute_f1(y, ŷ, average)` | Macro/weighted F1 |
| `compute_roc_auc(y, proba)` | Per-class + macro ROC-AUC |
| `plot_confusion_matrix(...)` | Save confusion matrix PNG |
| `evaluate(model, ds, ...)` | Full evaluation → JSON + PNG |

---

## Output Files

| File | Description |
|---|---|
| `dme_finetuned.weights.h5` | Trained DME head weights |
| `training_history.json` | Per-epoch loss/accuracy logs |
| `training_log.txt` | CSV training log (epoch-level) |
| `evaluation_metrics.json` | Accuracy, F1, ROC-AUC, per-class stats |
| `confusion_matrix.png` | Normalised confusion matrix heatmap |
| `checkpoints/dme_best.weights.h5` | Best checkpoint (by val accuracy) |

---

## Kaggle Dataset Paths (Reference)

When running on Kaggle, dataset paths follow this convention:

```python
# IDRiD dataset
CSV_PATH   = "/kaggle/input/idrid/B. Disease Grading/2. Groundtruths/b. DIARETDB1 DME Grading.csv"
IMAGE_DIR  = "/kaggle/input/idrid/B. Disease Grading/1. Original Images/b. Testing Set"

# EyePACS (backbone pretraining reference)
EYEPACS_DIR = "/kaggle/input/diabetic-retinopathy-detection/train"
```

---

## Class Distribution

Expected IRDID distribution (handles imbalance automatically):

| Class | Label | Expected % |
|---|---|---|
| No DME | 0 | 60–70 % |
| Mild | 1 | 15–20 % |
| Moderate | 2 | 8–12 % |
| Severe | 3 | 2–5 % |

Class weights are computed automatically using scikit-learn's `compute_class_weight("balanced", ...)`.

---

## Success Criteria

- ✅ All 5 modules importable
- ✅ Dataset pipeline works with real and mock data
- ✅ Model trains without errors
- ✅ Training convergence (val_loss decreases)
- ✅ Evaluation metrics computed and visualised
- ✅ F1-score > 0.70 on balanced assessment
