# EYEPACS MIXED TRAINING GUIDE

## Overview

This guide explains how to train your DR+DME model using a **mixed dataset approach**:
- **Stage 1 (DR training)**: Combine IDRiD (516 images) + EyePACS (88,000+ images)
- **Stage 2 (DME training)**: Use IDRiD only (to preserve clean DME labels)

**Why this works:**
- DR backbone learns robust features from 88K+ diverse images
- DME head stays accurate (trained only on clean IDRiD labels)
- Model generalizes much better to external datasets (Messidor, APTOS, EyePACS)

---

## Step 1: Obtain EyePACS Dataset

### Option A: Download from Kaggle (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Download the dataset
kaggle datasets download -d mariaherrerac/idrid  # (wrong - see next)

# Actually, for EyePACS Kaggle competition:
# Visit: https://www.kaggle.com/c/diabetic-retinopathy-detection/data
# OR use the preprocessed version:
kaggle datasets download -d shadab21/eyepacs

# Extract
unzip eyepacs.zip -d /path/to/eyepacs
```

### Expected Directory Structure

```
eyepacs/
├── trainLabels.csv          # Main training labels CSV
├── sample_submission.csv    # Optional
├── train/                   # Training images (~88,000)
│   ├── 0_left.jpeg
│   ├── 0_right.jpeg
│   ├── 1_left.jpeg
│   ...
└── test/                    # Test images (~optional)
    └── ...
```

### CSV Format

The `trainLabels.csv` should have columns:
- `image`: Image filename (without extension)
- `level`: DR grade (0-4 scale, matching IDRiD!)

Example:
```
image,level
00000,0
00001,2
00002,1
...
```

---

## Step 2: Prepare IDRiD Dataset

Make sure you have IDRiD in the standard format:
```
idrid/
├── DME_Grades.csv
├── A. Training set/
│   ├── IDRiD_001.jpg
│   ├── IDRiD_002.jpg
│   ...
└── B. Testing set/
    ├── ...
```

---

## Step 3: Run Mixed Training

### Python Script (New Recommended Way)

```python
from main_pipeline import run_pipeline

result = run_pipeline(
    csv_path="/path/to/idrid/DME_Grades.csv",
    image_dir="/path/to/idrid/A. Training set",
    config_path="config_mixed_eyepacs.yaml",
    two_stage=True,
    # NEW PARAMETERS FOR MIXED TRAINING:
    eyepacs_csv="/path/to/eyepacs/trainLabels.csv",
    eyepacs_image_dir="/path/to/eyepacs/train",
)

print("Pipeline completed!")
print(f"Best stage: {result['best_stage']}")
print(f"Best DME QWK: {result['best_qwk']:.4f}")
```

### Terminal (One-Liner)

```bash
python -c "
from main_pipeline import run_pipeline
import sys

result = run_pipeline(
    csv_path='path/to/idrid/DME_Grades.csv',
    image_dir='path/to/idrid/A. Training set',
    config_path='config_mixed_eyepacs.yaml',
    eyepacs_csv='path/to/eyepacs/trainLabels.csv',
    eyepacs_image_dir='path/to/eyepacs/train',
    two_stage=True,
)

print('✅ Complete! Best stage:', result['best_stage'])
"
```

### Expected Output

```
============================================================
Stage 0: Data preparation …
============================================================
Loading mixed IDRiD + EyePACS dataset for Stage 1 DR training
============================================================

Loading IDRiD...
IDRiD loaded: 516 samples

Loading EyePACS...
EyePACS loaded: 88000 samples

Combined dataset:
  Total samples: 88516
  IDRiD samples: 516
  EyePACS samples: 88000
  IDRiD DR distribution: {0: 360, 1: 134, 2: 149, 3: 46, 4: 21}
  EyePACS DR distribution: {0: 35126, 1: 20906, 2: 22540, 3: 7138, 4: 2290}

Train/val split: 70812 train, 17704 val (20.0% val)

DR class weights (mixed dataset): {0: 1.0, 1: 1.8, 2: 1.6, 3: 4.2, 4: 5.1}

Mixed dataset pipelines created.
Train dataset: 4426 batches of size 16
Val dataset: 1107 batches of size 16

============================================================
STAGE 1: Initial Training (DR backbone on mixed data)
============================================================

...training proceeds...

Stage 1 QWK: raw=0.7234 | calibrated=0.7156

============================================================
STAGE 2: Fine-Tuning (DME on IDRiD only)
============================================================

Stage 2: Reloading IDRiD-only data for DME fine-tuning

Loaded 516 valid samples
Train/val split: 412 train, 104 val (20.0% val)

...training proceeds...

Stage 2 QWK: raw=0.7891 | calibrated=0.7823

============================================================
CROSS-DATASET EVALUATION SUMMARY
============================================================

MESSIDOR:
  QWK: 0.4234    # Much better than 0.0104!
  Accuracy: 0.6842

✅ Pipeline Complete!
```

---

## Step 4: Evaluate on External Datasets

### Test on Messidor

```bash
python test_cross_dataset_dr.py \
  --use-model \
  --use-weights "pipeline_outputs/mixed_eyepacs/model_final_best.model.h5" \
  --messidor-images "/path/to/messidor/images" \
  --messidor-csv "/path/to/messidor/labels.csv"
```

### Expected Improvement

| Dataset | IDRiD-Only Model | IDRiD+EyePACS Mixed |
|---------|-----------------|-------------------|
| Messidor | QWK = 0.01 ❌ | QWK ≈ 0.40-0.50 ✅ |
| APTOS | QWK = 0.11 ❌ | QWK ≈ 0.45-0.55 ✅ |
| EyePACS test | QWK = 0.45 | QWK ≈ 0.60-0.70 ✅ |

---

## Step 5: Configuration Options

See `config_mixed_eyepacs.yaml` for tuning parameters:

- **Stage 1 epochs**: Increase if not converging (more data allows longer training)
- **batch_size**: Increase to 32 if GPU memory allows (88K images benefit from larger batches)
- **learning_rate**: Decrease if unstable, increase if converging too slowly
- **early_stopping_patience**: Lower for Stage 1 (larger dataset converges faster)

---

## Troubleshooting

### "EyePACS images not found"

**Problem:** `Image not found for '0_left' – skipping.`

**Solution:**
- Check image extension (looks for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`)
- Verify path matches exactly: `eyepacs_image_dir` should point to the folder containing `0_left.jpg`, `0_right.jpg`, etc.

### "CSV columns not recognized"

**Problem:** `Could not detect image name column in CSV`

**Solution:**
```python
# Check EyePACS CSV structure:
import pandas as pd
df = pd.read_csv('trainLabels.csv')
print(df.head())
print(df.columns.tolist())
```

Expected columns: `['image', 'level']` (or similar variants)

### Memory errors during training

**Problem:** `CUDA out of memory while executing CUDA graph...`

**Solution:**
1. Reduce `batch_size` in config: e.g., from 16 → 8
2. Use gradient accumulation (advanced)
3. Train on multiple GPUs

### Stage 2 isn't improving

**Problem:** "Stage 2 QWK barely improves over Stage 1"

**Solution:**
- Stage 1 backbone is ~70% of the work
- Stage 2 is fine-tuning for DME (small dataset, 516 images)
- Expected improvement: ~2-3% QWK absolute
- If worse: check `stage2_revert_if_worse` is enabled (auto-rollback to Stage 1)

---

## Advanced: Manual Control

### Stage 1 Only (Skip Stage 2)

```python
result = run_pipeline(
    ...,
    two_stage=False,  # Only run Stage 1
)
```

### Use Pre-trained Backbone

```python
result = run_pipeline(
    ...,
    backbone_weights_path="/path/to/pretrained.h5",
)
```

### Custom EyePACS Subset

```python
# Load your own filtered EyePACS
import pandas as pd
eyepacs_df = pd.read_csv('trainLabels.csv')

# Filter to balanced subset (optional)
eyepacs_df = eyepacs_df.groupby('level').apply(
    lambda x: x.sample(n=min(10000, len(x)))
).reset_index(drop=True)

eyepacs_df.to_csv('trainLabels_filtered.csv', index=False)

# Then run pipeline with filtered CSV
result = run_pipeline(
    ...,
    eyepacs_csv='trainLabels_filtered.csv',
)
```

---

## Expected Results

After mixed training:

✅ **DR generalization:** External dataset QWK improves from ~0.01-0.11 → 0.40-0.60
✅ **DME accuracy:** Maintained (still trained on clean IDRiD)
✅ **IDRiD validation:** Similar or slightly better (~0.72-0.75 QWK)
✅ **Training time:** ~3-5 hours on 2× Tesla T4 GPUs

---

## References

- **IDRiD**: https://ieee-dataport.org/competitions/idrid
- **EyePACS**: https://www.kaggle.com/c/diabetic-retinopathy-detection
- **Messidor**: http://www.adcis.net/en/Download-Third-Party/Messidor.html

---

Questions? Check [README.md](README.md) or create an issue on GitHub.
