# Quickstart Guide

Copy-paste runnable examples to get the DR+DME detection system running.

---

## Prerequisites

```bash
pip install -r requirements.txt
```

---

## 1. Smoke Test with Mock Data (No dataset required)

```bash
# Generate 80 synthetic fundus images and run the full pipeline
python main_pipeline.py --mock --epochs 5 --single-stage --output-dir /tmp/smoke_test

# Expected output:
# INFO: Stage 1 QWK: 0.xxxx
# INFO: Pipeline report saved
```

---

## 2. Compute QWK on Existing Predictions

```bash
# If you have y_true.npy and y_pred.npy
python qwk_metrics.py \
  --y-true /path/to/y_true.npy \
  --y-pred /path/to/y_pred.npy \
  --output-dir results/
```

---

## 3. Train DME Model (Standard)

```bash
python train.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --output dme_standard.weights.h5
```

---

## 4. Train with Enhanced QWK-Aware Pipeline

```bash
python train_enhanced.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --output-dir results/ \
  --output dme_enhanced.weights.h5
```

---

## 5. Comprehensive QWK Evaluation

```bash
python evaluate_comprehensive.py \
  --weights dme_enhanced.weights.h5 \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --output-dir results/
```

---

## 6. Advanced Dataset Loader with Ordinal Stratification

```bash
# Create mock dataset and inspect split
python dataset_loader_advanced.py \
  --mock \
  --output-dir /tmp/advanced_loader_test

# Run K-fold cross-validation split inspection
python dataset_loader_advanced.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --kfold 5 \
  --output-dir /tmp/kfold_splits
```

---

## 7. Ablation Study (Compare Model A / B / C)

```bash
# Quick ablation on mock data (5 epochs per model)
python ablation_study.py \
  --mock \
  --epochs 5 \
  --output-dir ablation_results/

# Real data ablation
python ablation_study.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --epochs 10 \
  --n-bootstrap 500 \
  --output-dir ablation_results/
```

---

## 8. Full End-to-End Pipeline (Recommended)

```bash
# Using config.yaml
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --output-dir pipeline_outputs/

# Mock data, 2-stage pipeline
python main_pipeline.py \
  --mock \
  --epochs 10 \
  --config config.yaml \
  --output-dir /tmp/pipeline_test/
```

Minimal runnable command:

```bash
python main_pipeline.py --mock --epochs 5
```

---

## 9. Jupyter Notebook

```bash
jupyter notebook train_dme_model.ipynb
```

---

## Expected File Structure After Running Pipeline

```
pipeline_outputs/
├── model_stage1.weights.h5
├── model_stage2.weights.h5
├── pipeline_report.json          ← QWK summary
├── eval_stage2/
│   ├── comprehensive_metrics.json
│   └── comprehensive_dashboard.png
└── checkpoints/
    └── stage2/best_qwk.weights.h5
```

---

## Reading the Results

```python
import json

with open("pipeline_outputs/pipeline_report.json") as f:
    report = json.load(f)

print(f"Best QWK: {report['best_qwk']:.4f}")
print(f"Target met: {report['target_met']}")
```

```python
# Read comprehensive evaluation metrics
with open("pipeline_outputs/eval_stage2/comprehensive_metrics.json") as f:
    metrics = json.load(f)

print(f"QWK:      {metrics['qwk']:.4f}")
print(f"MAE:      {metrics['mae']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"Clinical: {metrics['interpretation']['clinical_recommendation']}")
```

---

## Configuration Quick Reference

Edit `config.yaml` to tune the pipeline:

```yaml
# Key parameters to adjust:
batch_size: 8           # Reduce to 4 if OOM
stage1:
  epochs: 30            # Increase for better QWK
  learning_rate: 1.0e-4

stage2:
  epochs: 20
  learning_rate: 5.0e-5  # Always lower than stage1

# QWKCallback behavior:
max_batches: null        # null = full validation set (recommended)
# max_batches: 10         # optional approximation for very large val sets
```
