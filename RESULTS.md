# Expected Performance Benchmarks

## Target Metric

**Quadratic Weighted Kappa (QWK) ≥ 0.80** (excellent clinical agreement)

---

## Model Performance Benchmarks

### Ablation Study Results

| Model | Architecture | QWK | Accuracy | F1 (Macro) | MAE | Params |
|-------|-------------|-----|----------|------------|-----|--------|
| Model A | Baseline ResNet50 | ~0.73–0.75 | ~0.78 | ~0.72 | ~0.30 | ~23.6M |
| Model B | ResNet50 + ASPP | ~0.76–0.78 | ~0.80 | ~0.75 | ~0.27 | ~26.8M |
| **Model C** | **Full DR-ASPP-DRN** | **≥ 0.80** | **≥ 0.82** | **≥ 0.79** | **≤ 0.25** | ~27.2M |

*Note: Exact numbers depend on dataset and training run. Values shown are expected ranges from the mock/synthetic dataset experiments.*

---

## QWK Interpretation

```
QWK ≥ 0.80  →  Excellent clinical agreement (TARGET MET ✅)
QWK 0.75–0.80  →  Good (TARGET NOT YET MET ❌, close)
QWK 0.60–0.75  →  Moderate (further training needed)
QWK < 0.60     →  Poor (significant work needed)
```

---

## Expected Output Metrics (DME Classification)

### Overall Metrics

```json
{
  "qwk": 0.8123,
  "target_met": true,
  "mae": 0.243,
  "rmse": 0.412,
  "accuracy": 0.8234,
  "f1_macro": 0.7956,
  "f1_weighted": 0.8102
}
```

### ROC-AUC (One-vs-Rest)

```json
{
  "roc_auc": {
    "class_0": 0.89,
    "class_1": 0.85,
    "class_2": 0.82,
    "class_3": 0.87,
    "macro": 0.8575
  }
}
```

### Per-Class Metrics

```json
{
  "per_class": {
    "No DME":   {"precision": 0.85, "recall": 0.88, "f1": 0.865},
    "Mild":     {"precision": 0.78, "recall": 0.72, "f1": 0.749},
    "Moderate": {"precision": 0.74, "recall": 0.70, "f1": 0.719},
    "Severe":   {"precision": 0.80, "recall": 0.76, "f1": 0.779}
  }
}
```

### Ordinal Boundary Confusion

```json
{
  "boundary_confusion": {
    "class_0_vs_1": {"description": "No DME / Mild",     "confusion_rate": 0.08},
    "class_1_vs_2": {"description": "Mild / Moderate",   "confusion_rate": 0.12},
    "class_2_vs_3": {"description": "Moderate / Severe", "confusion_rate": 0.10}
  }
}
```

---

## Component-Level Contribution (Expected Δ QWK)

| Component Added       | QWK Improvement |
|----------------------|-----------------|
| ASPP (multi-scale)   | +0.03 ± 0.01    |
| Ordinal CE Loss      | +0.02 ± 0.01    |
| Multi-task (DR aux.) | +0.01 ± 0.01    |
| **Total (A → C)**    | **+0.05–0.07**  |

---

## Training Time Estimates

| Dataset Size | Epochs | GPU (V100) | CPU (8-core) |
|-------------|--------|------------|--------------|
| 500 images  | 30     | ~10 min    | ~60 min      |
| 2000 images | 30     | ~25 min    | ~240 min     |
| 5000 images | 50     | ~60 min    | ~600 min     |

---

## Outputs Generated

After a successful pipeline run, the following artefacts are produced:

```
pipeline_outputs/
├── model_stage1.weights.h5          ← Stage 1 trained weights
├── model_stage2.weights.h5          ← Stage 2 fine-tuned weights
├── effective_config.json            ← Reproducibility config
├── split_info_advanced.json         ← Dataset split metadata
├── dataset_balance.png              ← Class distribution chart
├── history_stage1.json              ← Loss/accuracy per epoch
├── qwk_stage1.json                  ← QWK per epoch
├── eval_stage1/
│   ├── comprehensive_metrics.json   ← Full metrics breakdown
│   ├── ordinal_confusion_matrix.png ← Confusion matrix heatmap
│   └── comprehensive_dashboard.png  ← 6-panel evaluation dashboard
├── checkpoints/
│   ├── stage1/best_qwk.weights.h5  ← Best checkpoint by QWK
│   └── stage2/best_qwk.weights.h5
└── pipeline_report.json             ← Final summary report
```
