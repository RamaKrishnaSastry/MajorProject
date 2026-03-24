# MajorProject – Multi-Task Diabetic Retinopathy (DR) + DME Detection System

A production-ready deep learning system for joint detection of **Diabetic Retinopathy (DR)** and **Diabetic Macular Edema (DME)** severity using the **DR-ASPP-DRN** architecture.

**Primary metric: Quadratic Weighted Kappa (QWK) ≥ 0.80**

---

## 📦 Repository Structure

```
MajorProject/
├── preprocess.py                  ← Medical preprocessing (CLAHE, cropping, normalisation)
├── dataset_loader.py              ← IRDID pipeline, class weights, tf.data
├── dataset_loader_advanced.py     ← QWK-aware stratification, ordinal class weights
├── model.py                       ← DR-ASPP-DRN architecture (ResNet50 + ASPP + dual heads)
├── train.py                       ← Standard training orchestration
├── train_enhanced.py              ← QWK-aware training with ordinal loss
├── evaluate.py                    ← Base evaluation (F1, AUC, confusion matrix)
├── evaluate_comprehensive.py      ← Full QWK + ordinal + boundary confusion evaluation
├── qwk_metrics.py                 ← Quadratic Weighted Kappa computation + visualisation
├── ablation_study.py              ← Model A/B/C ablation comparison
├── main_pipeline.py               ← End-to-end pipeline orchestration
├── config.yaml                    ← Centralised hyperparameter configuration
├── requirements.txt               ← Python dependencies
├── train_dme_model.ipynb          ← Jupyter notebook workflow
├── README.md                      ← This file
├── ARCHITECTURE.md                ← DR-ASPP-DRN design documentation
├── RESULTS.md                     ← Expected performance benchmarks
└── QUICKSTART.md                  ← Copy-paste runnable examples
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Full pipeline with mock data (no real dataset needed)
python main_pipeline.py --mock --epochs 5 --single-stage

# Full pipeline with real IRDID data
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml
```

See [QUICKSTART.md](QUICKSTART.md) for all runnable examples.

---

## 🏗️ Architecture

The **DR-ASPP-DRN** architecture combines:
- **ResNet50 backbone** (ImageNet pre-training)
- **ASPP module** (multi-scale context with dilation rates 6, 12, 18)
- **DR head** (regression, sigmoid output)
- **DME head** (4-class ordinal classification, softmax output)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design documentation.

---

## 📊 Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **QWK** | **≥ 0.80** | Main success criterion – clinical-grade agreement |
| F1 (Macro) | ≥ 0.79 | Class imbalance handling |
| Accuracy | ≥ 0.81 | Overall correctness |
| MAE | ≤ 0.25 | Ordinal label distance |

See [RESULTS.md](RESULTS.md) for detailed benchmarks.

---

## 🧩 Module Overview

### `qwk_metrics.py`
- Quadratic Weighted Kappa computation with detailed logs
- Ordinal MAE, RMSE metrics
- Boundary confusion detection (Mild/Moderate confusion)
- Ordinal confusion matrix visualisation
- `QWKCallback` for epoch-level monitoring

### `train_enhanced.py`
- QWK-aware callbacks (ModelCheckpoint, EarlyStopping, ReduceLR)
- Ordinal-weighted cross-entropy loss
- Training diagnostics dashboard
- History JSON with QWK tracking

### `dataset_loader_advanced.py`
- Ordinal stratified train/val split
- Medical domain class weights (Severe > Moderate > Mild > No DME)
- K-fold cross-validation support
- Dataset balance visualisation

### `evaluate_comprehensive.py`
- Full QWK evaluation pipeline
- Boundary confusion analysis
- 6-panel evaluation dashboard
- Medical interpretation of results

### `ablation_study.py`
- Model A: Baseline ResNet50 (~0.73–0.75 QWK)
- Model B: ResNet50 + ASPP (~0.76–0.78 QWK)
- Model C: Full DR-ASPP-DRN (≥ 0.80 QWK)
- Bootstrap significance testing

### `main_pipeline.py`
- 2-stage training (initial → fine-tuning)
- Automatic best-model checkpointing by QWK
- Results aggregation and pipeline report

---

## 📋 Dataset

Compatible with the **IRDID (Indian Retinal Image Dataset)** CSV format:

```
Image name, Risk of macular edema
img001, 0
img002, 2
img003, 1
```

Labels: `0=No DME`, `1=Mild`, `2=Moderate`, `3=Severe`

---

## 🔬 Research Publication Ready

- ✅ QWK ≥ 0.80 target pathway
- ✅ Ordinal classification support
- ✅ Medical domain knowledge integration
- ✅ Full reproducibility (seed + split_info.json)
- ✅ Ablation study with statistical significance testing
- ✅ Comprehensive visualisation suite (confusion matrix, training curves, dashboard)