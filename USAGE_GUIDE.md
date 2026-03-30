# Usage Guide

Comprehensive guide for running training, evaluation, and troubleshooting in `MajorProject`.

---

## 1) Setup

```bash
pip install -r requirements.txt
```

---

## 2) Fast verification with mock data

```bash
python main_pipeline.py --mock --epochs 5
```

This is the recommended smoke test. It validates dataset creation, model build, training, and QWK reporting end-to-end.

---

## 3) Real dataset run

```bash
python main_pipeline.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --config config.yaml \
  --epochs 30
```

Expected label spaces:
- DME classes: `0=No DME`, `1=Mild`, `2=Moderate` (**3 classes**)
- DR classes: `0=No DR`, `1=Mild`, `2=Moderate`, `3=Severe NPDR`, `4=Proliferative` (**5 classes**)

DME labels are treated as **ordinal** (order matters): No DME < Mild < Moderate.

---

## 4) QWK behavior and `max_batches`

QWK is computed by `QWKCallback` during training.

- Default: `max_batches: null` in `config.yaml` → compute on **full validation set** (accurate)
- Optional: set `max_batches` to a positive integer to reduce memory/time on very large validation datasets (approximate QWK)

Example:

```yaml
max_batches: null   # recommended
# max_batches: 10   # optional approximation mode
```

---

## 5) Class imbalance handling

The enhanced training path supports class imbalance via:

1. `ordinal_loss_weighting: true` (default in stage configs)
2. DME class weights passed into `OrdinalWeightedCrossEntropy`
3. Debug logs for train/val class distributions (sampled batches)

No extra CLI flags are required for default behavior.

---

## 6) Typical commands

### Enhanced training
```bash
python train_enhanced.py \
  --csv /path/to/DME_Grades.csv \
  --image-dir /path/to/images \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --output dme_enhanced.weights.h5
```

### QWK tests
```bash
python test_qwk_calculation.py
python test_qwk_multioutput.py
```

### Ordinal loss test
```bash
python test_ordinal_weighted_loss.py
```

---

## 7) Troubleshooting

### QWK appears unstable or too low early in training
- Ensure `max_batches: null` so full validation set is used.
- Check logs for class distribution skew in train/val.

### Out-of-memory during validation callback
- Set `max_batches` to a smaller positive value (e.g., 10–50).
- Reduce `batch_size`.

### Class mismatch errors
- Confirm `num_dme_classes: 3` and `num_dr_classes: 5` in `config.yaml`.
- Ensure CSV labels match expected value ranges.

### TensorFlow import errors in tests
- Install dependencies from `requirements.txt` in the active environment.
