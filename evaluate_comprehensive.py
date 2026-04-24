"""
evaluate_comprehensive.py - Comprehensive QWK-focused evaluation for DR+DME classification.

KEY CHANGES FROM ORIGINAL:
- Evaluation runs ONCE on the best saved joint model (not per-epoch)
- Full DR visualisations: confusion matrices, dashboard, ordinal plots
- DME QWK (raw) + DR QWK (calib) shown side-by-side in all summary panels
- Single inference pass collects both DR and DME predictions (no double scan)
- New plots: ordinal error histograms, per-class prediction distribution,
  multi-class ROC curves, joint QWK comparison bar chart
- evaluate_on_best_joint_model() helper to load checkpoint and run everything

Extends evaluate.py with:
- Quadratic Weighted Kappa with detailed computation log
- Ordinal classification metrics (MAE, RMSE on ordinal scale)
- Boundary confusion detection (which ordinal classes get confused)
- Per-class ordinal metrics and confusion patterns
- Heatmap visualisations with ordinal annotations
- JSON export with complete metric breakdown
- Medical interpretation of results
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from evaluate import (
    compute_accuracy,
    compute_f1,
    compute_roc_auc,
    compute_per_class_metrics,
    DME_CLASS_NAMES,
    NUM_DME_CLASSES,
    DR_CLASS_NAMES,
    NUM_DR_CLASSES,
)
from qwk_metrics import (
    compute_quadratic_weighted_kappa,
    compute_qwk_with_details,
    compute_ordinal_metrics,
    plot_ordinal_confusion_matrix,
    BOUNDARY_PAIRS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Clinical QWK target
QWK_TARGET_THRESHOLD = 0.80

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

try:
    from sklearn.metrics import (
        confusion_matrix as sk_cm,
        roc_curve,
        auc,
        cohen_kappa_score,
        accuracy_score,
    )
    from sklearn.preprocessing import label_binarize
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Medical interpretation thresholds
# ---------------------------------------------------------------------------

QWK_THRESHOLDS = {
    "poor":      (float("-inf"), 0.40),
    "fair":      (0.40,         0.60),
    "moderate":  (0.60,         0.75),
    "good":      (0.75,         0.80),
    "excellent": (0.80,         1.00),
}


def interpret_qwk(qwk: float) -> str:
    for label, (lo, hi) in QWK_THRESHOLDS.items():
        if lo <= qwk < hi:
            return label
    return "excellent" if qwk >= 1.00 else "poor"


def generate_medical_interpretation(metrics: Dict) -> Dict[str, str]:
    qwk     = metrics.get("qwk", 0.0)
    mae     = metrics.get("mae", float("nan"))
    accuracy= metrics.get("accuracy", 0.0)
    f1      = metrics.get("f1_macro", 0.0)
    dr_qwk_cal = metrics.get("dr_qwk_calib", 0.0)
    interp  = interpret_qwk(qwk)

    return {
        "dme_qwk_interpretation": (
            f"DME QWK (raw) = {qwk:.4f} ({interp}). "
            + ("Model meets the clinical-grade agreement target (≥0.80)."
               if qwk >= QWK_TARGET_THRESHOLD
               else f"Deficit: {QWK_TARGET_THRESHOLD - qwk:.4f}. Consider more training.")
        ),
        "dr_qwk_interpretation": (
            f"DR QWK (calib) = {dr_qwk_cal:.4f} ({interpret_qwk(dr_qwk_cal)}). "
            + ("DR grading meets the clinical-grade agreement target (≥0.80)."
               if dr_qwk_cal >= QWK_TARGET_THRESHOLD
               else f"DR deficit: {QWK_TARGET_THRESHOLD - dr_qwk_cal:.4f}.")
        ),
        "mae_interpretation": (
            f"DME MAE = {mae:.3f} ordinal units. "
            + ("Average prediction error < 1 severity class – clinically acceptable."
               if not np.isnan(mae) and mae < 1.0
               else "Average prediction error spans more than 1 severity class.")
        ),
        "accuracy_interpretation": (
            f"Overall DME accuracy = {accuracy:.4f}. "
            + ("Strong pixel-level agreement." if accuracy >= 0.80
               else "Accuracy below 80% – class imbalance or data quality issues likely.")
        ),
        "f1_interpretation": (
            f"Macro F1 = {f1:.4f}. "
            + ("Good handling of class imbalance." if f1 >= 0.75
               else "Macro F1 below 0.75 – minor classes may be under-represented.")
        ),
        "clinical_recommendation": (
            "PASS: Model is suitable for clinical decision support screening."
            if (qwk >= QWK_TARGET_THRESHOLD and dr_qwk_cal >= QWK_TARGET_THRESHOLD)
            else "FAIL: One or both heads require further training before clinical deployment."
        ),
    }


# ---------------------------------------------------------------------------
# Single-pass inference — collects BOTH DR and DME predictions
# ---------------------------------------------------------------------------

def get_all_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_dme_classes: int = NUM_DME_CLASSES,
    num_dr_classes: int = NUM_DR_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single inference pass collecting DR and DME predictions simultaneously.

    Returns
    -------
    tuple
        (dme_true, dme_proba, dme_pred, dr_true, dr_proba, dr_pred)
        All integer label arrays are shape (N,); proba arrays are (N, C).
    """
    dme_true_all, dme_proba_all = [], []
    dr_true_all,  dr_proba_all  = [], []

    for batch_images, batch_labels in dataset:
        preds = model(batch_images, training=False)

        # ---- Extract DME predictions ----
        if isinstance(preds, dict):
            dme_proba = preds.get("dme_risk", preds.get("dme_output"))
            dr_proba  = preds.get("dr_output")
        elif isinstance(preds, (list, tuple)):
            dr_proba  = preds[0]
            dme_proba = preds[1] if len(preds) > 1 else preds[0]
        else:
            dme_proba = preds
            dr_proba  = None

        # ---- Extract DME labels ----
        if isinstance(batch_labels, dict):
            dme_labels = batch_labels.get("dme_risk", batch_labels.get("dme_output"))
            dr_labels  = batch_labels.get("dr_output")
        elif isinstance(batch_labels, (list, tuple)):
            dr_labels  = batch_labels[0]
            dme_labels = batch_labels[1] if len(batch_labels) > 1 else batch_labels[0]
        else:
            dme_labels = batch_labels
            dr_labels  = None

        # ---- DME ----
        dme_proba_np = dme_proba.numpy() if hasattr(dme_proba, "numpy") else np.asarray(dme_proba)
        dme_labels_np = dme_labels.numpy() if hasattr(dme_labels, "numpy") else np.asarray(dme_labels)
        dme_true_all.append(np.argmax(dme_labels_np, axis=-1))
        dme_proba_all.append(dme_proba_np)

        # ---- DR ----
        if dr_proba is not None and dr_labels is not None:
            dr_proba_np  = dr_proba.numpy()  if hasattr(dr_proba,  "numpy") else np.asarray(dr_proba)
            dr_labels_np = dr_labels.numpy() if hasattr(dr_labels, "numpy") else np.asarray(dr_labels)

            if dr_proba_np.ndim > 1 and dr_proba_np.shape[-1] > 1:
                dr_pred_batch = np.argmax(dr_proba_np, axis=-1).astype(int)
            else:
                dr_pred_batch = np.clip(np.rint(dr_proba_np.reshape(-1)).astype(int),
                                        0, num_dr_classes - 1)

            if dr_labels_np.ndim > 1 and dr_labels_np.shape[-1] > 1:
                dr_true_batch = np.argmax(dr_labels_np, axis=-1).astype(int)
            else:
                dr_true_batch = np.clip(np.rint(dr_labels_np.reshape(-1)).astype(int),
                                        0, num_dr_classes - 1)

            dr_true_all.append(dr_true_batch)
            # Store full softmax if available, else repeat one-hot for AUC
            if dr_proba_np.ndim > 1 and dr_proba_np.shape[-1] == num_dr_classes:
                dr_proba_all.append(dr_proba_np)
            else:
                # Fallback: pseudo-proba from grade
                pseudo = np.zeros((len(dr_pred_batch), num_dr_classes), dtype=np.float32)
                for i, g in enumerate(dr_pred_batch):
                    pseudo[i, g] = 1.0
                dr_proba_all.append(pseudo)

    dme_true  = np.concatenate(dme_true_all)
    dme_proba = np.concatenate(dme_proba_all)
    dme_pred  = np.argmax(dme_proba, axis=-1)

    if dr_true_all:
        dr_true  = np.concatenate(dr_true_all)
        dr_proba = np.concatenate(dr_proba_all)
        dr_pred  = np.argmax(dr_proba, axis=-1).astype(int)
    else:
        dr_true  = np.array([], dtype=np.int32)
        dr_proba = np.zeros((0, num_dr_classes), dtype=np.float32)
        dr_pred  = np.array([], dtype=np.int32)

    return dme_true, dme_proba, dme_pred, dr_true, dr_proba, dr_pred


# ---------------------------------------------------------------------------
# Threshold calibration helpers
# ---------------------------------------------------------------------------

def _predict_from_expected_score(
    y_proba: np.ndarray,
    boundaries: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    expected_score = np.sum(
        y_proba * np.arange(num_classes, dtype=np.float32)[None, :], axis=1
    )
    pred = np.searchsorted(boundaries, expected_score, side="right")
    return np.clip(pred.astype(np.int32), 0, num_classes - 1)


def _optimize_expected_score_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    grid_size: int = 41,
    n_passes: int = 3,
    min_gap: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if num_classes <= 1:
        return np.array([], dtype=np.float32), np.zeros_like(y_true, dtype=np.int32), 0.0

    boundaries = np.arange(0.5, float(num_classes - 0.5), 1.0, dtype=np.float32)

    def _objective(b):
        p = _predict_from_expected_score(y_proba, b, num_classes)
        s = compute_quadratic_weighted_kappa(y_true, p, num_classes)
        return float(s), p

    best_qwk, best_pred = _objective(boundaries)

    for _ in range(max(1, int(n_passes))):
        improved = False
        for idx in range(len(boundaries)):
            lower = 0.0 if idx == 0 else boundaries[idx - 1] + min_gap
            upper = (float(num_classes - 1) if idx == len(boundaries) - 1
                     else boundaries[idx + 1] - min_gap)
            if upper <= lower:
                continue
            for candidate in np.linspace(lower, upper, num=max(5, int(grid_size)), dtype=np.float32):
                trial = boundaries.copy()
                trial[idx] = float(candidate)
                qwk_val, pred_val = _objective(trial)
                if qwk_val > best_qwk:
                    best_qwk = qwk_val
                    boundaries[idx] = float(candidate)
                    best_pred = pred_val
                    improved = True
        if not improved:
            break

    return boundaries.astype(np.float32), best_pred.astype(np.int32), float(best_qwk)


# ---------------------------------------------------------------------------
# ── NEW PLOTS ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def plot_ordinal_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Ordinal Prediction Error Distribution",
    output_path: str = "ordinal_error_hist.png",
) -> None:
    """Histogram of (y_pred - y_true) showing off-by-1, off-by-2 etc."""
    if not _PLOTTING_AVAILABLE:
        return
    errors = y_pred.astype(int) - y_true.astype(int)
    unique_errors = np.arange(errors.min(), errors.max() + 1)
    counts = [int(np.sum(errors == e)) for e in unique_errors]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if e == 0 else "orange" if abs(e) == 1 else "red" for e in unique_errors]
    bars = ax.bar(unique_errors, counts, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_xlabel("Prediction Error  (predicted − true)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(unique_errors)
    ax.set_xticklabels([str(e) for e in unique_errors])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green",  label="Correct (0)"),
        Patch(facecolor="orange", label="Off-by-1 (±1)"),
        Patch(facecolor="red",    label="Off-by-2+ (≥±2)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ordinal error histogram saved to '%s'.", output_path)


def plot_per_class_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Per-Class Prediction Distribution",
    output_path: str = "per_class_pred_dist.png",
) -> None:
    """Stacked bar: for each true class, what % was predicted as each class."""
    if not _PLOTTING_AVAILABLE:
        return
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        for j in range(num_classes):
            matrix[i, j] = np.sum((y_true == i) & (y_pred == j)) / mask.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    bottoms = np.zeros(num_classes)
    for j in range(num_classes):
        ax.bar(class_names, matrix[:, j], bottom=bottoms,
               label=f"Predicted: {class_names[j]}", color=colors[j],
               edgecolor="black", linewidth=0.5, alpha=0.88)
        bottoms += matrix[:, j]

    ax.set_xlabel("True Class", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Per-class prediction distribution saved to '%s'.", output_path)


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    title: str = "Multi-Class ROC Curves (One-vs-Rest)",
    output_path: str = "roc_curves.png",
) -> None:
    """One-vs-rest ROC curves for each class."""
    if not (_PLOTTING_AVAILABLE and _SKLEARN_AVAILABLE):
        return
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if y_bin.ndim == 1:
        y_bin = y_bin[:, None]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin.shape[1] <= i:
            continue
        col = y_bin[:, i]
        if len(np.unique(col)) < 2:
            continue
        prob_col = y_proba[:, i] if y_proba.shape[1] > i else np.zeros(len(y_true))
        fpr, tpr, _ = roc_curve(col, prob_col)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to '%s'.", output_path)


def plot_joint_qwk_comparison(
    dme_qwk_raw: float,
    dr_qwk_raw: float,
    dr_qwk_calib: float,
    stage_label: str = "Current Stage",
    output_path: str = "joint_qwk_comparison.png",
) -> None:
    """Grouped bar: DME QWK (raw), DR QWK (raw), DR QWK (calib)."""
    if not _PLOTTING_AVAILABLE:
        return
    labels  = ["DME QWK\n(raw)", "DR QWK\n(raw)", "DR QWK\n(calib)"]
    values  = [dme_qwk_raw, dr_qwk_raw, dr_qwk_calib]
    colors  = ["steelblue", "sandybrown", "darkorange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.88, width=0.45)
    ax.bar_label(bars, fmt="%.4f", fontsize=11, padding=3)
    ax.axhline(QWK_TARGET_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Clinical target = {QWK_TARGET_THRESHOLD}")
    ax.set_ylim([0, 1.10])
    ax.set_ylabel("QWK Score", fontsize=12)
    ax.set_title(f"Joint QWK Comparison — {stage_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Joint QWK comparison saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# DR dashboard
# ---------------------------------------------------------------------------

def plot_dr_dashboard(
    dr_metrics: Dict,
    output_dir: str,
    num_dr_classes: int = NUM_DR_CLASSES,
    dr_true: Optional[np.ndarray] = None,
    dr_pred: Optional[np.ndarray] = None,
    dr_proba: Optional[np.ndarray] = None,
) -> None:
    """2×3 DR grading evaluation dashboard.

    Panels:
    1. Confusion matrix (counts)
    2. Confusion matrix (normalised)
    3. Per-class prediction distribution (stacked bar)
    4. Ordinal error histogram
    5. Boundary confusion rates
    6. DR summary panel — DR QWK (raw) and DR QWK (calib)
    """
    if not _PLOTTING_AVAILABLE:
        return

    names = DR_CLASS_NAMES[:num_dr_classes]

    # Build confusion matrix from stored data or passed arrays
    cm_data = dr_metrics.get("confusion_matrix")
    if cm_data is not None:
        cm = np.array(cm_data)
    elif dr_true is not None and dr_pred is not None:
        if _SKLEARN_AVAILABLE:
            cm = sk_cm(dr_true, dr_pred, labels=list(range(num_dr_classes)))
        else:
            cm = np.array([[int(np.sum((dr_true == i) & (dr_pred == j)))
                            for j in range(num_dr_classes)]
                           for i in range(num_dr_classes)])
    else:
        cm = np.zeros((num_dr_classes, num_dr_classes), dtype=int)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Confusion matrix counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0, 0])
    axes[0, 0].set_title("DR Confusion Matrix (Counts)", fontsize=12)
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    # 2. Confusion matrix normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("DR Confusion Matrix (Normalised)", fontsize=12)
    axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("True")

    # 3. Per-class prediction distribution
    if dr_true is not None and dr_pred is not None and len(dr_true) > 0:
        matrix = np.zeros((num_dr_classes, num_dr_classes), dtype=float)
        for i in range(num_dr_classes):
            mask = dr_true == i
            if mask.sum() == 0:
                continue
            for j in range(num_dr_classes):
                matrix[i, j] = np.sum((dr_true == i) & (dr_pred == j)) / mask.sum()
        colors_bar = plt.cm.Set2(np.linspace(0, 1, num_dr_classes))
        bottoms = np.zeros(num_dr_classes)
        for j in range(num_dr_classes):
            axes[0, 2].bar(names, matrix[:, j], bottom=bottoms,
                           label=f"Pred: {names[j]}", color=colors_bar[j],
                           edgecolor="black", linewidth=0.4, alpha=0.88)
            bottoms += matrix[:, j]
        axes[0, 2].set_title("Per-Class Prediction Distribution", fontsize=12)
        axes[0, 2].set_xlabel("True Class")
        axes[0, 2].set_ylabel("Proportion")
        axes[0, 2].set_ylim([0, 1.05])
        axes[0, 2].legend(fontsize=7, loc="upper right")
    else:
        axes[0, 2].text(0.5, 0.5, "No prediction data", ha="center", va="center")

    # 4. Ordinal error histogram
    if dr_true is not None and dr_pred is not None and len(dr_true) > 0:
        errors = dr_pred.astype(int) - dr_true.astype(int)
        min_e, max_e = int(errors.min()), int(errors.max())
        unique_errors = np.arange(min_e, max_e + 1)
        counts = [int(np.sum(errors == e)) for e in unique_errors]
        colors_err = ["green" if e == 0 else "orange" if abs(e) == 1 else "red"
                      for e in unique_errors]
        bars = axes[1, 0].bar(unique_errors, counts, color=colors_err,
                              edgecolor="black", alpha=0.85)
        axes[1, 0].bar_label(bars, fmt="%d", fontsize=8)
        axes[1, 0].set_xlabel("Prediction Error (pred − true)", fontsize=10)
        axes[1, 0].set_ylabel("Count", fontsize=10)
        axes[1, 0].set_title("DR Ordinal Error Distribution", fontsize=12)
        axes[1, 0].set_xticks(unique_errors)
        axes[1, 0].grid(axis="y", alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No error data", ha="center", va="center")

    # 5. Boundary confusion rates
    ordinal = dr_metrics.get("ordinal", {})
    boundary = ordinal.get("boundary_confusion", {})
    if boundary:
        b_labels = [v.get("description", k) for k, v in boundary.items()]
        b_rates  = [v.get("confusion_rate", 0.0) for v in boundary.values()]
        axes[1, 1].bar(b_labels, b_rates, color="darkorchid",
                       edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("DR Boundary Confusion Rates", fontsize=12)
        axes[1, 1].set_xlabel("Class Boundary")
        axes[1, 1].set_ylabel("Confusion Rate")
        axes[1, 1].set_ylim([0, 1.05])
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(20); tick.set_ha("right")
    else:
        axes[1, 1].text(0.5, 0.5, "No boundary data", ha="center", va="center")

    # 6. DR summary panel
    calib       = dr_metrics.get("calibration", {})
    qwk_raw     = calib.get("baseline_qwk",  dr_metrics.get("dr_qwk",      float("nan")))
    qwk_cal     = calib.get("calibrated_qwk", qwk_raw)
    dr_mae      = dr_metrics.get("dr_mae",      float("nan"))
    dr_rmse     = dr_metrics.get("dr_rmse",     float("nan"))
    dr_acc      = dr_metrics.get("dr_accuracy", float("nan"))
    cal_applied = calib.get("applied", False)

    axes[1, 2].axis("off")
    summary = (
        f"DR QWK (raw):   {qwk_raw:.4f}  {'✅' if qwk_raw  >= QWK_TARGET_THRESHOLD else '❌'}\n"
        f"DR QWK (calib): {qwk_cal:.4f}  {'✅' if qwk_cal  >= QWK_TARGET_THRESHOLD else '❌'}\n"
        f"Calibration applied: {'Yes' if cal_applied else 'No'}\n"
        f"DR MAE:      {dr_mae:.3f}\n"
        f"DR RMSE:     {dr_rmse:.3f}\n"
        f"DR Accuracy: {dr_acc:.4f}\n\n"
        f"Interpretation: {interpret_qwk(qwk_cal).upper()}\n\n"
        f"Clinical: {'PASS ✅' if qwk_cal >= QWK_TARGET_THRESHOLD else 'FAIL ❌'}"
    )
    axes[1, 2].text(0.05, 0.5, summary,
                    transform=axes[1, 2].transAxes, fontsize=12,
                    verticalalignment="center",
                    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
                    fontfamily="monospace")
    axes[1, 2].set_title("DR Grading Summary", fontsize=12)

    fig.suptitle("DR Grading Evaluation Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "dr_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("DR dashboard saved to '%s'.", path)


# ---------------------------------------------------------------------------
# DME comprehensive dashboard
# ---------------------------------------------------------------------------

def plot_comprehensive_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: Dict,
    class_names: Optional[List[str]] = None,
    output_path: str = "comprehensive_dashboard.png",
    num_classes: int = NUM_DME_CLASSES,
) -> None:
    """2×3 DME evaluation dashboard.

    Summary panel shows DME QWK (raw) AND DR QWK (calib) side-by-side.
    """
    if not _PLOTTING_AVAILABLE:
        return

    names = class_names or DME_CLASS_NAMES[:num_classes]
    if _SKLEARN_AVAILABLE:
        cm = sk_cm(y_true, y_pred, labels=list(range(num_classes)))
    else:
        cm = np.array([[int(np.sum((y_true == i) & (y_pred == j)))
                        for j in range(num_classes)]
                       for i in range(num_classes)])

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Confusion matrix counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0, 0])
    axes[0, 0].set_title("DME Confusion Matrix (Counts)", fontsize=12)
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    # 2. Confusion matrix normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("DME Confusion Matrix (Normalised)", fontsize=12)
    axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("True")

    # 3. Per-class F1
    per_class = metrics.get("per_class", {})
    pc_names  = list(per_class.keys())
    pc_f1     = [per_class[n].get("f1", 0.0) for n in pc_names]
    axes[0, 2].bar(pc_names, pc_f1, color="royalblue", edgecolor="black", alpha=0.85)
    axes[0, 2].set_title("DME Per-Class F1 Score", fontsize=12)
    axes[0, 2].set_xlabel("Class"); axes[0, 2].set_ylabel("F1")
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].axhline(0.75, color="red", linestyle="--", linewidth=1, label="Target 0.75")
    axes[0, 2].legend(fontsize=8)

    # 4. Per-class ordinal misclassification distance
    pc_ordinal = metrics.get("ordinal", {}).get("per_class", {})
    if pc_ordinal:
        oc_names = list(pc_ordinal.keys())
        oc_dist  = [pc_ordinal[n].get("mean_misclassification_distance", 0.0) for n in oc_names]
        axes[1, 0].bar(oc_names, oc_dist, color="tomato", edgecolor="black", alpha=0.85)
        axes[1, 0].set_title("Mean Misclassification Distance (DME)", fontsize=12)
        axes[1, 0].set_xlabel("True Class"); axes[1, 0].set_ylabel("Mean |true − pred|")
    else:
        axes[1, 0].text(0.5, 0.5, "No ordinal data", ha="center", va="center")

    # 5. DME boundary confusion rates
    boundary = metrics.get("ordinal", {}).get("boundary_confusion", {})
    if boundary:
        b_labels = [v.get("description", k) for k, v in boundary.items()]
        b_rates  = [v.get("confusion_rate", 0.0) for v in boundary.values()]
        axes[1, 1].bar(b_labels, b_rates, color="darkorchid",
                       edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("DME Boundary Confusion Rates", fontsize=12)
        axes[1, 1].set_xlabel("Class Boundary"); axes[1, 1].set_ylabel("Confusion Rate")
        axes[1, 1].set_ylim([0, 1.05])
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(20); tick.set_ha("right")
    else:
        axes[1, 1].text(0.5, 0.5, "No boundary data", ha="center", va="center")

    # 6. Joint summary panel — DME QWK (raw) + DR QWK (calib)
    dme_qwk_raw = metrics.get("qwk", 0.0)
    dr_info     = metrics.get("dr", {})
    dr_calib    = dr_info.get("calibration", {})
    dr_qwk_raw  = dr_calib.get("baseline_qwk",   dr_info.get("dr_qwk",  float("nan")))
    dr_qwk_cal  = dr_calib.get("calibrated_qwk", dr_qwk_raw)
    mae         = metrics.get("mae",      float("nan"))
    rmse        = metrics.get("rmse",     float("nan"))
    acc         = metrics.get("accuracy", 0.0)
    f1_macro    = metrics.get("f1_macro", 0.0)

    axes[1, 2].axis("off")
    summary_text = (
        f"DME QWK (raw):   {dme_qwk_raw:.4f}  {'✅' if dme_qwk_raw >= QWK_TARGET_THRESHOLD else '❌'}\n"
        f"DR  QWK (raw):   {dr_qwk_raw:.4f}\n"
        f"DR  QWK (calib): {dr_qwk_cal:.4f}  {'✅' if dr_qwk_cal  >= QWK_TARGET_THRESHOLD else '❌'}\n"
        f"DME MAE:     {mae:.3f}\n"
        f"DME RMSE:    {rmse:.3f}\n"
        f"DME Acc:     {acc:.4f}\n"
        f"F1 Macro:    {f1_macro:.4f}\n\n"
        f"DME: {interpret_qwk(dme_qwk_raw).upper()}\n"
        f"DR:  {interpret_qwk(dr_qwk_cal).upper()}"
    )
    axes[1, 2].text(
        0.05, 0.5, summary_text,
        transform=axes[1, 2].transAxes, fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )
    axes[1, 2].set_title("Joint Summary (DME + DR)", fontsize=12)

    fig.suptitle("Comprehensive DME Evaluation Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comprehensive dashboard saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# DR grading evaluation (uses pre-collected arrays — no second dataset scan)
# ---------------------------------------------------------------------------

def evaluate_dr_grading(
    dr_true: np.ndarray,
    dr_pred: np.ndarray,
    dr_proba: np.ndarray,
    output_dir: str,
    num_dr_classes: int = NUM_DR_CLASSES,
    calibrate_thresholds: bool = False,
    min_qwk_gain: float = 1e-4,
    max_accuracy_drop: float = 0.0,
) -> Dict:
    """Evaluate DR grading using pre-collected predictions.

    NOTE: Accepts pre-collected arrays rather than re-running the dataset,
    avoiding a costly second inference pass.

    Parameters
    ----------
    dr_true : np.ndarray
        Ground-truth DR grades.
    dr_pred : np.ndarray
        Predicted DR grades (argmax of softmax).
    dr_proba : np.ndarray
        Predicted DR probabilities shape (N, num_dr_classes).
    output_dir : str
        Directory for dr_metrics.json.
    num_dr_classes : int
        Number of DR grades.
    calibrate_thresholds : bool
        Whether to run expected-score threshold calibration.
    """
    if len(dr_true) == 0:
        logger.warning("No DR labels found in dataset — skipping DR evaluation.")
        return {}

    dr_true = np.asarray(dr_true, dtype=np.int32)
    dr_pred_final = np.asarray(dr_pred, dtype=np.int32)

    base_qwk = float(compute_quadratic_weighted_kappa(dr_true, dr_pred_final, num_dr_classes))
    base_acc = float(np.mean(dr_true == dr_pred_final))

    calibration = {
        "enabled":          bool(calibrate_thresholds),
        "applied":          False,
        "baseline_qwk":     base_qwk,
        "baseline_accuracy": base_acc,
    }

    if calibrate_thresholds and dr_proba is not None and len(dr_proba) > 0:
        boundaries, calibrated_pred, calibrated_qwk = _optimize_expected_score_thresholds(
            dr_true, dr_proba, num_classes=num_dr_classes,
        )
        calibrated_acc = float(np.mean(dr_true == calibrated_pred))
        calibration.update({
            "calibrated_qwk":      float(calibrated_qwk),
            "calibrated_accuracy": calibrated_acc,
            "boundaries":          [float(x) for x in boundaries.tolist()],
            "gain":                float(calibrated_qwk - base_qwk),
            "accuracy_delta":      float(calibrated_acc - base_acc),
        })
        if (calibrated_qwk >= base_qwk + float(min_qwk_gain) and
                calibrated_acc >= base_acc - float(max_accuracy_drop)):
            dr_pred_final = calibrated_pred.astype(np.int32)
            calibration["applied"] = True
            logger.info("DR threshold calibration applied: QWK %.4f -> %.4f",
                        base_qwk, calibrated_qwk)
        else:
            logger.info("DR threshold calibration not applied: gain %.4f < %.4f",
                        calibrated_qwk - base_qwk, min_qwk_gain)

    if _SKLEARN_AVAILABLE:
        dr_qwk = float(cohen_kappa_score(dr_true, dr_pred_final, weights="quadratic"))
        dr_acc = float(accuracy_score(dr_true, dr_pred_final))
        cm = sk_cm(dr_true, dr_pred_final, labels=list(range(num_dr_classes)))
    else:
        dr_qwk = float(compute_quadratic_weighted_kappa(dr_true, dr_pred_final, num_dr_classes))
        dr_acc = float(np.mean(dr_true == dr_pred_final))
        cm = np.array([[int(np.sum((dr_true == i) & (dr_pred_final == j)))
                        for j in range(num_dr_classes)]
                       for i in range(num_dr_classes)])

    dr_mae  = float(np.mean(np.abs(dr_true - dr_pred_final)))
    dr_rmse = float(np.sqrt(np.mean((dr_true - dr_pred_final) ** 2)))

    dr_ordinal = compute_ordinal_metrics(
        dr_true, dr_pred_final, num_dr_classes,
        class_names=DR_CLASS_NAMES[:num_dr_classes]
    )

    logger.info("DR Grading Results:")
    logger.info("  QWK (raw):   %.4f", base_qwk)
    logger.info("  QWK (calib): %.4f", dr_qwk)
    logger.info("  Accuracy:    %.4f", dr_acc)
    logger.info("  MAE:         %.4f", dr_mae)
    logger.info("  RMSE:        %.4f", dr_rmse)

    results = {
        "dr_qwk":          dr_qwk,
        "dr_accuracy":     dr_acc,
        "dr_mae":          dr_mae,
        "dr_rmse":         dr_rmse,
        "confusion_matrix": cm.tolist(),
        "calibration":     calibration,
        "ordinal":         dr_ordinal,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dr_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("DR metrics saved.")

    return results


# ---------------------------------------------------------------------------
# Comprehensive evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_comprehensive(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    output_dir: str = ".",
    metrics_path: str = "comprehensive_metrics.json",
    num_dme_classes: int = NUM_DME_CLASSES,
    num_dr_classes: int = NUM_DR_CLASSES,
    calibrate_dme_thresholds: bool = False,
    calibrate_dr_thresholds: bool = False,
    calibration_min_qwk_gain: float = 1e-4,
    dr_calibration_max_accuracy_drop: float = 0.0,
    stage_label: str = "Evaluation",
) -> Dict:
    """Full evaluation pipeline for BOTH DR and DME heads.

    Single inference pass collects predictions for both heads.
    Evaluation is meant to be called ONCE on the best saved joint model,
    not inside the training epoch loop.

    Parameters
    ----------
    model : keras.Model
        Best joint model (loaded from checkpoint).
    dataset : tf.data.Dataset
        Validation / test dataset.
    stage_label : str
        Label for plot titles (e.g. "Stage 1", "Stage 2").
    """
    os.makedirs(output_dir, exist_ok=True)
    dme_names = class_names or DME_CLASS_NAMES[:num_dme_classes]
    dr_names  = DR_CLASS_NAMES[:num_dr_classes]

    logger.info("=== %s: Running single-pass inference ===", stage_label)
    dme_true, dme_proba, dme_pred, dr_true, dr_proba, dr_pred = get_all_predictions(
        model, dataset, num_dme_classes=num_dme_classes, num_dr_classes=num_dr_classes
    )
    logger.info("Collected %d DME samples, %d DR samples.", len(dme_true), len(dr_true))

    # ---- Optional DME threshold calibration ----
    calibration = {
        "dme": {"enabled": bool(calibrate_dme_thresholds), "applied": False},
        "dr":  {"enabled": bool(calibrate_dr_thresholds),  "applied": False},
    }

    if calibrate_dme_thresholds:
        baseline_qwk = float(compute_quadratic_weighted_kappa(dme_true, dme_pred, num_dme_classes))
        boundaries, calibrated_pred, calibrated_qwk = _optimize_expected_score_thresholds(
            dme_true, dme_proba, num_classes=num_dme_classes,
        )
        calibration["dme"].update({
            "baseline_qwk":  baseline_qwk,
            "calibrated_qwk": float(calibrated_qwk),
            "boundaries":    [float(x) for x in boundaries.tolist()],
            "gain":          float(calibrated_qwk - baseline_qwk),
        })
        if calibrated_qwk >= baseline_qwk + float(calibration_min_qwk_gain):
            dme_pred = calibrated_pred.astype(np.int32)
            calibration["dme"]["applied"] = True
            logger.info("DME calibration applied: QWK %.4f -> %.4f", baseline_qwk, calibrated_qwk)

    # ---- DME standard metrics ----
    accuracy      = compute_accuracy(dme_true, dme_pred)
    f1_macro      = compute_f1(dme_true, dme_pred, average="macro")
    f1_weighted   = compute_f1(dme_true, dme_pred, average="weighted")
    roc_auc_dme   = compute_roc_auc(dme_true, dme_proba, num_classes=num_dme_classes)
    per_class     = compute_per_class_metrics(dme_true, dme_pred, dme_names)
    qwk_details   = compute_qwk_with_details(dme_true, dme_pred, num_dme_classes)
    ordinal_metrics = compute_ordinal_metrics(dme_true, dme_pred, num_dme_classes, dme_names)

    # ---- DR evaluation (uses pre-collected arrays) ----
    dr_metrics = evaluate_dr_grading(
        dr_true, dr_pred, dr_proba,
        output_dir=output_dir,
        num_dr_classes=num_dr_classes,
        calibrate_thresholds=calibrate_dr_thresholds,
        min_qwk_gain=calibration_min_qwk_gain,
        max_accuracy_drop=dr_calibration_max_accuracy_drop,
    )
    if isinstance(dr_metrics.get("calibration"), dict):
        calibration["dr"] = dr_metrics["calibration"]

    # Pull DR QWK values for combined interpretation
    dr_calib_info = dr_metrics.get("calibration", {})
    dr_qwk_raw    = dr_calib_info.get("baseline_qwk",   dr_metrics.get("dr_qwk", 0.0))
    dr_qwk_calib  = dr_calib_info.get("calibrated_qwk", dr_qwk_raw)

    # ---- Medical interpretation ----
    interpretation = generate_medical_interpretation({
        "qwk":          qwk_details["qwk"],
        "dr_qwk_calib": dr_qwk_calib,
        "mae":          ordinal_metrics["mae"],
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "per_class":    per_class,
    })

    metrics = {
        # DME main metric
        "qwk":          qwk_details["qwk"],
        "target_met":   qwk_details["qwk"] >= QWK_TARGET_THRESHOLD,
        # DME ordinal
        "mae":          ordinal_metrics["mae"],
        "rmse":         ordinal_metrics["rmse"],
        # DME standard classification
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "roc_auc":      roc_auc_dme,
        "per_class":    per_class,
        "ordinal":      ordinal_metrics,
        "qwk_details":  qwk_details,
        # DR metrics
        "dr":           dr_metrics,
        "dr_qwk_raw":   dr_qwk_raw,
        "dr_qwk_calib": dr_qwk_calib,
        # Calibration info
        "calibration":  calibration,
        # Interpretation
        "interpretation": interpretation,
        # Meta
        "num_samples":  int(len(dme_true)),
        "num_dr_samples": int(len(dr_true)),
        "class_counts": {dme_names[i]: int(np.sum(dme_true == i))
                         for i in range(num_dme_classes)},
        "stage_label":  stage_label,
    }

    # ---- Logging ----
    logger.info("=== %s Results ===", stage_label)
    logger.info("DME QWK (raw):   %.4f %s", metrics["qwk"],
                "(✅ TARGET MET)" if metrics["target_met"] else f"(❌ below {QWK_TARGET_THRESHOLD})")
    logger.info("DR  QWK (raw):   %.4f", dr_qwk_raw)
    logger.info("DR  QWK (calib): %.4f %s", dr_qwk_calib,
                "(✅)" if dr_qwk_calib >= QWK_TARGET_THRESHOLD else "(❌)")
    logger.info("DME MAE:         %.3f", metrics["mae"])
    logger.info("DME Accuracy:    %.4f", accuracy)
    logger.info("DME F1 (macro):  %.4f", f1_macro)

    # ---- Save JSON ----
    json_path = os.path.join(output_dir, metrics_path)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Comprehensive metrics saved to '%s'.", json_path)

    # ================================================================
    # VISUALISATIONS
    # ================================================================

    # 1. DME ordinal confusion matrix
    plot_ordinal_confusion_matrix(
        dme_true, dme_pred,
        class_names=dme_names,
        output_path=os.path.join(output_dir, "dme_ordinal_confusion_matrix.png"),
        num_classes=num_dme_classes,
        title=f"DME Ordinal Confusion Matrix — {stage_label}",
    )

    # 2. DME comprehensive dashboard
    plot_comprehensive_dashboard(
        dme_true, dme_pred, dme_proba,
        metrics=metrics,
        class_names=dme_names,
        output_path=os.path.join(output_dir, "dme_comprehensive_dashboard.png"),
        num_classes=num_dme_classes,
    )

    # 3. DR ordinal confusion matrix
    if len(dr_true) > 0:
        plot_ordinal_confusion_matrix(
            dr_true, dr_pred,
            class_names=dr_names,
            output_path=os.path.join(output_dir, "dr_ordinal_confusion_matrix.png"),
            num_classes=num_dr_classes,
            title=f"DR Ordinal Confusion Matrix — {stage_label}",
        )

    # 4. DR comprehensive dashboard
    if len(dr_true) > 0:
        plot_dr_dashboard(
            dr_metrics,
            output_dir=output_dir,
            num_dr_classes=num_dr_classes,
            dr_true=dr_true,
            dr_pred=dr_pred,
            dr_proba=dr_proba,
        )

    # 5. Joint QWK comparison bar chart
    plot_joint_qwk_comparison(
        dme_qwk_raw=metrics["qwk"],
        dr_qwk_raw=dr_qwk_raw,
        dr_qwk_calib=dr_qwk_calib,
        stage_label=stage_label,
        output_path=os.path.join(output_dir, "joint_qwk_comparison.png"),
    )

    # 6. DME ordinal error histogram
    plot_ordinal_error_histogram(
        dme_true, dme_pred,
        title=f"DME Ordinal Error Distribution — {stage_label}",
        output_path=os.path.join(output_dir, "dme_ordinal_error_hist.png"),
    )

    # 7. DR ordinal error histogram
    if len(dr_true) > 0:
        plot_ordinal_error_histogram(
            dr_true, dr_pred,
            title=f"DR Ordinal Error Distribution — {stage_label}",
            output_path=os.path.join(output_dir, "dr_ordinal_error_hist.png"),
        )

    # 8. DME per-class prediction distribution
    plot_per_class_prediction_distribution(
        dme_true, dme_pred,
        class_names=dme_names,
        title=f"DME Per-Class Prediction Distribution — {stage_label}",
        output_path=os.path.join(output_dir, "dme_per_class_pred_dist.png"),
    )

    # 9. DR per-class prediction distribution
    if len(dr_true) > 0:
        plot_per_class_prediction_distribution(
            dr_true, dr_pred,
            class_names=dr_names,
            title=f"DR Per-Class Prediction Distribution — {stage_label}",
            output_path=os.path.join(output_dir, "dr_per_class_pred_dist.png"),
        )

    # 10. DME multi-class ROC curves
    plot_multiclass_roc(
        dme_true, dme_proba,
        class_names=dme_names,
        title=f"DME ROC Curves — {stage_label}",
        output_path=os.path.join(output_dir, "dme_roc_curves.png"),
    )

    # 11. DR multi-class ROC curves
    if len(dr_true) > 0 and dr_proba is not None and dr_proba.shape[1] == num_dr_classes:
        plot_multiclass_roc(
            dr_true, dr_proba,
            class_names=dr_names,
            title=f"DR ROC Curves — {stage_label}",
            output_path=os.path.join(output_dir, "dr_roc_curves.png"),
        )

    return metrics


# ---------------------------------------------------------------------------
# Top-level helper: load best joint model and evaluate once
# ---------------------------------------------------------------------------

def evaluate_on_best_joint_model(
    checkpoint_path: str,
    dataset: tf.data.Dataset,
    output_dir: str,
    stage_label: str = "Stage",
    num_dme_classes: int = NUM_DME_CLASSES,
    num_dr_classes: int = NUM_DR_CLASSES,
    calibrate_dr_thresholds: bool = True,
    calibrate_dme_thresholds: bool = False,
) -> Dict:
    """Load the best joint model checkpoint and run full evaluation once.

    This is the function to call from train.py at the END of each stage,
    NOT inside the epoch loop.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved model file. Accepts full Keras SavedModel
        (``model_stage1.keras`` / ``model_stage2.keras``) or HDF5
        (``best_joint.h5`` / ``model_stage2.model.h5``).
    dataset : tf.data.Dataset
        Validation dataset.
    output_dir : str
        Where to save all plots and JSON.
    stage_label : str
        Label used in plot titles, e.g. "Stage 1" or "Stage 2".

    Returns
    -------
    dict
        Full metrics dict from evaluate_comprehensive().
    """
    logger.info("Loading best joint model from: %s", checkpoint_path)
    model = keras.models.load_model(
        checkpoint_path,
        compile=False,
        custom_objects={"ResizeToMatch": _get_resize_to_match()},
    )
    logger.info("Model loaded. Running evaluation for %s …", stage_label)

    return evaluate_comprehensive(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        stage_label=stage_label,
        num_dme_classes=num_dme_classes,
        num_dr_classes=num_dr_classes,
        calibrate_dr_thresholds=calibrate_dr_thresholds,
        calibrate_dme_thresholds=calibrate_dme_thresholds,
    )


def _get_resize_to_match():
    """Return ResizeToMatch class if model.py is importable, else a stub."""
    try:
        from model import ResizeToMatch
        return ResizeToMatch
    except ImportError:
        pass
    try:
        from tensorflow.keras import layers
        import tensorflow as tf

        class ResizeToMatch(layers.Layer):
            def call(self, inputs):
                source, reference = inputs
                target_hw = tf.shape(reference)[1:3]
                return tf.image.resize(source, target_hw)
            def get_config(self):
                return super().get_config()
        return ResizeToMatch
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Comprehensive evaluation CLI — evaluates best joint model."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Comprehensive QWK evaluation — best joint model")
    parser.add_argument("--model",      type=str, default=None,
                        help="Path to saved best joint model (.h5 or SavedModel dir)")
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--image-dir",  type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results_comprehensive")
    parser.add_argument("--stage",      type=str, default="Evaluation",
                        help="Stage label for plot titles, e.g. 'Stage 1'")
    parser.add_argument("--mock",       action="store_true")
    parser.add_argument("--calibrate-dr",  action="store_true",
                        help="Run DR threshold calibration")
    parser.add_argument("--calibrate-dme", action="store_true",
                        help="Run DME threshold calibration")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_comp", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    _, val_ds, _ = build_datasets(csv_path=csv_path, image_dir=image_dir)

    if args.model:
        # Load and evaluate the specified checkpoint
        metrics = evaluate_on_best_joint_model(
            checkpoint_path=args.model,
            dataset=val_ds,
            output_dir=args.output_dir,
            stage_label=args.stage,
            calibrate_dr_thresholds=args.calibrate_dr,
            calibrate_dme_thresholds=args.calibrate_dme,
        )
    else:
        # Build a fresh model (mock/smoke-test path)
        from model import build_model
        model = build_model()
        metrics = evaluate_comprehensive(
            model, val_ds,
            output_dir=args.output_dir,
            stage_label=args.stage,
            calibrate_dr_thresholds=args.calibrate_dr,
            calibrate_dme_thresholds=args.calibrate_dme,
        )

    print(json.dumps({
        "stage":        metrics.get("stage_label"),
        "dme_qwk_raw":  metrics["qwk"],
        "dme_target_met": metrics["target_met"],
        "dr_qwk_raw":   metrics.get("dr_qwk_raw"),
        "dr_qwk_calib": metrics.get("dr_qwk_calib"),
        "dme_mae":      metrics["mae"],
        "dme_accuracy": metrics["accuracy"],
        "dme_f1_macro": metrics["f1_macro"],
        "clinical":     metrics["interpretation"]["clinical_recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()