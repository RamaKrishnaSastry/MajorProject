"""
evaluate_comprehensive.py - Comprehensive QWK-focused evaluation for DR+DME classification.

KEY CHANGES:
- Single-pass inference collects BOTH DR and DME predictions (no double dataset scan)
- Full DR visualisations matching DME: confusion matrices, dashboard, ordinal plots,
  per-class F1, ROC curves, error histograms, per-class distribution, reliability diagrams
- All plots and summaries display RAW QWK only (no calibration applied or shown)
- evaluate_on_best_joint_model() loads best joint checkpoint and runs everything once
- New plots: ordinal error histograms, per-class F1 bars, multi-class ROC,
  joint QWK comparison bar chart, calibration reliability diagrams

Extends evaluate.py with:
- Quadratic Weighted Kappa with detailed computation log (RAW only)
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

QWK_TARGET_THRESHOLD = 0.80

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
        classification_report,
    )
    from sklearn.preprocessing import label_binarize
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

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
    """Generate medical interpretation using RAW QWK values only."""
    qwk      = metrics.get("qwk", 0.0)
    mae      = metrics.get("mae", float("nan"))
    accuracy = metrics.get("accuracy", 0.0)
    f1       = metrics.get("f1_macro", 0.0)
    dr_qwk   = metrics.get("dr_qwk_raw", 0.0)   # RAW only
    interp   = interpret_qwk(qwk)
    return {
        "dme_qwk_interpretation": (
            f"DME QWK (raw) = {qwk:.4f} ({interp}). "
            + ("Model meets the clinical-grade agreement target (>=0.80)."
               if qwk >= QWK_TARGET_THRESHOLD
               else f"Deficit: {QWK_TARGET_THRESHOLD - qwk:.4f}. Consider more training.")
        ),
        "dr_qwk_interpretation": (
            f"DR QWK (raw) = {dr_qwk:.4f} ({interpret_qwk(dr_qwk)}). "
            + ("DR grading meets the clinical-grade agreement target (>=0.80)."
               if dr_qwk >= QWK_TARGET_THRESHOLD
               else f"DR deficit: {QWK_TARGET_THRESHOLD - dr_qwk:.4f}.")
        ),
        "mae_interpretation": (
            f"DME MAE = {mae:.3f} ordinal units. "
            + ("Average prediction error < 1 severity class - clinically acceptable."
               if not np.isnan(mae) and mae < 1.0
               else "Average prediction error spans more than 1 severity class.")
        ),
        "accuracy_interpretation": (
            f"Overall DME accuracy = {accuracy:.4f}. "
            + ("Strong pixel-level agreement." if accuracy >= 0.80
               else "Accuracy below 80% - class imbalance or data quality issues likely.")
        ),
        "f1_interpretation": (
            f"Macro F1 = {f1:.4f}. "
            + ("Good handling of class imbalance." if f1 >= 0.75
               else "Macro F1 below 0.75 - minor classes may be under-represented.")
        ),
        "clinical_recommendation": (
            "PASS: Model is suitable for clinical decision support screening."
            if (qwk >= QWK_TARGET_THRESHOLD and dr_qwk >= QWK_TARGET_THRESHOLD)
            else "FAIL: One or both heads require further training before clinical deployment."
        ),
    }


# ---------------------------------------------------------------------------
# Single-pass inference
# ---------------------------------------------------------------------------

def get_all_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_dme_classes: int = NUM_DME_CLASSES,
    num_dr_classes: int = NUM_DR_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single inference pass collecting DR and DME predictions simultaneously."""
    dme_true_all, dme_proba_all = [], []
    dr_true_all,  dr_proba_all  = [], []

    for batch_images, batch_labels in dataset:
        preds = model(batch_images, training=False)

        if isinstance(preds, dict):
            dme_proba = preds.get("dme_risk", preds.get("dme_output"))
            dr_proba  = preds.get("dr_output")
        elif isinstance(preds, (list, tuple)):
            dr_proba  = preds[0]
            dme_proba = preds[1] if len(preds) > 1 else preds[0]
        else:
            dme_proba = preds
            dr_proba  = None

        if isinstance(batch_labels, dict):
            dme_labels = batch_labels.get("dme_risk", batch_labels.get("dme_output"))
            dr_labels  = batch_labels.get("dr_output")
        elif isinstance(batch_labels, (list, tuple)):
            dr_labels  = batch_labels[0]
            dme_labels = batch_labels[1] if len(batch_labels) > 1 else batch_labels[0]
        else:
            dme_labels = batch_labels
            dr_labels  = None

        dme_proba_np  = dme_proba.numpy() if hasattr(dme_proba, "numpy") else np.asarray(dme_proba)
        dme_labels_np = dme_labels.numpy() if hasattr(dme_labels, "numpy") else np.asarray(dme_labels)
        dme_true_all.append(np.argmax(dme_labels_np, axis=-1))
        dme_proba_all.append(dme_proba_np)

        if dr_proba is not None and dr_labels is not None:
            dr_proba_np  = dr_proba.numpy()  if hasattr(dr_proba,  "numpy") else np.asarray(dr_proba)
            dr_labels_np = dr_labels.numpy() if hasattr(dr_labels, "numpy") else np.asarray(dr_labels)
            if dr_proba_np.ndim > 1 and dr_proba_np.shape[-1] > 1:
                dr_pred_batch = np.argmax(dr_proba_np, axis=-1).astype(int)
            else:
                dr_pred_batch = np.clip(np.rint(dr_proba_np.reshape(-1)).astype(int), 0, num_dr_classes - 1)
            if dr_labels_np.ndim > 1 and dr_labels_np.shape[-1] > 1:
                dr_true_batch = np.argmax(dr_labels_np, axis=-1).astype(int)
            else:
                dr_true_batch = np.clip(np.rint(dr_labels_np.reshape(-1)).astype(int), 0, num_dr_classes - 1)
            dr_true_all.append(dr_true_batch)
            if dr_proba_np.ndim > 1 and dr_proba_np.shape[-1] == num_dr_classes:
                dr_proba_all.append(dr_proba_np)
            else:
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
# Plot helpers (shared for DME and DR) — RAW QWK only
# ---------------------------------------------------------------------------

def plot_ordinal_error_histogram(
    y_true, y_pred,
    title="Ordinal Prediction Error Distribution",
    output_path="ordinal_error_hist.png",
):
    if not _PLOTTING_AVAILABLE:
        return
    errors = y_pred.astype(int) - y_true.astype(int)
    unique_errors = np.arange(errors.min(), errors.max() + 1)
    counts = [int(np.sum(errors == e)) for e in unique_errors]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if e == 0 else "orange" if abs(e) == 1 else "red" for e in unique_errors]
    bars = ax.bar(unique_errors, counts, color=colors, edgecolor="black", alpha=0.85)
    ax.bar_label(bars, fmt="%d", fontsize=9)
    ax.set_xlabel("Prediction Error  (predicted - true)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(unique_errors)
    ax.set_xticklabels([str(e) for e in unique_errors])
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="green",  label="Correct (0)"),
        Patch(facecolor="orange", label="Off-by-1"),
        Patch(facecolor="red",    label="Off-by-2+"),
    ], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ordinal error histogram saved to '%s'.", output_path)


def plot_per_class_prediction_distribution(
    y_true, y_pred, class_names,
    title="Per-Class Prediction Distribution",
    output_path="per_class_pred_dist.png",
):
    if not _PLOTTING_AVAILABLE:
        return
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        mask = y_true == i
        if mask.sum() == 0:
            continue
        for j in range(n):
            matrix[i, j] = np.sum((y_true == i) & (y_pred == j)) / mask.sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    bottoms = np.zeros(n)
    for j in range(n):
        ax.bar(class_names, matrix[:, j], bottom=bottoms,
               label=f"Predicted: {class_names[j]}",
               color=colors[j], edgecolor="black", linewidth=0.5, alpha=0.88)
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
    y_true, y_proba, class_names,
    title="Multi-Class ROC Curves (One-vs-Rest)",
    output_path="roc_curves.png",
):
    if not (_PLOTTING_AVAILABLE and _SKLEARN_AVAILABLE):
        return
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if y_bin.ndim == 1:
        y_bin = y_bin[:, None]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin.shape[1] <= i:
            continue
        col = y_bin[:, i]
        if len(np.unique(col)) < 2:
            continue
        prob_col = y_proba[:, i] if y_proba.shape[1] > i else np.zeros(len(y_true))
        fpr, tpr, _ = roc_curve(col, prob_col)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
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
    dme_qwk_raw, dr_qwk_raw,
    stage_label="Current Stage",
    output_path="joint_qwk_comparison.png",
):
    """Bar chart comparing DME and DR raw QWK values only."""
    if not _PLOTTING_AVAILABLE:
        return
    labels = ["DME QWK\n(raw)", "DR QWK\n(raw)"]
    values = [dme_qwk_raw, dr_qwk_raw]
    colors = ["steelblue", "darkorange"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.88, width=0.40)
    ax.bar_label(bars, fmt="%.4f", fontsize=12, padding=4)
    ax.axhline(
        QWK_TARGET_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
        label=f"Clinical target = {QWK_TARGET_THRESHOLD}",
    )
    ax.set_ylim([0, 1.10])
    ax.set_ylabel("QWK Score (raw)", fontsize=12)
    ax.set_title(f"Joint Raw QWK Comparison - {stage_label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Joint QWK comparison saved to '%s'.", output_path)


def plot_reliability_diagram(
    y_true, y_proba, class_names,
    title="Calibration Reliability Diagram",
    output_path="reliability_diagram.png",
    n_bins=10,
):
    """Reliability diagram: mean predicted confidence vs actual accuracy per bin."""
    if not _PLOTTING_AVAILABLE:
        return
    confidences = np.max(y_proba, axis=1)
    correct = (np.argmax(y_proba, axis=1) == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(correct[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(int(mask.sum()))
    if not bin_confs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].plot(bin_confs, bin_accs, "o-", color="steelblue",
                 linewidth=2, markersize=6, label="Model")
    axes[0].fill_between(bin_confs, bin_accs, bin_confs,
                         alpha=0.2, color="tomato", label="Calibration gap")
    axes[0].set_xlabel("Mean Predicted Confidence", fontsize=12)
    axes[0].set_ylabel("Fraction Correct", fontsize=12)
    axes[0].set_title(f"{title}\n(Reliability Curve)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3)
    axes[1].bar(bin_confs, bin_counts,
                width=(1.0 / n_bins) * 0.85,
                color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Mean Predicted Confidence", fontsize=12)
    axes[1].set_ylabel("Sample Count", fontsize=12)
    axes[1].set_title("Confidence Distribution", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Reliability diagram saved to '%s'.", output_path)


def plot_per_class_f1_bar(
    y_true, y_pred, class_names,
    title="Per-Class F1 Score",
    output_path="per_class_f1.png",
):
    """Bar chart of per-class F1 scores with colour-coded thresholds."""
    if not (_PLOTTING_AVAILABLE and _SKLEARN_AVAILABLE):
        return
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    f1_scores = [report.get(name, {}).get("f1-score", 0.0) for name in class_names]
    colors = ["green" if f >= 0.80 else "orange" if f >= 0.70 else "tomato" for f in f1_scores]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, f1_scores, color=colors, edgecolor="black", alpha=0.88)
    ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.axhline(0.80, color="red",    linestyle="--", linewidth=1.5, label="Target 0.80")
    ax.axhline(0.70, color="orange", linestyle=":",  linewidth=1.2, label="Acceptable 0.70")
    ax.set_ylim([0, 1.10])
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Per-class F1 bar chart saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# DR comprehensive dashboard — RAW QWK only
# ---------------------------------------------------------------------------

def plot_dr_dashboard(
    dr_metrics, output_dir,
    num_dr_classes=NUM_DR_CLASSES,
    dr_true=None, dr_pred=None, dr_proba=None,
):
    """2x3 DR grading evaluation dashboard showing RAW QWK only."""
    if not _PLOTTING_AVAILABLE:
        return
    names = DR_CLASS_NAMES[:num_dr_classes]
    cm_data = dr_metrics.get("confusion_matrix")
    if cm_data is not None:
        cm = np.array(cm_data)
    elif dr_true is not None and dr_pred is not None:
        cm = (
            sk_cm(dr_true, dr_pred, labels=list(range(num_dr_classes)))
            if _SKLEARN_AVAILABLE
            else np.array([
                [int(np.sum((dr_true == i) & (dr_pred == j))) for j in range(num_dr_classes)]
                for i in range(num_dr_classes)
            ])
        )
    else:
        cm = np.zeros((num_dr_classes, num_dr_classes), dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0, 0])
    axes[0, 0].set_title("DR Confusion Matrix (Counts)", fontsize=12)
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names, vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("DR Confusion Matrix (Normalised)", fontsize=12)
    axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("True")

    if dr_true is not None and dr_pred is not None and len(dr_true) > 0 and _SKLEARN_AVAILABLE:
        report = classification_report(
            dr_true, dr_pred,
            labels=list(range(num_dr_classes)),
            target_names=names,
            output_dict=True, zero_division=0,
        )
        f1_scores = [report.get(n, {}).get("f1-score", 0.0) for n in names]
        colors_f1 = ["green" if f >= 0.80 else "orange" if f >= 0.70 else "tomato" for f in f1_scores]
        bars_f1 = axes[0, 2].bar(names, f1_scores, color=colors_f1, edgecolor="black", alpha=0.88)
        axes[0, 2].bar_label(bars_f1, fmt="%.3f", fontsize=8, padding=2)
        axes[0, 2].axhline(0.80, color="red", linestyle="--", linewidth=1, label="Target 0.80")
        axes[0, 2].set_title("DR Per-Class F1 Score", fontsize=12)
        axes[0, 2].set_xlabel("Class"); axes[0, 2].set_ylabel("F1")
        axes[0, 2].set_ylim([0, 1.10]); axes[0, 2].legend(fontsize=7)
    else:
        axes[0, 2].text(0.5, 0.5, "No prediction data", ha="center", va="center")

    if dr_true is not None and dr_pred is not None and len(dr_true) > 0:
        errors = dr_pred.astype(int) - dr_true.astype(int)
        ue = np.arange(int(errors.min()), int(errors.max()) + 1)
        counts = [int(np.sum(errors == e)) for e in ue]
        colors_e = ["green" if e == 0 else "orange" if abs(e) == 1 else "red" for e in ue]
        bars = axes[1, 0].bar(ue, counts, color=colors_e, edgecolor="black", alpha=0.85)
        axes[1, 0].bar_label(bars, fmt="%d", fontsize=8)
        axes[1, 0].set_xlabel("Prediction Error (pred - true)", fontsize=10)
        axes[1, 0].set_ylabel("Count", fontsize=10)
        axes[1, 0].set_title("DR Ordinal Error Distribution", fontsize=12)
        axes[1, 0].set_xticks(ue); axes[1, 0].grid(axis="y", alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "No error data", ha="center", va="center")

    ordinal   = dr_metrics.get("ordinal", {})
    boundary  = ordinal.get("boundary_confusion", {})
    if boundary:
        b_labels = [v.get("description", k) for k, v in boundary.items()]
        b_rates  = [v.get("confusion_rate", 0.0) for v in boundary.values()]
        axes[1, 1].bar(b_labels, b_rates, color="darkorchid", edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("DR Boundary Confusion Rates", fontsize=12)
        axes[1, 1].set_xlabel("Class Boundary")
        axes[1, 1].set_ylabel("Confusion Rate")
        axes[1, 1].set_ylim([0, 1.05])
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(20); tick.set_ha("right")
    else:
        axes[1, 1].text(0.5, 0.5, "No boundary data", ha="center", va="center")

    # Summary panel — RAW QWK only
    dr_qwk_raw = dr_metrics.get("dr_qwk", float("nan"))
    dr_mae     = dr_metrics.get("dr_mae",      float("nan"))
    dr_rmse    = dr_metrics.get("dr_rmse",     float("nan"))
    dr_acc     = dr_metrics.get("dr_accuracy", float("nan"))
    axes[1, 2].axis("off")
    summary = (
        f"DR QWK (raw):  {dr_qwk_raw:.4f}  {'OK' if dr_qwk_raw >= QWK_TARGET_THRESHOLD else 'X'}\n"
        f"DR MAE:        {dr_mae:.3f}\n"
        f"DR RMSE:       {dr_rmse:.3f}\n"
        f"DR Accuracy:   {dr_acc:.4f}\n\n"
        f"Interpretation: {interpret_qwk(dr_qwk_raw).upper()}\n\n"
        f"Clinical: {'PASS' if dr_qwk_raw >= QWK_TARGET_THRESHOLD else 'FAIL'}"
    )
    axes[1, 2].text(
        0.05, 0.5, summary,
        transform=axes[1, 2].transAxes, fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        fontfamily="monospace",
    )
    axes[1, 2].set_title("DR Grading Summary (Raw QWK)", fontsize=12)
    fig.suptitle("DR Grading Evaluation Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "dr_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("DR dashboard saved to '%s'.", path)


# ---------------------------------------------------------------------------
# DME comprehensive dashboard — RAW QWK only
# ---------------------------------------------------------------------------

def plot_comprehensive_dashboard(
    y_true, y_pred, y_proba, metrics,
    class_names=None,
    output_path="comprehensive_dashboard.png",
    num_classes=NUM_DME_CLASSES,
):
    """2x3 DME evaluation dashboard. Summary panel shows raw QWK for both DME and DR."""
    if not _PLOTTING_AVAILABLE:
        return
    names = class_names or DME_CLASS_NAMES[:num_classes]
    cm = (
        sk_cm(y_true, y_pred, labels=list(range(num_classes)))
        if _SKLEARN_AVAILABLE
        else np.array([
            [int(np.sum((y_true == i) & (y_pred == j))) for j in range(num_classes)]
            for i in range(num_classes)
        ])
    )
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0, 0])
    axes[0, 0].set_title("DME Confusion Matrix (Counts)", fontsize=12)
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names, vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("DME Confusion Matrix (Normalised)", fontsize=12)
    axes[0, 1].set_xlabel("Predicted"); axes[0, 1].set_ylabel("True")

    per_class = metrics.get("per_class", {})
    pc_names  = list(per_class.keys())
    pc_f1     = [per_class[n].get("f1", 0.0) for n in pc_names]
    colors_f1 = ["green" if f >= 0.80 else "orange" if f >= 0.70 else "tomato" for f in pc_f1]
    axes[0, 2].bar(pc_names, pc_f1, color=colors_f1, edgecolor="black", alpha=0.85)
    axes[0, 2].set_title("DME Per-Class F1 Score", fontsize=12)
    axes[0, 2].set_xlabel("Class"); axes[0, 2].set_ylabel("F1")
    axes[0, 2].set_ylim([0, 1.10])
    axes[0, 2].axhline(0.80, color="red", linestyle="--", linewidth=1, label="Target 0.80")
    axes[0, 2].legend(fontsize=8)

    pc_ordinal = metrics.get("ordinal", {}).get("per_class", {})
    if pc_ordinal:
        oc_names = list(pc_ordinal.keys())
        oc_dist  = [pc_ordinal[n].get("mean_misclassification_distance", 0.0) for n in oc_names]
        axes[1, 0].bar(oc_names, oc_dist, color="tomato", edgecolor="black", alpha=0.85)
        axes[1, 0].set_title("Mean Misclassification Distance (DME)", fontsize=12)
        axes[1, 0].set_xlabel("True Class"); axes[1, 0].set_ylabel("Mean |true - pred|")
    else:
        axes[1, 0].text(0.5, 0.5, "No ordinal data", ha="center", va="center")

    boundary = metrics.get("ordinal", {}).get("boundary_confusion", {})
    if boundary:
        b_labels = [v.get("description", k) for k, v in boundary.items()]
        b_rates  = [v.get("confusion_rate", 0.0) for v in boundary.values()]
        axes[1, 1].bar(b_labels, b_rates, color="darkorchid", edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("DME Boundary Confusion Rates", fontsize=12)
        axes[1, 1].set_xlabel("Class Boundary")
        axes[1, 1].set_ylabel("Confusion Rate")
        axes[1, 1].set_ylim([0, 1.05])
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(20); tick.set_ha("right")
    else:
        axes[1, 1].text(0.5, 0.5, "No boundary data", ha="center", va="center")

    # Summary panel — RAW QWK only for both heads
    dme_qwk_raw = metrics.get("qwk", 0.0)
    dr_qwk_raw  = metrics.get("dr_qwk_raw", float("nan"))
    axes[1, 2].axis("off")
    summary_text = (
        f"DME QWK (raw):  {dme_qwk_raw:.4f}  {'OK' if dme_qwk_raw >= QWK_TARGET_THRESHOLD else 'X'}\n"
        f"DR  QWK (raw):  {dr_qwk_raw:.4f}  {'OK' if dr_qwk_raw >= QWK_TARGET_THRESHOLD else 'X'}\n"
        f"DME MAE:     {metrics.get('mae', float('nan')):.3f}\n"
        f"DME RMSE:    {metrics.get('rmse', float('nan')):.3f}\n"
        f"DME Acc:     {metrics.get('accuracy', 0.0):.4f}\n"
        f"F1 Macro:    {metrics.get('f1_macro', 0.0):.4f}\n\n"
        f"DME: {interpret_qwk(dme_qwk_raw).upper()}\n"
        f"DR:  {interpret_qwk(dr_qwk_raw).upper()}"
    )
    axes[1, 2].text(
        0.05, 0.5, summary_text,
        transform=axes[1, 2].transAxes, fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )
    axes[1, 2].set_title("Joint Summary — Raw QWK (DME + DR)", fontsize=12)
    fig.suptitle("Comprehensive DME Evaluation Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comprehensive dashboard saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# DR grading evaluation (pre-collected arrays — no second dataset scan)
# RAW QWK only; no threshold calibration applied or stored in plots.
# ---------------------------------------------------------------------------

def evaluate_dr_grading(
    dr_true, dr_pred, dr_proba, output_dir,
    num_dr_classes=NUM_DR_CLASSES,
):
    """Evaluate DR head using RAW argmax predictions only.

    Calibration has been removed from this function entirely so that every
    metric and plot reflects the model's unmodified output distribution.
    """
    if len(dr_true) == 0:
        logger.warning("No DR labels found - skipping DR evaluation.")
        return {}
    dr_true      = np.asarray(dr_true, dtype=np.int32)
    dr_pred_final = np.asarray(dr_pred, dtype=np.int32)

    if _SKLEARN_AVAILABLE:
        dr_qwk = float(cohen_kappa_score(dr_true, dr_pred_final, weights="quadratic"))
        dr_acc = float(accuracy_score(dr_true, dr_pred_final))
        cm     = sk_cm(dr_true, dr_pred_final, labels=list(range(num_dr_classes)))
    else:
        dr_qwk = float(compute_quadratic_weighted_kappa(dr_true, dr_pred_final, num_dr_classes))
        dr_acc = float(np.mean(dr_true == dr_pred_final))
        cm     = np.array([
            [int(np.sum((dr_true == i) & (dr_pred_final == j))) for j in range(num_dr_classes)]
            for i in range(num_dr_classes)
        ])

    dr_mae     = float(np.mean(np.abs(dr_true - dr_pred_final)))
    dr_rmse    = float(np.sqrt(np.mean((dr_true - dr_pred_final) ** 2)))
    dr_ordinal = compute_ordinal_metrics(
        dr_true, dr_pred_final, num_dr_classes,
        class_names=DR_CLASS_NAMES[:num_dr_classes],
    )

    logger.info(
        "DR Grading Results (raw): QWK=%.4f  Acc=%.4f  MAE=%.4f  RMSE=%.4f",
        dr_qwk, dr_acc, dr_mae, dr_rmse,
    )

    results = {
        "dr_qwk":          dr_qwk,
        "dr_accuracy":     dr_acc,
        "dr_mae":          dr_mae,
        "dr_rmse":         dr_rmse,
        "confusion_matrix": cm.tolist(),
        "ordinal":         dr_ordinal,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dr_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("DR metrics saved.")
    return results


# ---------------------------------------------------------------------------
# Comprehensive evaluation pipeline — RAW QWK throughout
# ---------------------------------------------------------------------------

def evaluate_comprehensive(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    output_dir: str = ".",
    metrics_path: str = "comprehensive_metrics.json",
    num_dme_classes: int = NUM_DME_CLASSES,
    num_dr_classes: int = NUM_DR_CLASSES,
    stage_label: str = "Evaluation",
    # Kept for API compatibility — ignored (no calibration applied)
    calibrate_dme_thresholds: bool = False,
    calibrate_dr_thresholds: bool = False,
    calibration_min_qwk_gain: float = 1e-4,
    dr_calibration_max_accuracy_drop: float = 0.0,
) -> Dict:
    """Full evaluation pipeline for BOTH DR and DME heads via single inference pass.

    Produces RAW QWK metrics and plots only.  No threshold calibration is
    applied regardless of the ``calibrate_*`` flags (kept for backward
    compatibility with call sites in train_enhanced.py).

    Call this ONCE on the best saved joint model — not inside the epoch loop.
    """
    os.makedirs(output_dir, exist_ok=True)
    dme_names = class_names or DME_CLASS_NAMES[:num_dme_classes]
    dr_names  = DR_CLASS_NAMES[:num_dr_classes]

    logger.info("=== %s: Running single-pass inference ===", stage_label)
    dme_true, dme_proba, dme_pred, dr_true, dr_proba, dr_pred = get_all_predictions(
        model, dataset,
        num_dme_classes=num_dme_classes,
        num_dr_classes=num_dr_classes,
    )
    logger.info(
        "Collected %d DME samples, %d DR samples.", len(dme_true), len(dr_true)
    )

    # ---- DME metrics (raw argmax) ----------------------------------------
    accuracy    = compute_accuracy(dme_true, dme_pred)
    f1_macro    = compute_f1(dme_true, dme_pred, average="macro")
    f1_weighted = compute_f1(dme_true, dme_pred, average="weighted")
    roc_auc_dme = compute_roc_auc(dme_true, dme_proba, num_classes=num_dme_classes)
    per_class   = compute_per_class_metrics(dme_true, dme_pred, dme_names)
    qwk_details = compute_qwk_with_details(dme_true, dme_pred, num_dme_classes)
    ordinal_metrics = compute_ordinal_metrics(dme_true, dme_pred, num_dme_classes, dme_names)

    # ---- DR metrics (raw argmax, no calibration) -------------------------
    dr_metrics = evaluate_dr_grading(
        dr_true, dr_pred, dr_proba,
        output_dir=output_dir,
        num_dr_classes=num_dr_classes,
    )
    dr_qwk_raw = dr_metrics.get("dr_qwk", float("nan"))

    interpretation = generate_medical_interpretation({
        "qwk":        qwk_details["qwk"],
        "dr_qwk_raw": dr_qwk_raw,
        "mae":        ordinal_metrics["mae"],
        "accuracy":   accuracy,
        "f1_macro":   f1_macro,
        "per_class":  per_class,
    })

    metrics = {
        "qwk":          qwk_details["qwk"],
        "target_met":   qwk_details["qwk"] >= QWK_TARGET_THRESHOLD,
        "mae":          ordinal_metrics["mae"],
        "rmse":         ordinal_metrics["rmse"],
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "roc_auc":      roc_auc_dme,
        "per_class":    per_class,
        "ordinal":      ordinal_metrics,
        "qwk_details":  qwk_details,
        "dr":           dr_metrics,
        "dr_qwk_raw":   dr_qwk_raw,
        "interpretation": interpretation,
        "num_samples":    int(len(dme_true)),
        "num_dr_samples": int(len(dr_true)),
        "class_counts": {
            dme_names[i]: int(np.sum(dme_true == i))
            for i in range(num_dme_classes)
        },
        "stage_label":  stage_label,
    }

    logger.info("=== %s Results ===", stage_label)
    logger.info(
        "DME QWK (raw):  %.4f %s",
        metrics["qwk"],
        "(TARGET MET)" if metrics["target_met"] else f"(below {QWK_TARGET_THRESHOLD})",
    )
    logger.info(
        "DR  QWK (raw):  %.4f %s",
        dr_qwk_raw,
        "(OK)" if dr_qwk_raw >= QWK_TARGET_THRESHOLD else "(below target)",
    )
    logger.info(
        "DME MAE: %.3f  Accuracy: %.4f  F1(macro): %.4f",
        metrics["mae"], accuracy, f1_macro,
    )

    with open(os.path.join(output_dir, metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Comprehensive metrics saved.")

    # ================================================================
    # VISUALISATIONS — DME and DR parity, RAW QWK only
    # ================================================================

    # DME: ordinal confusion matrix
    plot_ordinal_confusion_matrix(
        dme_true, dme_pred,
        class_names=dme_names,
        output_path=os.path.join(output_dir, "dme_ordinal_confusion_matrix.png"),
        num_classes=num_dme_classes,
        title=f"DME Ordinal Confusion Matrix - {stage_label}",
    )

    # DME: comprehensive dashboard (2x3)
    plot_comprehensive_dashboard(
        dme_true, dme_pred, dme_proba,
        metrics=metrics,
        class_names=dme_names,
        output_path=os.path.join(output_dir, "dme_comprehensive_dashboard.png"),
        num_classes=num_dme_classes,
    )

    # DR: ordinal confusion matrix
    if len(dr_true) > 0:
        plot_ordinal_confusion_matrix(
            dr_true, dr_pred,
            class_names=dr_names,
            output_path=os.path.join(output_dir, "dr_ordinal_confusion_matrix.png"),
            num_classes=num_dr_classes,
            title=f"DR Ordinal Confusion Matrix - {stage_label}",
        )

    # DR: comprehensive dashboard (2x3)
    if len(dr_true) > 0:
        plot_dr_dashboard(
            dr_metrics, output_dir=output_dir,
            num_dr_classes=num_dr_classes,
            dr_true=dr_true, dr_pred=dr_pred, dr_proba=dr_proba,
        )

    # Joint QWK bar chart — RAW only (2 bars, not 3)
    plot_joint_qwk_comparison(
        dme_qwk_raw=metrics["qwk"],
        dr_qwk_raw=dr_qwk_raw,
        stage_label=stage_label,
        output_path=os.path.join(output_dir, "joint_qwk_comparison.png"),
    )

    # DME: ordinal error histogram
    plot_ordinal_error_histogram(
        dme_true, dme_pred,
        title=f"DME Ordinal Error Distribution - {stage_label}",
        output_path=os.path.join(output_dir, "dme_ordinal_error_hist.png"),
    )

    # DR: ordinal error histogram
    if len(dr_true) > 0:
        plot_ordinal_error_histogram(
            dr_true, dr_pred,
            title=f"DR Ordinal Error Distribution - {stage_label}",
            output_path=os.path.join(output_dir, "dr_ordinal_error_hist.png"),
        )

    # DME: per-class prediction distribution
    plot_per_class_prediction_distribution(
        dme_true, dme_pred,
        class_names=dme_names,
        title=f"DME Per-Class Prediction Distribution - {stage_label}",
        output_path=os.path.join(output_dir, "dme_per_class_pred_dist.png"),
    )

    # DR: per-class prediction distribution
    if len(dr_true) > 0:
        plot_per_class_prediction_distribution(
            dr_true, dr_pred,
            class_names=dr_names,
            title=f"DR Per-Class Prediction Distribution - {stage_label}",
            output_path=os.path.join(output_dir, "dr_per_class_pred_dist.png"),
        )

    # DME: multi-class ROC
    plot_multiclass_roc(
        dme_true, dme_proba,
        class_names=dme_names,
        title=f"DME ROC Curves - {stage_label}",
        output_path=os.path.join(output_dir, "dme_roc_curves.png"),
    )

    # DR: multi-class ROC
    if len(dr_true) > 0 and dr_proba is not None and dr_proba.shape[1] == num_dr_classes:
        plot_multiclass_roc(
            dr_true, dr_proba,
            class_names=dr_names,
            title=f"DR ROC Curves - {stage_label}",
            output_path=os.path.join(output_dir, "dr_roc_curves.png"),
        )

    # DME: reliability / calibration diagram
    plot_reliability_diagram(
        dme_true, dme_proba,
        class_names=dme_names,
        title=f"DME Calibration - {stage_label}",
        output_path=os.path.join(output_dir, "dme_reliability_diagram.png"),
    )

    # DR: reliability / calibration diagram
    if len(dr_true) > 0 and dr_proba is not None and dr_proba.shape[1] == num_dr_classes:
        plot_reliability_diagram(
            dr_true, dr_proba,
            class_names=dr_names,
            title=f"DR Calibration - {stage_label}",
            output_path=os.path.join(output_dir, "dr_reliability_diagram.png"),
        )

    # DME: standalone per-class F1 bar chart
    plot_per_class_f1_bar(
        dme_true, dme_pred,
        class_names=dme_names,
        title=f"DME Per-Class F1 Score - {stage_label}",
        output_path=os.path.join(output_dir, "dme_per_class_f1.png"),
    )

    # DR: standalone per-class F1 bar chart
    if len(dr_true) > 0:
        plot_per_class_f1_bar(
            dr_true, dr_pred,
            class_names=dr_names,
            title=f"DR Per-Class F1 Score - {stage_label}",
            output_path=os.path.join(output_dir, "dr_per_class_f1.png"),
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
    # Kept for API compatibility — ignored (no calibration applied)
    calibrate_dr_thresholds: bool = False,
    calibrate_dme_thresholds: bool = False,
) -> Dict:
    """Load the best joint model checkpoint and run full RAW evaluation once.

    Call from train_enhanced.py at the END of each stage, not inside the epoch loop.
    """
    logger.info("Loading best joint model from: %s", checkpoint_path)
    model = keras.models.load_model(
        checkpoint_path,
        compile=False,
        custom_objects={"ResizeToMatch": _get_resize_to_match()},
    )
    logger.info("Model loaded. Running evaluation for %s ...", stage_label)
    return evaluate_comprehensive(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        stage_label=stage_label,
        num_dme_classes=num_dme_classes,
        num_dr_classes=num_dr_classes,
    )


def _get_resize_to_match():
    try:
        from model import ResizeToMatch
        return ResizeToMatch
    except ImportError:
        pass
    try:
        from tensorflow.keras import layers

        class ResizeToMatch(layers.Layer):
            def call(self, inputs):
                source, reference = inputs
                return tf.image.resize(source, tf.shape(reference)[1:3])

            def get_config(self):
                return super().get_config()

        return ResizeToMatch
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(
        description="Comprehensive RAW QWK evaluation - best joint model"
    )
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--image-dir",  type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results_comprehensive")
    parser.add_argument("--stage",      type=str, default="Evaluation")
    parser.add_argument("--mock",       action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_comp", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    _, val_ds, _ = build_datasets(csv_path=csv_path, image_dir=image_dir)

    if args.model:
        metrics = evaluate_on_best_joint_model(
            checkpoint_path=args.model,
            dataset=val_ds,
            output_dir=args.output_dir,
            stage_label=args.stage,
        )
    else:
        from model import build_model
        model = build_model()
        metrics = evaluate_comprehensive(
            model, val_ds,
            output_dir=args.output_dir,
            stage_label=args.stage,
        )

    print(json.dumps({
        "stage":          metrics.get("stage_label"),
        "dme_qwk_raw":    metrics["qwk"],
        "dme_target_met": metrics["target_met"],
        "dr_qwk_raw":     metrics.get("dr_qwk_raw"),
        "dme_mae":        metrics["mae"],
        "dme_accuracy":   metrics["accuracy"],
        "dme_f1_macro":   metrics["f1_macro"],
        "clinical":       metrics["interpretation"]["clinical_recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()