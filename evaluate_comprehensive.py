"""
evaluate_comprehensive.py - Comprehensive QWK-focused evaluation for DR+DME classification.

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
    get_predictions,
    compute_accuracy,
    compute_f1,
    compute_roc_auc,
    compute_per_class_metrics,
    DME_CLASS_NAMES,
    NUM_DME_CLASSES,
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

# Clinical QWK target – models must meet or exceed this for screening use
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
    """Return a clinical interpretation label for a QWK score.

    Parameters
    ----------
    qwk : float
        QWK score.

    Returns
    -------
    str
        Interpretation category string.
    """
    for label, (lo, hi) in QWK_THRESHOLDS.items():
        if lo <= qwk < hi:
            return label
    return "excellent" if qwk >= QWK_TARGET_THRESHOLD else "poor"


def generate_medical_interpretation(metrics: Dict) -> Dict[str, str]:
    """Generate a plain-language clinical interpretation of evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Metrics dict from :func:`evaluate_comprehensive`.

    Returns
    -------
    dict
        Human-readable interpretation for each key metric.
    """
    qwk = metrics.get("qwk", 0.0)
    mae = metrics.get("mae", float("nan"))
    accuracy = metrics.get("accuracy", 0.0)
    f1 = metrics.get("f1_macro", 0.0)
    interp = interpret_qwk(qwk)

    interpretation = {
        "qwk_interpretation": (
            f"QWK = {qwk:.4f} ({interp}). "
            + (
                "Model meets the clinical-grade agreement target (≥0.80). "
                "Suitable for screening assistance."
                if qwk >= QWK_TARGET_THRESHOLD
                else f"Model has not yet reached the ≥0.80 QWK target "
                     f"(deficit: {QWK_TARGET_THRESHOLD - qwk:.4f}). Consider more training."
            )
        ),
        "mae_interpretation": (
            f"MAE = {mae:.3f} ordinal units. "
            + (
                "Average prediction error is less than 1 severity class – clinically acceptable."
                if not np.isnan(mae) and mae < 1.0
                else "Average prediction error spans more than 1 severity class – needs improvement."
            )
        ),
        "accuracy_interpretation": (
            f"Overall accuracy = {accuracy:.4f}. "
            + (
                "Strong pixel-level agreement across all classes."
                if accuracy >= 0.80
                else "Accuracy below 80% – class imbalance or data quality issues likely."
            )
        ),
        "f1_interpretation": (
            f"Macro F1 = {f1:.4f}. "
            + (
                "Good handling of class imbalance."
                if f1 >= 0.75
                else "Macro F1 below 0.75 – minor classes may be under-represented."
            )
        ),
        "clinical_recommendation": (
            "PASS: Model is suitable for clinical decision support screening."
            if qwk >= QWK_TARGET_THRESHOLD
            else "FAIL: Model requires further training before clinical deployment."
        ),
    }
    return interpretation


# ---------------------------------------------------------------------------
# Comprehensive visualisation
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
    """Generate a 2×3 evaluation dashboard figure.

    Panels:
    1. Confusion matrix (counts)
    2. Confusion matrix (normalised)
    3. Per-class F1 bar chart
    4. Per-class ordinal MAE
    5. Boundary confusion rates
    6. QWK gauge / annotation

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities.
    metrics : dict
        Pre-computed metrics dict.
    class_names : list, optional
        Class names.
    output_path : str
        Output PNG path.
    num_classes : int
        Number of ordinal classes.
    """
    if not _PLOTTING_AVAILABLE:
        logger.warning("Plotting unavailable – skipping dashboard.")
        return

    from sklearn.metrics import confusion_matrix as sk_cm

    names = class_names or DME_CLASS_NAMES[:num_classes]
    try:
        cm = sk_cm(y_true, y_pred, labels=list(range(num_classes)))
    except Exception:
        cm = np.array(
            [
                [int(np.sum((y_true == i) & (y_pred == j))) for j in range(num_classes)]
                for i in range(num_classes)
            ]
        )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Confusion matrix – counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix (Counts)", fontsize=12)
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")

    # 2. Confusion matrix – normalised
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names,
                vmin=0, vmax=1, ax=axes[0, 1])
    axes[0, 1].set_title("Confusion Matrix (Normalised)", fontsize=12)
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")

    # 3. Per-class F1
    per_class = metrics.get("per_class", {})
    pc_names = list(per_class.keys())
    pc_f1 = [per_class[n].get("f1", 0.0) for n in pc_names]
    axes[0, 2].bar(pc_names, pc_f1, color="royalblue", edgecolor="black", alpha=0.85)
    axes[0, 2].set_title("Per-Class F1 Score", fontsize=12)
    axes[0, 2].set_xlabel("Class")
    axes[0, 2].set_ylabel("F1")
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].axhline(0.75, color="red", linestyle="--", linewidth=1, label="Target 0.75")
    axes[0, 2].legend(fontsize=8)

    # 4. Per-class ordinal misclassification distance
    pc_ordinal = metrics.get("ordinal", {}).get("per_class", {})
    if pc_ordinal:
        oc_names = list(pc_ordinal.keys())
        oc_dist = [pc_ordinal[n].get("mean_misclassification_distance", 0.0) for n in oc_names]
        axes[1, 0].bar(oc_names, oc_dist, color="tomato", edgecolor="black", alpha=0.85)
        axes[1, 0].set_title("Mean Misclassification Distance (Ordinal)", fontsize=12)
        axes[1, 0].set_xlabel("True Class")
        axes[1, 0].set_ylabel("Mean |true − pred|")
    else:
        axes[1, 0].text(0.5, 0.5, "No ordinal data", ha="center", va="center", fontsize=12)

    # 5. Boundary confusion rates
    boundary = metrics.get("ordinal", {}).get("boundary_confusion", {})
    if boundary:
        b_labels = [v.get("description", k) for k, v in boundary.items()]
        b_rates = [v.get("confusion_rate", 0.0) for v in boundary.values()]
        axes[1, 1].bar(b_labels, b_rates, color="darkorchid", edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("Boundary Confusion Rates", fontsize=12)
        axes[1, 1].set_xlabel("Class Boundary")
        axes[1, 1].set_ylabel("Confusion Rate")
        axes[1, 1].set_ylim([0, 1.05])
        for tick in axes[1, 1].get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    else:
        axes[1, 1].text(0.5, 0.5, "No boundary data", ha="center", va="center", fontsize=12)

    # 6. QWK summary panel
    qwk = metrics.get("qwk", 0.0)
    mae = metrics.get("mae", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    acc = metrics.get("accuracy", 0.0)
    f1_macro = metrics.get("f1_macro", 0.0)

    axes[1, 2].axis("off")
    summary_text = (
        f"QWK:      {qwk:.4f}  {'✅ TARGET MET' if qwk >= QWK_TARGET_THRESHOLD else '❌ BELOW TARGET'}\n"
        f"MAE:      {mae:.3f}\n"
        f"RMSE:     {rmse:.3f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"F1 Macro: {f1_macro:.4f}\n\n"
        f"Interpretation:\n{interpret_qwk(qwk).upper()}"
    )
    axes[1, 2].text(
        0.1, 0.5, summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=13, verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )
    axes[1, 2].set_title("QWK Summary", fontsize=12)

    fig.suptitle("Comprehensive DME Evaluation Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comprehensive dashboard saved to '%s'.", output_path)


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
) -> Dict:
    """Full evaluation pipeline including QWK, ordinal metrics, and visualisations.

    Parameters
    ----------
    model : keras.Model
        Trained DME model.
    dataset : tf.data.Dataset
        Validation / test dataset yielding (images, one_hot_labels).
    class_names : list, optional
        Human-readable class names.
    output_dir : str
        Output directory for artefacts.
    metrics_path : str
        Filename for the JSON metrics report (inside *output_dir*).
    num_dme_classes : int
        Number of DME classes.

    Returns
    -------
    dict
        Comprehensive metrics dict including QWK, ordinal, classification metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = class_names or DME_CLASS_NAMES[:num_dme_classes]

    logger.info("Running comprehensive inference …")
    y_true, y_proba, y_pred = get_predictions(model, dataset, num_dme_classes)

    # --- Standard classification metrics ---
    accuracy = compute_accuracy(y_true, y_pred)
    f1_macro = compute_f1(y_true, y_pred, average="macro")
    f1_weighted = compute_f1(y_true, y_pred, average="weighted")
    roc_auc = compute_roc_auc(y_true, y_proba, num_classes=num_dme_classes)
    per_class = compute_per_class_metrics(y_true, y_pred, names)

    # --- Ordinal / QWK metrics ---
    qwk_details = compute_qwk_with_details(y_true, y_pred, num_dme_classes)
    ordinal_metrics = compute_ordinal_metrics(y_true, y_pred, num_dme_classes, names)

    # --- Medical interpretation ---
    combined_metrics = {
        "qwk": qwk_details["qwk"],
        "mae": ordinal_metrics["mae"],
        "rmse": ordinal_metrics["rmse"],
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }
    interpretation = generate_medical_interpretation(combined_metrics)

    metrics = {
        # Top-level QWK (main metric)
        "qwk": qwk_details["qwk"],
        "target_met": qwk_details["qwk"] >= QWK_TARGET_THRESHOLD,
        # Ordinal metrics
        "mae": ordinal_metrics["mae"],
        "rmse": ordinal_metrics["rmse"],
        # Standard classification metrics
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        # Per-class
        "per_class": per_class,
        # Ordinal detail
        "ordinal": ordinal_metrics,
        # QWK computation detail
        "qwk_details": qwk_details,
        # Medical interpretation
        "interpretation": interpretation,
        # Meta
        "num_samples": int(len(y_true)),
        "class_counts": {
            names[i]: int(np.sum(y_true == i)) for i in range(num_dme_classes)
        },
    }

    logger.info("QWK:        %.4f %s", metrics["qwk"], "(✅ TARGET MET)" if metrics["target_met"] else f"(❌ below {QWK_TARGET_THRESHOLD})")
    logger.info("MAE:        %.3f", metrics["mae"])
    logger.info("RMSE:       %.3f", metrics["rmse"])
    logger.info("Accuracy:   %.4f", accuracy)
    logger.info("F1 (macro): %.4f", f1_macro)

    # --- Save JSON ---
    json_path = os.path.join(output_dir, metrics_path)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Comprehensive metrics saved to '%s'.", json_path)

    # --- Visualisations ---
    plot_ordinal_confusion_matrix(
        y_true, y_pred,
        class_names=names,
        output_path=os.path.join(output_dir, "ordinal_confusion_matrix.png"),
        num_classes=num_dme_classes,
        title="DME Ordinal Confusion Matrix",
    )

    plot_comprehensive_dashboard(
        y_true, y_pred, y_proba,
        metrics=metrics,
        class_names=names,
        output_path=os.path.join(output_dir, "comprehensive_dashboard.png"),
        num_classes=num_dme_classes,
    )

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Comprehensive evaluation CLI."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset
    from model import build_model_dme_tuning

    parser = argparse.ArgumentParser(description="Comprehensive QWK evaluation")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results_comprehensive")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_comp", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    _, val_ds, _ = build_datasets(csv_path=csv_path, image_dir=image_dir)
    model = build_model_dme_tuning(pretrained_weights=args.weights)

    metrics = evaluate_comprehensive(model, val_ds, output_dir=args.output_dir)
    print(json.dumps({
        "qwk": metrics["qwk"],
        "target_met": metrics["target_met"],
        "mae": metrics["mae"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "interpretation": metrics["interpretation"]["clinical_recommendation"],
    }, indent=2))


if __name__ == "__main__":
    main()
