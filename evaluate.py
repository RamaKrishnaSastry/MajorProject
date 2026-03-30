"""
evaluate.py - Comprehensive evaluation for the DME classification model.

Computes:
- Overall accuracy
- Confusion matrix (with visualisation)
- F1-score (macro) for handling class imbalance
- Per-class precision, recall, F1
- ROC-AUC per class (one-vs-rest)
- Saves all results to JSON and PNG files
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional visualisation imports
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    logger.warning("matplotlib / seaborn not installed – plots will be skipped.")

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed – some metrics will be skipped.")

DME_CLASS_NAMES = ["No DME", "Mild", "Moderate"]
NUM_DME_CLASSES = len(DME_CLASS_NAMES)

DR_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe NPDR", "Proliferative DR"]
NUM_DR_CLASSES = len(DR_CLASS_NAMES)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def dr_regression_to_grade(
    dr_pred: np.ndarray,
    num_dr_classes: int = NUM_DR_CLASSES,
) -> np.ndarray:
    """Convert DR regression outputs to integer DR grades.

    Applies ``round`` then clips to valid class range ``[0, num_dr_classes-1]``.
    Works with scalar, vector, or ``(N, 1)`` array-like inputs.
    """
    grades = np.rint(np.asarray(dr_pred, dtype=np.float32))
    grades = np.clip(grades, 0, num_dr_classes - 1)
    return grades.astype(np.int32)


def get_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_dme_classes: int = NUM_DME_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over *dataset* and collect ground-truth labels.

    Parameters
    ----------
    model : keras.Model
        Trained model.  Its output dict must contain the key ``"dme_risk"``.
    dataset : tf.data.Dataset
        Batched dataset yielding ``(images, labels)`` where labels can be:
        - A tensor: (batch_size, num_classes) one-hot encoded
        - A dict: {'dme_risk': tensor, 'dr_output': tensor, ...}
    num_dme_classes : int
        Number of DME classes.

    Returns
    -------
    tuple
        ``(y_true, y_pred_proba, y_pred_class)``

        - ``y_true`` – integer class labels, shape ``(N,)``
        - ``y_pred_proba`` – softmax probabilities, shape ``(N, C)``
        - ``y_pred_class`` – argmax class predictions, shape ``(N,)``
    """
    all_true = []
    all_proba = []

    for batch_images, batch_labels in dataset:
        # =====================================================
        # Step 1: Extract DME predictions from model output
        # =====================================================
        preds = model(batch_images, training=False)
        
        # Handle both dict and tuple outputs
        if isinstance(preds, dict):
            dme_proba = preds["dme_risk"]
        elif isinstance(preds, (list, tuple)):
            dme_proba = preds[1]
        else:
            dme_proba = preds
        
        # =====================================================
        # Step 2: Extract DME labels from batch_labels
        # =====================================================
        # Handle both tensor and dict inputs for labels
        if isinstance(batch_labels, dict):
            # Multi-output case: labels are in a dict
            labels_tensor = batch_labels["dme_risk"]
        else:
            # Single-output case: labels are already a tensor
            labels_tensor = batch_labels
        
        # =====================================================
        # Step 3: Convert to numpy and get class indices
        # =====================================================
        all_true.append(np.argmax(labels_tensor.numpy(), axis=-1))
        all_proba.append(dme_proba.numpy())

    # =====================================================
    # Step 4: Concatenate all batches and compute predictions
    # =====================================================
    y_true = np.concatenate(all_true)
    y_pred_proba = np.concatenate(all_proba)
    y_pred_class = np.argmax(y_pred_proba, axis=-1)
    
    return y_true, y_pred_proba, y_pred_class


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    if _SKLEARN_AVAILABLE:
        return float(accuracy_score(y_true, y_pred))
    return float(np.mean(y_true == y_pred))


def compute_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> float:
    """Compute F1-score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    average : str
        Averaging strategy (``"macro"`` recommended for imbalanced data).

    Returns
    -------
    float
        F1-score.
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("scikit-learn required for F1 computation.")
        return float("nan")
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
) -> Dict[str, float]:
    """Compute ROC-AUC for each class (one-vs-rest).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred_proba : np.ndarray
        Predicted probabilities of shape ``(N, C)``.
    num_classes : int
        Number of classes.

    Returns
    -------
    dict
        ``{"class_0": auc, ..., "macro": macro_auc}``
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("scikit-learn required for ROC-AUC computation.")
        return {}

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    result = {}
    macro_aucs = []
    for i in range(num_classes):
        if len(np.unique(y_bin[:, i])) < 2:
            logger.warning("Class %d has only one label in eval set – AUC skipped.", i)
            result[f"class_{i}"] = float("nan")
            continue
        auc = float(roc_auc_score(y_bin[:, i], y_pred_proba[:, i]))
        result[f"class_{i}"] = auc
        macro_aucs.append(auc)

    result["macro"] = float(np.mean(macro_aucs)) if macro_aucs else float("nan")
    return result


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return per-class precision, recall, and F1.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Human-readable class names.

    Returns
    -------
    dict
        ``{class_name: {"precision": .., "recall": .., "f1": ..}}``.
    """
    if not _SKLEARN_AVAILABLE:
        return {}

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    names = class_names or [f"class_{i}" for i in range(NUM_DME_CLASSES)]
    per_class = {}
    for i, name in enumerate(names):
        key = str(i)
        if key in report:
            per_class[name] = {
                "precision": report[key]["precision"],
                "recall": report[key]["recall"],
                "f1": report[key]["f1-score"],
                "support": int(report[key]["support"]),
            }
    return per_class


# ---------------------------------------------------------------------------
# Confusion matrix visualisation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "confusion_matrix.png",
    normalise: bool = True,
) -> None:
    """Plot and save the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Labels for each class.
    output_path : str
        File path for the saved figure.
    normalise : bool
        Whether to normalise values to proportions.
    """
    if not (_PLOTTING_AVAILABLE and _SKLEARN_AVAILABLE):
        logger.warning("matplotlib / seaborn / sklearn needed for confusion matrix plot.")
        return

    names = class_names or DME_CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("DME Classification – Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model: keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    output_dir: str = ".",
    confusion_matrix_path: str = "confusion_matrix.png",
    metrics_path: str = "evaluation_metrics.json",
    num_dme_classes: int = NUM_DME_CLASSES,
) -> Dict:
    """Run the complete evaluation pipeline.

    Computes accuracy, confusion matrix, F1-score, per-class metrics, and
    ROC-AUC.  Results are saved to disk.

    Parameters
    ----------
    model : keras.Model
        Trained DME model.
    dataset : tf.data.Dataset
        Validation / test dataset.
    class_names : list, optional
        Human-readable class names.
    output_dir : str
        Directory for output artefacts.
    confusion_matrix_path : str
        Filename for the confusion matrix PNG (inside *output_dir*).
    metrics_path : str
        Filename for the JSON metrics report (inside *output_dir*).
    num_dme_classes : int
        Number of DME classes.

    Returns
    -------
    dict
        All computed metrics as a serialisable dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    names = class_names or DME_CLASS_NAMES[:num_dme_classes]

    logger.info("Running inference …")
    y_true, y_proba, y_pred = get_predictions(model, dataset, num_dme_classes)

    accuracy = compute_accuracy(y_true, y_pred)
    f1_macro = compute_f1(y_true, y_pred, average="macro")
    f1_weighted = compute_f1(y_true, y_pred, average="weighted")
    roc_auc = compute_roc_auc(y_true, y_proba, num_classes=num_dme_classes)
    per_class = compute_per_class_metrics(y_true, y_pred, names)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "per_class": per_class,
        "num_samples": int(len(y_true)),
        "class_counts": {
            names[i]: int(np.sum(y_true == i)) for i in range(num_dme_classes)
        },
    }

    logger.info("Accuracy:   %.4f", accuracy)
    logger.info("F1 (macro): %.4f", f1_macro)
    logger.info("ROC-AUC:    %s", roc_auc)

    # Save metrics JSON
    json_path = os.path.join(output_dir, metrics_path)
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Evaluation metrics saved to '%s'.", json_path)

    # Save confusion matrix
    cm_path = os.path.join(output_dir, confusion_matrix_path)
    plot_confusion_matrix(y_true, y_pred, class_names=names, output_path=cm_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Evaluate a saved DME model from the command line."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset
    from model import build_model_dme_tuning

    parser = argparse.ArgumentParser(description="Evaluate trained DME model")
    parser.add_argument("--weights", type=str, required=False,
                        help="Path to trained weights .h5 file")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_eval", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    _, val_ds, _ = build_datasets(csv_path=csv_path, image_dir=image_dir)

    model = build_model_dme_tuning(pretrained_weights=args.weights)

    metrics = evaluate(model, val_ds, output_dir=args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
