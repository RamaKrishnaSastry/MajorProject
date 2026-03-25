"""
qwk_metrics.py - Quadratic Weighted Kappa (QWK) metrics for ordinal DR classification.

Provides:
- QWK computation (main success metric, target ≥ 0.80)
- Ordinal classification metrics (MAE, RMSE)
- Boundary confusion detection (e.g., Moderate vs Severe)
- Per-class ordinal analysis
- Visualization of ordinal confusion patterns
- Keras callback for epoch-level QWK monitoring
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional imports
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available – plots will be skipped.")

try:
    from sklearn.metrics import confusion_matrix
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ---------------------------------------------------------------------------
# DME class labels
# ---------------------------------------------------------------------------

DME_CLASS_NAMES = ["No DME", "Mild", "Moderate", "Severe"]
NUM_DME_CLASSES = len(DME_CLASS_NAMES)

# Ordinal boundary pairs (adjacent classes that are clinically ambiguous)
BOUNDARY_PAIRS = [
    (0, 1, "No DME / Mild"),
    (1, 2, "Mild / Moderate"),
    (2, 3, "Moderate / Severe"),
]


# ---------------------------------------------------------------------------
# Core QWK computation
# ---------------------------------------------------------------------------

def compute_quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
) -> float:
    """Compute Quadratic Weighted Kappa (QWK).

    QWK = 1 - Σ(w_ij × O_ij) / Σ(w_ij × E_ij)

    where:
    - w_ij = (i-j)² / (K-1)²  (quadratic penalty, adjacent errors penalised less)
    - O_ij = observed confusion matrix (normalised)
    - E_ij = expected under independent/random agreement (outer product of marginals)

    Ranges from -1 (complete disagreement) to 1 (perfect agreement).
    Clinical target: ≥ 0.80.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels in [0, num_classes-1], shape (N,).
    y_pred : np.ndarray
        Predicted integer labels in [0, num_classes-1], shape (N,).
    num_classes : int
        Number of ordinal classes.

    Returns
    -------
    float
        QWK score in [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf_matrix = np.array(
        [
            [np.sum((y_true == i) & (y_pred == j)) for j in range(num_classes)]
            for i in range(num_classes)
        ],
        dtype=float,
    )

    # Quadratic weight matrix: w_ij = (i-j)^2 / (K-1)^2
    weights = np.array(
        [
            [(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
            for i in range(num_classes)
        ],
        dtype=float,
    )

    n = conf_matrix.sum()
    if n == 0:
        return 0.0

    # Normalised observed
    observed = conf_matrix / n

    # Expected under independence
    row_totals = conf_matrix.sum(axis=1)
    col_totals = conf_matrix.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / (n * n)

    denom = np.sum(weights * expected)
    if denom == 0:
        return 0.0

    qwk = 1.0 - np.sum(weights * observed) / denom
    return float(qwk)


def compute_qwk_with_details(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
) -> Dict:
    """Compute QWK with a full computation log for debugging/reporting.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    num_classes : int
        Number of ordinal classes.

    Returns
    -------
    dict
        Contains ``qwk``, ``weight_matrix``, ``observed_matrix``,
        ``expected_matrix``, ``numerator``, and ``denominator``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    conf_matrix = np.array(
        [
            [np.sum((y_true == i) & (y_pred == j)) for j in range(num_classes)]
            for i in range(num_classes)
        ],
        dtype=float,
    )

    weights = np.array(
        [
            [(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
            for i in range(num_classes)
        ],
        dtype=float,
    )

    n = conf_matrix.sum()
    observed = conf_matrix / max(n, 1)
    row_totals = conf_matrix.sum(axis=1)
    col_totals = conf_matrix.sum(axis=0)
    expected = np.outer(row_totals, col_totals) / max(n * n, 1)

    numerator = float(np.sum(weights * observed))
    denominator = float(np.sum(weights * expected))
    qwk = 1.0 - numerator / denominator if denominator > 0 else 0.0

    return {
        "qwk": float(qwk),
        "weight_matrix": weights.tolist(),
        "conf_matrix": conf_matrix.tolist(),
        "observed_matrix": observed.tolist(),
        "expected_matrix": expected.tolist(),
        "numerator": numerator,
        "denominator": denominator,
        "n_samples": int(n),
    }


# ---------------------------------------------------------------------------
# Ordinal classification metrics
# ---------------------------------------------------------------------------

def compute_ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error on ordinal label scale.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    float
        MAE value.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def compute_ordinal_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error on ordinal label scale.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute full suite of ordinal classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    num_classes : int
        Number of ordinal classes.
    class_names : list, optional
        Human-readable class names.

    Returns
    -------
    dict
        Dict with ``qwk``, ``mae``, ``rmse``, ``accuracy``,
        ``boundary_confusion``, and ``per_class``.
    """
    names = class_names or DME_CLASS_NAMES[:num_classes]
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    qwk = compute_quadratic_weighted_kappa(y_true, y_pred, num_classes)
    mae = compute_ordinal_mae(y_true, y_pred)
    rmse = compute_ordinal_rmse(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred))

    boundary = detect_boundary_confusion(y_true, y_pred, num_classes)
    per_class = compute_per_class_ordinal_metrics(y_true, y_pred, num_classes, names)

    logger.info(
        "QWK=%.4f | MAE=%.3f | RMSE=%.3f | Acc=%.3f",
        qwk, mae, rmse, accuracy,
    )

    return {
        "qwk": qwk,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "boundary_confusion": boundary,
        "per_class": per_class,
        "target_met": qwk >= 0.80,
    }


# ---------------------------------------------------------------------------
# Boundary confusion detection
# ---------------------------------------------------------------------------

def detect_boundary_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
) -> Dict[str, float]:
    """Detect confusion rates at clinically important ordinal boundaries.

    For each adjacent pair (i, i+1), computes the fraction of samples from
    class i predicted as i+1 and vice versa.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    num_classes : int
        Number of ordinal classes.

    Returns
    -------
    dict
        Boundary confusion rates per pair.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    results = {}
    for i in range(num_classes - 1):
        j = i + 1
        # Mask samples that truly belong to class i or j
        mask = (y_true == i) | (y_true == j)
        if not np.any(mask):
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        # Confusion: class i predicted as j, or class j predicted as i
        confused = np.sum(((yt == i) & (yp == j)) | ((yt == j) & (yp == i)))
        total = np.sum(mask)
        rate = float(confused) / float(total) if total > 0 else 0.0

        pair_name = f"class_{i}_vs_{j}"
        readable = f"{DME_CLASS_NAMES[i] if i < len(DME_CLASS_NAMES) else i} / {DME_CLASS_NAMES[j] if j < len(DME_CLASS_NAMES) else j}"
        results[pair_name] = {
            "description": readable,
            "confusion_rate": rate,
            "confused_count": int(confused),
            "total_boundary_samples": int(total),
        }
        logger.info("Boundary confusion %s: %.3f (%d/%d)", readable, rate, confused, total)

    return results


# ---------------------------------------------------------------------------
# Per-class ordinal metrics
# ---------------------------------------------------------------------------

def compute_per_class_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_DME_CLASSES,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Per-class ordinal error analysis.

    For each class, computes: precision, recall, mean prediction error, and
    the average ordinal distance of misclassifications.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    num_classes : int
        Number of ordinal classes.
    class_names : list, optional
        Human-readable class names.

    Returns
    -------
    dict
        Per-class metrics dict.
    """
    names = class_names or DME_CLASS_NAMES[:num_classes]
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    per_class = {}
    for i in range(num_classes):
        name = names[i] if i < len(names) else f"class_{i}"
        mask_true = y_true == i

        n_true = int(np.sum(mask_true))
        if n_true == 0:
            per_class[name] = {"support": 0}
            continue

        n_correct = int(np.sum(mask_true & (y_pred == i)))
        recall = float(n_correct) / float(n_true)

        mask_pred = y_pred == i
        n_pred = int(np.sum(mask_pred))
        precision = float(n_correct) / float(n_pred) if n_pred > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Mean ordinal distance of misclassifications
        misclassified_mask = mask_true & (y_pred != i)
        if np.any(misclassified_mask):
            mean_error_distance = float(
                np.mean(np.abs(y_pred[misclassified_mask] - i))
            )
        else:
            mean_error_distance = 0.0

        per_class[name] = {
            "support": n_true,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_misclassification_distance": round(mean_error_distance, 3),
        }

    return per_class


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ordinal_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "ordinal_confusion_matrix.png",
    num_classes: int = NUM_DME_CLASSES,
    title: str = "Ordinal Confusion Matrix",
) -> None:
    """Plot confusion matrix with ordinal penalty overlay.

    Cells are coloured by confusion frequency; diagonal and off-diagonal
    cells are annotated to highlight ordinal severity of errors.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Human-readable class names.
    output_path : str
        Output PNG file path.
    num_classes : int
        Number of ordinal classes.
    title : str
        Plot title.
    """
    if not _PLOTTING_AVAILABLE:
        logger.warning("Plotting not available – skipping confusion matrix.")
        return

    names = class_names or DME_CLASS_NAMES[:num_classes]
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if _SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    else:
        cm = np.array(
            [
                [np.sum((y_true == i) & (y_pred == j)) for j in range(num_classes)]
                for i in range(num_classes)
            ]
        )

    # Normalise by row (true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} (Counts)", fontsize=13)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)

    # Right: normalised proportions
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=names,
        yticklabels=names,
        vmin=0,
        vmax=1,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} (Normalised)", fontsize=13)
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)

    qwk = compute_quadratic_weighted_kappa(y_true, y_pred, num_classes)
    fig.suptitle(f"QWK = {qwk:.4f}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ordinal confusion matrix saved to '%s'.", output_path)


def plot_qwk_per_epoch(
    qwk_history: List[float],
    val_qwk_history: Optional[List[float]] = None,
    output_path: str = "qwk_per_epoch.png",
) -> None:
    """Plot QWK score over training epochs.

    Parameters
    ----------
    qwk_history : list
        Train QWK per epoch.
    val_qwk_history : list, optional
        Validation QWK per epoch.
    output_path : str
        Output PNG path.
    """
    if not _PLOTTING_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = list(range(1, len(qwk_history) + 1))
    ax.plot(epochs, qwk_history, label="Train QWK", marker="o", markersize=4)
    if val_qwk_history:
        ax.plot(epochs, val_qwk_history, label="Val QWK", marker="s", markersize=4)

    ax.axhline(y=0.80, color="red", linestyle="--", label="Target QWK=0.80")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("QWK", fontsize=12)
    ax.set_title("Quadratic Weighted Kappa over Training", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("QWK curve saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Keras callback
# ---------------------------------------------------------------------------

if _TF_AVAILABLE:
    class QWKCallback(keras.callbacks.Callback):
        """Keras callback that computes QWK at the end of each epoch.

        Stores QWK history and triggers plateau detection for LR scheduling.

        Parameters
        ----------
        val_dataset : tf.data.Dataset
            Validation dataset yielding (images, one_hot_labels).
        num_classes : int
            Number of ordinal classes.
        history_path : str
            Path to persist QWK history JSON.
        verbose : int
            Verbosity level (0 = silent, 1 = per-epoch).
        """

        def __init__(
            self,
            val_dataset,
            num_classes: int = NUM_DME_CLASSES,
            history_path: str = "qwk_history.json",
            verbose: int = 1,
        ):
            super().__init__()
            self.val_dataset = val_dataset
            self.num_classes = num_classes
            self.history_path = history_path
            self.verbose = verbose
            self.qwk_history: List[float] = []
            self.best_qwk: float = -np.inf
            self.best_epoch: int = 0

        def on_epoch_end(self, epoch: int, logs=None):
            logs = logs or {}
            all_true, all_pred = [], []
            for batch_images, batch_labels in self.val_dataset:
                preds = self.model(batch_images, training=False)
                if isinstance(preds, dict):
                    dme_proba = preds["dme_risk"]
                elif isinstance(preds, (list, tuple)):
                    dme_proba = preds[1]
                else:
                    dme_proba = preds
                all_true.append(np.argmax(batch_labels.numpy(), axis=-1))
                all_pred.append(np.argmax(dme_proba.numpy(), axis=-1))

            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            qwk = compute_quadratic_weighted_kappa(y_true, y_pred, self.num_classes)

            self.qwk_history.append(qwk)
            logs["val_qwk"] = qwk

            if qwk > self.best_qwk:
                self.best_qwk = qwk
                self.best_epoch = epoch + 1

            if self.verbose:
                logger.info(
                    "Epoch %d – val_qwk=%.4f (best=%.4f @ epoch %d)",
                    epoch + 1, qwk, self.best_qwk, self.best_epoch,
                )

            # Persist history
            with open(self.history_path, "w") as f:
                json.dump(
                    {
                        "qwk_history": self.qwk_history,
                        "best_qwk": self.best_qwk,
                        "best_epoch": self.best_epoch,
                    },
                    f,
                    indent=2,
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI: compute QWK metrics for saved predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute QWK metrics from predictions")
    parser.add_argument("--y-true", type=str, required=True, help="Path to y_true npy/txt")
    parser.add_argument("--y-pred", type=str, required=True, help="Path to y_pred npy/txt")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.y_true.endswith(".npy"):
        y_true = np.load(args.y_true)
    else:
        y_true = np.loadtxt(args.y_true, dtype=int)

    if args.y_pred.endswith(".npy"):
        y_pred = np.load(args.y_pred)
    else:
        y_pred = np.loadtxt(args.y_pred, dtype=int)

    metrics = compute_ordinal_metrics(y_true, y_pred, args.num_classes)
    print(json.dumps(metrics, indent=2))

    plot_ordinal_confusion_matrix(
        y_true, y_pred,
        output_path=os.path.join(args.output_dir, "ordinal_confusion_matrix.png"),
        num_classes=args.num_classes,
    )

    metrics_path = os.path.join(args.output_dir, "qwk_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("QWK metrics saved to '%s'.", metrics_path)


if __name__ == "__main__":
    main()
