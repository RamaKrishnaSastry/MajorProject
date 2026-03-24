"""
Comprehensive evaluation module for DME / DR classification.

Generates:
    - evaluation_metrics.json   – accuracy, F1, per-class stats, QWK
    - confusion_matrix.png      – heatmap
    - metrics_report.txt        – human-readable report
    - training_curves.png       – loss / accuracy over epochs (optional)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

DME_CLASS_NAMES = ["No DME", "Possible DME", "Present DME"]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_dme(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple:
    """Run inference over a ``tf.data.Dataset`` and return arrays.

    Parameters
    ----------
    model:
        Compiled Keras model with ``dme_risk`` output.
    dataset:
        Batched ``tf.data.Dataset`` yielding ``(image, labels_dict)``.

    Returns
    -------
    y_true : np.ndarray  shape (N,)   integer class indices
    y_pred : np.ndarray  shape (N,)   integer class indices (argmax)
    y_prob : np.ndarray  shape (N, 3) softmax probabilities
    """
    y_true_list, y_prob_list = [], []
    for batch_images, batch_labels in dataset:
        preds = model.predict_on_batch(batch_images)
        # preds may be a list [dr_out, dme_out] or a single tensor
        if isinstance(preds, (list, tuple)):
            dme_probs = preds[1]
        else:
            dme_probs = preds

        dme_true = batch_labels["dme_risk"].numpy()   # one-hot
        y_true_list.append(np.argmax(dme_true, axis=1))
        y_prob_list.append(dme_probs)

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
    y_pred : array-like of int, shape (N,)

    Returns
    -------
    dict with keys:
        accuracy, macro_f1, weighted_f1, qwk,
        per_class (list of dicts per class),
        classification_report (str)
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError as exc:
        logger.warning("QWK calculation failed (insufficient class variance): %s", exc)
        qwk = 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(DME_CLASS_NAMES))),
        zero_division=0,
    )

    per_class = []
    for i, name in enumerate(DME_CLASS_NAMES):
        per_class.append({
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        })

    report = classification_report(
        y_true, y_pred,
        target_names=DME_CLASS_NAMES,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "per_class": per_class,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "confusion_matrix.png",
) -> None:
    """Save a normalised confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(DME_CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=DME_CLASS_NAMES, yticklabels=DME_CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalised
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=DME_CLASS_NAMES, yticklabels=DME_CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (normalised)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", output_path)


def plot_per_class_metrics(
    metrics: dict,
    output_path: str = "per_class_metrics.png",
) -> None:
    """Bar chart for per-class precision, recall, and F1."""
    names = [d["class"] for d in metrics["per_class"]]
    precision = [d["precision"] for d in metrics["per_class"]]
    recall = [d["recall"] for d in metrics["per_class"]]
    f1 = [d["f1"] for d in metrics["per_class"]]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Metrics")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class metrics chart saved → %s", output_path)


def plot_class_distribution(
    y_true: np.ndarray,
    output_path: str = "class_distribution.png",
) -> None:
    """Pie chart of class distribution."""
    counts = np.bincount(y_true, minlength=len(DME_CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=DME_CLASS_NAMES,
        autopct="%1.1f%%",
        startangle=140,
    )
    ax.set_title("DME Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Class distribution chart saved → %s", output_path)


def plot_training_curves(
    history_path: str,
    output_path: str = "training_curves.png",
) -> None:
    """Load a ``training_history.json`` file and plot loss / accuracy curves."""
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(next(iter(history.values()))) + 1)

    # Identify metric keys
    loss_key = "dme_risk_loss" if "dme_risk_loss" in history else "loss"
    val_loss_key = f"val_{loss_key}"
    acc_key = "dme_risk_accuracy" if "dme_risk_accuracy" in history else "accuracy"
    val_acc_key = f"val_{acc_key}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    if loss_key in history:
        axes[0].plot(epochs, history[loss_key], label="Train Loss")
    if val_loss_key in history:
        axes[0].plot(epochs, history[val_loss_key], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    if acc_key in history:
        axes[1].plot(epochs, history[acc_key], label="Train Accuracy")
    if val_acc_key in history:
        axes[1].plot(epochs, history[val_acc_key], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved → %s", output_path)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_dir: str = "outputs",
    history_path: Optional[str] = None,
) -> dict:
    """Run the complete evaluation pipeline and save all artefacts.

    Parameters
    ----------
    model:
        Trained Keras model.
    dataset:
        Batched validation / test ``tf.data.Dataset``.
    output_dir:
        Directory for all output files.
    history_path:
        Path to ``training_history.json`` (optional – for curve plots).

    Returns
    -------
    dict
        All computed metrics.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Running inference…")
    y_true, y_pred, _y_prob = predict_dme(model, dataset)

    logger.info("Computing metrics…")
    metrics = compute_metrics(y_true, y_pred)

    # ---- Save JSON --------------------------------------------------------
    json_path = out / "evaluation_metrics.json"
    # Remove non-serialisable classification_report string for clean JSON
    metrics_json = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info("Metrics JSON saved → %s", json_path)

    # ---- Text report ------------------------------------------------------
    report_path = out / "metrics_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DME Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy   : {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1   : {metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
        f.write(f"QWK        : {metrics['qwk']:.4f}\n\n")
        f.write(metrics["classification_report"])
    logger.info("Metrics report saved → %s", report_path)

    # ---- Visualisations ---------------------------------------------------
    plot_confusion_matrix(y_true, y_pred, str(out / "confusion_matrix.png"))
    plot_per_class_metrics(metrics, str(out / "per_class_metrics.png"))
    plot_class_distribution(y_true, str(out / "class_distribution.png"))

    if history_path and Path(history_path).exists():
        plot_training_curves(history_path, str(out / "training_curves.png"))

    # ---- Console summary --------------------------------------------------
    logger.info("Accuracy   : %.4f", metrics["accuracy"])
    logger.info("Macro F1   : %.4f", metrics["macro_f1"])
    logger.info("QWK        : %.4f", metrics["qwk"])
    logger.info("\n%s", metrics["classification_report"])

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dataset_loader import IDRiDDataset
    from model import build_model_dme_tuning

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate trained DME model")
    parser.add_argument("--dataset_path", default="dataset")
    parser.add_argument("--weights_path", default="outputs/dme_finetuned.weights.h5")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--history_path",
                        default="outputs/training_history.json")
    args = parser.parse_args()

    val_ds = IDRiDDataset(args.dataset_path, split="val", augment=False).get_dataset()
    model = build_model_dme_tuning()
    model.load_weights(args.weights_path, skip_mismatch=True)

    evaluate(
        model=model,
        dataset=val_ds,
        output_dir=args.output_dir,
        history_path=args.history_path,
    )
