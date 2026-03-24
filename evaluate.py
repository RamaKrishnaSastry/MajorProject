"""
evaluate.py
Comprehensive QWK-focused evaluation module for DR/DME models.

Primary metric: Quadratic Weighted Kappa (QWK) — target ≥ 0.80.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ── QWK helper ────────────────────────────────────────────────────────────────

def compute_quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = None,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    QWK = 1 - Σ(w_ij * O_ij) / Σ(w_ij * E_ij)

    where:
        O_ij  = observed confusion matrix (normalised)
        E_ij  = expected matrix under random agreement (normalised)
        w_ij  = (i - j)^2 / (K - 1)^2   (quadratic penalty)
        K     = number of classes

    Args:
        y_true: 1-D array of ground-truth integer labels.
        y_pred: 1-D array of predicted integer labels.
        num_classes: Total number of classes.  If None, inferred from data.

    Returns:
        QWK score in [-1, 1].  Target ≥ 0.80 for clinical grade.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if len(set(y_true)) <= 1:
        return 0.0

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# ── Metrics dict ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute a comprehensive metrics dictionary.

    Args:
        y_true: 1-D integer array of ground-truth labels.
        y_pred_proba: 2-D float array of predicted probabilities,
                      shape (N, num_classes).

    Returns:
        Dictionary with keys:
            qwk, accuracy, macro_f1, weighted_f1, per_class,
            confusion_matrix, roc_auc_ovr.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    num_classes = y_pred_proba.shape[1]
    y_pred = np.argmax(y_pred_proba, axis=1)

    qwk = compute_quadratic_weighted_kappa(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        "precision": [report.get(str(c), {}).get("precision", 0.0) for c in range(num_classes)],
        "recall":    [report.get(str(c), {}).get("recall",    0.0) for c in range(num_classes)],
        "f1":        [report.get(str(c), {}).get("f1-score",  0.0) for c in range(num_classes)],
        "support":   [int(report.get(str(c), {}).get("support", 0)) for c in range(num_classes)],
    }

    # ROC-AUC (one-vs-rest) — only meaningful if ≥2 classes present in y_true
    try:
        if num_classes == 2:
            roc_auc_ovr = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            roc_auc_ovr = roc_auc_score(
                y_true, y_pred_proba,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        roc_auc_ovr = None

    return {
        "qwk":            float(qwk),
        "accuracy":       float(acc),
        "macro_f1":       float(macro_f1),
        "weighted_f1":    float(weighted_f1),
        "per_class":      per_class,
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr":    float(roc_auc_ovr) if roc_auc_ovr is not None else None,
    }


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    qwk: float,
    output_path: str,
):
    """Save a normalised confusion matrix heatmap annotated with QWK."""
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Grade", fontsize=12)
    ax.set_ylabel("True Grade",      fontsize=12)
    ax.set_title(f"Confusion Matrix  (QWK = {qwk:.4f})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Confusion matrix → {output_path}")


def plot_training_curves(history: dict, output_path: str):
    """Plot training loss, accuracy, and val_qwk over epochs."""
    epochs = range(1, len(history.get("loss", [])) + 1)

    n_plots = 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Loss
    axes[0].plot(epochs, history.get("loss", []), label="train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # DME accuracy
    dme_acc_key = next((k for k in history if "dme_risk_accuracy" in k and "val" not in k), None)
    val_dme_acc_key = next((k for k in history if "dme_risk_accuracy" in k and "val" in k), None)
    if dme_acc_key:
        axes[1].plot(epochs, history[dme_acc_key], label="train")
    if val_dme_acc_key:
        axes[1].plot(epochs, history[val_dme_acc_key], label="val")
    axes[1].set_title("DME Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # QWK
    if "val_qwk" in history:
        qwk_epochs = range(1, len(history["val_qwk"]) + 1)
        axes[2].plot(qwk_epochs, history["val_qwk"], color="red", label="val QWK")
        axes[2].axhline(0.80, color="green", linestyle="--", label="Target 0.80")
        axes[2].set_ylim(-0.1, 1.05)
    axes[2].set_title("Validation QWK")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Training curves → {output_path}")


def plot_per_class_metrics(metrics: dict, class_names: list, output_path: str):
    """Bar chart of precision, recall, F1 per class."""
    pc = metrics.get("per_class", {})
    precision = pc.get("precision", [])
    recall    = pc.get("recall",    [])
    f1        = pc.get("f1",        [])

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1-Score")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Per-class metrics → {output_path}")


# ── Evaluator Class ───────────────────────────────────────────────────────────

class DMEEvaluator:
    """
    End-to-end evaluation of a trained DR-ASPP-DRN model on DME grading.

    Args:
        model: Trained Keras model with outputs [dr_output, dme_risk].
        test_dataset: ``tf.data.Dataset`` yielding ``(image, one_hot_label)``
                      batches.
        output_dir: Directory for plots and reports.
        class_names: Human-readable class names.
    """

    DME_CLASS_NAMES = ["No DME (0)", "Mild DME (1)", "Moderate DME (2)"]

    def __init__(
        self,
        model,
        test_dataset,
        output_dir: str = "outputs",
        class_names: list = None,
    ):
        self.model = model
        self.test_dataset = test_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or self.DME_CLASS_NAMES

    def _collect_predictions(self):
        """Run model inference and return (y_true, y_pred_proba) arrays."""
        y_true_list, y_proba_list = [], []

        for batch in self.test_dataset:
            x, y_true = batch[0], batch[1]
            preds = self.model(x, training=False)

            if isinstance(preds, (list, tuple)):
                dme_probs = preds[1].numpy()
            else:
                dme_probs = preds.numpy()

            if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
                y_true_cls = np.argmax(y_true.numpy(), axis=-1)
            else:
                y_true_cls = y_true.numpy().flatten().astype(int)

            y_true_list.extend(y_true_cls.tolist())
            y_proba_list.append(dme_probs)

        y_true = np.array(y_true_list, dtype=int)
        y_proba = np.vstack(y_proba_list)
        return y_true, y_proba

    def run(self) -> dict:
        """
        Run full evaluation pipeline.

        Steps:
            1. Collect predictions from test dataset.
            2. Compute comprehensive metrics.
            3. Generate visualisations.
            4. Write text report.

        Returns:
            Metrics dictionary (also saved to ``evaluation_metrics.json``).
        """
        print("\n[Eval] Collecting predictions …")
        y_true, y_proba = self._collect_predictions()
        y_pred = np.argmax(y_proba, axis=1)

        print("[Eval] Computing metrics …")
        metrics = compute_metrics(y_true, y_proba)

        # ── Visualisations ──────────────────────────────────────────────────
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            self.class_names,
            metrics["qwk"],
            str(self.output_dir / "confusion_matrix.png"),
        )
        plot_per_class_metrics(
            metrics,
            self.class_names,
            str(self.output_dir / "per_class_metrics.png"),
        )

        # ── Text report ─────────────────────────────────────────────────────
        self._write_report(metrics, y_true, y_pred)

        return metrics

    def _write_report(self, metrics: dict, y_true, y_pred):
        report_path = self.output_dir / "metrics_report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("  DR-ASPP-DRN  —  DME Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("PRIMARY METRIC\n")
            f.write(f"  Quadratic Weighted Kappa (QWK): {metrics['qwk']:.4f}\n")
            if metrics["qwk"] >= 0.85:
                f.write("  ✅ Exceeds stretch goal (≥0.85)\n")
            elif metrics["qwk"] >= 0.80:
                f.write("  ✅ Meets minimum excellent threshold (≥0.80)\n")
            else:
                f.write("  ⚠️  Below target (target ≥0.80)\n")

            f.write("\nOVERALL METRICS\n")
            f.write(f"  Accuracy        : {metrics['accuracy']:.4f}\n")
            f.write(f"  Macro F1-Score  : {metrics['macro_f1']:.4f}\n")
            f.write(f"  Weighted F1     : {metrics['weighted_f1']:.4f}\n")
            if metrics.get("roc_auc_ovr") is not None:
                f.write(f"  ROC-AUC (OvR)   : {metrics['roc_auc_ovr']:.4f}\n")

            f.write("\nPER-CLASS METRICS\n")
            f.write(f"  {'Class':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}\n")
            f.write("  " + "-" * 52 + "\n")
            pc = metrics["per_class"]
            for i, name in enumerate(self.class_names):
                f.write(
                    f"  {name:<20}"
                    f"{pc['precision'][i]:>10.3f}"
                    f"{pc['recall'][i]:>10.3f}"
                    f"{pc['f1'][i]:>10.3f}"
                    f"{pc['support'][i]:>10d}\n"
                )

            f.write("\nCONFUSION MATRIX\n")
            cm = np.array(metrics["confusion_matrix"])
            header = "  " + "".join(f"{n[:8]:>10}" for n in self.class_names) + "\n"
            f.write(header)
            for i, row in enumerate(cm):
                f.write(f"  {self.class_names[i][:8]:<8}" + "".join(f"{v:>10d}" for v in row) + "\n")

            f.write("\nCLINICAL INTERPRETATION\n")
            f.write("  QWK measures ordinal agreement penalising distant misclassifications.\n")
            f.write("  QWK ≥ 0.80 indicates excellent clinical reliability.\n")
            f.write("  Boundary errors (Grade 1 vs Grade 2) carry less penalty than grade 0↔2.\n")

        print(f"[Eval] Report saved → {report_path}")
