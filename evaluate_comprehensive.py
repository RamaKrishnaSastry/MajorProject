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
from typing import Dict, List, Optional, Sequence, Tuple

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
# Optional threshold calibration
# ---------------------------------------------------------------------------

def _predict_from_expected_score(
    y_proba: np.ndarray,
    boundaries: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Convert class probabilities to ordinal labels via expected-score boundaries."""
    expected_score = np.sum(
        y_proba * np.arange(num_classes, dtype=np.float32)[None, :],
        axis=1,
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
    """Coordinate-search boundaries that maximise QWK on validation data."""
    if num_classes <= 1:
        pred = np.zeros_like(y_true, dtype=np.int32)
        return np.array([], dtype=np.float32), pred, 0.0

    boundaries = np.arange(0.5, float(num_classes - 0.5), 1.0, dtype=np.float32)

    def _objective(b: np.ndarray) -> Tuple[float, np.ndarray]:
        pred_labels = _predict_from_expected_score(y_proba, b, num_classes)
        score = compute_quadratic_weighted_kappa(y_true, pred_labels, num_classes)
        return float(score), pred_labels

    best_qwk, best_pred = _objective(boundaries)

    for _ in range(max(1, int(n_passes))):
        improved = False
        for idx in range(len(boundaries)):
            lower = 0.0 if idx == 0 else boundaries[idx - 1] + min_gap
            upper = float(num_classes - 1) if idx == len(boundaries) - 1 else boundaries[idx + 1] - min_gap

            if upper <= lower:
                continue

            candidates = np.linspace(lower, upper, num=max(5, int(grid_size)), dtype=np.float32)

            local_best_qwk = best_qwk
            local_best_boundary = boundaries[idx]
            local_best_pred = best_pred

            for candidate in candidates:
                trial = boundaries.copy()
                trial[idx] = float(candidate)
                qwk_value, pred_value = _objective(trial)
                if qwk_value > local_best_qwk:
                    local_best_qwk = qwk_value
                    local_best_boundary = float(candidate)
                    local_best_pred = pred_value

            if local_best_qwk > best_qwk:
                boundaries[idx] = local_best_boundary
                best_qwk = local_best_qwk
                best_pred = local_best_pred
                improved = True

        if not improved:
            break

    return boundaries.astype(np.float32), best_pred.astype(np.int32), float(best_qwk)


def _extract_model_outputs(preds) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """Extract DME and DR outputs from model predictions."""
    if isinstance(preds, dict):
        dme_out = preds.get("dme_risk")
        dr_out = preds.get("dr_output")
        return dme_out, dr_out
    if isinstance(preds, (list, tuple)):
        if len(preds) >= 2:
            # Repo convention: [dr_output, dme_risk]
            return preds[1], preds[0]
        if len(preds) == 1:
            return preds[0], None
    return preds, None


def _extract_label_tensors(batch_y) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """Extract DME and DR targets from dataset labels."""
    if isinstance(batch_y, dict):
        return batch_y["dme_risk"], batch_y.get("dr_output")
    if isinstance(batch_y, (list, tuple)):
        if len(batch_y) >= 2:
            # Repo convention: [dr_output, dme_risk]
            return batch_y[1], batch_y[0]
        if len(batch_y) == 1:
            return batch_y[0], None
    return batch_y, None


def _build_tta_views(batch_x: tf.Tensor, tta_mode: str) -> List[tf.Tensor]:
    """Build TTA views for a batch (geometric-only to preserve lesion appearance)."""
    mode = str(tta_mode or "none").lower()
    if mode in {"", "none", "off", "false", "0"}:
        return [batch_x]

    if mode in {"flip", "hflip"}:
        return [batch_x, tf.image.flip_left_right(batch_x)]

    if mode in {"rot4", "dihedral8", "d4"}:
        rots = [tf.image.rot90(batch_x, k=i) for i in range(4)]
        flips = [tf.image.flip_left_right(r) for r in rots]
        if mode == "rot4":
            return rots
        return rots + flips

    logger.warning("Unknown TTA mode '%s'; falling back to no TTA.", tta_mode)
    return [batch_x]


def _predict_batch_with_tta(
    model: keras.Model,
    batch_x: tf.Tensor,
    tta_mode: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Predict one batch with optional TTA and average outputs."""
    views = _build_tta_views(batch_x, tta_mode)

    dme_accum = None
    dr_accum = None

    for view in views:
        preds = model(view, training=False)
        dme_out, dr_out = _extract_model_outputs(preds)

        dme_np = dme_out.numpy()
        dme_accum = dme_np if dme_accum is None else (dme_accum + dme_np)

        if dr_out is not None:
            dr_np = dr_out.numpy()
            dr_accum = dr_np if dr_accum is None else (dr_accum + dr_np)

    dme_mean = dme_accum / float(len(views))
    dr_mean = None if dr_accum is None else (dr_accum / float(len(views)))
    return dme_mean, dr_mean


def _collect_multitask_predictions(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_dme_classes: int,
    ensemble_weight_paths: Optional[Sequence[str]] = None,
    tta_mode: str = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collect DME and DR predictions with optional checkpoint ensembling + TTA."""
    paths: List[Optional[str]] = []
    for path in (ensemble_weight_paths or []):
        if not path:
            continue
        norm = os.path.normpath(str(path))
        if os.path.exists(norm) and norm not in paths:
            paths.append(norm)
    if not paths:
        paths = [None]

    y_true_dme = None
    y_true_dr = None
    dme_sum = None
    dr_sum = None

    for model_idx, weight_path in enumerate(paths):
        if weight_path is not None:
            model.load_weights(weight_path)

        dme_batches = []
        dr_batches = []
        batch_true_dme = []
        batch_true_dr = []

        for batch_x, batch_y in dataset:
            dme_np, dr_np = _predict_batch_with_tta(model, batch_x, tta_mode)
            dme_batches.append(dme_np)
            if dr_np is not None:
                dr_batches.append(dr_np)

            if model_idx == 0:
                dme_target, dr_target = _extract_label_tensors(batch_y)
                dme_target_np = dme_target.numpy()
                if dme_target_np.ndim > 1:
                    batch_true_dme.append(np.argmax(dme_target_np, axis=-1))
                else:
                    batch_true_dme.append(dme_target_np.astype(np.int32).reshape(-1))

                if dr_target is not None:
                    dr_target_np = dr_target.numpy()
                    if dr_target_np.ndim > 1 and dr_target_np.shape[-1] > 1:
                        batch_true_dr.append(np.argmax(dr_target_np, axis=-1))
                    else:
                        batch_true_dr.append(
                            np.clip(
                                np.round(dr_target_np.reshape(-1)).astype(np.int32),
                                0,
                                4,
                            )
                        )

        dme_np_all = np.concatenate(dme_batches, axis=0)
        dme_sum = dme_np_all if dme_sum is None else (dme_sum + dme_np_all)

        if dr_batches:
            dr_np_all = np.concatenate(dr_batches, axis=0)
            dr_sum = dr_np_all if dr_sum is None else (dr_sum + dr_np_all)

        if model_idx == 0:
            y_true_dme = np.concatenate(batch_true_dme, axis=0).astype(np.int32)
            if batch_true_dr:
                y_true_dr = np.concatenate(batch_true_dr, axis=0).astype(np.int32)

    dme_proba = dme_sum / float(len(paths))
    dr_output = None if dr_sum is None else (dr_sum / float(len(paths)))
    y_pred_dme = np.argmax(dme_proba, axis=-1).astype(np.int32)

    if y_true_dr is None:
        # DR labels may be absent in some custom datasets; keep shape consistent.
        y_true_dr = np.zeros_like(y_true_dme, dtype=np.int32)

    return y_true_dme, dme_proba, y_pred_dme, y_true_dr, dr_output


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
        f"QWK:      {qwk:.4f}  {'TARGET MET' if qwk >= QWK_TARGET_THRESHOLD else 'BELOW TARGET'}\n"
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


def evaluate_dr_grading(
    model: keras.Model,
    val_ds: tf.data.Dataset,
    output_dir: str,
    num_dr_classes: int = 5,
    calibrate_thresholds: bool = False,
    min_qwk_gain: float = 1e-4,
    max_accuracy_drop: float = 0.0,
    dr_true_cached: Optional[np.ndarray] = None,
    dr_output_cached: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate DR head by converting outputs to ordinal grades.

    Parameters
    ----------
    model : keras.Model
        Multi-output model containing DR head.
    val_ds : tf.data.Dataset
        Validation dataset yielding ``(images, labels)``.
    output_dir : str
        Directory where ``dr_metrics.json`` is written.
    num_dr_classes : int
        Number of DR grades (default 5 → labels 0..4).

    Returns
    -------
    dict
        DR grading metrics.
    """
    dr_true_all: List[int] = []
    dr_pred_all: List[int] = []
    dr_proba_all: List[np.ndarray] = []

    def _extract_dr_prediction(preds):
        if isinstance(preds, dict):
            return preds["dr_output"]
        if isinstance(preds, (list, tuple)):
            return preds[0]
        return preds

    def _extract_dr_target(batch_y):
        if isinstance(batch_y, dict):
            return batch_y["dr_output"]
        if isinstance(batch_y, (list, tuple)):
            for item in batch_y:
                shape = getattr(item, "shape", None)
                if shape is not None and (len(shape) == 1 or (len(shape) > 1 and shape[-1] == 1)):
                    return item
            return batch_y[0]
        return batch_y

    if dr_true_cached is not None and dr_output_cached is not None:
        dr_true_np = np.asarray(dr_true_cached, dtype=np.int32)
        dr_out = np.asarray(dr_output_cached)
        if dr_out.ndim > 1 and dr_out.shape[-1] > 1:
            dr_proba_all.append(dr_out)
            dr_pred_np = np.argmax(dr_out, axis=-1).astype(np.int32)
        else:
            dr_pred_np = np.clip(
                np.round(dr_out.reshape(-1)).astype(np.int32),
                0,
                num_dr_classes - 1,
            )
    else:
        for batch_x, batch_y in val_ds:
            preds = model(batch_x, training=False)
            dr_out = _extract_dr_prediction(preds).numpy()
            dr_gt = _extract_dr_target(batch_y).numpy()

            # Support both softmax DR classification and legacy scalar regression.
            if dr_out.ndim > 1 and dr_out.shape[-1] > 1:
                dr_proba_all.append(dr_out)
                dr_pred_grades = np.argmax(dr_out, axis=-1).astype(int)
            else:
                dr_pred_grades = np.clip(
                    np.round(dr_out.reshape(-1)).astype(int),
                    0,
                    num_dr_classes - 1,
                )

            if dr_gt.ndim > 1 and dr_gt.shape[-1] > 1:
                dr_true_grades = np.argmax(dr_gt, axis=-1).astype(int)
            else:
                dr_true_grades = np.clip(
                    np.round(dr_gt.reshape(-1)).astype(int),
                    0,
                    num_dr_classes - 1,
                )

            dr_true_all.extend(dr_true_grades.tolist())
            dr_pred_all.extend(dr_pred_grades.tolist())

        dr_true_np = np.asarray(dr_true_all, dtype=np.int32)
        dr_pred_np = np.asarray(dr_pred_all, dtype=np.int32)

    base_qwk = float(
        compute_quadratic_weighted_kappa(dr_true_np, dr_pred_np, num_dr_classes)
    )
    base_acc = float(np.mean(dr_true_np == dr_pred_np))

    calibration = {
        "enabled": bool(calibrate_thresholds),
        "applied": False,
        "baseline_qwk": base_qwk,
        "baseline_accuracy": base_acc,
    }

    if calibrate_thresholds and dr_proba_all:
        dr_proba_np = np.concatenate(dr_proba_all, axis=0)
        boundaries, calibrated_pred, calibrated_qwk = _optimize_expected_score_thresholds(
            dr_true_np,
            dr_proba_np,
            num_classes=num_dr_classes,
        )
        calibrated_acc = float(np.mean(dr_true_np == calibrated_pred))

        calibration.update(
            {
                "calibrated_qwk": float(calibrated_qwk),
                "calibrated_accuracy": calibrated_acc,
                "boundaries": [float(x) for x in boundaries.tolist()],
                "gain": float(calibrated_qwk - base_qwk),
                "accuracy_delta": float(calibrated_acc - base_acc),
            }
        )

        qwk_ok = calibrated_qwk >= base_qwk + float(min_qwk_gain)
        acc_ok = calibrated_acc >= base_acc - float(max_accuracy_drop)
        if qwk_ok and acc_ok:
            dr_pred_np = calibrated_pred.astype(np.int32)
            calibration["applied"] = True
            logger.info(
                "DR threshold calibration applied: QWK %.4f -> %.4f (gain=%.4f), "
                "Acc %.4f -> %.4f (delta=%+.4f)",
                base_qwk,
                float(calibrated_qwk),
                float(calibrated_qwk - base_qwk),
                base_acc,
                calibrated_acc,
                float(calibrated_acc - base_acc),
            )
        else:
            logger.info(
                "DR threshold calibration not applied: QWK %.4f -> %.4f (gain=%.4f, min=%.4f), "
                "Acc %.4f -> %.4f (delta=%+.4f, max_drop=%.4f)",
                base_qwk,
                float(calibrated_qwk),
                float(calibrated_qwk - base_qwk),
                float(min_qwk_gain),
                base_acc,
                calibrated_acc,
                float(calibrated_acc - base_acc),
                float(max_accuracy_drop),
            )

    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
        dr_qwk = float(cohen_kappa_score(dr_true_np, dr_pred_np, weights="quadratic"))
        dr_acc = float(accuracy_score(dr_true_np, dr_pred_np))
        cm = confusion_matrix(dr_true_np, dr_pred_np, labels=list(range(num_dr_classes)))
    except ImportError:
        logger.warning("scikit-learn unavailable; DR accuracy/confusion fallback to NumPy.")
        dr_qwk = float(compute_quadratic_weighted_kappa(dr_true_np, dr_pred_np, num_dr_classes))
        dr_acc = float(np.mean(dr_true_np == dr_pred_np))
        cm = np.array(
            [
                [
                    int(np.sum((dr_true_np == i) & (dr_pred_np == j)))
                    for j in range(num_dr_classes)
                ]
                for i in range(num_dr_classes)
            ]
        )

    dr_mae = float(np.mean(np.abs(dr_true_np - dr_pred_np)))
    dr_rmse = float(np.sqrt(np.mean((dr_true_np - dr_pred_np) ** 2)))

    logger.info("DR Grading Results:")
    logger.info("  QWK:      %.4f", dr_qwk)
    logger.info("  Accuracy: %.4f", dr_acc)
    logger.info("  MAE:      %.4f", dr_mae)
    logger.info("  RMSE:     %.4f", dr_rmse)
    logger.info("  Confusion:\n%s", cm)

    results = {
        "dr_qwk": dr_qwk,
        "dr_accuracy": dr_acc,
        "dr_mae": dr_mae,
        "dr_rmse": dr_rmse,
        "confusion_matrix": cm.tolist(),
        "calibration": calibration,
    }

    os.makedirs(output_dir, exist_ok=True)
    dr_json_path = os.path.join(output_dir, "dr_metrics.json")
    with open(dr_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("DR metrics saved to '%s'.", dr_json_path)

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
    calibrate_dme_thresholds: bool = False,
    calibrate_dr_thresholds: bool = False,
    calibration_min_qwk_gain: float = 1e-4,
    dr_calibration_max_accuracy_drop: float = 0.0,
    tta_mode: str = "none",
    ensemble_weight_paths: Optional[Sequence[str]] = None,
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
    y_true, y_proba, y_pred, dr_true_cached, dr_output_cached = _collect_multitask_predictions(
        model=model,
        dataset=dataset,
        num_dme_classes=num_dme_classes,
        ensemble_weight_paths=ensemble_weight_paths,
        tta_mode=tta_mode,
    )

    if ensemble_weight_paths:
        logger.info("Checkpoint ensemble enabled with %d model(s).", len([p for p in ensemble_weight_paths if p]))
    if str(tta_mode).lower() not in {"", "none", "off", "false", "0"}:
        logger.info("TTA enabled (mode=%s).", tta_mode)

    calibration = {
        "dme": {
            "enabled": bool(calibrate_dme_thresholds),
            "applied": False,
        },
        "dr": {
            "enabled": bool(calibrate_dr_thresholds),
            "applied": False,
        },
    }

    if calibrate_dme_thresholds:
        baseline_qwk = float(
            compute_quadratic_weighted_kappa(y_true, y_pred, num_dme_classes)
        )
        boundaries, calibrated_pred, calibrated_qwk = _optimize_expected_score_thresholds(
            y_true,
            y_proba,
            num_classes=num_dme_classes,
        )
        calibration["dme"].update(
            {
                "baseline_qwk": baseline_qwk,
                "calibrated_qwk": float(calibrated_qwk),
                "boundaries": [float(x) for x in boundaries.tolist()],
                "gain": float(calibrated_qwk - baseline_qwk),
            }
        )

        if calibrated_qwk >= baseline_qwk + float(calibration_min_qwk_gain):
            y_pred = calibrated_pred.astype(np.int32)
            calibration["dme"]["applied"] = True
            logger.info(
                "DME threshold calibration applied: QWK %.4f -> %.4f (gain=%.4f)",
                baseline_qwk,
                float(calibrated_qwk),
                float(calibrated_qwk - baseline_qwk),
            )
        else:
            logger.info(
                "DME threshold calibration not applied: QWK %.4f -> %.4f (gain=%.4f < %.4f)",
                baseline_qwk,
                float(calibrated_qwk),
                float(calibrated_qwk - baseline_qwk),
                float(calibration_min_qwk_gain),
            )

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

    dr_metrics = evaluate_dr_grading(
        model,
        dataset,
        output_dir,
        calibrate_thresholds=calibrate_dr_thresholds,
        min_qwk_gain=calibration_min_qwk_gain,
        max_accuracy_drop=dr_calibration_max_accuracy_drop,
        dr_true_cached=dr_true_cached,
        dr_output_cached=dr_output_cached,
    )
    metrics["dr"] = dr_metrics
    if isinstance(dr_metrics, dict) and isinstance(dr_metrics.get("calibration"), dict):
        calibration["dr"] = dr_metrics["calibration"]
    metrics["calibration"] = calibration

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
