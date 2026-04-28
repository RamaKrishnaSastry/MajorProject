"""
train_enhanced.py - QWK-aware enhanced training pipeline for DR+DME classification.

Extends the base train.py with:
- QWK-aware callback for epoch-level monitoring
- Advanced LR scheduling based on QWK plateau detection
- Ordinal-aware loss weighting (penalise large ordinal jumps more)
- Training diagnostics dashboard
- Comprehensive history logging including QWK
- Best model selection based on QWK (not just val_loss)

KEY CHANGE: DR threshold calibration is skipped automatically when the raw
DR QWK already meets or exceeds QWK_TARGET_THRESHOLD (0.80).  There is no
benefit to calibrating a model that is already clinically acceptable, and
doing so risks marginal degradation of accuracy for zero gain in QWK.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Reproducibility: seeds MUST be set before TensorFlow is imported.
# ---------------------------------------------------------------------------
_GLOBAL_SEED: int = int(os.environ.get("PYTHONHASHSEED", 42))
random.seed(_GLOBAL_SEED)

import numpy as np
np.random.seed(_GLOBAL_SEED)

os.environ["PYTHONHASHSEED"]         = str(_GLOBAL_SEED)
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
tf.random.set_seed(_GLOBAL_SEED)
from tensorflow import keras

from model import build_model, build_model_dme_tuning
from qwk_metrics import (
    QWKCallback,
    compute_quadratic_weighted_kappa,
    NUM_DME_CLASSES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False

# QWK threshold above which DR calibration is automatically skipped
QWK_TARGET_THRESHOLD: float = 0.80

# ---------------------------------------------------------------------------
# Default enhanced training configuration
# ---------------------------------------------------------------------------

DEFAULT_ENHANCED_CONFIG: Dict = {
    "input_shape": (512, 512, 3),
    "num_dme_classes": 3,
    "num_dr_classes": 5,
    "aspp_filters": 256,
    "dr_head_units": 256,
    "dme_head_units": 256,
    "dme_head_residual": False,
    "learning_rate": 1e-4,
    "dropout_rate": 0.5,
    "batch_size": 8,
    "epochs": 50,
    "early_stopping_patience": 8,
    "lr_reduce_patience": 4,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-7,
    "checkpoint_dir": "checkpoints_enhanced",
    "history_path": "training_history_enhanced.json",
    "log_path": "training_log_enhanced.csv",
    "qwk_history_path": "qwk_history.json",
    # Ordinal loss weight matrix: larger penalty for distant class misclassification
    "ordinal_loss_weighting": True,
    # Focal loss gamma: 0 = standard CE, 2.0 = standard focal loss.
    "focal_loss_gamma": 2.0,
    "dme_label_smoothing": 0.05,
    "dme_soft_ordinal_weights": True,
    "dr_loss_weight": 0.2,
    "dr_class_weighting": True,
    "dr_class_weight_clip_ratio": 6.0,
    "warmup_epochs": 5,
    "max_batches": None,
    "seed": 42,
    # Stage2 safety: if best stage2 QWK does not beat stage1 baseline,
    # restore stage1-init weights before returning/saving.
    "stage2_revert_if_worse": True,
    "stage2_min_improvement": 0.0,
    # Stage2 collapse guard defaults.
    "collapse_guard_ratio": 0.95,
    "collapse_guard_min_abs_qwk": 0.68,
    "collapse_guard_patience": 3,
    "collapse_guard_hard_drop": 0.20,
    "collapse_guard_start_epoch": 1,
    # Stage2 BN control.
    "stage2_freeze_aspp_bn": True,
    "stage2_freeze_head_bn": True,
    # Joint DME+DR checkpoint policy.
    "joint_checkpoint_enabled": True,
    "joint_qwk_thresholds": [
        [0.70, 0.80],
        [0.72, 0.78],
        [0.75, 0.75],
        [0.70, 0.75],
        [0.70, 0.72],
        [0.70, 0.70],
        [0.68, 0.70],
    ],
    "joint_qwk_fallback_step": 0.02,
    "joint_qwk_min_threshold": 0.60,
    "joint_dme_floor": 0.70,
    # DR calibration: set to True to ALLOW calibration; it will still be
    # skipped automatically when raw DR QWK >= QWK_TARGET_THRESHOLD.
    "calibrate_dr_thresholds": True,
    "calibrate_dme_thresholds": False,
}


# ---------------------------------------------------------------------------
# Ordinal-aware loss weighting
# ---------------------------------------------------------------------------

def build_ordinal_weight_matrix(num_classes: int = NUM_DME_CLASSES) -> np.ndarray:
    """Build a quadratic ordinal penalty matrix."""
    w = np.array(
        [
            [
                1 if i == j else (i - j) ** 2 / max((num_classes - 1) ** 2, 1)
                for j in range(num_classes)
            ]
            for i in range(num_classes)
        ],
        dtype=np.float32,
    )
    w = w / w.max()
    logger.info("Ordinal weight matrix:\n%s", np.round(w, 3))
    return w


class OrdinalWeightedCrossEntropy(keras.losses.Loss):
    """Ordinal-aware weighted cross-entropy loss with optional focal loss."""

    def __init__(
        self,
        num_classes=3,
        class_weights=None,
        focal_loss_gamma=2.0,
        label_smoothing=0.05,
        use_soft_ordinal_weights=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = float(label_smoothing)
        self.use_soft_ordinal_weights = bool(use_soft_ordinal_weights)
        self.ordinal_matrix = self._build_ordinal_matrix(num_classes)
        if class_weights is not None:
            self.class_weights = tf.constant(
                [class_weights.get(i, 1.0) for i in range(num_classes)],
                dtype=tf.float32,
            )
        else:
            self.class_weights = tf.ones(num_classes, dtype=tf.float32)

    def _build_ordinal_matrix(self, num_classes):
        matrix = []
        for i in range(num_classes):
            row = []
            for j in range(num_classes):
                if i == j:
                    weight = 1.0
                else:
                    distance = ((i - j) ** 2) / ((num_classes - 1) ** 2)
                    weight = 1.0 + distance
                row.append(weight)
            matrix.append(row)
        return tf.constant(matrix, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.label_smoothing > 0.0:
            num_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true_smooth = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_cls)
        else:
            y_true_smooth = y_true
        ce_loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        if self.focal_loss_gamma > 0:
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            focal_factor = tf.pow(1.0 - p_t, self.focal_loss_gamma)
            ce_loss = focal_factor * ce_loss
        y_true_class = tf.argmax(y_true, axis=1)
        if self.use_soft_ordinal_weights:
            true_class_rows = tf.gather(self.ordinal_matrix, y_true_class)
            ordinal_weights = tf.reduce_sum(true_class_rows * y_pred, axis=-1)
        else:
            y_pred_class = tf.argmax(y_pred, axis=1)
            ordinal_weights = tf.gather_nd(
                self.ordinal_matrix,
                tf.stack([y_true_class, y_pred_class], axis=1),
            )
        class_weights_per_sample = tf.gather(self.class_weights, y_true_class)
        weighted_loss = ce_loss * ordinal_weights * class_weights_per_sample
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        config = super().get_config()
        class_weights_list = None
        class_weights_tensor = getattr(self, "class_weights", None)
        if isinstance(class_weights_tensor, (tf.Tensor, tf.Variable)):
            try:
                class_weights_list = class_weights_tensor.numpy().tolist()
            except (AttributeError, TypeError, ValueError):
                class_weights_list = None
        config.update({
            "num_classes":            self.num_classes,
            "focal_loss_gamma":       float(getattr(self, "focal_loss_gamma", 0.0)),
            "label_smoothing":        float(getattr(self, "label_smoothing", 0.0)),
            "use_soft_ordinal_weights": bool(getattr(self, "use_soft_ordinal_weights", True)),
            "class_weights":          class_weights_list,
        })
        return config

    @classmethod
    def from_config(cls, config):
        class_weights = config.pop("class_weights", None)
        if isinstance(class_weights, list):
            try:
                class_weights = {i: float(w) for i, w in enumerate(class_weights)}
            except (TypeError, ValueError):
                class_weights = None
        elif not isinstance(class_weights, dict):
            class_weights = None
        return cls(**config, class_weights=class_weights)


class DRWeightedCategoricalCrossEntropy(keras.losses.Loss):
    """Class-weighted categorical cross-entropy for DR head."""

    def __init__(self, num_classes=5, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if class_weights is not None:
            self.class_weights = tf.constant(
                [class_weights.get(i, 1.0) for i in range(num_classes)],
                dtype=tf.float32,
            )
        else:
            self.class_weights = tf.ones(num_classes, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        y_true_class = tf.argmax(y_true, axis=1)
        sample_weights = tf.gather(self.class_weights, y_true_class)
        return tf.reduce_mean(ce * sample_weights)

    def get_config(self):
        config = super().get_config()
        weights_list = None
        try:
            weights_list = self.class_weights.numpy().tolist()
        except Exception:
            weights_list = None
        config.update({
            "num_classes":   self.num_classes,
            "class_weights": weights_list,
        })
        return config


def compute_balanced_class_weights_from_counts(
    counts: np.ndarray,
    clip_ratio: Optional[float] = 8.0,
) -> Optional[Dict[int, float]]:
    """Compute inverse-frequency class weights with optional max/min clipping."""
    if counts is None:
        return None
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        return None
    total = float(np.sum(counts))
    n_classes = int(counts.size)
    if total <= 0 or n_classes <= 0:
        return None
    safe_counts = np.maximum(counts, 1.0)
    weights = total / (n_classes * safe_counts)
    weights = weights / np.mean(weights)
    if clip_ratio is not None and clip_ratio > 1.0:
        min_w = float(np.min(weights))
        max_w = min_w * float(clip_ratio)
        weights = np.clip(weights, min_w, max_w)
    return {i: float(weights[i]) for i in range(n_classes)}


def _extract_dme_labels(batch_labels):
    if isinstance(batch_labels, dict):
        return batch_labels.get("dme_risk", batch_labels.get("dme_output"))
    if isinstance(batch_labels, (tuple, list)):
        return batch_labels[1] if len(batch_labels) > 1 else batch_labels[0]
    return batch_labels


def _extract_dr_labels(batch_labels):
    if isinstance(batch_labels, dict):
        return batch_labels.get("dr_output", batch_labels.get("dr"))
    if isinstance(batch_labels, (tuple, list)):
        return batch_labels[0]
    return batch_labels


def log_dataset_class_distribution(
    dataset: tf.data.Dataset,
    name: str,
    num_classes: int = NUM_DME_CLASSES,
    max_batches: int = 20,
) -> None:
    if num_classes <= 0:
        return
    counts = np.zeros(num_classes, dtype=np.int64)
    samples = 0
    for batch_idx, batch_data in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 2:
            continue
        _, batch_labels = batch_data
        dme_labels = _extract_dme_labels(batch_labels)
        try:
            arr = dme_labels.numpy()
        except (AttributeError, TypeError):
            arr = np.asarray(dme_labels)
        if arr.ndim > 1:
            classes = np.argmax(arr, axis=-1)
        else:
            classes = arr.astype(int).reshape(-1)
        classes = np.clip(classes, 0, num_classes - 1)
        counts += np.bincount(classes, minlength=num_classes)
        samples += int(classes.shape[0])
    if samples == 0:
        return
    fractions = {i: round((counts[i] / samples) * 100.0, 2) for i in range(num_classes)}
    logger.debug(
        "%s DME class distribution from first %d batches: counts=%s, pct=%s",
        name, max_batches, counts.tolist(), fractions,
    )


def compute_dataset_class_counts(
    dataset: tf.data.Dataset,
    num_classes: int = NUM_DME_CLASSES,
) -> Optional[np.ndarray]:
    """Compute exact DME class counts by iterating the full dataset once."""
    if num_classes <= 0:
        return None
    try:
        cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            logger.warning("Dataset has infinite cardinality; cannot compute exact class counts.")
            return None
    except Exception:
        cardinality = None
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch_data in dataset:
        if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 2:
            continue
        _, batch_labels = batch_data
        dme_labels = _extract_dme_labels(batch_labels)
        try:
            arr = dme_labels.numpy()
        except (AttributeError, TypeError):
            arr = np.asarray(dme_labels)
        if arr.ndim > 1:
            classes = np.argmax(arr, axis=-1)
        else:
            classes = arr.astype(int).reshape(-1)
        classes = np.clip(classes, 0, num_classes - 1)
        counts += np.bincount(classes, minlength=num_classes)
    total = int(np.sum(counts))
    if total == 0:
        logger.warning("Could not compute class counts from training dataset (zero samples read).")
        return None
    logger.info(
        "Computed exact train DME class counts from dataset: %s (samples=%d)",
        counts.tolist(), total,
    )
    return counts


def compute_dataset_dr_class_counts(
    dataset: tf.data.Dataset,
    num_classes: int = 5,
) -> Optional[np.ndarray]:
    """Compute exact DR class counts by iterating the full dataset once."""
    if num_classes <= 0:
        return None
    try:
        cardinality = int(tf.data.experimental.cardinality(dataset).numpy())
        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            logger.warning("Dataset has infinite cardinality; cannot compute exact DR class counts.")
            return None
    except Exception:
        cardinality = None
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch_data in dataset:
        if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 2:
            continue
        _, batch_labels = batch_data
        dr_labels = _extract_dr_labels(batch_labels)
        try:
            arr = dr_labels.numpy()
        except (AttributeError, TypeError):
            arr = np.asarray(dr_labels)
        if arr.ndim > 1:
            classes = np.argmax(arr, axis=-1)
        else:
            classes = arr.astype(int).reshape(-1)
        classes = np.clip(classes, 0, num_classes - 1)
        counts += np.bincount(classes, minlength=num_classes)
    total = int(np.sum(counts))
    if total == 0:
        logger.warning("Could not compute DR class counts from training dataset (zero samples read).")
        return None
    logger.info(
        "Computed exact train DR class counts from dataset: %s (samples=%d)",
        counts.tolist(), total,
    )
    return counts


# ---------------------------------------------------------------------------
# QWK-based model checkpointing callbacks
# ---------------------------------------------------------------------------

class QWKModelCheckpoint(keras.callbacks.Callback):
    """Save model weights when val_qwk improves."""

    def __init__(
        self,
        filepath: str,
        verbose: int = 1,
        initial_best_qwk: float = -np.inf,
        alias_filepaths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.best_qwk: float = float(initial_best_qwk)
        self.alias_filepaths = alias_filepaths or []

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return
        if current_qwk > self.best_qwk:
            if self.verbose:
                logger.info(
                    "QWKModelCheckpoint: val_qwk improved %.4f → %.4f, saving to '%s'",
                    self.best_qwk, current_qwk, self.filepath,
                )
            self.best_qwk = current_qwk
            self.model.save_weights(self.filepath)
            for alias_path in self.alias_filepaths:
                self.model.save_weights(alias_path)


class DRQWKModelCheckpoint(keras.callbacks.Callback):
    """Save model weights when val_dr_qwk improves."""

    def __init__(self, filepath: str, verbose: int = 1, initial_best_dr_qwk: float = -np.inf):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.best_dr_qwk: float = float(initial_best_dr_qwk)

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_dr_qwk = logs.get("val_dr_qwk", None)
        if current_dr_qwk is None:
            return
        if float(current_dr_qwk) > self.best_dr_qwk:
            if self.verbose:
                logger.info(
                    "DRQWKModelCheckpoint: val_dr_qwk improved %.4f → %.4f, saving to '%s'",
                    self.best_dr_qwk, float(current_dr_qwk), self.filepath,
                )
            self.best_dr_qwk = float(current_dr_qwk)
            self.model.save_weights(self.filepath)


def _build_joint_qwk_threshold_ladder(
    base_thresholds,
    fallback_step: float = 0.05,
    min_threshold: float = 0.0,
    max_extra_steps: int = 12,
) -> List[Tuple[float, float]]:
    ladder: List[Tuple[float, float]] = []
    for pair in base_thresholds or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            ladder.append((float(pair[0]), float(pair[1])))
        except (TypeError, ValueError):
            continue
    if not ladder:
        ladder = [(0.75, 0.70), (0.75, 0.65), (0.70, 0.60)]
    step = float(fallback_step)
    if step <= 0:
        return ladder
    min_t = float(min_threshold)
    last_dme, last_dr = ladder[-1]
    for i in range(1, max_extra_steps + 1):
        next_dme = max(min_t, last_dme - step * i)
        next_dr  = max(min_t, last_dr  - step * i)
        if (next_dme, next_dr) == ladder[-1]:
            break
        ladder.append((next_dme, next_dr))
        if next_dme <= min_t and next_dr <= min_t:
            break
    return ladder


class JointQWKModelCheckpoint(keras.callbacks.Callback):
    """Save model when joint DME+DR rank improves based on threshold ladder."""

    def __init__(
        self,
        filepath: str,
        thresholds: List[Tuple[float, float]],
        dme_floor: float = 0.0,
        verbose: int = 1,
    ):
        super().__init__()
        self.filepath   = filepath
        self.thresholds = thresholds
        self.dme_floor  = float(dme_floor)
        self.verbose    = verbose
        self.best       = None

    def _make_candidate(self, dme_qwk: float, dr_qwk: float) -> Dict[str, float]:
        tier_index = len(self.thresholds)
        for idx, (dme_t, dr_t) in enumerate(self.thresholds):
            if dme_qwk >= dme_t and dr_qwk >= dr_t:
                tier_index = idx
                break
        harmonic = 0.0
        if dme_qwk + dr_qwk > 0:
            harmonic = 2.0 * dme_qwk * dr_qwk / (dme_qwk + dr_qwk)
        return {
            "tier_index": float(tier_index),
            "harmonic":   float(harmonic),
            "dme_qwk":    float(dme_qwk),
            "dr_qwk":     float(dr_qwk),
        }

    @staticmethod
    def _is_better(candidate: Dict[str, float], best: Dict[str, float]) -> bool:
        if int(candidate["tier_index"]) != int(best["tier_index"]):
            return candidate["tier_index"] < best["tier_index"]
        if candidate["harmonic"] != best["harmonic"]:
            return candidate["harmonic"] > best["harmonic"]
        if candidate["dme_qwk"] != best["dme_qwk"]:
            return candidate["dme_qwk"] > best["dme_qwk"]
        return candidate["dr_qwk"] > best["dr_qwk"]

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        dme_qwk = logs.get("val_qwk",    None)
        dr_qwk  = logs.get("val_dr_qwk", None)
        if dme_qwk is None or dr_qwk is None:
            return
        dme_value        = float(dme_qwk)
        dr_value         = float(dr_qwk)
        blocked_by_floor = dme_value < self.dme_floor
        candidate = self._make_candidate(dme_value, dr_value)
        tier_idx  = int(candidate["tier_index"])
        if tier_idx < len(self.thresholds):
            dme_t, dr_t = self.thresholds[tier_idx]
            tier_label = f"tier={tier_idx+1} (dme>={dme_t:.2f}, dr>={dr_t:.2f})"
        else:
            tier_label = "tier=unmatched"
        saved = False
        if not blocked_by_floor and (self.best is None or self._is_better(candidate, self.best)):
            previous  = self.best
            self.best = candidate
            self.model.save_weights(self.filepath)
            saved = True
            if self.verbose:
                if previous is None:
                    logger.info(
                        "JointQWKCheckpoint: saved initial best (%s, dme=%.4f, dr=%.4f) to '%s'.",
                        tier_label, candidate["dme_qwk"], candidate["dr_qwk"], self.filepath,
                    )
                else:
                    logger.info(
                        "JointQWKCheckpoint: improved (%s, dme=%.4f, dr=%.4f, h=%.4f) -> saved to '%s'.",
                        tier_label, candidate["dme_qwk"], candidate["dr_qwk"],
                        candidate["harmonic"], self.filepath,
                    )
        if self.verbose:
            logger.info(
                "JointCheckpoint: epoch=%d dme=%.4f dr=%.4f %s floor=%.2f -> %s",
                epoch + 1, dme_value, dr_value, tier_label, self.dme_floor,
                "SAVED" if saved else ("blocked by dme_floor" if blocked_by_floor else "not saved"),
            )


class JointQWKFullModelCheckpoint(keras.callbacks.Callback):
    """Wraps JointQWKModelCheckpoint and additionally saves the full model."""

    def __init__(
        self,
        weights_filepath: str,
        full_model_filepath: str,
        thresholds: List[Tuple[float, float]],
        dme_floor: float = 0.0,
        verbose: int = 1,
    ):
        super().__init__()
        self._inner = JointQWKModelCheckpoint(
            filepath=weights_filepath,
            thresholds=thresholds,
            dme_floor=dme_floor,
            verbose=verbose,
        )
        self.full_model_filepath = full_model_filepath
        self.verbose = verbose
        self._prev_best_harmonic: float = -float("inf")

    def set_model(self, model):
        super().set_model(model)
        self._inner.set_model(model)

    def on_epoch_end(self, epoch: int, logs=None):
        self._inner.on_epoch_end(epoch, logs)
        current_best = self._inner.best
        if current_best is None:
            return
        current_harmonic = float(current_best.get("harmonic", -float("inf")))
        if current_harmonic > self._prev_best_harmonic:
            self._prev_best_harmonic = current_harmonic
            try:
                self.model.save(self.full_model_filepath)
                if self.verbose:
                    logger.info(
                        "JointQWKFullModelCheckpoint: full model saved to '%s' "
                        "(dme=%.4f, dr=%.4f, harmonic=%.4f).",
                        self.full_model_filepath,
                        float(current_best.get("dme_qwk", 0.0)),
                        float(current_best.get("dr_qwk", 0.0)),
                        current_harmonic,
                    )
            except Exception as exc:
                logger.warning(
                    "JointQWKFullModelCheckpoint: could not save full model: %s", exc
                )


# ---------------------------------------------------------------------------
# QWK-based early stopping callback
# ---------------------------------------------------------------------------

class QWKEarlyStopping(keras.callbacks.Callback):
    """Stop training when val_qwk stops improving."""

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.patience            = patience
        self.min_delta           = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose             = verbose
        self.best_qwk:   float   = -np.inf
        self.wait:       int     = 0
        self.best_weights        = None
        self.stopped_epoch: int  = 0

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return
        if current_qwk > self.best_qwk + self.min_delta:
            self.best_qwk = current_qwk
            self.wait     = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch     = epoch + 1
                self.model.stop_training = True
                if self.verbose:
                    logger.info(
                        "QWKEarlyStopping: no improvement for %d epochs. Stopping at epoch %d.",
                        self.patience, self.stopped_epoch,
                    )
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    logger.info("Restored best weights (val_qwk=%.4f).", self.best_qwk)


# ---------------------------------------------------------------------------
# QWK-based LR scheduler
# ---------------------------------------------------------------------------

class QWKReduceLROnPlateau(keras.callbacks.Callback):
    """Reduce LR when val_qwk stops improving."""

    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 4,
        min_lr: float = 1e-7,
        verbose: int = 1,
    ):
        super().__init__()
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr
        self.verbose  = verbose
        self.best_qwk: float = -np.inf
        self.wait: int = 0

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return
        if current_qwk > self.best_qwk:
            self.best_qwk = current_qwk
            self.wait     = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                if self.verbose:
                    logger.info(
                        "QWKReduceLROnPlateau: reducing LR from %.2e to %.2e",
                        old_lr, new_lr,
                    )
                self.wait = 0


# ---------------------------------------------------------------------------
# Training diagnostics dashboard
# ---------------------------------------------------------------------------

class TrainingDiagnosticsCallback(keras.callbacks.Callback):
    """Log and optionally plot training diagnostics per epoch."""

    def __init__(
        self,
        history_path: str = "training_history_enhanced.json",
        plot_dir: str = ".",
        plot_every: int = 0,
    ):
        super().__init__()
        self.history_path = history_path
        self.plot_dir     = plot_dir
        self.plot_every   = plot_every
        self._history: Dict[str, List] = {}

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self._history.setdefault(k, []).append(float(v))
        with open(self.history_path, "w") as f:
            json.dump(self._history, f, indent=2)
        if self.plot_every and _PLOTTING_AVAILABLE and (epoch + 1) % self.plot_every == 0:
            self._plot(epoch + 1)

    def on_train_end(self, logs=None):
        if _PLOTTING_AVAILABLE:
            self._plot("final")

    def _plot(self, suffix):
        if not _PLOTTING_AVAILABLE:
            return
        os.makedirs(self.plot_dir, exist_ok=True)
        keys       = list(self._history.keys())
        train_keys = [k for k in keys if not k.startswith("val_")]
        fig, axes  = plt.subplots(
            1, max(len(train_keys), 1),
            figsize=(6 * max(len(train_keys), 1), 4),
        )
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for ax, tk in zip(axes, train_keys):
            ax.plot(self._history[tk], label=tk)
            vk = f"val_{tk}"
            if vk in self._history:
                ax.plot(self._history[vk], label=vk)
            if "qwk" in tk:
                ax.axhline(y=0.80, color="red", linestyle="--", label="Target 0.80")
            ax.set_xlabel("Epoch"); ax.set_ylabel(tk)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.plot_dir, f"training_diagnostics_{suffix}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Diagnostics plot saved to '%s'.", path)


# ---------------------------------------------------------------------------
# Model compilation (enhanced)
# ---------------------------------------------------------------------------

def compile_model_enhanced(
    model: keras.Model,
    learning_rate: float = 1e-4,
    num_dme_classes: int = 3,
    num_dr_classes: int = 5,
    class_weights: Optional[Dict[int, float]] = None,
    dr_class_weights: Optional[Dict[int, float]] = None,
    ordinal_loss_weighting: bool = True,
    focal_loss_gamma: float = 2.0,
    dme_label_smoothing: float = 0.05,
    dme_soft_ordinal_weights: bool = True,
    dr_loss_weight: float = 0.2,
    dr_class_weighting: bool = True,
) -> keras.Model:
    """Compile with ordinal + class weighting baked into loss."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    if ordinal_loss_weighting:
        dme_loss = OrdinalWeightedCrossEntropy(
            num_classes=num_dme_classes,
            class_weights=class_weights,
            focal_loss_gamma=focal_loss_gamma,
            label_smoothing=dme_label_smoothing,
            use_soft_ordinal_weights=dme_soft_ordinal_weights,
        )
        logger.info("Using ordinal-weighted cross-entropy loss with class balancing.")
        logger.info("Focal loss gamma=%.1f.", focal_loss_gamma)
        if class_weights:
            logger.info("DME class weights applied in loss: %s", class_weights)
    else:
        dme_loss = "categorical_crossentropy"

    if dr_class_weighting and dr_class_weights:
        dr_loss = DRWeightedCategoricalCrossEntropy(
            num_classes=num_dr_classes, class_weights=dr_class_weights,
        )
        logger.info("DR class-weighted CE enabled: %s", dr_class_weights)
    else:
        dr_loss = "categorical_crossentropy"

    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": dr_loss,
            "dme_risk":  dme_loss,
        },
        loss_weights={
            "dr_output": float(dr_loss_weight),
            "dme_risk":  1.0,
        },
        metrics={
            "dr_output": [keras.metrics.CategoricalAccuracy(name="dr_accuracy")],
            "dme_risk":  [keras.metrics.CategoricalAccuracy(name="accuracy")],
        },
    )
    logger.info("Enhanced model compiled (lr=%.2e).", learning_rate)
    return model


# ---------------------------------------------------------------------------
# Callbacks builder
# ---------------------------------------------------------------------------

class LinearWarmupCallback(keras.callbacks.Callback):
    """Warm up LR linearly over first N epochs."""
    def __init__(self, target_lr: float, warmup_epochs: int = 5):
        super().__init__()
        self.target_lr     = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate = lr
            logger.info("Warmup epoch %d/%d: lr=%.2e", epoch+1, self.warmup_epochs, lr)
        elif epoch == self.warmup_epochs:
            self.model.optimizer.learning_rate = self.target_lr


class Stage2QWKCollapseGuard(keras.callbacks.Callback):
    """Stop stage2 and restore init checkpoint if QWK collapses."""

    def __init__(
        self,
        baseline_qwk: float,
        init_weights_path: str,
        min_ratio: float = 0.70,
        min_abs_qwk: float = 0.20,
        patience: int = 2,
        hard_drop: float = 0.20,
        start_epoch: int = 1,
        verbose: int = 1,
    ):
        super().__init__()
        self.baseline_qwk    = float(baseline_qwk)
        self.init_weights_path = init_weights_path
        self.min_ratio       = float(min_ratio)
        self.min_abs_qwk     = float(min_abs_qwk)
        self.patience        = int(patience)
        self.hard_drop       = float(hard_drop)
        self.start_epoch     = int(start_epoch)
        self.verbose         = verbose
        self.bad_epochs      = 0

    def _trigger_restore_stop(self, reason: str) -> None:
        if self.init_weights_path and os.path.exists(self.init_weights_path):
            self.model.load_weights(self.init_weights_path)
            logger.warning(
                "Stage2CollapseGuard triggered (%s): restored init weights from '%s'.",
                reason, self.init_weights_path,
            )
        else:
            logger.warning(
                "Stage2CollapseGuard triggered (%s) but init weights unavailable: '%s'.",
                reason, self.init_weights_path,
            )
        self.model.stop_training = True

    def _threshold(self) -> float:
        ratio_threshold = self.baseline_qwk * self.min_ratio
        if self.baseline_qwk >= self.min_abs_qwk:
            return max(ratio_threshold, self.min_abs_qwk)
        return ratio_threshold

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return
        epoch_num = epoch + 1
        if epoch_num < self.start_epoch:
            return
        current_qwk_value = float(current_qwk)
        hard_threshold = self.baseline_qwk - self.hard_drop
        if current_qwk_value < hard_threshold:
            if self.verbose:
                logger.warning(
                    "Stage2CollapseGuard: val_qwk=%.4f below hard threshold=%.4f.",
                    current_qwk_value, hard_threshold,
                )
            self._trigger_restore_stop(reason="hard_drop")
            return
        threshold = self._threshold()
        if current_qwk_value < threshold:
            self.bad_epochs += 1
            if self.verbose:
                logger.warning(
                    "Stage2CollapseGuard: val_qwk=%.4f below threshold=%.4f (%d/%d bad epochs).",
                    current_qwk_value, threshold, self.bad_epochs, self.patience,
                )
        else:
            self.bad_epochs = 0
        if self.bad_epochs >= self.patience:
            self._trigger_restore_stop(reason="patience")


def build_enhanced_callbacks(val_dataset: tf.data.Dataset, config: Dict) -> list:
    """Build the enhanced callback stack."""
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_qwk_path   = os.path.join(config["checkpoint_dir"], "best_qwk.weights.h5")
    best_dme_path   = os.path.join(config["checkpoint_dir"], "best_dme.weights.h5")
    best_dr_path    = os.path.join(config["checkpoint_dir"], "best_dr.weights.h5")
    best_joint_path = os.path.join(config["checkpoint_dir"], "best_joint.weights.h5")

    checkpoint_initial_best_qwk = -np.inf
    stage1_baseline_qwk = config.get("stage1_baseline_qwk", None)
    if bool(config.get("stage2_checkpoint_use_stage1_baseline", True)) and stage1_baseline_qwk is not None:
        try:
            checkpoint_initial_best_qwk = float(stage1_baseline_qwk)
            logger.info(
                "Stage 2 checkpoint baseline set to Stage 1 val_qwk=%.4f.",
                checkpoint_initial_best_qwk,
            )
        except (TypeError, ValueError):
            logger.warning("Invalid stage1_baseline_qwk=%r; using -inf.", stage1_baseline_qwk)

    callbacks = []

    warmup_epochs = int(config.get("warmup_epochs", 5))
    if warmup_epochs > 0:
        callbacks.append(LinearWarmupCallback(
            target_lr=config["learning_rate"], warmup_epochs=warmup_epochs,
        ))

    callbacks.append(QWKCallback(
        val_dataset=val_dataset,
        num_classes=config["num_dme_classes"],
        dr_num_classes=config.get("num_dr_classes", 5),
        compute_dr_metrics=True,
        history_path=config["qwk_history_path"],
        verbose=1,
        max_batches=config.get("max_batches", None),
    ))

    if bool(config.get("joint_checkpoint_enabled", True)):
        joint_thresholds = _build_joint_qwk_threshold_ladder(
            config.get("joint_qwk_thresholds", [
                [0.70, 0.80], [0.72, 0.78], [0.75, 0.75],
                [0.70, 0.75], [0.70, 0.72], [0.70, 0.70], [0.68, 0.70],
            ]),
            fallback_step=float(config.get("joint_qwk_fallback_step", 0.02)),
            min_threshold=float(config.get("joint_qwk_min_threshold", 0.60)),
        )
        dme_floor = float(config.get("joint_dme_floor", 0.70))
        best_joint_full_path = best_joint_path.replace(".weights.h5", ".full_model.keras")
        callbacks.append(JointQWKFullModelCheckpoint(
            weights_filepath=best_joint_path,
            full_model_filepath=best_joint_full_path,
            thresholds=joint_thresholds,
            dme_floor=dme_floor,
            verbose=1,
        ))
        config["_best_joint_full_model_path"] = best_joint_full_path
        logger.info(
            "JointQWKCheckpoint: ladder=%s, dme_floor=%.2f, full_model='%s'.",
            [tuple(np.round(t, 3)) for t in joint_thresholds],
            dme_floor, best_joint_full_path,
        )

    if (
        config.get("collapse_guard_enabled", False)
        and config.get("stage1_baseline_qwk") is not None
        and config.get("stage2_init_weights_path")
    ):
        callbacks.append(Stage2QWKCollapseGuard(
            baseline_qwk=float(config["stage1_baseline_qwk"]),
            init_weights_path=str(config["stage2_init_weights_path"]),
            min_ratio=float(config.get("collapse_guard_ratio", 0.95)),
            min_abs_qwk=float(config.get("collapse_guard_min_abs_qwk", 0.68)),
            patience=int(config.get("collapse_guard_patience", 3)),
            hard_drop=float(config.get("collapse_guard_hard_drop", 0.20)),
            start_epoch=int(config.get("collapse_guard_start_epoch", 1)),
            verbose=1,
        ))

    callbacks.extend([
        QWKModelCheckpoint(
            filepath=best_qwk_path, verbose=1,
            initial_best_qwk=checkpoint_initial_best_qwk,
            alias_filepaths=[best_dme_path],
        ),
        DRQWKModelCheckpoint(filepath=best_dr_path, verbose=1),
        QWKEarlyStopping(patience=config["early_stopping_patience"],
                         restore_best_weights=True, verbose=1),
        QWKReduceLROnPlateau(
            factor=config["lr_reduce_factor"],
            patience=config["lr_reduce_patience"],
            min_lr=config["min_lr"], verbose=1,
        ),
        keras.callbacks.CSVLogger(config["log_path"], append=False),
        TrainingDiagnosticsCallback(
            history_path=config["history_path"],
            plot_dir=config.get("output_dir", "."),
        ),
    ])
    return callbacks


def _freeze_backbone_batchnorm_layers(
    model: keras.Model,
    backbone_layer_name: str = "resnet50_conv4_backbone",
    freeze_aspp_bn: bool = True,
    freeze_head_bn: bool = False,
) -> int:
    count = 0
    stack = []
    try:
        stack.append(model.get_layer(backbone_layer_name))
    except Exception as e:
        logger.warning("Could not locate backbone layer '%s': %s", backbone_layer_name, e)
    if freeze_aspp_bn:
        for layer in model.layers:
            if layer.name.startswith("aspp_"):
                stack.append(layer)
    if freeze_head_bn:
        for layer in model.layers:
            if layer.name.startswith("dr_"):
                stack.append(layer)
    if not stack:
        return 0
    seen = set()
    while stack:
        current = stack.pop()
        obj_id  = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if isinstance(current, keras.layers.BatchNormalization):
            current.trainable = False
            count += 1
        child_layers = getattr(current, "layers", None)
        if child_layers:
            stack.extend(child_layers)
    return count


# ---------------------------------------------------------------------------
# Calibration skip helper
# ---------------------------------------------------------------------------

def _should_skip_dr_calibration(val_ds: tf.data.Dataset, model: keras.Model,
                                  num_dr_classes: int) -> bool:
    """Return True if raw DR QWK already meets the clinical threshold.

    Runs a lightweight single-pass inference over the validation set to
    compute the raw DR QWK.  If it is >= QWK_TARGET_THRESHOLD (0.80) the
    caller should skip threshold calibration entirely.
    """
    dr_true_all, dr_pred_all = [], []
    try:
        for batch_images, batch_labels in val_ds:
            preds = model(batch_images, training=False)
            if isinstance(preds, dict):
                dr_proba = preds.get("dr_output")
            elif isinstance(preds, (list, tuple)):
                dr_proba = preds[0]
            else:
                dr_proba = None
            if dr_proba is None:
                continue
            dr_proba_np = dr_proba.numpy() if hasattr(dr_proba, "numpy") else np.asarray(dr_proba)
            if isinstance(batch_labels, dict):
                dr_labels = batch_labels.get("dr_output", batch_labels.get("dr"))
            elif isinstance(batch_labels, (list, tuple)):
                dr_labels = batch_labels[0]
            else:
                dr_labels = None
            if dr_labels is None:
                continue
            dr_labels_np = dr_labels.numpy() if hasattr(dr_labels, "numpy") else np.asarray(dr_labels)
            if dr_labels_np.ndim > 1:
                dr_true_batch = np.argmax(dr_labels_np, axis=-1).astype(int)
            else:
                dr_true_batch = np.clip(np.rint(dr_labels_np.reshape(-1)).astype(int), 0, num_dr_classes - 1)
            dr_pred_batch = np.argmax(dr_proba_np, axis=-1).astype(int)
            dr_true_all.append(dr_true_batch)
            dr_pred_all.append(dr_pred_batch)
    except Exception as exc:
        logger.warning("Could not compute raw DR QWK for calibration check: %s", exc)
        return False

    if not dr_true_all:
        return False

    dr_true = np.concatenate(dr_true_all)
    dr_pred = np.concatenate(dr_pred_all)
    raw_dr_qwk = float(compute_quadratic_weighted_kappa(dr_true, dr_pred, num_dr_classes))
    logger.info(
        "Pre-calibration raw DR QWK = %.4f (threshold = %.2f).",
        raw_dr_qwk, QWK_TARGET_THRESHOLD,
    )
    if raw_dr_qwk >= QWK_TARGET_THRESHOLD:
        logger.info(
            "Raw DR QWK already meets clinical target (%.4f >= %.2f). "
            "Skipping DR threshold calibration.",
            raw_dr_qwk, QWK_TARGET_THRESHOLD,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Main training entry point (enhanced)
# ---------------------------------------------------------------------------

def train_enhanced(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    pretrained_weights: Optional[str] = None,
    eyepacs_backbone: Optional[str] = None,
    backbone_weights_path: Optional[str] = None,
    config: Optional[Dict] = None,
    output_weights: str = "dme_enhanced.weights.h5",
    use_dme_tuning: bool = False,
) -> Tuple[keras.Model, Dict]:

    cfg = {**DEFAULT_ENHANCED_CONFIG, **(config or {})}
    logger.info("Building enhanced model …")
    dr_class_counts_cache: Optional[np.ndarray] = None
    dr_class_weights_for_loss: Optional[Dict[int, float]] = None

    if use_dme_tuning:
        logger.info("Stage 2: DME head fine-tuning (backbone frozen)")
        model = build_model_dme_tuning(
            input_shape=tuple(cfg["input_shape"]),
            pretrained_weights=pretrained_weights,
            num_dme_classes=cfg["num_dme_classes"],
            num_dr_classes=cfg.get("num_dr_classes", 5),
            aspp_filters=int(cfg.get("aspp_filters", 256)),
            dr_head_units=int(cfg.get("dr_head_units", 256)),
            dme_head_units=int(cfg.get("dme_head_units", 256)),
            dme_head_residual=bool(cfg.get("dme_head_residual", False)),
            dropout_rate=float(cfg.get("dropout_rate", 0.5)),
            backbone_weights_path=backbone_weights_path,
        )
    else:
        if pretrained_weights is not None:
            logger.info("Stage 2: full-model fine-tuning from Stage 1 checkpoint")
            backbone_weights = None
        elif eyepacs_backbone is not None:
            logger.info("Stage 1: Backbone frozen, training ASPP + heads only (EyePACS)")
            backbone_weights = None
        else:
            logger.info("Stage 1: Backbone frozen, training ASPP + heads only")
            backbone_weights = "imagenet"

        model_backbone_weights_path = (
            None
            if (eyepacs_backbone is not None or pretrained_weights is not None)
            else backbone_weights_path
        )

        model = build_model(
            input_shape=tuple(cfg["input_shape"]),
            backbone_weights=backbone_weights,
            num_dme_classes=cfg["num_dme_classes"],
            num_dr_classes=cfg.get("num_dr_classes", 5),
            aspp_filters=int(cfg.get("aspp_filters", 256)),
            dr_head_units=int(cfg.get("dr_head_units", 256)),
            dme_head_units=int(cfg.get("dme_head_units", 256)),
            dme_head_residual=bool(cfg.get("dme_head_residual", False)),
            dropout_rate=float(cfg.get("dropout_rate", 0.5)),
            trainable=True,
            backbone_weights_path=model_backbone_weights_path,
        )

        if eyepacs_backbone is not None and pretrained_weights is None:
            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.load_weights(eyepacs_backbone, skip_mismatch=True)
                logger.info("Loaded EyePACS backbone weights from '%s'.", eyepacs_backbone)
            except Exception as e:
                logger.warning("Could not load EyePACS backbone weights: %s", e)

        if pretrained_weights is None:
            # Stage 1: freeze backbone, initialise output biases
            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.trainable = False
                trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
                logger.info("Backbone FROZEN. Trainable params: %d", trainable_count)
            except Exception as e:
                logger.warning("Could not freeze backbone: %s", e)

            try:
                dme_head = model.get_layer("dme_risk")
                class_counts = compute_dataset_class_counts(
                    train_ds, num_classes=int(cfg.get("num_dme_classes", NUM_DME_CLASSES)),
                )
                if class_counts is None:
                    raise ValueError("Training class counts unavailable for DME bias init.")
                class_counts  = class_counts.astype(np.float32)
                class_probs   = class_counts / class_counts.sum()
                log_probs     = np.log(class_probs + 1e-7)
                w = dme_head.get_weights()
                if len(w) >= 2:
                    w[1] = log_probs
                    dme_head.set_weights(w)
                    logger.info("DME head bias initialised: %s", np.round(log_probs, 3))
            except Exception as e:
                logger.warning("Could not initialise DME bias: %s", e)

            try:
                dr_head = model.get_layer("dr_output")
                dr_class_counts = compute_dataset_dr_class_counts(
                    train_ds, num_classes=int(cfg.get("num_dr_classes", 5)),
                )
                dr_class_counts_cache = dr_class_counts
                if dr_class_counts is None:
                    raise ValueError("Training class counts unavailable for DR bias init.")
                dr_class_counts  = dr_class_counts.astype(np.float32)
                dr_class_probs   = dr_class_counts / dr_class_counts.sum()
                dr_log_probs     = np.log(dr_class_probs + 1e-7)
                dr_weights = dr_head.get_weights()
                if len(dr_weights) >= 2:
                    dr_weights[1] = dr_log_probs
                    dr_head.set_weights(dr_weights)
                    logger.info("DR head bias initialised: %s", np.round(dr_log_probs, 3))
            except Exception as e:
                logger.warning("Could not initialise DR bias: %s", e)

        else:
            # Stage 2: restore Stage 1 checkpoint, unfreeze backbone
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(
                    f"Stage 2 checkpoint not found: '{pretrained_weights}'."
                )
            logger.info(
                "Stage 2 restoring full Stage 1 checkpoint from '%s'.", pretrained_weights,
            )
            dme_head = model.get_layer("dme_risk")
            dme_head_before = [np.copy(w) for w in dme_head.get_weights()]
            model.load_weights(pretrained_weights, skip_mismatch=True)
            dme_head_after = dme_head.get_weights()
            if dme_head_before and dme_head_after and len(dme_head_before) == len(dme_head_after):
                max_head_delta = max(
                    float(np.max(np.abs(before - after)))
                    for before, after in zip(dme_head_before, dme_head_after)
                )
                if max_head_delta < 1e-8:
                    raise RuntimeError(
                        "Stage 2 checkpoint restore failed: DME head unchanged after load."
                    )
                logger.info(
                    "Stage 2 restore verified (max |delta| DME head = %.3e).", max_head_delta,
                )
            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.trainable = True
                freeze_aspp_bn = bool(cfg.get("stage2_freeze_aspp_bn", True))
                freeze_head_bn = bool(cfg.get("stage2_freeze_head_bn", True))
                bn_frozen      = _freeze_backbone_batchnorm_layers(
                    model, freeze_aspp_bn=freeze_aspp_bn, freeze_head_bn=freeze_head_bn,
                )
                trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
                logger.info(
                    "Stage 2: Backbone UNFROZEN, %d BN layers frozen. Trainable params: %d",
                    bn_frozen, trainable_count,
                )
            except Exception as e:
                logger.warning("Could not unfreeze backbone for Stage 2: %s", e)

    if bool(cfg.get("dr_class_weighting", True)):
        if dr_class_counts_cache is None:
            dr_class_counts_cache = compute_dataset_dr_class_counts(
                train_ds, num_classes=int(cfg.get("num_dr_classes", 5)),
            )
        dr_class_weights_for_loss = compute_balanced_class_weights_from_counts(
            dr_class_counts_cache,
            clip_ratio=cfg.get("dr_class_weight_clip_ratio", 6.0),
        )
        if dr_class_weights_for_loss:
            logger.info("DR loss class weights: %s", dr_class_weights_for_loss)

    model = compile_model_enhanced(
        model,
        learning_rate=cfg["learning_rate"],
        num_dme_classes=cfg["num_dme_classes"],
        num_dr_classes=cfg.get("num_dr_classes", 5),
        class_weights=class_weights,
        dr_class_weights=dr_class_weights_for_loss,
        ordinal_loss_weighting=cfg.get("ordinal_loss_weighting", True),
        focal_loss_gamma=cfg.get("focal_loss_gamma", 2.0),
        dme_label_smoothing=float(cfg.get("dme_label_smoothing", 0.05)),
        dme_soft_ordinal_weights=bool(cfg.get("dme_soft_ordinal_weights", True)),
        dr_loss_weight=float(cfg.get("dr_loss_weight", 0.2)),
        dr_class_weighting=bool(cfg.get("dr_class_weighting", True)),
    )

    callbacks = build_enhanced_callbacks(val_ds, cfg)

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, dme_tuning=%s",
        cfg["epochs"], cfg["batch_size"], use_dme_tuning,
    )
    log_dataset_class_distribution(train_ds, "Train", num_classes=cfg["num_dme_classes"])
    log_dataset_class_distribution(val_ds,   "Val",   num_classes=cfg["num_dme_classes"])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Final stage2 guardrail
    if (
        pretrained_weights is not None
        and cfg.get("stage2_revert_if_worse", True)
        and cfg.get("stage1_baseline_qwk") is not None
    ):
        baseline_qwk      = float(cfg["stage1_baseline_qwk"])
        min_improvement   = float(cfg.get("stage2_min_improvement", 0.0))
        qwk_history       = history.history.get("val_qwk", [])
        best_stage2_qwk   = max(qwk_history) if qwk_history else -float("inf")
        if best_stage2_qwk < baseline_qwk + min_improvement:
            logger.warning(
                "Stage 2 best val_qwk=%.4f did not beat Stage 1 baseline=%.4f (required +%.4f). "
                "Restoring stage2 init weights from '%s'.",
                best_stage2_qwk, baseline_qwk, min_improvement, pretrained_weights,
            )
            if os.path.exists(pretrained_weights):
                model.load_weights(pretrained_weights)
                history.history["stage2_reverted_to_stage1"] = [1.0]
            else:
                logger.warning("Cannot restore: file missing: '%s'.", pretrained_weights)

    model.save_weights(output_weights)
    logger.info("Saved weights to '%s'.", output_weights)
    model_path = output_weights.replace(".weights.h5", ".model.h5")
    model.save(model_path)
    logger.info("Saved full model to '%s'.", model_path)

    # -----------------------------------------------------------------------
    # Post-stage evaluation on best joint checkpoint — RAW QWK only.
    #
    # DR threshold calibration is SKIPPED if raw DR QWK already meets the
    # clinical target (QWK_TARGET_THRESHOLD = 0.80).  This prevents the
    # calibration search from making unnecessary changes to a model that is
    # already clinically acceptable.
    # -----------------------------------------------------------------------
    best_joint_full  = cfg.get("_best_joint_full_model_path", None)
    eval_model_path  = (
        best_joint_full
        if (best_joint_full and os.path.exists(best_joint_full))
        else model_path
    )
    stage_label = "Stage 2" if pretrained_weights is not None else "Stage 1"

    if eval_model_path and os.path.exists(eval_model_path):
        eval_output_dir = os.path.join(
            cfg.get("output_dir", "."),
            f"eval_{stage_label.replace(' ', '_').lower()}",
        )
        os.makedirs(eval_output_dir, exist_ok=True)
        try:
            from evaluate_comprehensive import evaluate_on_best_joint_model

            # ---------------------------------------------------------------
            # Decide whether to calibrate DR thresholds.
            # Never calibrate when the raw DR QWK already meets the target.
            # ---------------------------------------------------------------
            allow_dr_calibration = bool(cfg.get("calibrate_dr_thresholds", True))
            if allow_dr_calibration:
                # Quick pass to check raw DR QWK on the current final model
                # weights (not necessarily the best-joint checkpoint).
                skip_calib = _should_skip_dr_calibration(
                    val_ds, model, num_dr_classes=cfg.get("num_dr_classes", 5),
                )
                if skip_calib:
                    allow_dr_calibration = False
            else:
                logger.info(
                    "DR calibration disabled in config (calibrate_dr_thresholds=False)."
                )

            logger.info(
                "Running post-stage evaluation: model='%s', output='%s', "
                "calibrate_dr=%s, calibrate_dme=%s",
                eval_model_path, eval_output_dir,
                allow_dr_calibration,
                bool(cfg.get("calibrate_dme_thresholds", False)),
            )

            eval_metrics = evaluate_on_best_joint_model(
                checkpoint_path=eval_model_path,
                dataset=val_ds,
                output_dir=eval_output_dir,
                stage_label=stage_label,
                num_dme_classes=cfg.get("num_dme_classes", NUM_DME_CLASSES),
                num_dr_classes=cfg.get("num_dr_classes", 5),
                # Pass resolved flags (evaluate_on_best_joint_model ignores them
                # in the updated evaluate_comprehensive — kept for compatibility)
                calibrate_dr_thresholds=allow_dr_calibration,
                calibrate_dme_thresholds=bool(cfg.get("calibrate_dme_thresholds", False)),
            )
            logger.info(
                "Post-stage eval complete: DME QWK(raw)=%.4f  DR QWK(raw)=%.4f",
                eval_metrics.get("qwk", float("nan")),
                eval_metrics.get("dr_qwk_raw", float("nan")),
            )
            history.history["post_eval_dme_qwk"]    = [eval_metrics.get("qwk", float("nan"))]
            history.history["post_eval_dr_qwk_raw"] = [eval_metrics.get("dr_qwk_raw", float("nan"))]
        except Exception as exc:
            logger.warning("Post-stage evaluation failed (non-fatal): %s", exc)
    else:
        logger.warning(
            "No best-joint full model found at '%s'; skipping post-stage evaluation.",
            eval_model_path,
        )

    return model, history.history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Enhanced QWK-aware DME training")
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--image-dir",  type=str, default=None)
    parser.add_argument("--weights",    type=str, default=None)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--output",     type=str, default="dme_enhanced.weights.h5")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--mock",       action="store_true")
    parser.add_argument("--no-ordinal-loss", action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        logger.info("Using mock dataset for testing.")
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_enhanced", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    train_ds, val_ds, class_weights = build_datasets(
        csv_path=csv_path, image_dir=image_dir, batch_size=args.batch_size,
    )

    config = {
        **DEFAULT_ENHANCED_CONFIG,
        "epochs":                args.epochs,
        "batch_size":            args.batch_size,
        "learning_rate":         args.lr,
        "output_dir":            args.output_dir,
        "ordinal_loss_weighting": not args.no_ordinal_loss,
    }

    model, history = train_enhanced(
        train_ds=train_ds, val_ds=val_ds,
        class_weights=class_weights,
        pretrained_weights=args.weights,
        config=config,
        output_weights=args.output,
    )
    logger.info("Enhanced training complete.")


if __name__ == "__main__":
    main()