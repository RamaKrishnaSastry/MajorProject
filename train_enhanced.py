"""
train_enhanced.py - QWK-aware enhanced training pipeline for DR+DME classification.

Extends the base train.py with:
- QWK-aware callback for epoch-level monitoring
- Advanced LR scheduling based on QWK plateau detection
- Ordinal-aware loss weighting (penalise large ordinal jumps more)
- Training diagnostics dashboard
- Comprehensive history logging including QWK
- Best model selection based on QWK (not just val_loss)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
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

# ---------------------------------------------------------------------------
# Default enhanced training configuration
# ---------------------------------------------------------------------------

DEFAULT_ENHANCED_CONFIG: Dict = {
    "input_shape": (512, 512, 3),
    "num_dme_classes": 3,
    "num_dr_classes": 5,
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
    # Higher gamma reduces gradient for easy majority-class samples.
    "focal_loss_gamma": 2.0,
    "max_batches": None,  # None = process entire validation set for accurate QWK
    "seed": 42,
    # Stage2 safety: if best stage2 QWK does not beat stage1 baseline,
    # restore stage1-init weights before returning/saving.
    "stage2_revert_if_worse": True,
    "stage2_min_improvement": 0.0,
}


# ---------------------------------------------------------------------------
# Ordinal-aware loss weighting
# ---------------------------------------------------------------------------

def build_ordinal_weight_matrix(num_classes: int = NUM_DME_CLASSES) -> np.ndarray:
    """Build a quadratic ordinal penalty matrix.

    Entry (i, j) = (i-j)^2 / (K-1)^2.  Used to weight cross-entropy loss
    so that Mild→Severe errors are penalised more than Mild→Moderate.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes.

    Returns
    -------
    np.ndarray
        Weight matrix of shape (num_classes, num_classes).
    """
    w = np.array(
        [
            [
                1 if i==j else (i - j) ** 2 / max((num_classes - 1) ** 2, 1)
                for j in range(num_classes)
            ]
            for i in range(num_classes)
        ],
        dtype=np.float32,
    )
    # Normalise so diagonal stays ~1.0
    w = w / w.max()
    logger.info("Ordinal weight matrix:\n%s", np.round(w, 3))
    return w


class OrdinalWeightedCrossEntropy(keras.losses.Loss):
    """Ordinal-aware weighted cross-entropy loss with optional focal loss.

    Combines three complementary mechanisms to address class imbalance and
    ordinal misclassification:

    1. **Ordinal penalty matrix** – off-diagonal entries are the normalised
       class distance ``(i-j)^2 / (K-1)^2`` while the diagonal remains 1.0 to
       keep a non-zero gradient for correct predictions.

    2. **Per-class weights** – scales the loss contribution of each true
       class to counteract frequency imbalance (minority classes get a higher
       multiplier).

    3. **Focal loss** (gamma > 0) – down-weights easy-to-predict samples
       (high predicted probability for the true class) so the optimiser
       focuses on hard, misclassified examples.  With severe class imbalance
       the majority class is "easy", so focal loss naturally shifts attention
       to minority classes.
    """

    def __init__(self, num_classes=3, class_weights=None, focal_loss_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.focal_loss_gamma = focal_loss_gamma

        # Create ordinal weight matrix
        self.ordinal_matrix = self._build_ordinal_matrix(num_classes)

        # Class weights (to handle imbalance)
        if class_weights is not None:
            # Convert to tensor: {0: 0.413, 1: 1.925, 2: 0.662} → [0.413, 1.925, 0.662]
            self.class_weights = tf.constant(
                [class_weights.get(i, 1.0) for i in range(num_classes)],
                dtype=tf.float32
            )
        else:
            self.class_weights = tf.ones(num_classes, dtype=tf.float32)

    def _build_ordinal_matrix(self, num_classes):
        """Ordinal penalty matrix: correct=1.0, adjacent error=1.25, far error=2.0.
        Higher values = model is penalized MORE for that prediction.
        This ensures the model prefers adjacent errors over distant errors,
        and prefers correct predictions over any error.
        
        This gives:
            [[1.00, 1.25, 2.00],
            [1.25, 1.00, 1.25],
            [2.00, 1.25, 1.00]]
        """
        matrix = []
        for i in range(num_classes):
            row = []
            for j in range(num_classes):
                if i == j:
                    weight = 1.0  # correct prediction: baseline penalty
                else:
                    distance = ((i - j) ** 2) / ((num_classes - 1) ** 2)
                    weight = 1.0 + distance  # adjacent=1.25, far=2.0
                row.append(weight)
            matrix.append(row)
        return tf.constant(matrix, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Label smoothing: prevents the model from being overconfident on one class
        smoothing = 0.1
        num_cls = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = y_true * (1.0 - smoothing) + (smoothing / num_cls)

        # Cross-entropy on smoothed labels
        ce_loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)

        # Focal modulation: down-weight easy correct predictions
        if self.focal_loss_gamma > 0:
            p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
            focal_factor = tf.pow(1.0 - p_t, self.focal_loss_gamma)
            ce_loss = focal_factor * ce_loss

        # Ordinal + class weights
        y_true_class = tf.argmax(y_true, axis=1)
        y_pred_class = tf.argmax(y_pred, axis=1)
        ordinal_weights = tf.gather_nd(
            self.ordinal_matrix,
            tf.stack([y_true_class, y_pred_class], axis=1)
        )
        class_weights_per_sample = tf.gather(self.class_weights, y_true_class)

        # NO entropy term — it was causing soft single-class collapse
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
            "num_classes": self.num_classes,
            "focal_loss_gamma": float(getattr(self, "focal_loss_gamma", 0.0)),
            "class_weights": class_weights_list,
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


def _extract_dme_labels(batch_labels):
    """Extract DME labels from dataset batch labels (dict/tuple/tensor)."""
    if isinstance(batch_labels, dict):
        return batch_labels.get("dme_risk", batch_labels.get("dme_output"))
    if isinstance(batch_labels, (tuple, list)):
        return batch_labels[1] if len(batch_labels) > 1 else batch_labels[0]
    return batch_labels


def log_dataset_class_distribution(
    dataset: tf.data.Dataset,
    name: str,
    num_classes: int = NUM_DME_CLASSES,
    max_batches: int = 20,
) -> None:
    """Log approximate class distribution from the first N batches of a dataset."""
    if num_classes <= 0:
        logger.debug("%s class distribution skipped: invalid num_classes=%s", name, num_classes)
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
            # Expected format is one-hot encoded labels from dataset loaders.
            # For debug logging, tolerate minor numeric noise but flag non one-hot rows.
            row_sums = np.sum(arr, axis=-1)
            mismatch = ~np.isclose(row_sums, 1.0, atol=1e-3)
            if np.any(mismatch):
                logger.debug(
                    "%s labels appear non one-hot in sampled batch %d: mismatched_rows=%d/%d, row_sum_range=[%.4f, %.4f]",
                    name, batch_idx, int(np.sum(mismatch)), int(row_sums.shape[0]),
                    float(np.min(row_sums)), float(np.max(row_sums)),
                )
            classes = np.argmax(arr, axis=-1)
        else:
            classes = arr.astype(int).reshape(-1)
        classes = np.clip(classes, 0, num_classes - 1)
        counts += np.bincount(classes, minlength=num_classes)
        samples += int(classes.shape[0])

    if samples == 0:
        logger.debug("%s class distribution unavailable (no samples read).", name)
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
        "Computed exact train DME class counts from dataset: %s (samples=%d, batches=%s)",
        counts.tolist(),
        total,
        "unknown" if cardinality in (None, tf.data.experimental.UNKNOWN_CARDINALITY) else cardinality,
    )
    return counts


# ---------------------------------------------------------------------------
# QWK-based model checkpointing callback
# ---------------------------------------------------------------------------

class QWKModelCheckpoint(keras.callbacks.Callback):
    """Save model weights when val_qwk improves.

    Parameters
    ----------
    filepath : str
        Path to save best model weights.
    verbose : int
        Verbosity level.
    """

    def __init__(self, filepath: str, verbose: int = 1, initial_best_qwk: float = -np.inf):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.best_qwk: float = float(initial_best_qwk)

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


# ---------------------------------------------------------------------------
# QWK-based early stopping callback
# ---------------------------------------------------------------------------

class QWKEarlyStopping(keras.callbacks.Callback):
    """Stop training when val_qwk stops improving.

    Parameters
    ----------
    patience : int
        Epochs to wait before stopping.
    min_delta : float
        Minimum QWK improvement to count as improvement.
    restore_best_weights : bool
        If True, restores weights from best epoch on stop.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_qwk: float = -np.inf
        self.wait: int = 0
        self.best_weights = None
        self.stopped_epoch: int = 0

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return

        if current_qwk > self.best_qwk + self.min_delta:
            self.best_qwk = current_qwk
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.verbose:
                    logger.info(
                        "QWKEarlyStopping: no improvement for %d epochs. Stopping at epoch %d.",
                        self.patience, self.stopped_epoch,
                    )
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    logger.info(
                        "Restored best weights (val_qwk=%.4f).", self.best_qwk
                    )


# ---------------------------------------------------------------------------
# QWK-based LR scheduler
# ---------------------------------------------------------------------------

class QWKReduceLROnPlateau(keras.callbacks.Callback):
    """Reduce LR when val_qwk stops improving.

    Parameters
    ----------
    factor : float
        Factor by which LR is reduced.
    patience : int
        Epochs to wait before reducing.
    min_lr : float
        Minimum learning rate.
    verbose : int
        Verbosity level.
    """
    """Reduce LR when val_qwk stops improving."""

    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 4,
        min_lr: float = 1e-7,
        verbose: int = 1,
    ):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_qwk: float = -np.inf
        self.wait: int = 0

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        current_qwk = logs.get("val_qwk", None)
        if current_qwk is None:
            return

        if current_qwk > self.best_qwk:
            self.best_qwk = current_qwk
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # FIX: Use direct assignment instead of keras.backend.set_value()
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
    """Log and optionally plot training diagnostics per epoch.

    Parameters
    ----------
    history_path : str
        Path to save running history JSON.
    plot_dir : str
        Directory to save training curve plots.
    plot_every : int
        Save plot every N epochs (0 = only at end).
    """

    def __init__(
        self,
        history_path: str = "training_history_enhanced.json",
        plot_dir: str = ".",
        plot_every: int = 0,
    ):
        super().__init__()
        self.history_path = history_path
        self.plot_dir = plot_dir
        self.plot_every = plot_every
        self._history: Dict[str, List] = {}

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self._history.setdefault(k, []).append(float(v))

        # Persist JSON each epoch
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
        keys = list(self._history.keys())

        # Group val / train pairs
        train_keys = [k for k in keys if not k.startswith("val_")]
        val_keys = [k for k in keys if k.startswith("val_")]

        fig, axes = plt.subplots(
            1, max(len(train_keys), 1), figsize=(6 * max(len(train_keys), 1), 4)
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
            ax.set_xlabel("Epoch")
            ax.set_ylabel(tk)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.plot_dir, f"training_diagnostics_{suffix}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Diagnostics plot saved to '%s'.", path)


# ---------------------------------------------------------------------------
# Model compilation (enhanced)
# ---------------------------------------------------------------------------
"""Compile model with optionally ordinal-weighted loss.

    Parameters
    ----------
    model : keras.Model
        DME model.
    learning_rate : float
        Initial learning rate.
    num_dme_classes : int
        Number of DME classes.
    ordinal_loss_weighting : bool
        Use ordinal-weighted cross-entropy for the DME head.

    Returns
    -------
    keras.Model
        Compiled model.
    """
def compile_model_enhanced(
    model: keras.Model,
    learning_rate: float = 1e-4,
    num_dme_classes: int = 3,
    class_weights: Optional[Dict[int, float]] = None,
    ordinal_loss_weighting: bool = True,
    focal_loss_gamma: float = 2.0,
) -> keras.Model:
    """Compile with ordinal + class weighting baked into loss."""
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )
    if ordinal_loss_weighting:
        dme_loss = OrdinalWeightedCrossEntropy(
            num_classes=num_dme_classes,
            class_weights=class_weights,
            focal_loss_gamma=focal_loss_gamma,
        )
        logger.info("Using ordinal-weighted cross-entropy loss with class balancing.")
        logger.info("Focal loss gamma=%.1f (0=disabled).", focal_loss_gamma)
        if class_weights:
            logger.info("DME class weights applied in loss: %s", class_weights)
    else:
        dme_loss = "categorical_crossentropy"

    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": "mse",
            "dme_risk": dme_loss,
        },
        loss_weights={
            "dr_output": 0.05,   # DR regression contributes to joint training
            "dme_risk": 1.0,
        },
        metrics={
            "dme_risk": [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        },
    )
    logger.info("Enhanced model compiled (lr=%.2e).", learning_rate)
    return model

# def compile_model_enhanced(
#     model: keras.Model,
#     learning_rate: float = 1e-4,
#     num_dme_classes: int = 4,
#     ordinal_loss_weighting: bool = True,
# ) -> keras.Model:
#     """Compile with class weights in loss."""
    
#     optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

#     if ordinal_loss_weighting:
#         dme_loss = OrdinalWeightedCrossEntropy(num_classes=num_dme_classes)
#         logger.info("Using ordinal-weighted cross-entropy loss.")
#     else:
#         dme_loss = "categorical_crossentropy"

#     model.compile(
#         optimizer=optimizer,
#         loss={
#             "dr_output": "mse",
#             "dme_risk": dme_loss,
#         },
#         loss_weights={
#             "dr_output": 0.0,
#             "dme_risk": 1.0,
#         },
#         metrics={
#             "dme_risk": [
#                 keras.metrics.CategoricalAccuracy(name="accuracy"),
#             ],
#         },
#     )
#     logger.info("Enhanced model compiled (lr=%.2e).", learning_rate)
#     return model

# ---------------------------------------------------------------------------
# Callbacks builder
# ---------------------------------------------------------------------------

class LinearWarmupCallback(keras.callbacks.Callback):
    """Warm up LR linearly over first N epochs to prevent epoch-1 collapse."""
    def __init__(self, target_lr: float, warmup_epochs: int = 5):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            self.model.optimizer.learning_rate = lr
            logger.info("Warmup epoch %d/%d: lr=%.2e", epoch+1, self.warmup_epochs, lr)
        elif epoch == self.warmup_epochs:
            self.model.optimizer.learning_rate = self.target_lr


class Stage2QWKCollapseGuard(keras.callbacks.Callback):
    """Stop stage2 and restore init checkpoint if QWK collapses.

    This protects a good stage1 model from catastrophic stage2 drift.
    """

    def __init__(
        self,
        baseline_qwk: float,
        init_weights_path: str,
        min_ratio: float = 0.70,
        min_abs_qwk: float = 0.20,
        patience: int = 2,
        start_epoch: int = 1,
        verbose: int = 1,
    ):
        super().__init__()
        self.baseline_qwk = float(baseline_qwk)
        self.init_weights_path = init_weights_path
        self.min_ratio = float(min_ratio)
        self.min_abs_qwk = float(min_abs_qwk)
        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.verbose = verbose
        self.bad_epochs = 0

    def _threshold(self) -> float:
        ratio_threshold = self.baseline_qwk * self.min_ratio
        # Only enforce absolute floor when baseline itself is reasonably high.
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

        threshold = self._threshold()
        if float(current_qwk) < threshold:
            self.bad_epochs += 1
            if self.verbose:
                logger.warning(
                    "Stage2CollapseGuard: val_qwk=%.4f below threshold=%.4f "
                    "(%d/%d bad epochs).",
                    float(current_qwk),
                    threshold,
                    self.bad_epochs,
                    self.patience,
                )
        else:
            self.bad_epochs = 0

        if self.bad_epochs >= self.patience:
            if self.init_weights_path and os.path.exists(self.init_weights_path):
                self.model.load_weights(self.init_weights_path)
                logger.warning(
                    "Stage2CollapseGuard triggered: restored stage2 init weights from '%s'.",
                    self.init_weights_path,
                )
            else:
                logger.warning(
                    "Stage2CollapseGuard triggered but init weights path is unavailable: '%s'.",
                    self.init_weights_path,
                )
            self.model.stop_training = True

def build_enhanced_callbacks(
    val_dataset: tf.data.Dataset,
    config: Dict,
) -> list:
    """Build the enhanced callback stack.

    Parameters
    ----------
    val_dataset : tf.data.Dataset
        Validation dataset for QWK computation.
    config : dict
        Training configuration dict.

    Returns
    -------
    list
        List of Keras callbacks.
    """
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_qwk_path = os.path.join(config["checkpoint_dir"], "best_qwk.weights.h5")

    checkpoint_initial_best_qwk = -np.inf
    use_stage1_baseline_for_checkpoint = bool(
        config.get("stage2_checkpoint_use_stage1_baseline", False)
    )
    stage1_baseline_qwk = config.get("stage1_baseline_qwk", None)
    if use_stage1_baseline_for_checkpoint and stage1_baseline_qwk is not None:
        try:
            checkpoint_initial_best_qwk = float(stage1_baseline_qwk)
            logger.info(
                "Stage 2 checkpoint baseline set to Stage 1 val_qwk=%.4f "
                "(save only on improvement).",
                checkpoint_initial_best_qwk,
            )
        except (TypeError, ValueError):
            logger.warning(
                "Invalid stage1_baseline_qwk=%r; using -inf checkpoint baseline.",
                stage1_baseline_qwk,
            )
    elif stage1_baseline_qwk is not None:
        logger.info(
            "Stage 2 checkpoint baseline is -inf (independent from Stage 1)."
        )

    callbacks = [
        LinearWarmupCallback(target_lr=config["learning_rate"], warmup_epochs=5),
        # QWK tracking callback (must run first to populate val_qwk in logs)
        QWKCallback(
            val_dataset=val_dataset,
            num_classes=config["num_dme_classes"],
            dr_num_classes=config.get("num_dr_classes", 5),
            compute_dr_metrics=True,
            history_path=config["qwk_history_path"],
            verbose=1,
            max_batches=config.get("max_batches", None),
        ),
    ]

    # Stage 2 safety guard: abort if QWK collapses far below stage1 baseline.
    if (
        config.get("collapse_guard_enabled", False)
        and config.get("stage1_baseline_qwk") is not None
        and config.get("stage2_init_weights_path")
    ):
        callbacks.append(
            Stage2QWKCollapseGuard(
                baseline_qwk=float(config["stage1_baseline_qwk"]),
                init_weights_path=str(config["stage2_init_weights_path"]),
                min_ratio=float(config.get("collapse_guard_ratio", 0.70)),
                min_abs_qwk=float(config.get("collapse_guard_min_abs_qwk", 0.20)),
                patience=int(config.get("collapse_guard_patience", 2)),
                start_epoch=1,
                verbose=1,
            )
        )

    callbacks.extend(
        [
            # Save best model by QWK
            QWKModelCheckpoint(
                filepath=best_qwk_path,
                verbose=1,
                initial_best_qwk=checkpoint_initial_best_qwk,
            ),
            # Early stopping on QWK
            QWKEarlyStopping(
                patience=config["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            # LR reduction on QWK plateau
            QWKReduceLROnPlateau(
                factor=config["lr_reduce_factor"],
                patience=config["lr_reduce_patience"],
                min_lr=config["min_lr"],
                verbose=1,
            ),
            # CSV logging
            keras.callbacks.CSVLogger(config["log_path"], append=False),
            # Comprehensive diagnostics
            TrainingDiagnosticsCallback(
                history_path=config["history_path"],
                plot_dir=config.get("output_dir", "."),
            ),
        ]
    )

    return callbacks


def _freeze_backbone_batchnorm_layers(
    model: keras.Model,
    backbone_layer_name: str = "resnet50_conv4_backbone",
) -> int:
    """Freeze BatchNorm layers inside backbone only and return the count."""
    count = 0
    try:
        backbone = model.get_layer(backbone_layer_name)
    except Exception as e:
        logger.warning("Could not locate backbone layer '%s': %s", backbone_layer_name, e)
        return 0

    stack = [backbone]
    seen = set()

    while stack:
        current = stack.pop()
        obj_id = id(current)
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

    if use_dme_tuning:
        logger.info("Stage 2: DME head fine-tuning (backbone frozen)")
        model = build_model_dme_tuning(
            input_shape=tuple(cfg["input_shape"]),
            pretrained_weights=pretrained_weights,
            num_dme_classes=cfg["num_dme_classes"],
            dropout_rate=float(cfg.get("dropout_rate", 0.5)),
            backbone_weights_path=backbone_weights_path,
        )
    else:
        if pretrained_weights is not None:
            logger.info("Stage 2: full-model fine-tuning from Stage 1 checkpoint")
            backbone_weights = None
        elif eyepacs_backbone is not None:
            logger.info("Stage 1: Backbone frozen, training ASPP + heads only")
            backbone_weights = None
            logger.info("EyePACS backbone weights will be loaded from '%s'.", eyepacs_backbone)
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
            dropout_rate=float(cfg.get("dropout_rate", 0.5)),
            trainable=True,
            backbone_weights_path=model_backbone_weights_path,
        )

        if eyepacs_backbone is not None and pretrained_weights is None:
            # EyePACS artifact stores backbone weights, so load directly into backbone.
            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.load_weights(eyepacs_backbone, skip_mismatch=True)
                logger.info("Loaded EyePACS backbone weights into backbone from '%s'.", eyepacs_backbone)
            except Exception as e:
                logger.warning("Could not load EyePACS backbone weights '%s': %s", eyepacs_backbone, e)

        if pretrained_weights is None:
            # Stage 1 only: freeze backbone and initialize DME bias
            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.trainable = False
                trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
                logger.info("✅ Backbone FROZEN. Trainable params: %d (~ASPP + heads only)", trainable_count)
            except Exception as e:
                logger.warning("Could not freeze backbone: %s", e)

            try:
                dme_head = model.get_layer("dme_risk")
                class_counts = compute_dataset_class_counts(
                    train_ds,
                    num_classes=int(cfg.get("num_dme_classes", NUM_DME_CLASSES)),
                )
                if class_counts is None:
                    raise ValueError("Training class counts unavailable for DME bias initialization.")
                class_counts = class_counts.astype(np.float32)
                class_probs = class_counts / class_counts.sum()
                log_probs = np.log(class_probs + 1e-7)
                w = dme_head.get_weights()
                if len(w) >= 2:
                    w[1] = log_probs
                    dme_head.set_weights(w)
                    logger.info(
                        "✅ DME head bias initialized from train split counts=%s -> log_probs=%s",
                        class_counts.astype(int).tolist(),
                        np.round(log_probs, 3),
                    )
            except Exception as e:
                logger.warning("Could not initialize DME bias: %s", e)

        else:
            # Stage 2: strictly restore full Stage 1 checkpoint (backbone + heads)
            if not os.path.exists(pretrained_weights):
                raise FileNotFoundError(
                    f"Stage 2 checkpoint not found: '{pretrained_weights}'. "
                    "Cannot start fine-tuning without Stage 1 weights."
                )

            logger.info(
                "Stage 2 restoring full Stage 1 checkpoint (backbone + heads) from '%s'.",
                pretrained_weights,
            )

            dme_head = model.get_layer("dme_risk")
            dme_head_before = [np.copy(w) for w in dme_head.get_weights()]

            # Strict load: do not allow silent partial restores.
            model.load_weights(pretrained_weights)

            dme_head_after = dme_head.get_weights()
            if dme_head_before and dme_head_after and len(dme_head_before) == len(dme_head_after):
                max_head_delta = max(
                    float(np.max(np.abs(before - after)))
                    for before, after in zip(dme_head_before, dme_head_after)
                )
                if max_head_delta < 1e-8:
                    raise RuntimeError(
                        "Stage 2 checkpoint restore verification failed: "
                        "DME head weights did not change after loading Stage 1 checkpoint."
                    )
                logger.info(
                    "Stage 2 checkpoint restore verified (max |delta| in DME head = %.3e).",
                    max_head_delta,
                )
            else:
                logger.warning(
                    "Could not verify DME head restore; layer weights unavailable for comparison."
                )

            try:
                backbone_layer = model.get_layer("resnet50_conv4_backbone")
                backbone_layer.trainable = True
                bn_frozen = _freeze_backbone_batchnorm_layers(model)
                trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
                logger.info(
                    "✅ Stage 2: Backbone UNFROZEN with %d backbone BN layers frozen (ASPP BN remains trainable). Trainable params: %d",
                    bn_frozen,
                    trainable_count,
                )
            except Exception as e:
                logger.warning("Could not unfreeze backbone for Stage 2: %s", e)

    model = compile_model_enhanced(
        model,
        learning_rate=cfg["learning_rate"],
        num_dme_classes=cfg["num_dme_classes"],
        class_weights=class_weights,
        ordinal_loss_weighting=cfg.get("ordinal_loss_weighting", True),
        focal_loss_gamma=cfg.get("focal_loss_gamma", 2.0),
    )

    callbacks = build_enhanced_callbacks(val_ds, cfg)

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, dme_tuning=%s",
        cfg["epochs"], cfg["batch_size"], use_dme_tuning,
    )
    log_dataset_class_distribution(train_ds, "Train", num_classes=cfg["num_dme_classes"], max_batches=20)
    log_dataset_class_distribution(val_ds, "Val", num_classes=cfg["num_dme_classes"], max_batches=20)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Final stage2 guardrail: never keep a model that is worse than stage1 baseline.
    if (
        pretrained_weights is not None
        and cfg.get("stage2_revert_if_worse", True)
        and cfg.get("stage1_baseline_qwk") is not None
    ):
        baseline_qwk = float(cfg["stage1_baseline_qwk"])
        min_improvement = float(cfg.get("stage2_min_improvement", 0.0))
        qwk_history = history.history.get("val_qwk", [])
        best_stage2_qwk = max(qwk_history) if qwk_history else -float("inf")

        if best_stage2_qwk < baseline_qwk + min_improvement:
            logger.warning(
                "Stage 2 best val_qwk=%.4f did not beat Stage 1 baseline=%.4f (required +%.4f). "
                "Restoring stage2 init weights from '%s'.",
                best_stage2_qwk,
                baseline_qwk,
                min_improvement,
                pretrained_weights,
            )
            if os.path.exists(pretrained_weights):
                model.load_weights(pretrained_weights)
                history.history["stage2_reverted_to_stage1"] = [1.0]
            else:
                logger.warning(
                    "Could not restore stage2 init weights because file is missing: '%s'.",
                    pretrained_weights,
                )

    model.save_weights(output_weights)
    logger.info("Saved weights to '%s'.", output_weights)

    return model, history.history

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for enhanced training."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Enhanced QWK-aware DME training")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="dme_enhanced.weights.h5")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-ordinal-loss", action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        logger.info("Using mock dataset for testing.")
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_enhanced", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    train_ds, val_ds, class_weights = build_datasets(
        csv_path=csv_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
    )

    config = {
        **DEFAULT_ENHANCED_CONFIG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "output_dir": args.output_dir,
        "ordinal_loss_weighting": not args.no_ordinal_loss,
    }

    model, history = train_enhanced(
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        pretrained_weights=args.weights,
        config=config,
        output_weights=args.output,
    )
    logger.info("Enhanced training complete.")


if __name__ == "__main__":
    main()
