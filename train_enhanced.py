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
    "max_batches": None,  # None = process entire validation set for accurate QWK
    "seed": 42,
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
    """Ordinal-aware weighted cross-entropy loss."""

    def __init__(self, num_classes=4, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Create ordinal weight matrix
        self.ordinal_matrix = self._build_ordinal_matrix(num_classes)
        
        # Class weights (to handle imbalance)
        if class_weights is not None:
            # Convert to tensor: {0: 0.067, 1: 0.157, ...} → [0.067, 0.157, ...]
            self.class_weights = tf.constant(
                [class_weights.get(i, 1.0) for i in range(num_classes)],
                dtype=tf.float32
            )
        else:
            self.class_weights = tf.ones(num_classes, dtype=tf.float32)

    def _build_ordinal_matrix(self, num_classes):
        """Build ordinal weight matrix (penalize distant misclassifications).

        IMPORTANT: diagonal weights must be non-zero so correct predictions still
        contribute cross-entropy gradients.
        """
        matrix = []
        for i in range(num_classes):
            row = []
            for j in range(num_classes):
                if i==j:
                    distance = 1.0  # No penalty for correct class
                else:
                    distance = ((i - j) ** 2) / ((num_classes - 1) ** 2)
                row.append(distance)
            matrix.append(row)
        return tf.constant(matrix, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """Compute loss with ordinal + class weighting."""
        # y_true: (batch, num_classes) one-hot encoded
        # y_pred: (batch, num_classes) probabilities
        
        # Standard cross-entropy
        ce_loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Get true class indices
        y_true_class = tf.argmax(y_true, axis=1)  # (batch,)
        y_pred_class = tf.argmax(y_pred, axis=1)  # (batch,)
        
        # Apply ordinal weighting
        ordinal_weights = tf.gather_nd(
            self.ordinal_matrix,
            tf.stack([y_true_class, y_pred_class], axis=1)
        )  # (batch,)
        
        # Apply class weights
        class_weights_per_sample = tf.gather(self.class_weights, y_true_class)  # (batch,)
        
        # Combine all weights
        total_weights = ordinal_weights * class_weights_per_sample
        
        # Weighted loss
        weighted_loss = ce_loss * total_weights
        
        return tf.reduce_mean(weighted_loss)


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

    def __init__(self, filepath: str, verbose: int = 1):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.best_qwk: float = -np.inf

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
) -> keras.Model:
    """Compile with ordinal + class weighting baked into loss."""
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    if ordinal_loss_weighting:
        dme_loss = OrdinalWeightedCrossEntropy(
            num_classes=num_dme_classes,
            class_weights=class_weights,
        )
        logger.info("Using ordinal-weighted cross-entropy loss with class balancing.")
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
            "dr_output": 0.3,   # DR regression contributes to joint training
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

    callbacks = [
        # QWK tracking callback (must run first to populate val_qwk in logs)
        QWKCallback(
            val_dataset=val_dataset,
            num_classes=config["num_dme_classes"],
            history_path=config["qwk_history_path"],
            verbose=1,
            max_batches=config.get("max_batches", None),
        ),
        # Save best model by QWK
        QWKModelCheckpoint(filepath=best_qwk_path, verbose=1),
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
    return callbacks


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
    """Run the enhanced DME training loop with QWK monitoring.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Batched training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset.
    class_weights : dict, optional
        Per-class loss weights.
    pretrained_weights : str, optional
        Path to pretrained full-model weights.
    eyepacs_backbone : str, optional
        Path to EyePACS backbone weights (Stage 1 only).
    backbone_weights_path : str, optional
        Path to a custom ``.h5`` backbone weights file.  When provided, these
        weights are loaded into the backbone at construction time (Stage 1
        and Stage 2 fine-tuning).  Ignored when ``eyepacs_backbone`` is set
        (EyePACS weights take priority).
    config : dict, optional
        Training configuration. Defaults to :data:`DEFAULT_ENHANCED_CONFIG`.
    output_weights : str
        Path to save final weights.
    use_dme_tuning : bool
        When True, only the DME head is trainable (Stage 2).

    Returns
    -------
    tuple
        ``(model, history_dict)``
    """

    cfg = {**DEFAULT_ENHANCED_CONFIG, **(config or {})}

    logger.info("Building enhanced model …")

    if use_dme_tuning:
        logger.info("Stage 2: DME head fine-tuning (backbone frozen)")
        model = build_model_dme_tuning(
            input_shape=tuple(cfg["input_shape"]),
            pretrained_weights=pretrained_weights,
            num_dme_classes=cfg["num_dme_classes"],
            backbone_weights_path=backbone_weights_path,
        )
    else:
        logger.info("Stage 1: Full model training (all layers trainable)")
        # Use EyePACS-pretrained backbone weights when provided, otherwise ImageNet
        if eyepacs_backbone is not None:
            backbone_weights = None  # Skip ImageNet; EyePACS weights loaded below
            logger.info("EyePACS backbone weights will be loaded from '%s'.", eyepacs_backbone)
        else:
            backbone_weights = "imagenet"
        model = build_model(
            input_shape=tuple(cfg["input_shape"]),
            backbone_weights=backbone_weights,
            num_dme_classes=cfg["num_dme_classes"],
            trainable=True,
            # EyePACS weights take priority; custom backbone path only used
            # when no EyePACS weights are provided.
            backbone_weights_path=None if eyepacs_backbone is not None else backbone_weights_path,
        )
        if eyepacs_backbone is not None:
            model.load_weights(eyepacs_backbone, skip_mismatch=True)
            logger.info("Loaded EyePACS backbone weights from '%s'.", eyepacs_backbone)
        elif pretrained_weights is not None:
            model.load_weights(pretrained_weights, skip_mismatch=True)

    model = compile_model_enhanced(
        model,
        learning_rate=cfg["learning_rate"],
        num_dme_classes=cfg["num_dme_classes"],
        class_weights=class_weights,
        ordinal_loss_weighting=cfg.get("ordinal_loss_weighting", True),
    )

    callbacks = build_enhanced_callbacks(val_ds, cfg)

    logger.info(
        "Starting training: epochs=%d, batch_size=%d, dme_tuning=%s",
        cfg["epochs"], cfg["batch_size"], use_dme_tuning,
    )
    log_dataset_class_distribution(
        train_ds, "Train", num_classes=cfg["num_dme_classes"], max_batches=20
    )
    log_dataset_class_distribution(
        val_ds, "Val", num_classes=cfg["num_dme_classes"], max_batches=20
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
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
