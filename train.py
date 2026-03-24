"""
train.py - Training orchestration for the DME fine-tuning pipeline.

Handles:
- Model compilation with appropriate losses and metrics
- Training loop with callbacks (early stopping, LR schedule, checkpointing)
- Learning rate scheduling
- Training history logging to JSON
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import build_model_dme_tuning

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    "input_shape": (512, 512, 3),
    "num_dme_classes": 4,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 30,
    "early_stopping_patience": 5,
    "lr_reduce_patience": 3,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-7,
    "checkpoint_dir": "checkpoints",
    "history_path": "training_history.json",
    "log_path": "training_log.txt",
}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(
    checkpoint_dir: str = "checkpoints",
    history_path: str = "training_history.json",
    log_path: str = "training_log.txt",
    early_stopping_patience: int = 5,
    lr_reduce_patience: int = 3,
    lr_reduce_factor: float = 0.5,
    min_lr: float = 1e-7,
    monitor: str = "val_dme_risk_accuracy",
) -> list:
    """Build the list of Keras callbacks for DME training.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save best model checkpoints.
    history_path : str
        Path to write training history JSON.
    log_path : str
        Path to write per-epoch CSV log.
    early_stopping_patience : int
        Number of epochs with no improvement before stopping.
    lr_reduce_patience : int
        Number of epochs with no improvement before reducing LR.
    lr_reduce_factor : float
        Factor by which LR is reduced.
    min_lr : float
        Minimum learning rate.
    monitor : str
        Metric to monitor for callbacks.

    Returns
    -------
    list
        List of ``tf.keras.callbacks.Callback`` objects.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "dme_best.weights.h5")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(log_path, append=False),
        _HistoryJsonCallback(history_path),
    ]
    return callbacks


class _HistoryJsonCallback(keras.callbacks.Callback):
    """Persist the full training history to a JSON file at end of training."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def on_train_end(self, logs=None):
        history = self.model.history.history
        # Convert numpy types to Python native for JSON serialisation
        serialisable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        with open(self.path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Training history saved to '%s'.", self.path)


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

def compile_dme_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    num_dme_classes: int = 4,
) -> keras.Model:
    """Compile the DME fine-tuning model.

    Uses ``categorical_crossentropy`` for the DME head and ``mse`` for the
    frozen DR head.

    Parameters
    ----------
    model : keras.Model
        Model returned by :func:`~model.build_model_dme_tuning`.
    learning_rate : float
        Adam optimiser learning rate.
    num_dme_classes : int
        Number of DME output classes (determines loss weights).

    Returns
    -------
    keras.Model
        The same model, compiled in-place.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": "mse",
            "dme_risk": "categorical_crossentropy",
        },
        loss_weights={
            "dr_output": 0.0,   # DR head is frozen – zero its gradient contribution
            "dme_risk": 1.0,
        },
        metrics={
            "dme_risk": [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.AUC(
                    name="auc",
                    multi_label=False,
                    num_thresholds=200,
                ),
            ],
        },
    )
    logger.info("Model compiled (lr=%.2e).", learning_rate)
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    pretrained_weights: Optional[str] = None,
    config: Optional[Dict] = None,
    output_weights: str = "dme_finetuned.weights.h5",
) -> Tuple[keras.Model, Dict]:
    """Run the complete DME fine-tuning training loop.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Batched training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset.
    class_weights : dict, optional
        Per-class loss weights for handling imbalance.
    pretrained_weights : str, optional
        Path to pre-trained backbone weights (``pretrain_final.weights.h5``).
    config : dict, optional
        Training hyper-parameters.  Defaults to :data:`DEFAULT_CONFIG`.
    output_weights : str
        Path to save the final tuned DME head weights.

    Returns
    -------
    tuple
        ``(model, history_dict)``
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info("Building DME fine-tuning model …")
    model = build_model_dme_tuning(
        input_shape=tuple(cfg["input_shape"]),
        pretrained_weights=pretrained_weights,
        num_dme_classes=cfg["num_dme_classes"],
    )

    model = compile_dme_model(
        model,
        learning_rate=cfg["learning_rate"],
        num_dme_classes=cfg["num_dme_classes"],
    )

    callbacks = build_callbacks(
        checkpoint_dir=cfg["checkpoint_dir"],
        history_path=cfg["history_path"],
        log_path=cfg["log_path"],
        early_stopping_patience=cfg["early_stopping_patience"],
        lr_reduce_patience=cfg["lr_reduce_patience"],
        lr_reduce_factor=cfg["lr_reduce_factor"],
        min_lr=cfg["min_lr"],
    )

    logger.info(
        "Starting training: epochs=%d, batch_size=%d", cfg["epochs"], cfg["batch_size"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final weights
    model.save_weights(output_weights)
    logger.info("Saved fine-tuned weights to '%s'.", output_weights)

    return model, history.history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for training.

    Expects real dataset when ``--csv`` and ``--image-dir`` are provided,
    otherwise falls back to mock data for a quick smoke-test.
    """
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Train DME fine-tuning model")
    parser.add_argument("--csv", type=str, default=None, help="Path to DME_Grades.csv")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to image directory")
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to pretrained backbone weights"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="dme_finetuned.weights.h5")
    parser.add_argument(
        "--mock", action="store_true", help="Use mock dataset (for testing)"
    )
    args = parser.parse_args()

    if args.mock or (args.csv is None or args.image_dir is None):
        logger.info("No real dataset provided – using mock data for smoke-test.")
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid", num_samples=40)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    train_ds, val_ds, class_weights = build_datasets(
        csv_path=csv_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
    )

    config = {
        **DEFAULT_CONFIG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }

    model, history = train(
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        pretrained_weights=args.weights,
        config=config,
        output_weights=args.output,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
