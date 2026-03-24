"""
DME Head Fine-Tuning Training Script

Usage
-----
    python train.py [--dataset_path DATASET_PATH]
                    [--weights_path BACKBONE_WEIGHTS]
                    [--epochs 30]
                    [--batch_size 8]
                    [--output_dir outputs]

If *dataset_path* is omitted, a synthetic dataset is generated automatically
so that the script can be validated without the real IRDID data.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import IDRiDDataset
from model import build_model_dme_tuning

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_file: str = "training_log.txt") -> None:
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DME head on IRDID dataset")
    p.add_argument("--dataset_path", default="dataset",
                   help="Root directory of the IRDID dataset (default: dataset/)")
    p.add_argument("--weights_path", default=None,
                   help="Path to pretrain_final.weights.h5 (optional)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    dataset_path: str = "dataset",
    weights_path: str = None,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    output_dir: str = "outputs",
) -> tf.keras.callbacks.History:
    """Run the DME fine-tuning training loop.

    Parameters
    ----------
    dataset_path:
        Root directory of the IRDID dataset.
    weights_path:
        Path to pre-trained backbone weights (``pretrain_final.weights.h5``).
        Pass *None* to train from scratch (ImageNet init).
    epochs:
        Maximum number of training epochs.
    batch_size:
        Mini-batch size.
    learning_rate:
        Adam learning rate.
    output_dir:
        Directory where all outputs are saved.

    Returns
    -------
    tf.keras.callbacks.History
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading dataset from: %s", dataset_path)
    train_ds_obj = IDRiDDataset(
        dataset_path, split="train", augment=True, batch_size=batch_size
    )
    val_ds_obj = IDRiDDataset(
        dataset_path, split="val", augment=False, batch_size=batch_size
    )

    train_ds = train_ds_obj.get_dataset()
    val_ds = val_ds_obj.get_dataset()

    class_weights = train_ds_obj.get_class_weights()
    logger.info("Class weights: %s", class_weights)
    logger.info("Train samples: %d  Val samples: %d",
                len(train_ds_obj), len(val_ds_obj))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Building model (DME fine-tuning variant)…")
    model = build_model_dme_tuning(backbone_weights_path=weights_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "dr_output": "mse",
            "dme_risk": "categorical_crossentropy",
        },
        loss_weights={
            "dr_output": 0.0,   # DR head is frozen/dummy
            "dme_risk": 1.0,
        },
        metrics={
            "dr_output": [],
            "dme_risk": ["accuracy"],
        },
    )

    model.summary(print_fn=logger.info)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    best_weights_path = str(out / "dme_finetuned.weights.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dme_risk_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_dme_risk_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(out / "tensorboard_logs"),
            histogram_freq=0,
        ),
    ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    logger.info("Starting training for up to %d epochs…", epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    hist_path = out / "training_history.json"
    # Convert numpy floats to Python floats for JSON serialisation
    hist_dict = {k: [float(v) for v in vals]
                 for k, v in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    logger.info("Training history saved → %s", hist_path)
    logger.info("Best model weights  → %s", best_weights_path)

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(os.path.join(args.output_dir, "training_log.txt"))
    train(
        dataset_path=args.dataset_path,
        weights_path=args.weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
