"""
train.py - Training orchestration for the DME fine-tuning pipeline.

KEY CHANGES FROM ORIGINAL:
- evaluate_comprehensive is called ONCE at the END of each stage on the
  best saved joint model, never inside the epoch loop.
- Best joint model is saved as a full model (not just weights) so it can
  be loaded cleanly for evaluation.
- Stage 1 and Stage 2 each produce their own results directory.
- DR threshold calibration is enabled by default during evaluation.

Handles:
- Model compilation with appropriate losses and metrics
- Training loop with callbacks (early stopping, LR schedule, checkpointing)
- Learning rate scheduling
- Training history logging to JSON
- Post-stage evaluation on best joint checkpoint
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import build_model_dme_tuning, build_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    "input_shape": (512, 512, 3),
    "num_dme_classes": 3,
    "num_dr_classes": 5,
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
    # Evaluation directories (one per stage)
    "eval_dir_stage1": "results_stage1",
    "eval_dir_stage2": "results_stage2",
    # Whether to run DR/DME threshold calibration during evaluation
    "calibrate_dr_thresholds":  True,
    "calibrate_dme_thresholds": False,
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
    stage_label: str = "stage",
) -> Tuple[list, str]:
    """Build the list of Keras callbacks.

    Returns
    -------
    tuple
        (callbacks_list, best_model_path)
        best_model_path is the full-model SavedModel/H5 path used by
        evaluate_on_best_joint_model() after training ends.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the FULL model (not just weights) so we can load it cleanly
    # for evaluation without rebuilding the architecture.
    best_model_path = os.path.join(checkpoint_dir, f"best_joint_{stage_label}.keras")

    callbacks = [
        # Full-model checkpoint — used by post-stage evaluation
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=False,   # ← full model, not just weights
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
    return callbacks, best_model_path


class _HistoryJsonCallback(keras.callbacks.Callback):
    """Persist training history to JSON at end of training."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def on_train_end(self, logs=None):
        history = self.model.history.history
        serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(self.path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Training history saved to '%s'.", self.path)


# ---------------------------------------------------------------------------
# Model compilation
# ---------------------------------------------------------------------------

def compile_dme_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
    num_dme_classes: int = 3,
) -> keras.Model:
    """Compile the DME fine-tuning model."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": "mse",
            "dme_risk":  "categorical_crossentropy",
        },
        loss_weights={
            "dr_output": 0.0,
            "dme_risk":  1.0,
        },
        metrics={
            "dme_risk": [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc", multi_label=False, num_thresholds=200),
            ],
        },
    )
    logger.info("Model compiled (lr=%.2e).", learning_rate)
    return model


def compile_joint_model(
    model: keras.Model,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Compile the full joint DR+DME model for Stage 2 / joint training."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": "sparse_categorical_crossentropy",
            "dme_risk":  "categorical_crossentropy",
        },
        loss_weights={
            "dr_output": 1.0,
            "dme_risk":  1.0,
        },
        metrics={
            "dr_output": [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
            "dme_risk": [
                keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        },
    )
    logger.info("Joint model compiled (lr=%.2e).", learning_rate)
    return model


# ---------------------------------------------------------------------------
# Helper: run post-stage evaluation on best saved checkpoint
# ---------------------------------------------------------------------------

def _run_post_stage_evaluation(
    best_model_path: str,
    val_ds: tf.data.Dataset,
    eval_dir: str,
    stage_label: str,
    calibrate_dr: bool = True,
    calibrate_dme: bool = False,
    num_dme_classes: int = 3,
    num_dr_classes: int = 5,
) -> Optional[Dict]:
    """Load best joint checkpoint and evaluate once.  Called after each stage."""
    if not os.path.exists(best_model_path):
        logger.warning(
            "Best model checkpoint not found at '%s' — skipping evaluation.",
            best_model_path,
        )
        return None

    try:
        from evaluate_comprehensive import evaluate_on_best_joint_model
    except ImportError:
        logger.error("evaluate_comprehensive.py not found — skipping evaluation.")
        return None

    logger.info("=== Post-stage evaluation: %s ===", stage_label)
    metrics = evaluate_on_best_joint_model(
        checkpoint_path=best_model_path,
        dataset=val_ds,
        output_dir=eval_dir,
        stage_label=stage_label,
        num_dme_classes=num_dme_classes,
        num_dr_classes=num_dr_classes,
        calibrate_dr_thresholds=calibrate_dr,
        calibrate_dme_thresholds=calibrate_dme,
    )
    logger.info(
        "%s evaluation complete. DME QWK=%.4f  DR QWK(calib)=%.4f",
        stage_label,
        metrics.get("qwk", float("nan")),
        metrics.get("dr_qwk_calib", float("nan")),
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage 1: DME-only fine-tuning (backbone + DR head frozen)
# ---------------------------------------------------------------------------

def train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    pretrained_weights: Optional[str] = None,
    config: Optional[Dict] = None,
    output_weights: str = "dme_finetuned.weights.h5",
) -> Tuple[keras.Model, Dict]:
    """Stage 1: DME fine-tuning (only DME head trainable).

    After training ends, loads the best joint checkpoint and calls
    evaluate_comprehensive ONCE to produce all plots and metrics.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Batched training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset.
    class_weights : dict, optional
        Per-class loss weights.
    pretrained_weights : str, optional
        Path to pre-trained backbone weights.
    config : dict, optional
        Training hyper-parameters.
    output_weights : str
        Path to save the final tuned DME head weights.

    Returns
    -------
    tuple
        (model, history_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info("=== Stage 1: DME fine-tuning ===")
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

    callbacks, best_model_path = build_callbacks(
        checkpoint_dir=cfg["checkpoint_dir"],
        history_path=cfg["history_path"],
        log_path=cfg["log_path"],
        early_stopping_patience=cfg["early_stopping_patience"],
        lr_reduce_patience=cfg["lr_reduce_patience"],
        lr_reduce_factor=cfg["lr_reduce_factor"],
        min_lr=cfg["min_lr"],
        stage_label="stage1",
    )

    logger.info("Starting Stage 1 training: epochs=%d", cfg["epochs"])
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
    logger.info("Saved Stage 1 weights to '%s'.", output_weights)

    # ── Evaluate ONCE on best saved joint model ──────────────────────────
    _run_post_stage_evaluation(
        best_model_path=best_model_path,
        val_ds=val_ds,
        eval_dir=cfg.get("eval_dir_stage1", "results_stage1"),
        stage_label="Stage 1",
        calibrate_dr=cfg.get("calibrate_dr_thresholds", True),
        calibrate_dme=cfg.get("calibrate_dme_thresholds", False),
        num_dme_classes=cfg["num_dme_classes"],
        num_dr_classes=cfg["num_dr_classes"],
    )

    return model, history.history


# ---------------------------------------------------------------------------
# Stage 2: Joint fine-tuning (both heads trainable)
# ---------------------------------------------------------------------------

def train_joint(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict[int, float]] = None,
    pretrained_weights: Optional[str] = None,
    config: Optional[Dict] = None,
    output_weights: str = "joint_finetuned.weights.h5",
) -> Tuple[keras.Model, Dict]:
    """Stage 2: Joint DR+DME fine-tuning (all weights trainable).

    After training ends, loads the best joint checkpoint and calls
    evaluate_comprehensive ONCE to produce all plots and metrics.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Batched training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset.
    class_weights : dict, optional
        Per-class loss weights.
    pretrained_weights : str, optional
        Path to weights from Stage 1 output.
    config : dict, optional
        Training hyper-parameters.
    output_weights : str
        Path to save the final joint weights.

    Returns
    -------
    tuple
        (model, history_dict)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    logger.info("=== Stage 2: Joint fine-tuning ===")
    model = build_model(
        input_shape=tuple(cfg["input_shape"]),
        num_dme_classes=cfg["num_dme_classes"],
        num_dr_classes=cfg["num_dr_classes"],
        trainable=True,
    )

    if pretrained_weights is not None and os.path.exists(pretrained_weights):
        model.load_weights(pretrained_weights, skip_mismatch=True)
        logger.info("Loaded Stage 1 weights from '%s'.", pretrained_weights)

    model = compile_joint_model(model, learning_rate=cfg.get("learning_rate_stage2",
                                                              cfg["learning_rate"] * 0.1))

    stage2_log  = cfg["log_path"].replace(".txt", "_stage2.txt")
    stage2_hist = cfg["history_path"].replace(".json", "_stage2.json")

    callbacks, best_model_path = build_callbacks(
        checkpoint_dir=cfg["checkpoint_dir"],
        history_path=stage2_hist,
        log_path=stage2_log,
        early_stopping_patience=cfg["early_stopping_patience"],
        lr_reduce_patience=cfg["lr_reduce_patience"],
        lr_reduce_factor=cfg["lr_reduce_factor"],
        min_lr=cfg["min_lr"],
        monitor="val_dme_risk_accuracy",
        stage_label="stage2",
    )

    logger.info("Starting Stage 2 training: epochs=%d", cfg["epochs"])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    model.save_weights(output_weights)
    logger.info("Saved Stage 2 weights to '%s'.", output_weights)

    # ── Evaluate ONCE on best saved joint model ──────────────────────────
    _run_post_stage_evaluation(
        best_model_path=best_model_path,
        val_ds=val_ds,
        eval_dir=cfg.get("eval_dir_stage2", "results_stage2"),
        stage_label="Stage 2",
        calibrate_dr=cfg.get("calibrate_dr_thresholds", True),
        calibrate_dme=cfg.get("calibrate_dme_thresholds", False),
        num_dme_classes=cfg["num_dme_classes"],
        num_dr_classes=cfg["num_dr_classes"],
    )

    return model, history.history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for training."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Train DME / joint DR+DME model")
    parser.add_argument("--csv",        type=str,   default=None)
    parser.add_argument("--image-dir",  type=str,   default=None)
    parser.add_argument("--weights",    type=str,   default=None,
                        help="Path to pretrained backbone weights")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--output",     type=str,   default="dme_finetuned.weights.h5")
    parser.add_argument("--stage",      type=str,   default="1",
                        choices=["1", "2"],
                        help="Training stage: 1=DME-only, 2=joint")
    parser.add_argument("--stage1-weights", type=str, default=None,
                        help="(Stage 2 only) Path to Stage 1 output weights")
    parser.add_argument("--mock",       action="store_true")
    args = parser.parse_args()

    if args.mock or (args.csv is None or args.image_dir is None):
        logger.info("No real dataset provided – using mock data.")
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
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
    }

    if args.stage == "1":
        model, history = train(
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            pretrained_weights=args.weights,
            config=config,
            output_weights=args.output,
        )
    else:
        model, history = train_joint(
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            pretrained_weights=args.stage1_weights or args.weights,
            config=config,
            output_weights=args.output,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()