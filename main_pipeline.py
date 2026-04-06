"""
main_pipeline.py - End-to-end workflow orchestration for DR+DME detection.

Implements:
- Multi-stage training (pretraining → fine-tuning → evaluation)
- Automatic model selection and checkpointing
- Results aggregation and reporting
- Reproducibility with seed management
- Configurable via config.yaml or CLI arguments
"""

import json
import logging
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Optional YAML support
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.warning("PyYAML not available. YAML config loading disabled.")


# ---------------------------------------------------------------------------
# Reproducibility seed management
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Set global random seeds for reproducibility.

    Covers Python random, NumPy, and TensorFlow.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info("Global seeds set to %d (Python, NumPy, TensorFlow).", seed)
    except ImportError:
        logger.info("Global seeds set to %d (Python, NumPy).", seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_CONFIG = {
    "seed": 42,
    "input_shape": [512, 512, 3],
    "num_dme_classes": 3,
    "num_dr_classes": 5,
    "batch_size": 8,
    "val_split": 0.2,
    "augment_train": True,
    # Stage 1: Initial training
    "stage1": {
        "epochs": 30,
        "learning_rate": 1e-4,
        "early_stopping_patience": 5,
        "lr_reduce_patience": 3,
        "lr_reduce_factor": 0.5,
        "min_lr": 1e-7,
        "ordinal_loss_weighting": True,
    },
    # Stage 2: Fine-tuning (after stage 1)
    "stage2": {
        "epochs": 20,
        "learning_rate": 5e-5,
        "early_stopping_patience": 5,
        "lr_reduce_patience": 3,
        "lr_reduce_factor": 0.3,
        "min_lr": 1e-8,
        "ordinal_loss_weighting": True,
        # Abort stage2 if QWK collapses too far below stage1 baseline.
        "collapse_guard_enabled": True,
        "collapse_guard_ratio": 0.70,
        "collapse_guard_min_abs_qwk": 0.20,
        "collapse_guard_patience": 2,
        # If True, stage2 checkpointing saves only when val_qwk beats stage1 baseline.
        # Default False keeps stage2 checkpoint tracking independent from stage1.
        "stage2_checkpoint_use_stage1_baseline": False,
    },
    "output_dir": "pipeline_outputs",
    "checkpoint_dir": "pipeline_outputs/checkpoints",
    "use_advanced_loader": True,
    "medical_importance": True,
    "ordinal_penalty": True,
}


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load pipeline configuration from a YAML file, merged with defaults.

    Parameters
    ----------
    config_path : str, optional
        Path to ``config.yaml``. If ``None`` or file missing, returns defaults.

    Returns
    -------
    dict
        Merged configuration dict.
    """
    config = dict(DEFAULT_PIPELINE_CONFIG)

    if config_path and os.path.exists(config_path) and _YAML_AVAILABLE:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        # Deep merge top-level keys
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k] = {**config[k], **v}
            else:
                config[k] = v
        logger.info("Loaded config from '%s'.", config_path)
    elif config_path:
        logger.warning("Config file '%s' not found or YAML unavailable; using defaults.", config_path)

    # Normalize max_batches to int or None (users often set malformed YAML values).
    max_batches = config.get("max_batches", None)
    if isinstance(max_batches, dict):
        logger.warning("Invalid 'max_batches' format in config; using None (full validation set).")
        config["max_batches"] = None
    elif isinstance(max_batches, str):
        token = max_batches.strip().lower()
        if token in {"", "none", "null"}:
            config["max_batches"] = None
        else:
            try:
                parsed = int(token)
                config["max_batches"] = parsed if parsed > 0 else None
            except ValueError:
                logger.warning("Could not parse 'max_batches=%s'; using None.", max_batches)
                config["max_batches"] = None
    elif isinstance(max_batches, int):
        if max_batches <= 0:
            config["max_batches"] = None
    elif max_batches is not None:
        logger.warning("Unsupported type for 'max_batches'; using None.")
        config["max_batches"] = None

    return config


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_data_preparation(
    csv_path: str,
    image_dir: str,
    config: Dict,
) -> tuple:
    """Stage 0: Load and prepare datasets.

    Parameters
    ----------
    csv_path : str
        Path to DME_Grades.csv.
    image_dir : str
        Directory with fundus images.
    config : dict
        Pipeline configuration.

    Returns
    -------
    tuple
        ``(train_ds, val_ds, class_weights, split_info)``
    """
    import tensorflow as tf

    os.makedirs(config["output_dir"], exist_ok=True)
    logger.info("Stage 0: Data preparation …")

    input_shape = tuple(config["input_shape"])
    target_size = input_shape[:2]

    if config.get("use_advanced_loader", True):
        from dataset_loader_advanced import build_datasets_advanced
        train_ds, val_ds, class_weights, split_info = build_datasets_advanced(
            csv_path=csv_path,
            image_dir=image_dir,
            target_size=target_size,
            batch_size=config["batch_size"],
            val_split=config["val_split"],
            augment_train=config["augment_train"],
            seed=config["seed"],
            medical_importance=config.get("medical_importance", True),
            ordinal_penalty=config.get("ordinal_penalty", True),
            output_dir=config["output_dir"],
        )
    else:
        from dataset_loader import build_datasets, save_split_info
        train_ds, val_ds, class_weights = build_datasets(
            csv_path=csv_path,
            image_dir=image_dir,
            target_size=target_size,
            batch_size=config["batch_size"],
            val_split=config["val_split"],
            augment_train=config["augment_train"],
            seed=config["seed"],
        )

        print(f"Class weights: {class_weights}")
        
        split_info = {
            "seed": config["seed"],
            "val_split": config["val_split"],
            "medical_importance": False,
            "ordinal_penalty": False,
            "note": "Standard loader used; for QWK-aware splits use use_advanced_loader=true",
        }

    return train_ds, val_ds, class_weights, split_info


def stage_training(
    train_ds,
    val_ds,
    class_weights: Dict,
    config: Dict,
    stage_name: str = "stage1",
    pretrained_weights: Optional[str] = None,
    stage1_baseline_qwk: Optional[float] = None,
    eyepacs_backbone: Optional[str] = None,
    backbone_weights_path: Optional[str] = None,
) -> tuple:
    """Stage 1 / Stage 2: Training.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    class_weights : dict
        Class weights.
    config : dict
        Pipeline config.
    stage_name : str
        Stage key in config (``"stage1"`` or ``"stage2"``).
    pretrained_weights : str, optional
        Pre-trained weights path.
    stage1_baseline_qwk : float, optional
        Stage 1 best QWK used by stage2 collapse guard.
    eyepacs_backbone : str, optional
        Path to EyePACS backbone weights saved by preprocessing.ipynb.
    backbone_weights_path : str, optional
        Path to a custom ``.h5`` backbone weights file.  When provided,
        the backbone is initialised with these weights instead of ImageNet
        (Stage 1 custom transfer learning).

    Returns
    -------
    tuple
        ``(model, history)``
    """
    from train_enhanced import train_enhanced, DEFAULT_ENHANCED_CONFIG

    stage_cfg = config.get(stage_name, {})
    checkpoint_dir = os.path.join(config.get("checkpoint_dir", "checkpoints"), stage_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_config = {
        **DEFAULT_ENHANCED_CONFIG,
        "input_shape": config["input_shape"],
        "num_dme_classes": config["num_dme_classes"],
        "batch_size": config["batch_size"],
        "epochs": stage_cfg.get("epochs", 30),
        "learning_rate": stage_cfg.get("learning_rate", 1e-4),
        "early_stopping_patience": stage_cfg.get("early_stopping_patience", 5),
        "lr_reduce_patience": stage_cfg.get("lr_reduce_patience", 3),
        "lr_reduce_factor": stage_cfg.get("lr_reduce_factor", 0.5),
        "min_lr": stage_cfg.get("min_lr", 1e-7),
        "ordinal_loss_weighting": stage_cfg.get("ordinal_loss_weighting", True),
        "focal_loss_gamma": stage_cfg.get("focal_loss_gamma", 2.0),
        "max_batches": config.get("max_batches", None),
        "checkpoint_dir": checkpoint_dir,
        "history_path": os.path.join(config["output_dir"], f"history_{stage_name}.json"),
        "log_path": os.path.join(config["output_dir"], f"log_{stage_name}.csv"),
        "qwk_history_path": os.path.join(config["output_dir"], f"qwk_{stage_name}.json"),
        "output_dir": config["output_dir"],
        "seed": config["seed"],
    }

    if stage_name == "stage2":
        train_config.update(
            {
                "stage1_baseline_qwk": stage1_baseline_qwk,
                "stage2_init_weights_path": pretrained_weights,
                "stage2_checkpoint_use_stage1_baseline": stage_cfg.get(
                    "stage2_checkpoint_use_stage1_baseline", False
                ),
                "collapse_guard_enabled": stage_cfg.get("collapse_guard_enabled", True),
                "collapse_guard_ratio": stage_cfg.get("collapse_guard_ratio", 0.70),
                "collapse_guard_min_abs_qwk": stage_cfg.get("collapse_guard_min_abs_qwk", 0.20),
                "collapse_guard_patience": stage_cfg.get("collapse_guard_patience", 2),
                "stage2_revert_if_worse": stage_cfg.get("stage2_revert_if_worse", True),
                "stage2_min_improvement": stage_cfg.get("stage2_min_improvement", 0.0),
            }
        )

    output_weights = os.path.join(config["output_dir"], f"model_{stage_name}.weights.h5")
    logger.info("Stage: %s (epochs=%d, lr=%.2e)", stage_name,
                train_config["epochs"], train_config["learning_rate"])

    model, history = train_enhanced(
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        pretrained_weights=pretrained_weights,
        eyepacs_backbone=eyepacs_backbone,
        backbone_weights_path=backbone_weights_path,
        config=train_config,
        output_weights=output_weights,
    )
    return model, history, output_weights


def stage_evaluation(
    model,
    val_ds,
    config: Dict,
    stage_name: str = "stage1",
) -> Dict:
    """Evaluation stage: comprehensive QWK metrics.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    val_ds : tf.data.Dataset
        Validation dataset.
    config : dict
        Pipeline config.
    stage_name : str
        Stage name for output file naming.

    Returns
    -------
    dict
        Comprehensive metrics.
    """
    from evaluate_comprehensive import evaluate_comprehensive

    eval_dir = os.path.join(config["output_dir"], f"eval_{stage_name}")
    metrics = evaluate_comprehensive(
        model=model,
        dataset=val_ds,
        output_dir=eval_dir,
        metrics_path="comprehensive_metrics.json",
        num_dme_classes=config["num_dme_classes"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_stage_metrics: Dict, output_dir: str) -> Dict:
    """Aggregate results from all pipeline stages.

    Parameters
    ----------
    all_stage_metrics : dict
        Metrics keyed by stage name.
    output_dir : str
        Output directory.

    Returns
    -------
    dict
        Aggregated report.
    """
    report = {
        "stages": {},
        "best_qwk": -float("inf"),
        "best_stage": None,
        "target_met": False,
    }

    for stage, metrics in all_stage_metrics.items():
        qwk = metrics.get("qwk", 0.0)
        report["stages"][stage] = {
            "qwk": qwk,
            "mae": metrics.get("mae", float("nan")),
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_macro": metrics.get("f1_macro", 0.0),
            "target_met": qwk >= 0.80,
        }
        if qwk > report["best_qwk"]:
            report["best_qwk"] = qwk
            report["best_stage"] = stage
            report["target_met"] = qwk >= 0.80

    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Pipeline report saved to '%s'.", report_path)

    logger.info(
        "\nPipeline Summary:\n  Best QWK: %.4f (%s)\n  Target ≥0.80 met: %s",
        report["best_qwk"],
        report["best_stage"],
        "✅ YES" if report["target_met"] else "❌ NO",
    )
    return report


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    csv_path: str,
    image_dir: str,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    two_stage: bool = True,
    eyepacs_backbone: Optional[str] = None,
    backbone_weights_path: Optional[str] = None,
) -> Dict:
    """Run the full multi-stage training and evaluation pipeline.

    Parameters
    ----------
    csv_path : str
        Path to DME_Grades.csv.
    image_dir : str
        Directory with fundus images.
    config : dict, optional
        Pipeline configuration dict. Takes priority over config_path.
    config_path : str, optional
        Path to config.yaml.
    two_stage : bool
        If True, run both stage1 (initial) and stage2 (fine-tuning).
    eyepacs_backbone : str, optional
        Path to EyePACS backbone weights file saved by preprocessing.ipynb.
        When provided, stage 1 uses EyePACS-pretrained weights instead of
        ImageNet, enabling transfer learning benchmarks.
    backbone_weights_path : str, optional
        Path to a custom ``.h5`` backbone weights file for Stage 1.  Enables
        transfer learning from any pre-trained backbone (e.g. EyePACS,
        proprietary).  Ignored when ``eyepacs_backbone`` is set.

    Returns
    -------
    dict
        Final pipeline report.
    """
    cfg = config or load_config(config_path)
    set_global_seed(cfg.get("seed", 42))

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Save effective config
    config_out = os.path.join(cfg["output_dir"], "effective_config.json")
    with open(config_out, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Effective config saved to '%s'.", config_out)

    t_total = time.time()

    # Data preparation
    train_ds, val_ds, class_weights, split_info = stage_data_preparation(
        csv_path, image_dir, cfg
    )

    all_metrics = {}

    # Stage 1: Initial training
    logger.info("\n" + "=" * 60 + "\nSTAGE 1: Initial Training\n" + "=" * 60)
    if eyepacs_backbone is not None:
        logger.info("Using EyePACS backbone weights: '%s'", eyepacs_backbone)
    elif backbone_weights_path is not None:
        logger.info("Using custom backbone weights: '%s'", backbone_weights_path)
    model, history1, weights1 = stage_training(
        train_ds, val_ds, class_weights, cfg, stage_name="stage1",
        eyepacs_backbone=eyepacs_backbone,
        backbone_weights_path=backbone_weights_path,
    )
    metrics1 = stage_evaluation(model, val_ds, cfg, stage_name="stage1")
    all_metrics["stage1"] = metrics1
    logger.info("Stage 1 QWK: %.4f", metrics1["qwk"])

    best_stage1_weights = os.path.join(cfg["checkpoint_dir"], "stage1", "best_qwk.weights.h5")
    if os.path.exists(best_stage1_weights):
        stage1_qwk = float(metrics1.get("qwk", float("nan")))
        if np.isfinite(stage1_qwk):
            stage1_snapshot_name = f"stage1_best_{stage1_qwk:.4f}.weights.h5"
            stage1_snapshot_path = os.path.join(cfg["checkpoint_dir"], "stage1", stage1_snapshot_name)
            if os.path.abspath(stage1_snapshot_path) != os.path.abspath(best_stage1_weights):
                shutil.copy2(best_stage1_weights, stage1_snapshot_path)
                logger.info(
                    "Archived Stage 1 best checkpoint as '%s'.",
                    stage1_snapshot_path,
                )
            stage2_init_weights = stage1_snapshot_path
        else:
            stage2_init_weights = best_stage1_weights
        logger.info(
            "Stage 2 will start from Stage 1 best full-model checkpoint: '%s'",
            stage2_init_weights,
        )
    else:
        stage2_init_weights = weights1
        logger.warning(
            "Stage 1 best checkpoint not found at '%s'. Falling back to final Stage 1 weights '%s'.",
            best_stage1_weights,
            stage2_init_weights,
        )

    # Stage 2: Fine-tuning (lower LR, starting from stage1 weights)
    if two_stage:
        logger.info("\n" + "=" * 60 + "\nSTAGE 2: Fine-Tuning\n" + "=" * 60)
        logger.info("Stage 2 starting from: %s", stage2_init_weights)
        model, history2, weights2 = stage_training(
            train_ds, val_ds, class_weights, cfg,
            stage_name="stage2",
            pretrained_weights=stage2_init_weights,
            stage1_baseline_qwk=metrics1.get("qwk"),
        )
        metrics2 = stage_evaluation(model, val_ds, cfg, stage_name="stage2")
        all_metrics["stage2"] = metrics2
        logger.info("Stage 2 QWK: %.4f", metrics2["qwk"])

    report = aggregate_results(all_metrics, cfg["output_dir"])
    elapsed = time.time() - t_total
    logger.info("Total pipeline time: %.1f seconds.", elapsed)
    report["elapsed_seconds"] = round(elapsed, 1)
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for the full pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run end-to-end DR+DME pipeline")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="pipeline_outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--single-stage", action="store_true",
                        help="Run only stage 1 (no fine-tuning)")
    parser.add_argument("--use-eyepacs", type=str, default=None, metavar="WEIGHTS_PATH",
                        help="Path to EyePACS backbone weights (.weights.h5) saved by "
                             "preprocessing.ipynb. When provided, stage 1 uses EyePACS "
                             "transfer learning instead of ImageNet.")
    parser.add_argument("--backbone-weights-path", type=str, default=None,
                        metavar="WEIGHTS_PATH",
                        help="Path to a custom backbone weights file (.h5) for Stage 1. "
                             "Enables transfer learning from any pre-trained backbone "
                             "(e.g. EyePACS, proprietary). Ignored when --use-eyepacs is set.")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        from dataset_loader import create_mock_dataset
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_pipeline")
        logger.info("Using mock dataset.")
    else:
        csv_path, image_dir = args.csv, args.image_dir

    cfg = load_config(args.config)
    cfg["output_dir"] = args.output_dir
    cfg["checkpoint_dir"] = os.path.join(args.output_dir, "checkpoints")
    cfg["seed"] = args.seed

    if args.epochs:
        cfg["stage1"]["epochs"] = args.epochs
        cfg["stage2"]["epochs"] = max(args.epochs // 2, 5)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    report = run_pipeline(
        csv_path=csv_path,
        image_dir=image_dir,
        config=cfg,
        two_stage=not args.single_stage,
        eyepacs_backbone=args.use_eyepacs,
        backbone_weights_path=args.backbone_weights_path,
    )

    print("\nPipeline Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
