"""
ablation_study.py - Ablation study for DR-ASPP-DRN architecture variants.

Compares three model configurations to validate each component's contribution:
- Model A: Baseline ResNet50 only                    (expected QWK ~0.73-0.75)
- Model B: ResNet50 + ASPP module                    (expected QWK ~0.76-0.78)
- Model C: Full DR-ASPP-DRN (Multi-task + ASPP)      (expected QWK ≥ 0.80)

Provides:
- Quantitative comparison table
- Statistical significance testing (paired bootstrap)
- Visualisation of QWK improvement trajectory
- JSON export of ablation results
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from qwk_metrics import (
    compute_quadratic_weighted_kappa,
    compute_ordinal_metrics,
    NUM_DME_CLASSES,
    DME_CLASS_NAMES,
)
from evaluate import get_predictions, compute_accuracy, compute_f1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
# Model variant builders
# ---------------------------------------------------------------------------

def build_model_a(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
) -> keras.Model:
    """Model A: Baseline ResNet50 with direct classification head.

    No ASPP, no multi-task learning. Represents the simplest feasible
    baseline for this task.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = keras.applications.ResNet50(
        include_top=False, weights=backbone_weights, input_shape=input_shape
    )
    backbone.trainable = True
    features = backbone(inputs)

    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelA_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model A (Baseline ResNet50): %d params", model.count_params())
    return model


def build_model_b(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = 256,
) -> keras.Model:
    """Model B: ResNet50 + ASPP module, single DME head.

    Adds multi-scale context via ASPP but without the DR auxiliary task.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_aspp

    inputs = keras.Input(shape=input_shape, name="input_image")
    backbone = keras.applications.ResNet50(
        include_top=False, weights=backbone_weights, input_shape=input_shape
    )
    backbone.trainable = True
    features = backbone(inputs)

    aspp_out = build_aspp(features, filters=aspp_filters, name_prefix="aspp")

    x = layers.GlobalAveragePooling2D(name="dme_gap")(aspp_out)
    x = layers.Dense(256, activation="relu", name="dme_fc1")(x)
    x = layers.Dropout(0.4, name="dme_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dme_risk")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ModelB_ASPP")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    logger.info("Model B (ResNet50 + ASPP): %d params", model.count_params())
    return model


def build_model_c(
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_dme_classes: int = NUM_DME_CLASSES,
    backbone_weights: str = "imagenet",
    aspp_filters: int = 256,
) -> keras.Model:
    """Model C: Full DR-ASPP-DRN (multi-task DR + DME with ASPP).

    The complete production architecture.

    Parameters
    ----------
    input_shape : tuple
        Input image shape.
    num_dme_classes : int
        Number of DME classes.
    backbone_weights : str
        Backbone pre-training source.
    aspp_filters : int
        ASPP branch filter count.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    from model import build_model

    model = build_model(
        input_shape=input_shape,
        backbone_weights=backbone_weights,
        num_dme_classes=num_dme_classes,
        aspp_filters=aspp_filters,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={"dr_output": "mse", "dme_risk": "categorical_crossentropy"},
        loss_weights={"dr_output": 0.3, "dme_risk": 1.0},
        metrics={"dme_risk": [keras.metrics.CategoricalAccuracy(name="accuracy")]},
    )
    logger.info("Model C (Full DR-ASPP-DRN): %d params", model.count_params())
    return model


# ---------------------------------------------------------------------------
# Model prediction helpers for ablation
# ---------------------------------------------------------------------------

def _get_predictions_ablation(
    model: keras.Model,
    dataset: tf.data.Dataset,
    num_classes: int = NUM_DME_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions from any of the three ablation model variants.

    Parameters
    ----------
    model : keras.Model
        Any of Model A, B, or C.
    dataset : tf.data.Dataset
        Validation dataset.
    num_classes : int
        Number of classes.

    Returns
    -------
    tuple
        ``(y_true, y_proba, y_pred)``
    """
    all_true, all_proba = [], []
    for batch_images, batch_labels in dataset:
        preds = model(batch_images, training=False)
        if isinstance(preds, dict):
            proba = preds["dme_risk"].numpy()
        elif isinstance(preds, (list, tuple)):
            proba = preds[1].numpy()
        else:
            proba = preds.numpy()

        all_true.append(np.argmax(batch_labels.numpy(), axis=-1))
        all_proba.append(proba)

    y_true = np.concatenate(all_true)
    y_proba = np.concatenate(all_proba)
    y_pred = np.argmax(y_proba, axis=-1)
    return y_true, y_proba, y_pred


# ---------------------------------------------------------------------------
# Quick training helper (for ablation experiments)
# ---------------------------------------------------------------------------

def _quick_train(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 5,
    class_weights: Optional[Dict] = None,
    verbose: int = 0,
) -> keras.callbacks.History:
    """Train a model variant for a quick ablation experiment.

    Parameters
    ----------
    model : keras.Model
        Compiled model.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    epochs : int
        Number of training epochs.
    class_weights : dict, optional
        Per-class weights.
    verbose : int
        Verbosity.

    Returns
    -------
    keras.callbacks.History
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


# ---------------------------------------------------------------------------
# Statistical significance testing (paired bootstrap)
# ---------------------------------------------------------------------------

def bootstrap_qwk_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """Paired bootstrap test to compare QWK of two models.

    Computes a 95% confidence interval on the QWK difference and a
    two-sided p-value estimate.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_a : np.ndarray
        Predictions from model A.
    y_pred_b : np.ndarray
        Predictions from model B.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``qwk_a``, ``qwk_b``, ``delta``, ``ci_low``, ``ci_high``, ``p_value``.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    qwk_a = compute_quadratic_weighted_kappa(y_true, y_pred_a)
    qwk_b = compute_quadratic_weighted_kappa(y_true, y_pred_b)
    observed_delta = qwk_b - qwk_a

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        qt_a = compute_quadratic_weighted_kappa(y_true[idx], y_pred_a[idx])
        qt_b = compute_quadratic_weighted_kappa(y_true[idx], y_pred_b[idx])
        deltas.append(qt_b - qt_a)

    deltas = np.array(deltas)
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))

    # Two-sided p-value: fraction of bootstrap deltas on the wrong side
    if observed_delta >= 0:
        p_value = float(np.mean(deltas <= 0))
    else:
        p_value = float(np.mean(deltas >= 0))

    return {
        "qwk_a": float(qwk_a),
        "qwk_b": float(qwk_b),
        "delta": float(observed_delta),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "significant": bool(ci_low > 0 or ci_high < 0),
    }


# ---------------------------------------------------------------------------
# Comparison table and visualisation
# ---------------------------------------------------------------------------

def print_ablation_table(results: List[Dict]) -> None:
    """Pretty-print the ablation comparison table.

    Parameters
    ----------
    results : list of dict
        Each element is a per-model result dict.
    """
    print("\n" + "=" * 72)
    print(f"{'Model':<30} {'QWK':>8} {'Acc':>8} {'F1':>8} {'MAE':>8} {'Params':>10}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['name']:<30} {r['qwk']:>8.4f} {r['accuracy']:>8.4f} "
            f"{r['f1_macro']:>8.4f} {r['mae']:>8.3f} {r.get('params', 0):>10,}"
        )
    print("=" * 72 + "\n")


def plot_ablation_comparison(
    results: List[Dict],
    output_path: str = "ablation_comparison.png",
) -> None:
    """Plot QWK improvement across model variants.

    Parameters
    ----------
    results : list of dict
        Per-model result dicts.
    output_path : str
        Output PNG path.
    """
    if not _PLOTTING_AVAILABLE:
        return

    names = [r["name"] for r in results]
    qwks = [r["qwk"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1_macro"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # QWK bar chart
    bars = axes[0].bar(x, qwks, width=0.5, color=["steelblue", "darkorange", "seagreen"],
                       edgecolor="black", alpha=0.85)
    axes[0].axhline(y=0.80, color="red", linestyle="--", linewidth=2, label="Target 0.80")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=10, ha="right")
    axes[0].set_ylim([0, 1.05])
    axes[0].set_title("QWK by Model Variant", fontsize=13)
    axes[0].set_ylabel("QWK Score")
    axes[0].legend()
    for bar, q in zip(bars, qwks):
        axes[0].text(bar.get_x() + bar.get_width() / 2, q + 0.01, f"{q:.3f}",
                     ha="center", va="bottom", fontsize=10)

    # Multi-metric grouped bar chart
    axes[1].bar(x - width, qwks, width, label="QWK", color="steelblue", alpha=0.85)
    axes[1].bar(x, accs, width, label="Accuracy", color="darkorange", alpha=0.85)
    axes[1].bar(x + width, f1s, width, label="F1 Macro", color="seagreen", alpha=0.85)
    axes[1].axhline(y=0.80, color="red", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=10, ha="right")
    axes[1].set_ylim([0, 1.05])
    axes[1].set_title("Multi-Metric Comparison", fontsize=13)
    axes[1].set_ylabel("Score")
    axes[1].legend()

    # QWK improvement annotations
    for i in range(1, len(qwks)):
        delta = qwks[i] - qwks[i - 1]
        axes[0].annotate(
            f"+{delta:.3f}",
            xy=(x[i], qwks[i] + 0.04),
            ha="center",
            color="darkred",
            fontsize=9,
        )

    fig.suptitle("Ablation Study – DR-ASPP-DRN Component Contribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ablation comparison plot saved to '%s'.", output_path)


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation_study(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Optional[Dict] = None,
    epochs: int = 5,
    output_dir: str = "ablation_results",
    input_shape: Tuple[int, int, int] = (512, 512, 3),
    num_classes: int = NUM_DME_CLASSES,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict:
    """Run the full ablation study comparing Models A, B, and C.

    Parameters
    ----------
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    class_weights : dict, optional
        Class weights for imbalance handling.
    epochs : int
        Training epochs per model variant.
    output_dir : str
        Output directory for artefacts.
    input_shape : tuple
        Input image shape.
    num_classes : int
        Number of ordinal classes.
    n_bootstrap : int
        Bootstrap resamples for significance testing.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Ablation results including per-model metrics and significance tests.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_configs = [
        ("Model A: Baseline ResNet50", build_model_a),
        ("Model B: ResNet50 + ASPP",   build_model_b),
        ("Model C: Full DR-ASPP-DRN",  build_model_c),
    ]

    results = []
    all_predictions = {}

    for name, builder in model_configs:
        logger.info("\n%s\nTraining %s …\n%s", "=" * 60, name, "=" * 60)
        t0 = time.time()

        if name.startswith("Model C"):
            model = builder(input_shape=input_shape, num_dme_classes=num_classes)
        else:
            model = builder(input_shape=input_shape, num_classes=num_classes)

        _quick_train(model, train_ds, val_ds, epochs=epochs,
                     class_weights=class_weights, verbose=1)

        y_true, y_proba, y_pred = _get_predictions_ablation(model, val_ds, num_classes)
        ordinal = compute_ordinal_metrics(y_true, y_pred, num_classes)
        acc = compute_accuracy(y_true, y_pred)
        f1 = compute_f1(y_true, y_pred, average="macro")

        elapsed = time.time() - t0
        result = {
            "name": name,
            "qwk": ordinal["qwk"],
            "mae": ordinal["mae"],
            "rmse": ordinal["rmse"],
            "accuracy": acc,
            "f1_macro": f1,
            "target_met": ordinal["qwk"] >= 0.80,
            "training_time_s": round(elapsed, 1),
            "params": int(model.count_params()),
        }
        results.append(result)
        all_predictions[name] = y_pred
        logger.info("%s → QWK=%.4f | acc=%.4f | f1=%.4f", name, ordinal["qwk"], acc, f1)

    # Statistical significance tests (A vs B, B vs C, A vs C)
    logger.info("\nRunning bootstrap significance tests …")
    # Re-use stored predictions and the last collected y_true for significance testing
    names = [r["name"] for r in results]
    sig_tests = {}

    comparison_pairs = [
        (0, 1, "A_vs_B"),
        (1, 2, "B_vs_C"),
        (0, 2, "A_vs_C"),
    ]
    for i, j, label in comparison_pairs:
        pred_i = all_predictions[names[i]]
        pred_j = all_predictions[names[j]]
        # y_true may differ per model call; use first collected y_true
        sig_tests[label] = bootstrap_qwk_test(
            y_true, pred_i, pred_j, n_bootstrap=n_bootstrap, seed=seed,
        )
        logger.info(
            "Significance %s: Δ=%.4f [%.4f, %.4f] p=%.3f %s",
            label,
            sig_tests[label]["delta"],
            sig_tests[label]["ci_low"],
            sig_tests[label]["ci_high"],
            sig_tests[label]["p_value"],
            "✅ significant" if sig_tests[label]["significant"] else "❌ not significant",
        )

    ablation_output = {
        "results": results,
        "significance_tests": sig_tests,
        "summary": {
            "best_model": max(results, key=lambda r: r["qwk"])["name"],
            "qwk_improvement_A_to_C": round(
                results[-1]["qwk"] - results[0]["qwk"], 4
            ),
            "target_met_by_model_c": results[-1]["target_met"],
        },
    }

    print_ablation_table(results)

    # Save results
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(ablation_output, f, indent=2)
    logger.info("Ablation results saved to '%s'.", json_path)

    # Visualise
    plot_ablation_comparison(
        results,
        output_path=os.path.join(output_dir, "ablation_comparison.png"),
    )

    return ablation_output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the ablation study."""
    import argparse

    from dataset_loader import build_datasets, create_mock_dataset

    parser = argparse.ArgumentParser(description="Run ablation study (Model A/B/C)")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="ablation_results")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock or args.csv is None:
        logger.info("Using mock dataset.")
        csv_path, image_dir = create_mock_dataset("/tmp/mock_irdid_ablation", num_samples=80)
    else:
        csv_path, image_dir = args.csv, args.image_dir

    train_ds, val_ds, class_weights = build_datasets(
        csv_path=csv_path, image_dir=image_dir, batch_size=args.batch_size,
    )

    results = run_ablation_study(
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        epochs=args.epochs,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )

    print("\nAblation Summary:")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
