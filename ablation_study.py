"""
ablation_study.py
RFA (Receptive Field Augmentation) ablation study.

Compares three model variants on the IDRiD DME grading task:
    A — Baseline   : Standard ResNet50 + simple dense head (no ASPP, no DRN)
    B — RFA-Lite   : Standard ResNet50 + ASPP  (no DRN dilation)
    C — Full RFA   : DRN backbone + ASPP        (complete DR-ASPP-DRN)

Expected QWK:
    A ≈ 0.75  (baseline)
    B ≈ 0.78  (+ASPP only)
    C ≥ 0.80  (+DRN +ASPP)

Usage:
    python ablation_study.py --dataset_path /path/to/dataset \
                             --output_dir outputs/ablation \
                             --epochs 15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

from evaluate import compute_quadratic_weighted_kappa, compute_metrics
from model import aspp_module, dme_head, build_backbone


# ── Model Builders ────────────────────────────────────────────────────────────

def _build_baseline(input_shape=(512, 512, 3)) -> Model:
    """
    Model A — Standard ResNet50 (no DRN, no ASPP).
    Simple GlobalAveragePooling → Dense DME head.
    """
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    inp = layers.Input(shape=input_shape, name="input_image")
    features = base(inp, training=False)
    out = dme_head(features, num_classes=3)
    return Model(inp, out, name="Baseline_ResNet50")


def _build_rfa_lite(input_shape=(512, 512, 3)) -> Model:
    """
    Model B — Standard ResNet50 + ASPP (no DRN dilation).
    Tests whether ASPP alone improves QWK.
    """
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    inp = layers.Input(shape=input_shape, name="input_image")
    features = base(inp, training=False)
    aspp_out = aspp_module(features, name_prefix="aspp_b")
    out = dme_head(aspp_out, num_classes=3)
    return Model(inp, out, name="RFA_Lite_ResNet50_ASPP")


def _build_full_rfa(input_shape=(512, 512, 3)) -> Model:
    """
    Model C — DRN backbone (dilated convolutions) + ASPP.
    Full DR-ASPP-DRN DME-only variant.
    """
    inp = layers.Input(shape=input_shape, name="input_image")
    backbone = build_backbone(input_shape, pretrained_weights=None)
    backbone.trainable = False

    features = backbone(inp, training=False)
    aspp_out = aspp_module(features, name_prefix="aspp_c")
    out = dme_head(aspp_out, num_classes=3)
    return Model(inp, out, name="Full_RFA_DRN_ASPP")


# ── Training ──────────────────────────────────────────────────────────────────

def _train_model(
    model: Model,
    train_ds,
    val_ds,
    epochs: int,
    class_weights: dict,
    out_dir: Path,
    model_name: str,
) -> dict:
    """Train a single model and return its evaluation metrics."""
    from train import QuadraticWeightedKappa, QWKCallback

    qwk_metric = QuadraticWeightedKappa(num_classes=3, name="qwk")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", qwk_metric],
    )

    qwk_cb = QWKCallback(val_ds, num_classes=3)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_qwk", patience=4, mode="max",
        restore_best_weights=True, verbose=0,
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / f"{model_name}.weights.h5"),
        monitor="val_qwk",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=0,
    )

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[qwk_cb, es, ckpt],
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.time() - t0

    # ── Collect final metrics ───────────────────────────────────────────────
    y_true_all, y_proba_all = [], []
    for batch in val_ds:
        x, y_true = batch[0], batch[1]
        y_proba = model(x, training=False).numpy()
        y_true_cls = np.argmax(y_true.numpy(), axis=-1)
        y_true_all.extend(y_true_cls.tolist())
        y_proba_all.append(y_proba)

    y_true_np = np.array(y_true_all, dtype=int)
    y_proba_np = np.vstack(y_proba_all)
    metrics = compute_metrics(y_true_np, y_proba_np)
    metrics["training_seconds"] = round(elapsed, 1)
    metrics["best_val_qwk"] = (
        max(qwk_cb.history, key=lambda e: e["val_qwk"])["val_qwk"]
        if qwk_cb.history else 0.0
    )
    return metrics


# ── Ablation Study ────────────────────────────────────────────────────────────

def run_ablation_study(
    dataset_path: str,
    output_dir: str = "outputs/ablation",
    epochs: int = 15,
    batch_size: int = 8,
) -> dict:
    """
    Run the full RFA ablation study.

    Trains models A, B, C on IDRiD and compares QWK scores.

    Args:
        dataset_path: Root directory of the IDRiD dataset.
        output_dir: Directory for results.
        epochs: Maximum training epochs per model.
        batch_size: Mini-batch size.

    Returns:
        Dictionary mapping model names to their metrics.
    """
    from dataset_loader import IDRiDDataset

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Shared datasets ─────────────────────────────────────────────────────
    print("[Ablation] Loading IDRiD dataset …")
    train_obj = IDRiDDataset(dataset_path, split="train", augment=True,  batch_size=batch_size)
    val_obj   = IDRiDDataset(dataset_path, split="val",   augment=False, batch_size=batch_size)
    train_ds  = train_obj.get_dataset()
    val_ds    = val_obj.get_dataset()
    cw        = train_obj.get_class_weights()
    train_obj.print_summary()

    model_specs = [
        ("A_Baseline",  _build_baseline,  "Standard ResNet50 (no ASPP, no DRN)"),
        ("B_RFA_Lite",  _build_rfa_lite,  "ResNet50 + ASPP  (no DRN dilation)"),
        ("C_Full_RFA",  _build_full_rfa,  "DRN backbone + ASPP (Full DR-ASPP-DRN)"),
    ]

    all_results = {}
    for name, builder, description in model_specs:
        print(f"\n{'='*60}")
        print(f"  Training Model {name}: {description}")
        print(f"{'='*60}")
        model = builder()
        m = _train_model(model, train_ds, val_ds, epochs, cw, out_dir, name)
        m["description"] = description
        all_results[name] = m
        tf.keras.backend.clear_session()

    # ── Summary table ───────────────────────────────────────────────────────
    _print_summary(all_results, out_dir)

    results_path = out_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] Results saved → {results_path}")

    return all_results


def _print_summary(results: dict, out_dir: Path):
    """Print and save a formatted comparison table."""
    header = f"\n{'Model':<20}{'Description':<45}{'QWK':>8}{'Acc':>8}{'F1':>8}"
    sep = "-" * 90
    lines = [header, sep]

    sorted_items = sorted(results.items(), key=lambda kv: kv[1].get("qwk", 0))
    for name, m in sorted_items:
        line = (
            f"{name:<20}{m.get('description', ''):<45}"
            f"{m.get('qwk', 0):>8.4f}"
            f"{m.get('accuracy', 0):>8.4f}"
            f"{m.get('macro_f1', 0):>8.4f}"
        )
        lines.append(line)

    lines.append(sep)
    best = max(results.items(), key=lambda kv: kv[1].get("qwk", 0))
    lines.append(f"Best model: {best[0]}  (QWK = {best[1].get('qwk', 0):.4f})")

    output = "\n".join(lines)
    print(output)

    txt_path = out_dir / "ablation_results.txt"
    with open(txt_path, "w") as f:
        f.write(output + "\n")
    print(f"[Ablation] Table saved → {txt_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RFA Ablation Study")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir",   default="outputs/ablation")
    p.add_argument("--epochs",       type=int, default=15)
    p.add_argument("--batch_size",   type=int, default=8)
    args = p.parse_args()

    run_ablation_study(
        args.dataset_path,
        args.output_dir,
        args.epochs,
        args.batch_size,
    )
