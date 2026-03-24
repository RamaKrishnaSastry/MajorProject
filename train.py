"""
train.py
QWK-optimised training script for DME head fine-tuning.

Usage:
    python train.py --config config.yaml \
                    --dataset_path /path/to/dataset \
                    --output_dir outputs/

The script:
    1. Loads configuration from YAML.
    2. Builds IDRiD tf.data pipelines with class-weight handling.
    3. Constructs DR-ASPP-DRN with frozen backbone/ASPP.
    4. Trains with QWK-aware callbacks.
    5. Saves best model, history, and logs.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import cohen_kappa_score


# ── QWK Metric ────────────────────────────────────────────────────────────────

class QuadraticWeightedKappa(tf.keras.metrics.Metric):
    """
    Streaming Quadratic Weighted Kappa (QWK) metric.

    Accumulates the confusion matrix over an epoch and computes QWK
    from it at ``result()`` time.  Compatible with ``model.compile``
    ``metrics`` dict and with ``EarlyStopping(monitor='val_qwk')``.
    """

    def __init__(self, num_classes: int = 3, name: str = "qwk", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mat = self.add_weight(
            name="conf_mat",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.float32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: one-hot (B, C) or integer (B,) / (B, 1)
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # y_pred: softmax (B, C) → argmax
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
        else:
            y_pred = tf.clip_by_value(tf.round(y_pred), 0, self.num_classes - 1)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)

        cm = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32,
        )
        self.conf_mat.assign_add(cm)

    def result(self):
        n = self.num_classes
        cm = self.conf_mat

        # Quadratic weight matrix  w_ij = (i-j)^2 / (n-1)^2
        weights = tf.constant(
            [[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)] for i in range(n)],
            dtype=tf.float32,
        )

        act_hist = tf.reduce_sum(cm, axis=1)
        pred_hist = tf.reduce_sum(cm, axis=0)
        expected = tf.tensordot(act_hist, pred_hist, axes=0)
        expected /= tf.reduce_sum(expected) + tf.keras.backend.epsilon()

        cm_norm = cm / (tf.reduce_sum(cm) + tf.keras.backend.epsilon())

        num = tf.reduce_sum(weights * cm_norm)
        denom = tf.reduce_sum(weights * expected)
        return 1.0 - num / (denom + tf.keras.backend.epsilon())

    def reset_state(self):
        self.conf_mat.assign(tf.zeros((self.num_classes, self.num_classes)))


# ── QWK Callback ─────────────────────────────────────────────────────────────

class QWKCallback(tf.keras.callbacks.Callback):
    """
    Compute QWK on the validation set at the end of every epoch and
    write it to the training logs so that ``EarlyStopping`` and
    ``ModelCheckpoint`` can monitor ``val_qwk``.
    """

    def __init__(self, val_dataset, num_classes: int = 3, log_file: str = None):
        super().__init__()
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.log_file = log_file
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true_all, y_pred_all = [], []

        for batch in self.val_dataset:
            x, y_true = batch[0], batch[1]
            preds = self.model(x, training=False)
            # preds is [dr_out, dme_out] for multi-task model
            if isinstance(preds, (list, tuple)):
                dme_probs = preds[1]
            else:
                dme_probs = preds

            y_pred_cls = np.argmax(dme_probs.numpy(), axis=-1)

            if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
                y_true_cls = np.argmax(y_true.numpy(), axis=-1)
            else:
                y_true_cls = y_true.numpy().flatten().astype(int)

            y_true_all.extend(y_true_cls.tolist())
            y_pred_all.extend(y_pred_cls.tolist())

        if len(set(y_true_all)) > 1:
            qwk = cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic")
        else:
            qwk = 0.0

        logs["val_qwk"] = qwk
        self.history.append({"epoch": epoch + 1, "val_qwk": qwk})
        print(f"  val_qwk: {qwk:.4f}")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch + 1:03d}  val_qwk={qwk:.4f}\n")


# ── Training ──────────────────────────────────────────────────────────────────

def train_dme_model(
    config_path: str,
    dataset_path: str,
    output_dir: str,
) -> dict:
    """
    Main training entry point.

    Args:
        config_path: Path to ``config.yaml``.
        dataset_path: Root directory of the IDRiD dataset.
        output_dir: Directory where outputs are saved.

    Returns:
        Dictionary containing the final evaluation metrics.
    """
    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    from dataset_loader import IDRiDDataset

    train_ds_obj = IDRiDDataset(
        dataset_path,
        split="train",
        augment=cfg["training"].get("augment", True),
        batch_size=cfg["training"]["batch_size"],
    )
    val_ds_obj = IDRiDDataset(
        dataset_path,
        split="val",
        augment=False,
        batch_size=cfg["training"]["batch_size"],
    )

    train_ds_obj.print_summary()
    val_ds_obj.print_summary()

    train_ds = train_ds_obj.get_dataset()
    val_ds = val_ds_obj.get_dataset()
    class_weights = train_ds_obj.get_class_weights()

    # ── Model ────────────────────────────────────────────────────────────────
    from model import build_model_dme_tuning

    backbone_weights = cfg["model"].get("backbone_weights")
    freeze = cfg["model"].get("freeze_backbone", True)
    model = build_model_dme_tuning(backbone_weights, freeze_backbone=freeze)

    # ── Compile ──────────────────────────────────────────────────────────────
    lr = float(cfg["optimizer"]["learning_rate"])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    dme_weight = float(cfg["loss"]["dme_risk"]["weight"])
    dr_weight = float(cfg["loss"]["dr_output"]["weight"])

    qwk_metric_dme = QuadraticWeightedKappa(num_classes=3, name="qwk")

    model.compile(
        optimizer=optimizer,
        loss={
            "dr_output": "mse",
            "dme_risk": "categorical_crossentropy",
        },
        loss_weights={
            "dr_output": dr_weight,
            "dme_risk": dme_weight,
        },
        metrics={
            "dr_output": [],
            "dme_risk": ["accuracy", qwk_metric_dme],
        },
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    qwk_log_file = str(out_dir / "training_log.txt")
    qwk_callback = QWKCallback(val_ds, num_classes=3, log_file=qwk_log_file)

    es_cfg = cfg["training"].get("early_stopping", {})
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_qwk",
        patience=es_cfg.get("patience", 5),
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    best_weights_path = str(out_dir / "dme_finetuned.weights.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor="val_qwk",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_qwk",
        factor=0.5,
        patience=3,
        mode="max",
        min_lr=1e-7,
        verbose=1,
    )

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(out_dir / "tb_logs"),
        histogram_freq=0,
    )

    callbacks = [qwk_callback, early_stop, checkpoint, reduce_lr, tb_callback]

    # ── Train ────────────────────────────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    print(f"\n[Train] Starting DME fine-tuning for {epochs} epochs …\n")
    t0 = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n[Train] Finished in {elapsed / 60:.1f} min")

    # ── Save history ─────────────────────────────────────────────────────────
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [
            float(v) if not (isinstance(v, float) and v != v) else None
            for v in values
        ]
    # Append QWK history from callback
    history_dict["val_qwk"] = [e["val_qwk"] for e in qwk_callback.history]

    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[Train] History saved → {history_path}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    from evaluate import DMEEvaluator

    evaluator = DMEEvaluator(model, val_ds, str(out_dir))
    metrics = evaluator.run()

    metrics_path = out_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Train] Metrics saved → {metrics_path}")

    print(f"\n{'='*50}")
    print(f"  FINAL QWK : {metrics.get('qwk', 0.0):.4f}")
    print(f"  Accuracy  : {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Macro F1  : {metrics.get('macro_f1', 0.0):.4f}")
    print(f"{'='*50}\n")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Train DME head on IDRiD")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_dme_model(args.config, args.dataset_path, args.output_dir)
