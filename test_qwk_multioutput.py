"""Validate QWKCallback with multi-output list predictions."""

import numpy as np
import tensorflow as tf

from qwk_metrics import QWKCallback


class _DummyMultiOutputModel:
    """Minimal model-like object returning [dr_output, dme_risk]."""

    def __call__(self, images, training=False):
        # Encode class index in the first pixel channel and reconstruct one-hot probs.
        cls = tf.cast(tf.round(images[:, 0, 0, 0] * 3.0), tf.int32)
        dme = tf.one_hot(cls, depth=4, dtype=tf.float32)
        dr = tf.expand_dims(tf.cast(cls, tf.float32), axis=-1)
        return [dr, dme]


def main() -> None:
    # Four classes repeated twice.
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)

    # Images carry class id in first pixel, normalized to [0, 1].
    images = np.zeros((len(y_true), 4, 4, 3), dtype=np.float32)
    images[:, 0, 0, 0] = y_true / 3.0

    dme_one_hot = tf.one_hot(y_true, depth=4, dtype=tf.float32)
    dr_targets = tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1)
    targets = {"dr_output": dr_targets, "dme_risk": dme_one_hot}

    val_ds = tf.data.Dataset.from_tensor_slices((images, targets)).batch(4)

    cb = QWKCallback(val_dataset=val_ds, num_classes=4, verbose=0)
    cb.set_model(_DummyMultiOutputModel())
    logs = {}
    cb.on_epoch_end(epoch=0, logs=logs)

    qwk = float(logs.get("val_qwk", -1.0))
    assert qwk == 1.0, f"Expected val_qwk=1.0 with perfect list-output predictions, got {qwk}"
    print("PASS: QWKCallback correctly extracts DME from multi-output list predictions.")


if __name__ == "__main__":
    main()
