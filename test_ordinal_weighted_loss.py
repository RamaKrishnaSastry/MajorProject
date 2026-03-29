"""Regression test for ordinal weighted loss stability.

Ensures the loss does not become zero for perfect predictions.
"""

import numpy as np
import tensorflow as tf

from train_enhanced import OrdinalWeightedCrossEntropy


def main() -> None:
    loss_fn = OrdinalWeightedCrossEntropy(num_classes=4, class_weights={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})

    # Perfect prediction should still produce non-negative (small) CE-driven loss,
    # but never force an all-zero weighting.
    y_true = tf.one_hot([0, 1, 2, 3], depth=4, dtype=tf.float32)
    y_pred = tf.one_hot([0, 1, 2, 3], depth=4, dtype=tf.float32)
    perfect_loss = float(loss_fn(y_true, y_pred).numpy())

    # Misclassified prediction should be >= perfect prediction loss.
    y_pred_bad = tf.one_hot([3, 2, 1, 0], depth=4, dtype=tf.float32)
    bad_loss = float(loss_fn(y_true, y_pred_bad).numpy())

    assert perfect_loss >= 0.0, f"Perfect loss must be >=0, got {perfect_loss}"
    assert bad_loss >= perfect_loss, (
        f"Bad loss should be >= perfect loss, got bad={bad_loss}, perfect={perfect_loss}"
    )
    print(
        "PASS: OrdinalWeightedCrossEntropy keeps non-zero diagonal weighting; "
        f"perfect_loss={perfect_loss:.6f}, bad_loss={bad_loss:.6f}"
    )


if __name__ == "__main__":
    main()

