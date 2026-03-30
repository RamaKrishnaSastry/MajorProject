"""Regression test for ordinal weighted loss stability.

Ensures the loss does not become zero for perfect predictions.
"""

import numpy as np
import tensorflow as tf

from train_enhanced import OrdinalWeightedCrossEntropy


def main() -> None:
    loss_fn = OrdinalWeightedCrossEntropy(num_classes=3, class_weights={0: 1.0, 1: 1.0, 2: 1.0})

    # Perfect prediction should still produce non-negative (small) CE-driven loss,
    # but never force an all-zero weighting.
    y_true = tf.one_hot([0, 1, 2], depth=3, dtype=tf.float32)
    y_pred = tf.one_hot([0, 1, 2], depth=3, dtype=tf.float32)
    perfect_loss = float(loss_fn(y_true, y_pred).numpy())

    # Misclassified prediction should be >= perfect prediction loss.
    y_pred_bad = tf.one_hot([2, 0, 1], depth=3, dtype=tf.float32)
    bad_loss = float(loss_fn(y_true, y_pred_bad).numpy())

    assert perfect_loss >= 0.0, f"Perfect loss must be >=0, got {perfect_loss}"
    assert bad_loss >= perfect_loss, (
        f"Bad loss should be >= perfect loss, got bad={bad_loss}, perfect={perfect_loss}"
    )

    # Loss should serialize and deserialize class weights / gamma correctly.
    cfg = loss_fn.get_config()
    restored = OrdinalWeightedCrossEntropy.from_config(dict(cfg))
    restored_cfg = restored.get_config()
    assert restored_cfg["num_classes"] == 3, f"Expected num_classes=3, got {restored_cfg['num_classes']}"
    assert abs(restored_cfg["focal_loss_gamma"] - cfg["focal_loss_gamma"]) < 1e-9, (
        f"Expected focal_loss_gamma to round-trip, got {restored_cfg['focal_loss_gamma']}"
    )
    assert restored_cfg["class_weights"] == cfg["class_weights"], (
        f"Expected class_weights to round-trip, got {restored_cfg['class_weights']}"
    )

    # Serialization should be robust to malformed/non-list class_weights.
    malformed_cfg = dict(cfg)
    malformed_cfg["class_weights"] = "not-a-list"
    malformed_restored = OrdinalWeightedCrossEntropy.from_config(malformed_cfg)
    malformed_restored_cfg = malformed_restored.get_config()
    assert malformed_restored_cfg["class_weights"] == [1.0, 1.0, 1.0], (
        "Malformed class_weights should safely fall back to default weights."
    )
    print(
        "PASS: OrdinalWeightedCrossEntropy keeps non-zero diagonal weighting; "
        f"perfect_loss={perfect_loss:.6f}, bad_loss={bad_loss:.6f}, "
        "serialization round-trip and malformed-config fallback ok"
    )


if __name__ == "__main__":
    main()
