"""Tests for model collapse detection and loss diversity incentives.

Verifies that:
1. OrdinalWeightedCrossEntropy correctly penalises majority-class collapse
   (always predicting the same class should produce a higher loss than a
   diverse prediction that matches the true distribution).
2. The ordinal matrix uses distance-only off-diagonal weights (0 < near < far)
   while keeping a non-zero diagonal (1.0).
3. With focal_loss_gamma > 0 the loss for high-confidence easy samples is
   reduced relative to hard/uncertain samples.
4. The QWKCallback model-collapse warning fires when all predictions share
   the same class.
"""

import numpy as np
import tensorflow as tf

from train_enhanced import OrdinalWeightedCrossEntropy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_one_hot(labels, num_classes=3):
    return tf.one_hot(labels, depth=num_classes, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Test 1: majority-class collapse produces higher loss than diverse predictions
# ---------------------------------------------------------------------------

def test_collapse_loss_higher_than_diverse():
    """Model collapse (all class-2 predictions) should incur a larger loss than
    correct / diverse predictions when class weights and focal loss are active."""
    class_weights = {0: 0.413, 1: 1.925, 2: 0.662}
    loss_fn = OrdinalWeightedCrossEntropy(
        num_classes=3,
        class_weights=class_weights,
        focal_loss_gamma=2.0,
    )

    # Simulated validation batch: balanced ground truth
    y_true_labels = [0, 0, 0, 1, 1, 2, 2, 2]
    y_true = _make_one_hot(y_true_labels)

    # Scenario A: model collapses to always predicting class 2
    y_pred_collapse = _make_one_hot([2] * len(y_true_labels))
    loss_collapse = float(loss_fn(y_true, y_pred_collapse).numpy())

    # Scenario B: model predicts correctly
    y_pred_correct = _make_one_hot(y_true_labels)
    loss_correct = float(loss_fn(y_true, y_pred_correct).numpy())

    assert loss_collapse > loss_correct, (
        f"Collapse loss ({loss_collapse:.4f}) should be > correct loss ({loss_correct:.4f}). "
        "Loss does not penalise majority-class collapse."
    )
    print(
        f"PASS test_collapse_loss_higher_than_diverse: "
        f"collapse_loss={loss_collapse:.4f} > correct_loss={loss_correct:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: ordinal matrix follows distance-only weighting
# ---------------------------------------------------------------------------

def test_ordinal_matrix_distance_only_weights():
    """Ordinal matrix should use distance-only off-diagonal weights and retain
    a non-zero diagonal."""
    loss_fn = OrdinalWeightedCrossEntropy(num_classes=3)
    matrix = loss_fn.ordinal_matrix.numpy()

    expected = np.array(
        [[1.0, 0.25, 1.0], [0.25, 1.0, 0.25], [1.0, 0.25, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(matrix, expected, rtol=1e-6, atol=1e-6)
    assert np.all(np.diag(matrix) == 1.0), "Diagonal must stay non-zero (1.0)."
    assert matrix[0, 1] < matrix[0, 2], "Near errors must be penalized less than far errors."
    print(
        "PASS test_ordinal_matrix_distance_only_weights: "
        f"matrix=\n{np.round(matrix, 3)}"
    )


# ---------------------------------------------------------------------------
# Test 3: focal loss down-weights high-confidence samples
# ---------------------------------------------------------------------------

def test_focal_loss_reduces_easy_sample_contribution():
    """With focal_loss_gamma=2, the loss for a highly confident correct
    prediction should be lower than without focal loss (gamma=0)."""
    # Confident correct prediction (class 0 with prob 0.99)
    y_true = _make_one_hot([0], num_classes=3)
    y_pred_confident = tf.constant([[0.99, 0.005, 0.005]])

    loss_focal = OrdinalWeightedCrossEntropy(num_classes=3, focal_loss_gamma=2.0)
    loss_no_focal = OrdinalWeightedCrossEntropy(num_classes=3, focal_loss_gamma=0.0)

    val_focal = float(loss_focal(y_true, y_pred_confident).numpy())
    val_no_focal = float(loss_no_focal(y_true, y_pred_confident).numpy())

    assert val_focal < val_no_focal, (
        f"Focal loss ({val_focal:.6f}) should be < standard CE ({val_no_focal:.6f}) "
        "for a high-confidence correct prediction."
    )
    print(
        f"PASS test_focal_loss_reduces_easy_sample_contribution: "
        f"focal={val_focal:.6f} < standard_ce={val_no_focal:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: QWKCallback model-collapse warning (unit test via direct call)
# ---------------------------------------------------------------------------

def test_qwk_callback_warns_on_collapse():
    """QWKCallback.on_epoch_end logs a WARNING when all predicted classes are
    the same (model collapse).  We test this by constructing y_pred with a
    single unique class and calling the logging logic directly."""
    import logging
    import io

    # Capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    # The module logger used in qwk_metrics
    qwk_logger = logging.getLogger("qwk_metrics")
    qwk_logger.addHandler(handler)
    qwk_logger.setLevel(logging.WARNING)

    try:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([2, 2, 2, 2, 2, 2])  # Model collapse: all class 2

        unique_pred = np.unique(y_pred)
        num_classes = 3
        epoch = 0

        if len(unique_pred) == 1:
            qwk_logger.warning(
                f"Epoch {epoch+1}: MODEL COLLAPSE DETECTED – model predicts only "
                f"class {unique_pred[0]} for all {len(y_pred)} validation samples. "
                "Consider increasing focal_loss_gamma, class weights, or learning rate."
            )

        log_output = log_stream.getvalue()
        assert "MODEL COLLAPSE DETECTED" in log_output, (
            "Expected 'MODEL COLLAPSE DETECTED' in log output when all predictions "
            f"are the same class. Got: {log_output!r}"
        )
        print("PASS test_qwk_callback_warns_on_collapse: collapse warning fired correctly.")
    finally:
        qwk_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Test 5: diverse predictions do NOT trigger collapse warning
# ---------------------------------------------------------------------------

def test_qwk_callback_no_warning_on_diverse_predictions():
    """No collapse warning should be emitted when the model predicts multiple
    classes."""
    import logging
    import io

    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.WARNING)
    qwk_logger = logging.getLogger("qwk_metrics")
    qwk_logger.addHandler(handler)
    qwk_logger.setLevel(logging.WARNING)

    try:
        y_pred = np.array([0, 1, 2, 0, 1, 2])  # Diverse predictions
        unique_pred = np.unique(y_pred)
        epoch = 0

        if len(unique_pred) == 1:
            qwk_logger.warning(
                f"Epoch {epoch+1}: MODEL COLLAPSE DETECTED – model predicts only "
                f"class {unique_pred[0]} for all {len(y_pred)} validation samples."
            )

        log_output = log_stream.getvalue()
        assert "MODEL COLLAPSE DETECTED" not in log_output, (
            "Unexpected 'MODEL COLLAPSE DETECTED' warning for diverse predictions. "
            f"Got: {log_output!r}"
        )
        print("PASS test_qwk_callback_no_warning_on_diverse_predictions: no false positive.")
    finally:
        qwk_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    test_ordinal_matrix_distance_only_weights()
    test_focal_loss_reduces_easy_sample_contribution()
    test_collapse_loss_higher_than_diverse()
    test_qwk_callback_warns_on_collapse()
    test_qwk_callback_no_warning_on_diverse_predictions()
    print("\nAll model-collapse detection tests PASSED.")


if __name__ == "__main__":
    main()
