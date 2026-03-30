"""Tests for DR regression-to-grade conversion utility."""

import numpy as np

from evaluate import dr_regression_to_grade


def run_test() -> None:
    # Includes negatives, fractions near boundaries, and values above class max.
    preds = np.array([-1.2, -0.49, 0.49, 0.5, 1.6, 2.5, 3.51, 4.2, 7.9], dtype=np.float32)
    grades = dr_regression_to_grade(preds, num_dr_classes=5)

    expected = np.array([0, 0, 0, 0, 2, 2, 4, 4, 4], dtype=np.int32)
    assert np.array_equal(grades, expected), f"Unexpected grades: {grades} vs {expected}"

    # Works with (N,1) tensors/arrays too.
    preds_col = np.array([[0.1], [1.9], [3.6], [4.8]], dtype=np.float32)
    grades_col = dr_regression_to_grade(preds_col, num_dr_classes=5)
    expected_col = np.array([[0], [2], [4], [4]], dtype=np.int32)
    assert np.array_equal(grades_col, expected_col), f"Unexpected column grades: {grades_col}"

    print("PASS: dr_regression_to_grade rounds and clips DR regression outputs correctly.")


if __name__ == "__main__":
    run_test()
