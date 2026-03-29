# test_qwk_calculation.py
"""Test if QWK is being calculated correctly."""

import numpy as np
from qwk_metrics import compute_quadratic_weighted_kappa

print("=" * 80)
print("TESTING QWK CALCULATION")
print("=" * 80)

# Test 1: Perfect agreement (should be 1.0)
print("\n[Test 1] Perfect Agreement (should be 1.0)")
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 4)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected 1.0000)")
assert qwk == 1.0, "❌ Perfect agreement should give QWK=1.0"
print("  ✅ PASS")

# Test 2: Random guessing (should be ~0)
print("\n[Test 2] All Same Prediction (should be 0 or close)")
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Always predict class 0
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 4)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected ~0)")
print(f"  Status: {'✅ OK' if qwk <= 0.2 else '❌ WRONG'}")

# Test 3: Off-by-one (should be decent)
print("\n[Test 3] Off-by-One Errors (should be > 0.5)")
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([1, 2, 3, 2, 1, 2, 3, 2])  # Shifted by 1
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 4)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected > 0.5)")
print(f"  Status: {'✅ OK' if qwk > 0.5 else '❌ WRONG'}")

# Test 4: Partially correct
print("\n[Test 4] 50% Correct (should be > 0)")
y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
y_pred = np.array([0, 1, 2, 3, 3, 2, 1, 0])  # First 4 correct, last 4 opposite
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 4)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected 0 to 0.3)")
print(f"  Status: {'✅ OK' if qwk > -0.1 else '❌ WRONG'}")

print("\n" + "=" * 80)
print("If all tests PASS, QWK calculation is correct.")
print("If tests FAIL, there's a bug in compute_quadratic_weighted_kappa().")
print("=" * 80)