# test_qwk_calculation.py
"""Test if QWK is being calculated correctly using 3 DME classes."""

import numpy as np
from qwk_metrics import compute_quadratic_weighted_kappa

print("=" * 80)
print("TESTING QWK CALCULATION (3 DME classes: No DME, Mild, Moderate)")
print("=" * 80)

# Test 1: Perfect agreement (should be 1.0)
print("\n[Test 1] Perfect Agreement (should be 1.0)")
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 2, 0, 1, 2])
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 3)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected 1.0000)")
assert qwk == 1.0, "❌ Perfect agreement should give QWK=1.0"
print("  ✅ PASS")

# Test 2: All same prediction (should be ~0)
print("\n[Test 2] All Same Prediction (should be 0 or close)")
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 0, 0, 0, 0, 0])  # Always predict class 0
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 3)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected ~0)")
print(f"  Status: {'✅ OK' if qwk <= 0.2 else '❌ WRONG'}")

# Test 3: Mostly correct with some off-by-one errors (should be > 0)
print("\n[Test 3] Mostly Correct + Some Off-by-One (should be > 0)")
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 1, 0, 1, 2])  # 5/6 correct, 1 off by one (class 2 → 1)
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 3)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f} (expected > 0)")
print(f"  Status: {'✅ OK' if qwk > 0 else '❌ WRONG'}")

# Test 4: Partially correct
print("\n[Test 4] 50% Correct (should be > -0.1)")
y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 2, 2, 0, 1])  # First 3 correct, last 3 off
qwk = compute_quadratic_weighted_kappa(y_true, y_pred, 3)
print(f"  True:  {y_true}")
print(f"  Pred:  {y_pred}")
print(f"  QWK = {qwk:.4f}")
print(f"  Status: {'✅ OK' if qwk > -0.1 else '❌ WRONG'}")

print("\n" + "=" * 80)
print("If all tests PASS, QWK calculation is correct for 3 DME classes.")
print("=" * 80)
