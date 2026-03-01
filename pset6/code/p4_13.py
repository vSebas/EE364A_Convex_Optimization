"""
Problem 4.13: Robust LP with interval coefficients

This is a theoretical problem - the solution shows that the robust LP
can be reformulated as a standard LP.
"""
import numpy as np
import cvxpy as cp

# Example to verify the reformulation
np.random.seed(42)

# Problem dimensions
m, n = 5, 3

# Generate nominal matrix and variation
A_bar = np.random.randn(m, n)
V = np.abs(np.random.randn(m, n)) * 0.5  # Variation matrix (all positive)
b = np.random.randn(m)
c = np.random.randn(n)

# Method 1: Robust LP formulation (equivalent LP)
x = cp.Variable(n)
constraints = []
for i in range(m):
    # For row i: sum_j (A_bar[i,j] * x[j]) + sum_j (V[i,j] * |x[j]|) <= b[i]
    constraints.append(A_bar[i, :] @ x + V[i, :] @ cp.abs(x) <= b[i])

prob1 = cp.Problem(cp.Minimize(c @ x), constraints)
prob1.solve()
print(f"Robust LP optimal value: {prob1.value:.6f}")
print(f"Optimal x: {x.value}")

# Method 2: Explicit LP formulation with auxiliary variables
x2 = cp.Variable(n)
t = cp.Variable(n)  # t[j] >= |x[j]|
constraints2 = [
    A_bar @ x2 + V @ t <= b,
    t >= x2,
    t >= -x2
]

prob2 = cp.Problem(cp.Minimize(c @ x2), constraints2)
prob2.solve()
print(f"\nExplicit LP optimal value: {prob2.value:.6f}")
print(f"Optimal x: {x2.value}")

# Verify both methods give same result
print(f"\nDifference in optimal values: {abs(prob1.value - prob2.value):.2e}")
