"""
Problem A6.13: Fitting with censored data

We fit a linear model y = c^T x with censored data.
y^(1), ..., y^(M) are observed (uncensored)
y^(M+1), ..., y^(K) are censored (only known to be > D)
"""
import numpy as np
import cvxpy as cp

# Load data
from cens_fit_data import n, M, K, c_true, X, y, D

print(f"Problem dimensions: n={n}, M={M}, K={K}")
D_val = float(D) if np.isscalar(D) else float(D.flatten()[0])
print(f"Censoring threshold D = {D_val:.4f}")

# Part (a): Formulation
# We want to minimize J = sum_{k=1}^{K} (y^(k) - c^T x^(k))^2
# For k = 1, ..., M: y^(k) is known
# For k = M+1, ..., K: y^(k) > D (unknown, need to optimize over them)
#
# Variables: c (model parameter), y_cens (censored y values)
# Minimize: sum_{k=1}^{M} (y[k] - c^T X[:,k])^2 + sum_{k=M+1}^{K} (y_cens[k-M] - c^T X[:,k])^2
# Subject to: y_cens[k] >= D for all censored points

# Part (b): Solve using CVXPY
c_var = cp.Variable((n, 1))
y_cens = cp.Variable((K - M, 1))  # Censored y values

# Observed part: X[:, :M] corresponds to first M data points
X_obs = X[:, :M].reshape(n, M)
X_cens = X[:, M:].reshape(n, K - M)

# Residuals for observed data
residuals_obs = y - X_obs.T @ c_var  # Shape (M, 1)

# Residuals for censored data
residuals_cens = y_cens - X_cens.T @ c_var  # Shape (K-M, 1)

# Objective: sum of squared residuals
objective = cp.sum_squares(residuals_obs) + cp.sum_squares(residuals_cens)

# Constraints: censored values must be >= D
constraints = [y_cens >= D]

prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()

c_hat = c_var.value
y_cens_hat = y_cens.value

print(f"\nOptimal objective J = {prob.value:.6f}")
print(f"\nc_hat (all {n} components):")
print(c_hat.flatten())

# Compute relative errors
c_hat_flat = c_hat.flatten()
c_true_flat = c_true.flatten()

rel_error_censored = np.linalg.norm(c_true_flat - c_hat_flat) / np.linalg.norm(c_true_flat)
print(f"\nRelative error ||c_true - c_hat||_2 / ||c_true||_2 = {rel_error_censored:.6f}")

# Compare with least squares ignoring censored data (using only observed data)
X_obs_M = X_obs  # Already shaped properly
y_obs_M = y.flatten()

# Least squares solution: c_ls = (X X^T)^{-1} X y
c_ls = np.linalg.lstsq(X_obs_M.T, y_obs_M, rcond=None)[0]
c_ls_flat = c_ls.flatten()

print(f"\nc_ls (all {n} components, ignoring censored data):")
print(c_ls_flat)

rel_error_ls = np.linalg.norm(c_true_flat - c_ls_flat) / np.linalg.norm(c_true_flat)
print(f"Relative error (ignoring censored) ||c_true - c_ls||_2 / ||c_true||_2 = {rel_error_ls:.6f}")
