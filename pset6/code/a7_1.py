"""
Problem A7.1: Maximum likelihood estimation of x and noise mean and covariance

This is primarily a theoretical problem showing that if f is log-concave,
then the ML estimation is a convex optimization problem.

The key insight is that the log-likelihood:
  sum_i log p(y_i - a_i^T x) = -m log(sigma) + sum_i log f((y_i - a_i^T x - mu)/sigma)

If f is log-concave, then log f is concave, making the entire expression
concave in (x, mu, sigma) under appropriate constraints.

Here we demonstrate with Gaussian noise using the analytical solution.
"""
import numpy as np
import cvxpy as cp

# For Gaussian MLE, we can solve analytically or use a two-step approach:
# 1. For fixed sigma, optimize over (x, mu) - this is least squares
# 2. The optimal sigma^2 = ||y - Ax - mu||^2 / m

# Generate synthetic data
np.random.seed(42)
n = 5   # dimension of x
m = 20  # number of measurements

# True parameters
x_true = np.random.randn(n)
mu_true = 0.5    # true noise mean
sigma_true = 2.0  # true noise std

# Measurement matrix
A = np.random.randn(m, n)

# Generate measurements: y_i = a_i^T x + v_i where v_i ~ N(mu, sigma^2)
y = A @ x_true + mu_true + sigma_true * np.random.randn(m)

# MLE for Gaussian: minimize sum of squared residuals
# Variables: x, mu (sigma is determined analytically)
x = cp.Variable(n)
mu = cp.Variable()

# Residuals
residuals = y - A @ x - mu

# Minimize sum of squared residuals (equivalent to MLE for fixed sigma)
prob = cp.Problem(cp.Minimize(cp.sum_squares(residuals)))
prob.solve()

x_hat = x.value
mu_hat = mu.value

# Optimal sigma from closed-form solution
residuals_opt = y - A @ x_hat - mu_hat
sigma_hat = np.sqrt(np.sum(residuals_opt**2) / m)

print("Maximum Likelihood Estimates:")
print(f"  x_hat = {x_hat}")
print(f"  mu_hat = {mu_hat:.4f}")
print(f"  sigma_hat = {sigma_hat:.4f}")

print("\nTrue values:")
print(f"  x_true = {x_true}")
print(f"  mu_true = {mu_true:.4f}")
print(f"  sigma_true = {sigma_true:.4f}")

print(f"\nEstimation errors:")
print(f"  ||x_hat - x_true||_2 = {np.linalg.norm(x_hat - x_true):.4f}")
print(f"  |mu_hat - mu_true| = {abs(mu_hat - mu_true):.4f}")
print(f"  |sigma_hat - sigma_true| = {abs(sigma_hat - sigma_true):.4f}")

# Log-likelihood at optimal
log_lik = -m * np.log(sigma_hat) - m/2 - (m/2) * np.log(2*np.pi)
print(f"\nLog-likelihood at MLE: {log_lik:.4f}")
