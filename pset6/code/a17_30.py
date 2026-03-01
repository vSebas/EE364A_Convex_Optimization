"""
Problem A17.30: Maximum Sharpe ratio portfolio

Maximize Sharpe ratio = (mu^T x) / ||Sigma^{1/2} x||_2
subject to: 1^T x = 1, ||x||_1 <= L^max
"""
import numpy as np
import cvxpy as cp

# Generate example data
np.random.seed(42)
n = 10  # number of assets

# Expected returns
mu = np.random.randn(n) * 0.1 + 0.05  # mean around 5%

# Covariance matrix (generate positive definite)
A = np.random.randn(n, n) * 0.1
Sigma = A @ A.T + 0.01 * np.eye(n)

L_max = 1.5  # leverage limit

print(f"Number of assets: n = {n}")
print(f"Leverage limit: L^max = {L_max}")

# Part (a): Show quasiconvexity
# The Sharpe ratio S(x) = mu^T x / ||Sigma^{1/2} x||_2 is quasiconvex in x
# because:
# - The sublevel set {x : S(x) <= t} = {x : mu^T x <= t * ||Sigma^{1/2} x||_2}
# - For t >= 0 and mu^T x > 0, this is equivalent to ||Sigma^{1/2} x||_2 >= (mu^T x)/t
# - This is a second-order cone constraint (convex)

# Part (b): Solve via transformation
# Let y = x / (mu^T x) (assuming mu^T x > 0)
# Then 1^T y = 1/(mu^T x) and mu^T y = 1
# Original problem becomes:
#   minimize ||Sigma^{1/2} y||_2
#   subject to: mu^T y = 1
#               ||y||_1 <= L^max * (1^T y)  [from leverage constraint]
#               1^T y >= 0 (implied by mu^T x > 0)
# Then x = y / (1^T y)

y = cp.Variable(n)

# Constraints
constraints = [
    mu @ y == 1,        # Normalization from transformation
    cp.sum(y) >= 0,     # mu^T x > 0 implies sum(y) > 0
    cp.norm1(y) <= L_max * cp.sum(y)  # Leverage constraint
]

# Objective: minimize portfolio risk
objective = cp.Minimize(cp.quad_form(y, Sigma))

prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status == 'optimal':
    y_opt = y.value
    scale = np.sum(y_opt)
    x_opt = y_opt / scale

    # Compute Sharpe ratio
    portfolio_return = mu @ x_opt
    portfolio_risk = np.sqrt(x_opt @ Sigma @ x_opt)
    sharpe_ratio = portfolio_return / portfolio_risk

    print(f"\nOptimal portfolio weights:")
    for i in range(n):
        print(f"  Asset {i+1}: {x_opt[i]:+.4f}")

    print(f"\nPortfolio statistics:")
    print(f"  Expected return: {portfolio_return:.4f}")
    print(f"  Standard deviation: {portfolio_risk:.4f}")
    print(f"  Sharpe ratio: {sharpe_ratio:.4f}")
    print(f"  Leverage ||x||_1: {np.sum(np.abs(x_opt)):.4f}")
    print(f"  Sum of weights: {np.sum(x_opt):.4f}")
else:
    print(f"Problem status: {prob.status}")

# Compare with unconstrained (L_max = infinity) case
y2 = cp.Variable(n)
constraints2 = [mu @ y2 == 1]
prob2 = cp.Problem(cp.Minimize(cp.quad_form(y2, Sigma)), constraints2)
prob2.solve()

y2_opt = y2.value
scale2 = np.sum(y2_opt)
x2_opt = y2_opt / scale2

portfolio_return2 = mu @ x2_opt
portfolio_risk2 = np.sqrt(x2_opt @ Sigma @ x2_opt)
sharpe_ratio2 = portfolio_return2 / portfolio_risk2

print(f"\nUnconstrained leverage portfolio:")
print(f"  Sharpe ratio: {sharpe_ratio2:.4f}")
print(f"  Leverage ||x||_1: {np.sum(np.abs(x2_opt)):.4f}")
