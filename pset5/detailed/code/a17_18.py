import numpy as np
import cvxpy as cp

# Problem data
r = 1.05  # risk-free rate
S0 = 1.0  # current stock price
m = 200   # number of scenarios

# Stock prices in each scenario (uniformly spaced from 0.5 to 2)
S = np.linspace(0.5, 2.0, m)

# Known asset prices
prices_known = np.array([1.0, 1.0, 0.06, 0.03, 0.02, 0.01])

# Payoff matrix for known assets (m x 6)
V_known = np.zeros((m, 6))
V_known[:, 0] = r                          # Risk-free asset
V_known[:, 1] = S                          # Stock
V_known[:, 2] = np.maximum(0, S - 1.1)     # Call K=1.1
V_known[:, 3] = np.maximum(0, S - 1.2)     # Call K=1.2
V_known[:, 4] = np.maximum(0, 0.8 - S)     # Put K=0.8
V_known[:, 5] = np.maximum(0, 0.7 - S)     # Put K=0.7

# Collar payoff: min(C, max(F, S)) with F=0.9, C=1.15
F, C = 0.9, 1.15
v_collar = np.minimum(C, np.maximum(F, S))

# State-price vector variable
pi = cp.Variable(m)

# Constraints: prices match and pi >= 0
constraints = [
    V_known.T @ pi == prices_known,
    pi >= 0
]

# Lower bound on collar price
prob_lower = cp.Problem(cp.Minimize(v_collar @ pi), constraints)
prob_lower.solve()
price_lower = prob_lower.value

# Upper bound on collar price
prob_upper = cp.Problem(cp.Maximize(v_collar @ pi), constraints)
prob_upper.solve()
price_upper = prob_upper.value

print("Option Price Bounds for Collar (F=0.9, C=1.15)")
print("=" * 50)
print(f"Lower bound: {price_lower:.4f}")
print(f"Upper bound: {price_upper:.4f}")
print(f"\nArbitrage-free price range: [{price_lower:.4f}, {price_upper:.4f}]")

# Verify with a feasible state-price vector
print("\n--- Verification ---")
print(f"Sum of pi (should be close to 1/r = {1/r:.4f}): {np.sum(pi.value):.4f}")
print(f"All pi >= 0: {np.all(pi.value >= -1e-8)}")
