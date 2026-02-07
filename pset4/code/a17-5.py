import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from simple_portfolio_data import n, pbar, S, x_unif

# Target return = same as uniform portfolio
r_unif = (pbar.T @ x_unif)[0, 0]
var_unif = (x_unif.T @ S @ x_unif)[0, 0]
std_unif = np.sqrt(var_unif)

print("=== Part (a): Minimum-risk portfolios ===")
print(f"Uniform portfolio: return = {r_unif:.6f}, std = {std_unif:.6f}")

# (a) Minimum variance portfolios with same expected return as uniform
def min_variance_portfolio(pbar, S, r_target, long_only=False, short_limit=None):
    """Find minimum variance portfolio with target return."""
    n = len(pbar)
    x = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(x, S))
    constraints = [
        cp.sum(x) == 1,
        pbar.flatten() @ x == r_target
    ]

    if long_only:
        constraints.append(x >= 0)

    if short_limit is not None:
        # (x_-)_i = max(-x_i, 0), so 1^T x_- = sum of negative parts
        constraints.append(cp.sum(cp.neg(x)) <= short_limit)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value, np.sqrt(prob.value)

# No constraints
x_no_const, std_no_const = min_variance_portfolio(pbar, S, r_unif)
print(f"\nNo constraints: std = {std_no_const:.6f}")

# Long-only
x_long, std_long = min_variance_portfolio(pbar, S, r_unif, long_only=True)
print(f"Long-only: std = {std_long:.6f}")

# Short limit 0.5
x_short_lim, std_short_lim = min_variance_portfolio(pbar, S, r_unif, short_limit=0.5)
print(f"Short limit 0.5: std = {std_short_lim:.6f}")

print(f"\nComparison to uniform (std = {std_unif:.6f}):")
print(f"  No constraints: {100*(std_unif - std_no_const)/std_unif:.1f}% lower risk")
print(f"  Long-only: {100*(std_unif - std_long)/std_unif:.1f}% lower risk")
print(f"  Short limit: {100*(std_unif - std_short_lim)/std_unif:.1f}% lower risk")

# (b) Risk-return trade-off curves
print("\n=== Part (b): Risk-return trade-off curves ===")

def efficient_frontier(pbar, S, long_only=False, short_limit=None, n_points=50):
    """Compute efficient frontier."""
    n = len(pbar)

    # Find return range
    x = cp.Variable(n)
    constraints = [cp.sum(x) == 1]
    if long_only:
        constraints.append(x >= 0)
    if short_limit is not None:
        constraints.append(cp.sum(cp.neg(x)) <= short_limit)

    # Min return
    prob = cp.Problem(cp.Minimize(pbar.flatten() @ x), constraints)
    prob.solve()
    r_min = prob.value

    # Max return
    prob = cp.Problem(cp.Maximize(pbar.flatten() @ x), constraints)
    prob.solve()
    r_max = prob.value

    returns = np.linspace(r_min, r_max, n_points)
    stds = []

    for r_target in returns:
        try:
            _, std = min_variance_portfolio(pbar, S, r_target, long_only, short_limit)
            stds.append(std)
        except:
            stds.append(np.nan)

    return np.array(stds), returns

# Compute frontiers
std_long_frontier, ret_long_frontier = efficient_frontier(pbar, S, long_only=True)
std_short_frontier, ret_short_frontier = efficient_frontier(pbar, S, short_limit=0.5)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(std_long_frontier, ret_long_frontier, 'b-', linewidth=2, label='Long-only')
plt.plot(std_short_frontier, ret_short_frontier, 'r--', linewidth=2, label='Short limit 0.5')
plt.scatter([std_unif], [r_unif], s=100, c='green', marker='o', label='Uniform portfolio', zorder=5)
plt.xlabel('Standard deviation of return')
plt.ylabel('Expected return')
plt.title('Efficient Frontier: Risk-Return Trade-off')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../latex/img/a17_5_frontier.png', dpi=150, bbox_inches='tight')
plt.close()

print("Plot saved to ../latex/img/a17_5_frontier.png")

# Print summary table
print("\n=== Summary Table ===")
print(f"{'Portfolio':<20} {'Return':>10} {'Std Dev':>10}")
print("-" * 42)
print(f"{'Uniform':<20} {r_unif:>10.6f} {std_unif:>10.6f}")
print(f"{'No constraints':<20} {r_unif:>10.6f} {std_no_const:>10.6f}")
print(f"{'Long-only':<20} {r_unif:>10.6f} {std_long:>10.6f}")
print(f"{'Short limit 0.5':<20} {r_unif:>10.6f} {std_short_lim:>10.6f}")
