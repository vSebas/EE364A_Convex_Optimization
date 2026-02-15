import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Data generation
k = 201
t = np.linspace(-3, 3, k)
y = np.exp(t)

# Build the Vandermonde-like matrix for powers of t
# Tpowers = [1, t, t^2] for each data point
Tpowers = np.column_stack([np.ones(k), t, t**2])

def check_feasibility(gamma, Tpowers, y):
    """
    Check if there exists a rational function f(t) = p(t)/q(t) such that
    |f(t_i) - y_i| <= gamma for all i, with q(t_i) > 0.

    p(t) = a0 + a1*t + a2*t^2  (coefficients a = [a0, a1, a2])
    q(t) = 1 + b1*t + b2*t^2   (coefficients [1, b1, b2], so b = [b1, b2])

    The constraint |p(t_i)/q(t_i) - y_i| <= gamma becomes:
    |p(t_i) - y_i*q(t_i)| <= gamma*q(t_i)

    With Tpowers @ a = p(t) and Tpowers @ [1; b] = q(t):
    |Tpowers @ a - y * (Tpowers @ [1; b])| <= gamma * (Tpowers @ [1; b])
    """
    k = len(y)

    # Variables: a = [a0, a1, a2], b = [b1, b2]
    a = cp.Variable(3)
    b = cp.Variable(2)

    # q_coeffs = [1, b1, b2]
    q_coeffs = cp.hstack([1, b])

    # p(t_i) = Tpowers @ a
    # q(t_i) = Tpowers @ q_coeffs
    p_vals = Tpowers @ a
    q_vals = Tpowers @ q_coeffs

    # Constraints: |p - y*q| <= gamma*q, which splits into two linear constraints
    # p - y*q <= gamma*q  =>  p <= (y + gamma)*q
    # p - y*q >= -gamma*q =>  p >= (y - gamma)*q
    constraints = [
        p_vals <= cp.multiply(y + gamma, q_vals),
        p_vals >= cp.multiply(y - gamma, q_vals),
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    if prob.status == 'optimal':
        return True, a.value, b.value
    else:
        return False, None, None

# Bisection to find optimal gamma
l, u = 0.0, np.exp(3)
a_opt, b_opt, gamma_opt = None, None, None

while u - l >= 1e-3:
    gamma = (l + u) / 2
    feasible, a, b = check_feasibility(gamma, Tpowers, y)
    if feasible:
        u = gamma
        a_opt = a
        b_opt = b
        gamma_opt = gamma
    else:
        l = gamma

print(f"Optimal gamma: {gamma_opt:.4f}")
print(f"a0 = {a_opt[0]:.4f}")
print(f"a1 = {a_opt[1]:.4f}")
print(f"a2 = {a_opt[2]:.4f}")
print(f"b1 = {b_opt[0]:.4f}")
print(f"b2 = {b_opt[1]:.4f}")

# Compute the rational function fit
def f(t_val, a, b):
    p = a[0] + a[1]*t_val + a[2]*t_val**2
    q = 1 + b[0]*t_val + b[1]*t_val**2
    return p / q

y_fit = f(t, a_opt, b_opt)

# Plot 1: Data and fit
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b.', markersize=3, label=r'$y = e^t$ (data)')
plt.plot(t, y_fit, 'r-', linewidth=1.5, label=r'Rational fit $f(t)$')
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.title('Minimax Rational Fit to Exponential')
plt.legend()
plt.grid(True)
plt.savefig('latex/img/a6_2_fit.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Fitting error
error = y_fit - y
plt.figure(figsize=(10, 6))
plt.plot(t, error, 'b-', linewidth=1)
plt.axhline(y=gamma_opt, color='r', linestyle='--', alpha=0.7, label=f'$\\pm\\gamma^* = \\pm{gamma_opt:.4f}$')
plt.axhline(y=-gamma_opt, color='r', linestyle='--', alpha=0.7)
plt.xlabel('$t$')
plt.ylabel('$f(t_i) - y_i$')
plt.title('Fitting Error')
plt.legend()
plt.grid(True)
plt.savefig('latex/img/a6_2_error.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nMax absolute error: {np.max(np.abs(error)):.4f}")
print("Plots saved to latex/img/")
