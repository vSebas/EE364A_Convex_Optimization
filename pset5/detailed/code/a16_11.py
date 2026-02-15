import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Import problem data
from various_obj_regulator_data import A, B, x_init, T, n, m

# Variables
X = cp.Variable((n, T+1))  # states x_0, ..., x_T
U = cp.Variable((m, T))    # inputs u_0, ..., u_{T-1}

# Common constraints: dynamics, initial and final state
constraints = [
    X[:, 0] == x_init,
    X[:, T] == 0,
]
for t in range(T):
    constraints.append(X[:, t+1] == A @ X[:, t] + B @ U[:, t])

# (a) Sum of squares of 2-norms: sum ||u_t||_2^2
obj_a = cp.sum_squares(U)
prob_a = cp.Problem(cp.Minimize(obj_a), constraints)
prob_a.solve()
U_a = U.value.copy()
print(f"(a) Sum of squares of 2-norms: optimal value = {prob_a.value:.4f}")

# (b) Sum of 2-norms: sum ||u_t||_2
obj_b = cp.sum([cp.norm(U[:, t], 2) for t in range(T)])
prob_b = cp.Problem(cp.Minimize(obj_b), constraints)
prob_b.solve()
U_b = U.value.copy()
print(f"(b) Sum of 2-norms: optimal value = {prob_b.value:.4f}")

# (c) Max of 2-norms: max ||u_t||_2
obj_c = cp.max(cp.norm(U, 2, axis=0))
prob_c = cp.Problem(cp.Minimize(obj_c), constraints)
prob_c.solve()
U_c = U.value.copy()
print(f"(c) Max of 2-norms: optimal value = {prob_c.value:.4f}")

# (d) Sum of 1-norms: sum ||u_t||_1 (fuel use approximation)
obj_d = cp.sum([cp.norm(U[:, t], 1) for t in range(T)])
prob_d = cp.Problem(cp.Minimize(obj_d), constraints)
prob_d.solve()
U_d = U.value.copy()
print(f"(d) Sum of 1-norms: optimal value = {prob_d.value:.4f}")

# Plotting
t_range = np.arange(T)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

def plot_control(ax, U_opt, title, label):
    u_norms = np.linalg.norm(U_opt, axis=0)
    ax.plot(t_range, U_opt[0, :], 'b-', label='$u_1$', alpha=0.7)
    ax.plot(t_range, U_opt[1, :], 'r-', label='$u_2$', alpha=0.7)
    ax.plot(t_range, u_norms, 'k--', label=r'$\|u_t\|_2$', alpha=0.7)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Control input')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_control(axes[0, 0], U_a, '(a) Sum of squares of 2-norms', 'a')
plot_control(axes[0, 1], U_b, '(b) Sum of 2-norms', 'b')
plot_control(axes[1, 0], U_c, '(c) Max of 2-norms', 'c')
plot_control(axes[1, 1], U_d, '(d) Sum of 1-norms', 'd')

plt.tight_layout()
plt.savefig('../latex/img/a16_11_controls.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlot saved to ../latex/img/a16_11_controls.png")
