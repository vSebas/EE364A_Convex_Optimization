"""
Problem A8.16: Fitting a sphere to data

We fit a sphere {x in R^n | ||x - c||_2 = r} to m points u_1, ..., u_m
by minimizing the loss function:
    sum_{i=1}^m (||u_i - c||_2^2 - r^2)^2

Using the substitution t = r^2 - ||c||_2^2, this becomes a least squares problem.
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
from sphere_fit_data import n, m, U

print(f"Problem dimensions: n={n}, m={m}")

# Part (b): Solve the least squares problem
# With t = r^2 - ||c||^2, the loss becomes:
#   sum_{i=1}^m (||u_i||^2 - 2 u_i^T c - t)^2
#
# In matrix form: minimize ||Ax - b||_2^2
# where x = [c; t], and
#   A[i,:] = [-2*u_i^T, -1]
#   b[i] = -||u_i||_2^2

A = np.zeros((m, n + 1))
b = np.zeros(m)

for i in range(m):
    u_i = U[:, i]
    A[i, :n] = -2 * u_i
    A[i, n] = -1
    b[i] = -np.dot(u_i, u_i)

# Solve least squares
x_star = np.linalg.lstsq(A, b, rcond=None)[0]

c_star = x_star[:n]
t_star = x_star[n]

# Recover r from t = r^2 - ||c||^2
r_squared = t_star + np.dot(c_star, c_star)
r_star = np.sqrt(r_squared)

print(f"\nResults:")
print(f"c* = ({c_star[0]:.4f}, {c_star[1]:.4f})")
print(f"t* = {t_star:.4f}")
print(f"r* = {r_star:.4f}")

# Compute optimal objective value
residuals = np.array([np.linalg.norm(U[:, i] - c_star)**2 - r_star**2 for i in range(m)])
obj_value = np.sum(residuals**2)
print(f"Optimal objective value = {obj_value:.4f}")

# Plot the fitted circle and data points
fig, ax = plt.subplots(figsize=(8, 8))

# Plot data points
ax.scatter(U[0, :], U[1, :], c='red', s=30, label='Data points', zorder=2)

# Plot fitted circle
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = c_star[0] + r_star * np.cos(theta)
circle_y = c_star[1] + r_star * np.sin(theta)
ax.plot(circle_x, circle_y, 'b-', linewidth=2, label='Fitted circle', zorder=1)

# Plot center
ax.scatter([c_star[0]], [c_star[1]], c='blue', s=100, marker='+', linewidths=2,
           label=f'Center ({c_star[0]:.2f}, {c_star[1]:.2f})', zorder=3)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'Sphere Fitting: $r^* = {r_star:.4f}$')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../latex/img/a8_16.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlot saved to ../latex/img/a8_16.png")
