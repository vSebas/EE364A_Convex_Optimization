"""
Problem A15.13: Bandlimited signal recovery from zero-crossings

Given signs s of a bandlimited signal y, recover y up to a positive scale.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Load data
from zero_crossings_data import n, f_min, B, s, y as y_true

print(f"Signal length n = {n}")
print(f"Lowest frequency f_min = {f_min}")
print(f"Bandwidth B = {B}")

# Part (a): Formulation
# y is bandlimited: y_t = sum_{j=1}^{B} [a_j cos(2*pi*(f_min+j-1)*t/n) + b_j sin(2*pi*(f_min+j-1)*t/n)]
# for t = 1, ..., n
#
# Build the Fourier basis matrix F such that y = F @ [a; b]

t = np.arange(1, n + 1)
F = np.zeros((n, 2 * B))
for j in range(B):
    freq = f_min + j
    F[:, j] = np.cos(2 * np.pi * freq * t / n)
    F[:, B + j] = np.sin(2 * np.pi * freq * t / n)

# Variables: coefficients a, b (combined as coeff)
coeff = cp.Variable(2 * B)

# y = F @ coeff
y_var = F @ coeff

# Reformulation for normalization:
# Instead of ||y||_1 = n (not DCP), we use auxiliary variables
# t_i >= |y_i|, and sum(t_i) = n
# But sum(t) = n with t >= y, t >= -y is also not directly DCP as equality.
#
# Alternative: Fix one element of y to be positive (at a point where s > 0)
# Find first index where s = 1
pos_idx = np.where(s == 1)[0][0]
print(f"Using y[{pos_idx}] = 1 for normalization")

# Constraints:
# 1. Sign constraints: s_t * y_t >= 0
# 2. Normalization: y[pos_idx] = 1 (fix one positive value)
constraints = [
    cp.multiply(s, y_var) >= 0,  # Sign constraints
    y_var[pos_idx] == 1          # Normalization
]

# Objective: minimize ||y||_2 (among signals consistent with signs and normalization)
objective = cp.Minimize(cp.norm2(y_var))

prob = cp.Problem(objective, constraints)
prob.solve()

if prob.status == 'optimal':
    y_hat_raw = (F @ coeff.value).flatten()

    # Rescale to have ||y||_1 = n
    scale = n / np.sum(np.abs(y_hat_raw))
    y_hat = y_hat_raw * scale

    print(f"\nOptimal ||y_hat||_2 = {np.linalg.norm(y_hat):.4f}")
    print(f"||y_hat||_1 = {np.sum(np.abs(y_hat)):.4f} (should be {n})")

    # Part (b): Compare with true signal
    # Scale true signal to have same normalization
    y_true_scaled = y_true * n / np.sum(np.abs(y_true))

    # Compute relative recovery error
    rel_error = np.linalg.norm(y_true_scaled - y_hat) / np.linalg.norm(y_true_scaled)
    print(f"\nRelative recovery error ||y - y_hat||_2 / ||y||_2 = {rel_error:.6f}")

    # Check sign consistency
    sign_match = np.sum(np.sign(y_hat) == s)
    print(f"Sign matches: {sign_match}/{n} ({100*sign_match/n:.1f}%)")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(y_true_scaled, 'b-', linewidth=0.5, label='True signal (scaled)')
    axes[0].plot(y_hat, 'r--', linewidth=0.5, label='Recovered signal')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('y')
    axes[0].set_title('Signal Recovery from Zero-Crossings')
    axes[0].legend()
    axes[0].set_xlim([0, n])

    # Zoom in on a portion
    zoom_start, zoom_end = 0, 500
    axes[1].plot(range(zoom_start, zoom_end), y_true_scaled[zoom_start:zoom_end], 'b-', linewidth=1, label='True signal (scaled)')
    axes[1].plot(range(zoom_start, zoom_end), y_hat[zoom_start:zoom_end], 'r--', linewidth=1, label='Recovered signal')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('y')
    axes[1].set_title(f'Zoomed view (t = {zoom_start} to {zoom_end})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../latex/img/a15_13.png', dpi=150)
    plt.close()

    print(f"\nThe recovery quality is {'good' if rel_error < 0.1 else 'moderate' if rel_error < 0.3 else 'poor'}.")
else:
    print(f"Problem status: {prob.status}")
