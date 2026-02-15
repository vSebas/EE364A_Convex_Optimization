import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Import problem data (suppress plots during import)
import matplotlib
matplotlib.use('Agg')

# Manually set PLOT_FIGURES to False before importing
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("microgrid_data", "microgrid_data.py")
# Load the module but override PLOT_FIGURES
exec(open("microgrid_data.py").read().replace("PLOT_FIGURES = True", "PLOT_FIGURES = False"))

# Now we have: N, R_buy, R_sell, p_pv, p_ld, D, C, Q

print(f"Problem size: N = {N} periods")
print(f"Battery: Q = {Q} kWh, C = {C} kW (max charge), D = {D} kW (max discharge)")

# Variables
p_buy = cp.Variable(N, nonneg=True)   # power bought from grid
p_sell = cp.Variable(N, nonneg=True)  # power sold to grid
p_grid = p_buy - p_sell               # net grid power
p_batt = cp.Variable(N)               # battery power (positive = discharging)
q = cp.Variable(N)                    # battery state of charge

# Power balance: p_ld = p_grid + p_batt + p_pv
constraints = [
    p_ld == p_buy - p_sell + p_batt + p_pv
]

# Battery dynamics: q_{i+1} = q_i - (1/4) p_batt_i
# For i = 1, ..., N-1: q[i] = q[i-1] - 0.25 * p_batt[i-1]
for i in range(1, N):
    constraints.append(q[i] == q[i-1] - 0.25 * p_batt[i-1])

# Periodicity: q[0] = q[N-1] - 0.25 * p_batt[N-1]
constraints.append(q[0] == q[N-1] - 0.25 * p_batt[N-1])

# Battery constraints
constraints.append(q >= 0)
constraints.append(q <= Q)
constraints.append(p_batt >= -C)  # charging limit
constraints.append(p_batt <= D)   # discharging limit

# Grid cost: (1/4) * (R_buy^T p_buy - R_sell^T p_sell)
cost = (1/4) * (R_buy @ p_buy - R_sell @ p_sell)

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

print(f"\n--- Part (a) Results ---")
print(f"Optimal grid cost: ${prob.value:.2f}")

p_grid_opt = p_buy.value - p_sell.value
p_batt_opt = p_batt.value
q_opt = q.value

# Get dual variable for power balance (LMP)
# Note: sign may be flipped depending on constraint formulation
nu = constraints[0].dual_value
LMP = -4 * nu  # Convert to $/kWh (negate due to constraint form)

print(f"\n--- Part (b) LMP Analysis ---")
print(f"Average LMP: ${np.mean(LMP):.4f}/kWh")
print(f"Max LMP: ${np.max(LMP):.4f}/kWh")
print(f"Min LMP: ${np.min(LMP):.4f}/kWh")

# Part (c) - LMP payments
payment_load = np.dot(LMP, p_ld)
payment_pv = np.dot(LMP, p_pv)
payment_batt = np.dot(LMP, p_batt_opt)
payment_grid = np.dot(LMP, p_grid_opt)

print(f"\n--- Part (c) LMP Payments ---")
print(f"Load pays: ${payment_load:.2f}")
print(f"PV array is paid: ${payment_pv:.2f}")
print(f"Battery is paid: ${payment_batt:.2f}")
print(f"Grid is paid: ${payment_grid:.2f}")
print(f"Balance check (should be ~0): ${payment_load - payment_pv - payment_batt - payment_grid:.4f}")

# Plotting
fig_size = (12, 3)
xtick_vals = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]
xtick_labels = ["0:00", "2:00am", "4:00am", "6:00am", "8:00am", "10:00am",
                "12:00pm", "2:00pm", "4:00pm", "6:00pm", "8:00pm", "10:00pm", "12:00am"]
t = np.arange(N)

# Plot 1: Powers
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(t, p_grid_opt, 'b-', label='$p^{grid}$')
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_ylabel('Power (kW)')
axes[0, 0].set_title('Grid Power')
axes[0, 0].set_xticks(xtick_vals)
axes[0, 0].set_xticklabels(xtick_labels, rotation=45, ha='right')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, p_ld, 'r-', label='$p^{load}$')
axes[0, 1].set_ylabel('Power (kW)')
axes[0, 1].set_title('Load Power')
axes[0, 1].set_xticks(xtick_vals)
axes[0, 1].set_xticklabels(xtick_labels, rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t, p_pv, 'g-', label='$p^{pv}$')
axes[1, 0].set_ylabel('Power (kW)')
axes[1, 0].set_title('PV Power')
axes[1, 0].set_xticks(xtick_vals)
axes[1, 0].set_xticklabels(xtick_labels, rotation=45, ha='right')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, p_batt_opt, 'm-', label='$p^{batt}$')
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_ylabel('Power (kW)')
axes[1, 1].set_title('Battery Power (+ = discharge)')
axes[1, 1].set_xticks(xtick_vals)
axes[1, 1].set_xticklabels(xtick_labels, rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../latex/img/a20_14_powers.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Battery state of charge
plt.figure(figsize=(12, 4))
plt.plot(t, q_opt, 'b-', linewidth=2)
plt.axhline(Q, color='r', linestyle='--', alpha=0.5, label=f'Max capacity Q={Q} kWh')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.ylabel('State of Charge (kWh)')
plt.title('Battery State of Charge')
plt.xticks(xtick_vals, xtick_labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../latex/img/a20_14_soc.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: LMP vs grid prices
plt.figure(figsize=(12, 4))
plt.plot(t, LMP, 'b-', linewidth=2, label='LMP')
plt.plot(t, R_buy, 'r--', alpha=0.7, label='$R^{buy}$')
plt.plot(t, R_sell, 'g--', alpha=0.7, label='$R^{sell}$')
plt.ylabel('Price ($/kWh)')
plt.title('Locational Marginal Price vs Grid Prices')
plt.xticks(xtick_vals, xtick_labels, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../latex/img/a20_14_lmp.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved to ../latex/img/")
