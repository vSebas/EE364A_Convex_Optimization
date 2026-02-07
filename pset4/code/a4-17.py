import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Problem data
np.random.seed(0)
m, n = 300, 100
A, c = np.random.rand(m, n), -np.random.rand(n)
b = A @ np.ones(n) / 2

# LP relaxation
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, 0 <= x, x <= 1])
prob.solve()
x_rlx, L = x.value, prob.value

# Threshold rounding
t_vals = np.linspace(0, 1, 100)
objs = np.array([c @ (x_rlx >= t) for t in t_vals])
viols = np.array([np.max(A @ (x_rlx >= t) - b) for t in t_vals])
feas = viols <= 1e-6

# Best feasible
best_idx = np.where(feas)[0][np.argmin(objs[feas])]
U, t_best = objs[best_idx], t_vals[best_idx]

print(f"L = {L:.2f}, U = {U:.2f}, gap = {U-L:.2f}, t* = {t_best:.2f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax1.plot(t_vals, objs, 'b-')
ax1.axhline(L, color='g', ls='--', label=f'L={L:.1f}')
ax1.axhline(U, color='r', ls='--', label=f'U={U:.1f}')
ax1.fill_between(t_vals, objs.min(), objs.max(), where=feas, alpha=0.2, color='g')
ax1.set_ylabel(r'Objective $c^T\hat{x}$'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(t_vals, viols, 'b-'); ax2.axhline(0, color='k', lw=0.5)
ax2.fill_between(t_vals, viols.min(), 0, where=feas, alpha=0.2, color='g', label='Feasible')
ax2.set_xlabel('Threshold $t$'); ax2.set_ylabel('Max violation'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../latex/img/a4_17.png', dpi=150)
