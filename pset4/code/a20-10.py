import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from storage_tradeoff_data import T, t, p, u

def solve_storage(Q, C, D, p, u):
    """
    Solve energy storage optimization problem.

    minimize    p^T (u + c)
    subject to  q_{t+1} = q_t + c_t,  t = 1,...,T-1
                q_1 = q_T + c_T  (cyclic)
                0 <= q_t <= Q
                -D <= c_t <= C
                u_t + c_t >= 0
    """
    T = len(p)
    c = cp.Variable(T)
    q = cp.Variable(T)

    constraints = [
        # Dynamics: q_{t+1} = q_t + c_t
        q[1:] == q[:-1] + c[:-1],
        # Cyclic: q_1 = q_T + c_T
        q[0] == q[-1] + c[-1],
        # Capacity constraints
        q >= 0,
        q <= Q,
        # Charge/discharge rate limits
        c >= -D,
        c <= C,
        # Net consumption nonnegative
        u.flatten() + c >= 0
    ]

    objective = cp.Minimize(p.flatten() @ (u.flatten() + c))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return {
        'c': c.value,
        'q': q.value,
        'cost': prob.value,
        'status': prob.status
    }

# Part (b): Solve with Q=35, C=D=3
print("=== Part (b): Q=35, C=D=3 ===")
Q, C, D = 35, 3, 3
result = solve_storage(Q, C, D, p, u)

cost_no_battery = float((p.T @ u)[0, 0])
cost_with_battery = result['cost']
savings = cost_no_battery - cost_with_battery

print(f"Cost without battery: {cost_no_battery:.2f}")
print(f"Cost with battery: {cost_with_battery:.2f}")
print(f"Savings: {savings:.2f} ({100*savings/cost_no_battery:.1f}%)")

# Plot for part (b)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(t/4, u, 'b-', linewidth=1.5)
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Usage $u_t$')
axes[0, 0].set_title('Usage')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t/4, p, 'r-', linewidth=1.5)
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Price $p_t$')
axes[0, 1].set_title('Electricity Price')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t/4, result['c'], 'g-', linewidth=1.5)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Charge $c_t$')
axes[1, 0].set_title('Battery Charging (>0) / Discharging (<0)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t/4, result['q'], 'm-', linewidth=1.5)
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Stored energy $q_t$')
axes[1, 1].set_title('Battery State of Charge')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../latex/img/a20_10_partb.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (b) plot saved.")

# Part (c): Trade-off curves
print("\n=== Part (c): Trade-off curves ===")
Q_values = np.linspace(0, 100, 50)

costs_CD3 = []
costs_CD1 = []

for Q_val in Q_values:
    # C = D = 3
    res = solve_storage(Q_val, 3, 3, p, u)
    costs_CD3.append(res['cost'])

    # C = D = 1
    res = solve_storage(Q_val, 1, 1, p, u)
    costs_CD1.append(res['cost'])

plt.figure(figsize=(10, 6))
plt.plot(Q_values, costs_CD3, 'b-', linewidth=2, label='C = D = 3')
plt.plot(Q_values, costs_CD1, 'r--', linewidth=2, label='C = D = 1')
plt.axhline(y=cost_no_battery, color='k', linestyle=':', label='No battery')
plt.xlabel('Storage capacity Q')
plt.ylabel('Total cost')
plt.title('Energy Storage Trade-off: Cost vs Capacity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../latex/img/a20_10_partc.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (c) plot saved.")

print(f"\nEndpoints interpretation:")
print(f"  Q=0: Cost = {costs_CD3[0]:.2f} (no storage benefit)")
print(f"  Q=100, C=D=3: Cost = {costs_CD3[-1]:.2f}")
print(f"  Q=100, C=D=1: Cost = {costs_CD1[-1]:.2f}")
print(f"  Higher C,D allows faster charge/discharge, better cost reduction")
