"""
Problem A19.10: Scheduling

Find optimal trade-off between cost C and completion time T.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Load data
from opt_schedule_data import n, alpha, m, P, visualize_schedule

print(f"Number of tasks: n = {n}")
print(f"Alpha (cost coefficients): {alpha}")
print(f"m (minimum durations): {m}")
print(f"Precedence constraints: {P}")

# Part (a): Formulation
# Variables: s_i (start time), f_i (finish time) for each task
# Duration: d_i = f_i - s_i >= m_i
# Cost: phi_i(d_i) = alpha_i / (d_i - m_i) if d_i > m_i, else infinity
# Total cost: C = sum_i phi_i(d_i)
# Completion time: T = max_i f_i
#
# Reformulation using d_i:
# minimize sum_i alpha_i / (d_i - m_i)
# subject to: d_i >= m_i + epsilon (small positive)
#             s_i >= 0
#             f_i = s_i + d_i
#             s_j >= f_i for (i,j) in P (precedence)
#             f_i <= T
#
# For fixed T, this is a convex problem (sum of 1/x terms is convex for x > 0)

def solve_schedule(T_max, verbose=False):
    """Solve the scheduling problem for a given completion time T."""
    s = cp.Variable(n)  # start times
    d = cp.Variable(n)  # durations

    # Finish times
    f = s + d

    # Cost: sum of alpha_i / (d_i - m_i)
    # This is convex in d for d > m
    cost = cp.sum(cp.multiply(alpha, cp.inv_pos(d - m)))

    constraints = [
        s >= 0,              # Non-negative start times
        d >= m + 1e-4,       # Duration > minimum (small margin for numerical stability)
        f <= T_max,          # Completion time constraint
    ]

    # Precedence constraints: s_j >= f_i for each (i,j) in P
    for i, j in P:
        constraints.append(s[j] >= f[i])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    if prob.status in ['optimal', 'optimal_inaccurate']:
        if verbose:
            print(f"T = {T_max:.2f}, C = {prob.value:.4f}")
        return prob.value, s.value, (s.value + d.value)
    else:
        return None, None, None

# Part (b): Compute Pareto frontier
T_values = np.linspace(10, 30, 50)
C_values = []
valid_T = []

for T in T_values:
    C, _, _ = solve_schedule(T)
    if C is not None:
        C_values.append(C)
        valid_T.append(T)

valid_T = np.array(valid_T)
C_values = np.array(C_values)

# Plot trade-off curve
plt.figure(figsize=(10, 6))
plt.plot(valid_T, C_values, 'b-', linewidth=2)
plt.xlabel('Completion time T')
plt.ylabel('Total cost C')
plt.title('Optimal Trade-off Curve: Cost vs Completion Time')
plt.grid(True, alpha=0.3)
plt.xlim([10, 30])
plt.savefig('../latex/img/a19_10_tradeoff.png', dpi=150)
plt.close()

print(f"\nPareto frontier computed for T in [{min(valid_T):.1f}, {max(valid_T):.1f}]")
print(f"Cost range: [{min(C_values):.2f}, {max(C_values):.2f}]")

# Visualize schedule for T = 20
T_example = 20
C_opt, s_opt, f_opt = solve_schedule(T_example, verbose=True)

print(f"\nOptimal schedule for T = {T_example}:")
print(f"Total cost C = {C_opt:.4f}")
print("\nTask schedule:")
print("Task | Start | Finish | Duration | Cost")
print("-" * 45)
for i in range(n):
    d_i = f_opt[i] - s_opt[i]
    cost_i = alpha[i] / (d_i - m[i])
    print(f"  {i+1:2d} | {s_opt[i]:5.2f} | {f_opt[i]:6.2f} | {d_i:8.2f} | {cost_i:.4f}")

# Visualize the schedule
visualize_schedule(s_opt, f_opt, save='../latex/img/a19_10_schedule.png')
