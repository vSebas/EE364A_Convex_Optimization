import cvxpy as cp
import numpy as np

def solve_qp(u1, u2):
    x = cp.Variable(2)
    x1, x2 = x[0], x[1]

    # Objective: x1^2 + 2*x2^2 - x1*x2 - x1
    # Rewrite as (1/2)*x^T*P*x + q^T*x where:
    # P = [[2, -1], [-1, 4]], q = [-1, 0]
    # (1/2)*x^T*P*x = x1^2 + 2*x2^2 - x1*x2
    P = np.array([[2, -1], [-1, 4]])
    q = np.array([-1, 0])
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)

    constraints = [
        x1 + 2*x2 <= u1,
        x1 - 4*x2 <= u2,
        5*x1 + 76*x2 <= 1
    ]

    problem = cp.Problem(objective, constraints)
    p_star = problem.solve()

    lambda1 = constraints[0].dual_value
    lambda2 = constraints[1].dual_value
    lambda3 = constraints[2].dual_value

    return {
        'x1': x1.value,
        'x2': x2.value,
        'p_star': p_star,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'lambda3': lambda3
    }

def verify_kkt(sol, u1, u2):
    x1, x2 = sol['x1'], sol['x2']
    l1, l2, l3 = sol['lambda1'], sol['lambda2'], sol['lambda3']

    print("\n=== KKT Conditions Verification ===")

    # 1. Primal constraints: f_i <= 0
    f1 = x1 + 2*x2 - u1
    f2 = x1 - 4*x2 - u2
    f3 = 5*x1 + 76*x2 - 1
    print(f"1. Primal constraints (f_i <= 0):")
    print(f"   f1 = x1 + 2*x2 - u1 = {f1:.6e}")
    print(f"   f2 = x1 - 4*x2 - u2 = {f2:.6e}")
    print(f"   f3 = 5*x1 + 76*x2 - 1 = {f3:.6e}")

    # 2. Dual constraints: lambda >= 0
    print(f"2. Dual feasibility (lambda >= 0):")
    print(f"   lambda1 = {l1:.6f}")
    print(f"   lambda2 = {l2:.6f}")
    print(f"   lambda3 = {l3:.6f}")

    # 3. Complementary slackness: lambda_i * f_i = 0
    cs1 = l1 * f1
    cs2 = l2 * f2
    cs3 = l3 * f3
    print(f"3. Complementary slackness (lambda_i * f_i ~0):")
    print(f"   lambda1 * f1 = {cs1:.6e}")
    print(f"   lambda2 * f2 = {cs2:.6e}")
    print(f"   lambda3 * f3 = {cs3:.6e}")

    # 4. Stationarity: gradient of Lagrangian w.r.t x = 0
    # L = x1^2 + 2*x2^2 - x1*x2 - x1 + l1*(x1 + 2*x2 - u1) + l2*(x1 - 4*x2 - u2) + l3*(5*x1 + 76*x2 - 1)
    # dL/dx1 = 2*x1 - x2 - 1 + l1 + l2 + 5*l3 = 0
    # dL/dx2 = 4*x2 - x1 + 2*l1 - 4*l2 + 76*l3 = 0
    grad_x1 = 2*x1 - x2 - 1 + l1 + l2 + 5*l3
    grad_x2 = 4*x2 - x1 + 2*l1 - 4*l2 + 76*l3
    print(f"4. Gradient ~0:")
    print(f"   dL/dx1 = {grad_x1:.6e}")
    print(f"   dL/dx2 = {grad_x2:.6e}")

# ============================================================
# Part (a): Solve with u1 = -2, u2 = -3
# ============================================================
print("=" * 60)
print("Part (a): Solve QP with u1 = -2, u2 = -3")
print("=" * 60)

u1_base, u2_base = -2, -3
sol_base = solve_qp(u1_base, u2_base)

print(f"\nOptimal primal variables:")
print(f"  x1* = {sol_base['x1']:.6f}")
print(f"  x2* = {sol_base['x2']:.6f}")

print(f"\nOptimal dual variables:")
print(f"  lambda1* = {sol_base['lambda1']:.6f}")
print(f"  lambda2* = {sol_base['lambda2']:.6f}")
print(f"  lambda3* = {sol_base['lambda3']:.6f}")

print(f"\nOptimal objective value:")
print(f"  p* = {sol_base['p_star']:.6f}")

verify_kkt(sol_base, u1_base, u2_base)

# ============================================================
# Part (b): Perturbed problems
# ============================================================
print("\n" + "=" * 60)
print("Part (b): Perturbed QP solutions")
print("=" * 60)

# Perturbation values
deltas = [-0.1, 0, 0.1]

# For sensitivity analysis with constraints g(x) <= u:
# p*(u + delta) approx p*(u) - lambda^T * delta
# The negative sign is because relaxing constraints (larger u) decreases optimal value

print("\nSensitivity analysis: p*_pred = p* - lambda1*delta1 - lambda2*delta2")
print("(For constraints gi(x) <= ui, sensitivity is dp*/dui = -lambda_i)")
print()

# Table header
print(f"{'delta1':>8} {'delta2':>8} {'p*_pred':>12} {'p*_exact':>12} {'p*_pred <= p*_exact':>20}")
print("-" * 64)

for d1 in deltas:
    for d2 in deltas:
        u1_pert = u1_base + d1
        u2_pert = u2_base + d2

        # Solve perturbed problem
        sol_pert = solve_qp(u1_pert, u2_pert)
        p_exact = sol_pert['p_star']

        # Predicted value using sensitivity (first-order approximation)
        # For constraint gi(x) <= ui, sensitivity is dp*/dui = -lambda_i
        p_pred = sol_base['p_star'] - sol_base['lambda1']*d1 - sol_base['lambda2']*d2

        # Check inequality
        inequality_holds = p_pred <= p_exact + 1e-6  # small tolerance

        print(f"{d1:>8.1f} {d2:>8.1f} {p_pred:>12.6f} {p_exact:>12.6f} {str(inequality_holds):>20}")

# print()
# print("Note: p*_pred <= p*_exact should hold because the dual function")
# print("provides a lower bound on the optimal value, and the first-order")
# print("approximation based on Lagrange multipliers underestimates for")
# print("convex problems when constraints are tightened.")
