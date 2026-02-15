import numpy as np
import cvxpy as cp

# Problem data
A_bar = np.array([
    [60, 45, -8],
    [90, 30, -30],
    [0, -8, -4],
    [30, 10, -10]
], dtype=float)

d = 0.05  # radius
R = d * np.ones((4, 3))

b = np.array([-6, -3, 18, -9], dtype=float)

m, n = A_bar.shape

# Nominal least-squares solution
x_ls = np.linalg.lstsq(A_bar, b, rcond=None)[0]
print("Nominal LS solution x_ls:", x_ls)

# Robust least-squares solution using QP formulation
x = cp.Variable(n)
y = cp.Variable(m)
z = cp.Variable(n)

constraints = [
    A_bar @ x + R @ z - b <= y,
    A_bar @ x - R @ z - b >= -y,
    x <= z,
    x + z >= 0  # equivalent to -z <= x
]

prob = cp.Problem(cp.Minimize(cp.norm(y, 2)), constraints)
prob.solve()

x_rls = x.value
print("Robust LS solution x_rls:", x_rls)

# Alternative formulation: minimize ||abs(A_bar*x - b) + R*abs(x)||_2
x2 = cp.Variable(n)
t = cp.Variable(m)

constraints2 = [
    cp.abs(A_bar @ x2 - b) + R @ cp.abs(x2) <= t
]

prob2 = cp.Problem(cp.Minimize(cp.norm(t, 2)), constraints2)
prob2.solve()

print("Robust LS solution (alt):", x2.value)

# Compute residual norms
def nominal_residual(x):
    return np.linalg.norm(A_bar @ x - b)

def worst_case_residual(x):
    r = A_bar @ x - b
    return np.linalg.norm(np.abs(r) + R @ np.abs(x))

nom_res_ls = nominal_residual(x_ls)
nom_res_rls = nominal_residual(x_rls)
wc_res_ls = worst_case_residual(x_ls)
wc_res_rls = worst_case_residual(x_rls)

print("\n--- Results ---")
print(f"Nominal residual norm (LS solution): {nom_res_ls:.2f}")
print(f"Nominal residual norm (Robust solution): {nom_res_rls:.2f}")
print(f"Worst-case residual norm (LS solution): {wc_res_ls:.2f}")
print(f"Worst-case residual norm (Robust solution): {wc_res_rls:.2f}")
