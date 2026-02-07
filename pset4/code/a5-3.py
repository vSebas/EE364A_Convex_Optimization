import numpy as np
import cvxpy as cp

def solve_relative_entropy(A, b, y, solver_primal=cp.SCS, solver_dual=cp.SCS):
    """
    Solve:
        minimize    sum_k x_k log(x_k / y_k)
        subject to  A x = b
                    1^T x = 1
                    x >= 0
    and also solve the dual:
        maximize    b^T z - log( sum_k y_k * exp(a_k^T z) )

    Parameters
    ----------
    A : (m, n) ndarray
    b : (m,) ndarray
    y : (n,) ndarray, strictly positive

    Returns dict with primal/dual solutions and consistency checks.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, n = A.shape
    assert b.shape == (m,)
    assert y.shape == (n,)
    if np.any(y <= 0):
        raise ValueError("y must be strictly positive (domain of relative entropy).")

    # ---------- Primal ----------
    x = cp.Variable(n)

    # Objective: sum x_k log(x_k / y_k) = sum rel_entr(x_k, y_k)
    obj_primal = cp.Minimize(cp.sum(cp.rel_entr(x, y)))

    constraints = [
        A @ x == b,
        cp.sum(x) == 1,
        x >= 0
    ]

    prob_primal = cp.Problem(obj_primal, constraints)
    p_star = prob_primal.solve(solver=solver_primal)

    x_star = x.value

    # ---------- Dual ----------
    z = cp.Variable(m)

    # log( sum_k y_k * exp(a_k^T z) )
    # where (a_k^T z) are entries of A^T z
    log_sum = cp.log_sum_exp(cp.log(y) + (A.T @ z))

    obj_dual = cp.Maximize(b @ z - log_sum)
    prob_dual = cp.Problem(obj_dual)
    d_star = prob_dual.solve(solver=solver_dual)

    z_star = z.value

    # Recover x from dual (softmax-like formula)
    # x_k = y_k exp(a_k^T z) / sum_j y_j exp(a_j^T z)
    Az = A.T @ z_star
    w = np.log(y) + Az
    w = w - np.max(w)                  # stabilize
    expw = np.exp(w)
    x_from_dual = expw / np.sum(expw)

    # Diagnostics
    primal_feas = {
        "Ax_minus_b_norm": float(np.linalg.norm(A @ x_star - b)),
        "sumx_minus_1": float(np.sum(x_star) - 1.0),
        "min_x": float(np.min(x_star)),
    }
    gap = None
    if p_star is not None and d_star is not None:
        gap = float(p_star - d_star)

    return {
        "status_primal": prob_primal.status,
        "p_star": float(p_star),
        "x_star": x_star,
        "status_dual": prob_dual.status,
        "d_star": float(d_star),
        "z_star": z_star,
        "x_from_dual": x_from_dual,
        "duality_gap": gap,
        "primal_feas": primal_feas,
    }


if __name__ == "__main__":
    np.random.seed(42)
    n, m = 4, 1
    A = np.random.randn(m, n)
    b = np.array([0.5])
    y = np.abs(np.random.randn(n)) + 0.1

    result = solve_relative_entropy(A, b, y)

    print("=== A5.3 Results ===")
    print(f"p* = {result['p_star']:.6f}")
    print(f"d* = {result['d_star']:.6f}")
    print(f"Duality gap: {result['duality_gap']:.2e}")
    print(f"x* (primal): {np.round(result['x_star'], 6)}")
    print(f"x* (from dual): {np.round(result['x_from_dual'], 6)}")
    print(f"z* = {result['z_star']}")
    print(f"Strong duality holds: {abs(result['duality_gap']) < 1e-5}")
