"""
Problem A17.16: Efficient Solution of Basic Portfolio Optimization Problem

maximize   mu^T w - (lambda/2) w^T Sigma w
subject to 1^T w = 1

where Sigma = F Q F^T + D (factor model)
- n = 2500 assets, k = 30 factors
- F: n x k factor loading matrix
- Q: k x k factor covariance (SPD)
- D: n x n diagonal (idiosyncratic risk)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

np.random.seed(0)

# Problem dimensions
n = 2500
k = 30

# Generate random data (with conditioning as per hint)
F = np.random.randn(n, k)
Q = np.random.randn(k, k)
Q = Q @ Q.T + 0.1 * np.eye(k)  # Add positive multiple of I for conditioning
D_diag = np.abs(np.random.randn(n)) + 0.1  # Positive diagonal + conditioning
mu = np.random.randn(n)

# Form full Sigma for dense method
D = np.diag(D_diag)
Sigma = F @ Q @ F.T + D

print("=" * 60)
print("Part (c): Comparing dense vs. structured methods")
print(f"n = {n}, k = {k}")
print("=" * 60)

# === Method (a): Dense method ===
# Treats Sigma as a full n x n matrix. Complexity: O(n^3)

start_dense = time.time()

KKT_dense = np.zeros((n + 1, n + 1))
KKT_dense[:n, :n] = Sigma
KKT_dense[:n, n] = 1
KKT_dense[n, :n] = 1
rhs_dense = np.zeros(n + 1)
rhs_dense[:n] = mu
rhs_dense[n] = 1

sol_dense = np.linalg.solve(KKT_dense, rhs_dense)
w_dense = sol_dense[:n]

time_dense = time.time() - start_dense
print(f"\nMethod (a) - Dense: {time_dense:.4f} seconds")

# === Method (b1): Sparse augmented KKT system ===
# Introduces y = F^T w and solves the (n+k+1) x (n+k+1) sparse system:
# [D, FQ, 1; -F^T, I, 0; 1^T, 0, 0] [w; y; nu] = [mu; 0; 1]
# Complexity: O(nk^2 + k^3)
#
# This approach follows the hint in the problem: "you'll need to recast it as sparse."
# D is stored as a sparse diagonal, making the full KKT matrix sparse.

start_sparse = time.time()

D_sparse = sp.diags(D_diag, format='csr')          # (n,n)
FQ = F @ Q                                          # (n,k) dense
FQ_sparse = sp.csr_matrix(FQ)                       # (n,k)
ones_n = sp.csr_matrix(np.ones((n, 1)))             # (n,1)
zeros_k1 = sp.csr_matrix((k, 1))                    # (k,1)
zeros_1k = sp.csr_matrix((1, k))                    # (1,k)
zero_11 = sp.csr_matrix([[0.0]])                    # (1,1)

KKT_sparse = sp.bmat([
    [D_sparse,            FQ_sparse,  ones_n],
    [sp.csr_matrix(-F.T), sp.eye(k),  zeros_k1],
    [sp.csr_matrix(np.ones((1, n))), zeros_1k, zero_11]
], format='csr')

rhs_sparse = np.concatenate([mu, np.zeros(k), [1.0]])
sol_sparse = spla.spsolve(KKT_sparse, rhs_sparse)
w_sparse = sol_sparse[:n]

time_sparse = time.time() - start_sparse
print(f"Method (b1) - Sparse KKT: {time_sparse:.4f} seconds (speedup: {time_dense/time_sparse:.1f}x)")

# === Method (b2): Woodbury identity ===
# Uses Sigma^{-1} = D^{-1} - D^{-1} F (Q^{-1} + F^T D^{-1} F)^{-1} F^T D^{-1}
# Complexity: O(nk^2 + k^3)
#
# COMPARISON OF APPROACHES:
# - Sparse KKT: Forms (n+k+1) x (n+k+1) system, uses sparse LU factorization.
#   Follows the problem hint but has overhead from sparse matrix construction.
# - Woodbury: Only works with k x k matrices for the inner solve.
#   Much faster in practice (~100x) because it avoids the full system entirely.
# Both achieve O(nk^2 + k^3) complexity, but Woodbury has smaller constants.

start_woodbury = time.time()

D_inv = 1.0 / D_diag
Q_inv = np.linalg.inv(Q)
F_scaled = F * D_inv[:, np.newaxis]  # D^{-1} F
M = Q_inv + F.T @ F_scaled           # Q^{-1} + F^T D^{-1} F (k x k)
M_chol = np.linalg.cholesky(M)

def apply_Sigma_inv(v):
    """Apply Sigma^{-1} to vector v using Woodbury."""
    D_inv_v = D_inv * v
    z = np.linalg.solve(M_chol.T, np.linalg.solve(M_chol, F.T @ D_inv_v))
    return D_inv_v - F_scaled @ z

q = apply_Sigma_inv(mu)
p = apply_Sigma_inv(np.ones(n))
nu = (np.ones(n) @ q - 1) / (np.ones(n) @ p)
w_woodbury = q - nu * p

time_woodbury = time.time() - start_woodbury
print(f"Method (b2) - Woodbury: {time_woodbury:.4f} seconds (speedup: {time_dense/time_woodbury:.1f}x)")

# === Verify agreement ===
print(f"\nRelative differences from dense solution:")
print(f"  Sparse KKT: {np.linalg.norm(w_dense - w_sparse) / np.linalg.norm(w_dense):.2e}")
print(f"  Woodbury:   {np.linalg.norm(w_dense - w_woodbury) / np.linalg.norm(w_dense):.2e}")

print(f"\nConstraint check (1^T w = 1):")
print(f"  Dense:      {np.sum(w_dense):.10f}")
print(f"  Sparse KKT: {np.sum(w_sparse):.10f}")
print(f"  Woodbury:   {np.sum(w_woodbury):.10f}")

obj_dense = mu @ w_dense - 0.5 * w_dense @ Sigma @ w_dense
obj_sparse = mu @ w_sparse - 0.5 * w_sparse @ Sigma @ w_sparse
obj_woodbury = mu @ w_woodbury - 0.5 * w_woodbury @ Sigma @ w_woodbury
print(f"\nObjective value:")
print(f"  Dense:      {obj_dense:.6f}")
print(f"  Sparse KKT: {obj_sparse:.6f}")
print(f"  Woodbury:   {obj_woodbury:.6f}")
