import numpy as np

np.random.seed(1)
n = 20
pbar = np.ones((n, 1)) * 0.03 
pbar += np.r_[np.random.rand(n - 1, 1), np.zeros((1, 1))] * 0.12
S = np.random.randn(n, n)
S = S.T @ S
S = S / max(np.abs(np.diag(S))) * 0.2
S[:, -1] = np.zeros(n)
S[-1, :] = np.zeros(n)
x_unif = np.ones((n, 1)) / n