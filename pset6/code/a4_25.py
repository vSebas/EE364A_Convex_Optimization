"""
Problem A4.25: Probability bounds

Find min and max of prob(X4 = 1) given marginal and conditional probabilities.
"""
import numpy as np
import cvxpy as cp

# Create a 4-dimensional array of joint probabilities
# p[i,j,k,l] = prob(X1=i, X2=j, X3=k, X4=l) for i,j,k,l in {0,1}
p = cp.Variable((2, 2, 2, 2), nonneg=True) # so each p is >= 0

# Constraints from the given data
constraints = []

# 1. Probabilities sum to 1
constraints.append(cp.sum(p) == 1)

# 2. prob(X1 = 1) = 0.9
# Sum over all X2, X3, X4 where X1=1
constraints.append(cp.sum(p[1, :, :, :]) == 0.9)

# 3. prob(X2 = 1) = 0.9
constraints.append(cp.sum(p[:, 1, :, :]) == 0.9)

# 4. prob(X3 = 1) = 0.1
constraints.append(cp.sum(p[:, :, 1, :]) == 0.1)

# 5. prob(X1 = 1, X4 = 0 | X3 = 1) = 0.7
# This means: prob(X1=1, X4=0, X3=1) / prob(X3=1) = 0.7
# So: prob(X1=1, X4=0, X3=1) = 0.7 * 0.1 = 0.07
constraints.append(cp.sum(p[1, :, 1, 0]) == 0.7 * 0.1)

# 6. prob(X4 = 1 | X2 = 1, X3 = 0) = 0.6
# This means: prob(X4=1, X2=1, X3=0) / prob(X2=1, X3=0) = 0.6
# prob(X2=1, X3=0) = prob(X2=1) - prob(X2=1, X3=1)
#                  = 0.9 - prob(X2=1, X3=1)
# We need: prob(X4=1, X2=1, X3=0) = 0.6 * prob(X2=1, X3=0)
# Rearranging: prob(X4=1, X2=1, X3=0) - 0.6 * prob(X2=1, X3=0) = 0
# i.e., sum over X1 of p[X1, 1, 0, 1] = 0.6 * sum over X1, X4 of p[X1, 1, 0, X4]
constraints.append(
    cp.sum(p[:, 1, 0, 1]) == 0.6 * cp.sum(p[:, 1, 0, :])
)

# Objective: prob(X4 = 1)
prob_X4_1 = cp.sum(p[:, :, :, 1])

# Minimize prob(X4 = 1)
prob_min = cp.Problem(cp.Minimize(prob_X4_1), constraints)
prob_min.solve()
p_min = prob_min.value
print(f"Minimum prob(X4 = 1) = {p_min:.6f}")

# Maximize prob(X4 = 1)
prob_max = cp.Problem(cp.Maximize(prob_X4_1), constraints)
prob_max.solve()
p_max = prob_max.value
print(f"Maximum prob(X4 = 1) = {p_max:.6f}")

# Print the joint distribution for min case
print("\nJoint distribution achieving minimum:")
p_array = p.value
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                if p_array[i, j, k, l] > 1e-6:
                    print(f"  P(X1={i}, X2={j}, X3={k}, X4={l}) = {p_array[i,j,k,l]:.6f}")

# Verify constraints
print("\nVerification of constraints:")
print(f"  prob(X1=1) = {np.sum(p_array[1,:,:,:]):.4f} (should be 0.9)")
print(f"  prob(X2=1) = {np.sum(p_array[:,1,:,:]):.4f} (should be 0.9)")
print(f"  prob(X3=1) = {np.sum(p_array[:,:,1,:]):.4f} (should be 0.1)")
print(f"  prob(X1=1,X4=0|X3=1) = {np.sum(p_array[1,:,1,0])/np.sum(p_array[:,:,1,:]):.4f} (should be 0.7)")
print(f"  prob(X4=1|X2=1,X3=0) = {np.sum(p_array[:,1,0,1])/np.sum(p_array[:,1,0,:]):.4f} (should be 0.6)")
