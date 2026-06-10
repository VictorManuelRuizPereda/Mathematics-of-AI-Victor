import numpy as np

A = np.array([[3, 1], 
              [2, 2]], dtype=float)

S = np.dot(A.T, A)
values, vectors = np.linalg.eigh(S)

print("=== COVARIANCE SYSTEM ===")
print(S)
print("\nEIGENVALUES:", values)
print("EIGENVECTORS:\n", vectors)


def compute_bernoulli_kl(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

p_param = 0.4
kl_p_to_q = compute_bernoulli_kl(p_param, 1 - p_param)
kl_q_to_p = compute_bernoulli_kl(1 - p_param, p_param)

print("\n=== DIVERGENCE METRICS ===")
print(f"KL(P || Q): {kl_p_to_q:.4f}")
print(f"KL(Q || P): {kl_q_to_p:.4f}")
print(f"IDENTITY VERIFICATION: {np.isclose(kl_p_to_q, kl_q_to_p)}")