import tensorflow as tf
import pandas as pd

import tensorflow as tf
import numpy as np

A = np.array([
    [2, 0],
    [1, 1],
    [0, 2]
], dtype=float)

print("=== VICTOR'S ORIGINAL MATRIX A ===")
print(A)


S = np.dot(A.T, A)
print("\n=== ASSOCIATED MATRIX S = A^T * A ===")
print(S)


is_symmetric = np.allclose(S, S.T)
print(f"Is matrix S strictly symmetric?: {is_symmetric}")


eigenvalues, eigenvectors = np.linalg.eigh(S)
print("\n=== EIGENVALUES OF S ===")
print(eigenvalues)
print("\n=== EIGENVECTORS OF S (V Columns) ===")
print(eigenvectors)


U, s, Vt = np.linalg.svd(A, full_matrices=True)

print("\n=== SVD FACTORIZATION (NUMPY) ===")
print("Matrix U (Left Singular Vectors):\n", U)
print("\nSingular Values (Sigma):\n", s)
print("\nMatrix V^T (Right Singular Vectors Transposed):\n", Vt)


df = pd.DataFrame(summary_data)
print("\n=== VICTOR'S EXPERIMENTAL RESULTS ===")
print(df.to_string(index=False))