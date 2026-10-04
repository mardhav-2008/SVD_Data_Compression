import time

import numpy as np

from core.svd import Calculate_Eigenvalues, Calculate_SVD


def Calculate_Rank(matrix: np.ndarray) -> int:
    eigenvalues = Calculate_Eigenvalues(matrix.T @ matrix)
    tolerance = 1e-10
    return int(np.count_nonzero(np.abs(eigenvalues) > tolerance))


def low_rank(matrix: np.ndarray, k: int):
    U, Sigma, V = Calculate_SVD(matrix)

    U_k = U[:, :k]
    Sigma_k = Sigma[:k, :k]
    V_k = V[:, :k]

    return U_k, Sigma_k, V_k


def Compare_matrix(matrix: np.ndarray, reconstructed_matrix: np.ndarray):
    return np.allclose(matrix, reconstructed_matrix, atol=1e-2)


def Approximate(A: np.ndarray):
    start_time = time.perf_counter()

    A_size = A.shape[0] * A.shape[1]

    U, Sigma, V = Calculate_SVD(A)

    tolerance = 1e-10
    rank = np.count_nonzero(Sigma > tolerance)

    for i in range(1, rank + 1):
        U_k = U[:, :i]
        S_k = Sigma[:i, :i]
        V_k = V[:, :i]

        reconstructed = U_k @ S_k @ V_k.T

        compressed_size = U_k.size + S_k.size + V_k.size

        if Compare_matrix(A, reconstructed):
            break

        if compressed_size >= A_size:
            i -= 1

            U_k = U[:, :i]
            S_k = Sigma[:i, :i]
            V_k = V[:, :i]

            reconstructed = U_k @ S_k @ V_k.T
            compressed_size = U_k.size + S_k.size + V_k.size
            break

    else:
        raise ValueError("Reconstruction was not found")

    relative_error = np.linalg.norm(A - reconstructed, ord="fro") / np.linalg.norm(
        A, ord="fro"
    )

    error_percent = relative_error * 100
    compression_percent = 100 * (1 - compressed_size / A_size)

    end_time = time.perf_counter()

    print(
        f"k = {i}\n"
        f"Original size = {A_size}\n"
        f"Compressed size = {compressed_size}\n"
        f"Compression percentage = {compression_percent:.2f}%\n"
        f"Relative error = {error_percent:.6f}%\n"
        f"Time taken = {end_time - start_time:.4f} seconds\n"
    )

    return reconstructed
