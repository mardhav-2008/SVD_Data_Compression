import numpy as np


def Householder_Reflection(matrix: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.sum(matrix**2))

    if np.isclose(norm, 0):
        return np.eye(matrix.shape[0])

    y = np.zeros(matrix.shape, float)
    y[0] = norm

    u = matrix - y
    denominator = u.T @ u

    if np.isclose(denominator, 0):
        return np.eye(matrix.shape[0])

    return np.eye(matrix.shape[0]) - (2 * np.outer(u, u) / denominator)


def Tridiagonalize(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    tridiagonal = matrix.copy()

    n = matrix.shape[0]
    Q_tri = np.eye(n)

    for i in range(n - 2):
        x = tridiagonal[i + 1 :, i]

        H = Householder_Reflection(x)

        Q = np.eye(n)
        Q[i + 1 :, i + 1 :] = H

        tridiagonal = Q @ tridiagonal @ Q.T
        Q_tri = Q_tri @ Q

    return tridiagonal, Q_tri


def QR_Decomposition(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    n = matrix.shape[0]

    R = matrix.copy()
    Q = np.eye(n)

    for i in range(n - 1):
        x = R[i:, i]

        H_small = Householder_Reflection(x)

        H = np.eye(n)
        H[i:, i:] = H_small

        R = H @ R
        Q = Q @ H

    return Q, R


def RQ_Iterator(
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    Q_qr = np.eye(T.shape[0])

    for iterations in range(1000):
        if np.allclose(
            T,
            np.diag(np.diag(T)),
            atol=1e-10,
        ):
            print("QR iterations:", iterations)
            break

        Q, R = QR_Decomposition(T)

        T = R @ Q

        Q_qr = Q_qr @ Q

    T[np.abs(T) < 1e-6] = 0

    return T, Q_qr


def Calculate_Eigenvalues(
    matrix: np.ndarray,
) -> np.ndarray:

    T, _ = Tridiagonalize(matrix)
    T, _ = RQ_Iterator(T)

    return np.diag(T).copy()


def Calculate_Eigenvectors(
    matrix: np.ndarray,
) -> np.ndarray:

    T, Q_tri = Tridiagonalize(matrix)
    T, Q_qr = RQ_Iterator(T)

    return Q_tri @ Q_qr


def Calculate_SVD(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if matrix.ndim != 2:
        raise ValueError("Matrix should be a 2D matrix")

    m, _ = matrix.shape

    # AᵀA
    symmetric_matrix = matrix.T @ matrix

    # Eigenvalues and eigenvectors of AᵀA
    eigenvalues = Calculate_Eigenvalues(symmetric_matrix)
    V = Calculate_Eigenvectors(symmetric_matrix)

    # Numerical cleanup
    eigenvalues[np.abs(eigenvalues) < 1e-10] = 0

    # Sort eigenvalues in descending order
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    V = V[:, order]

    # Singular values
    singular_values = np.sqrt(np.maximum(eigenvalues, 0))

    # Keep only non-zero singular values
    tolerance = 1e-10
    nonzero = singular_values > tolerance

    singular_values = singular_values[nonzero]
    V = V[:, nonzero]

    # Number of non-zero singular values
    r = len(singular_values)

    # U is m × r
    U = np.zeros((m, r))

    # uᵢ = Avᵢ / σᵢ
    for i in range(r):
        v = V[:, i]
        U[:, i] = matrix @ v / singular_values[i]

    # Reduced Σ
    Sigma = np.diag(singular_values)

    return U, Sigma, V
