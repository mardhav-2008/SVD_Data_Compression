from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class SVD:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix
        self.U, self.Sigma, self.VT = np.linalg.svd(matrix, full_matrices=False)

    def approximate(self, k: int):
        U_k = self.U[:, :k]
        Sigma_k = np.diag(self.Sigma[:k])
        VT_k = self.VT[:k, :]

        reconstructed = U_k @ Sigma_k @ VT_k

        original_size = self.matrix.size
        reconstructed_size = U_k.size + Sigma_k.size + VT_k.size

        error = np.linalg.norm(self.matrix - reconstructed) / np.linalg.norm(
            self.matrix
        )

        compression_ratio = reconstructed_size / original_size
        error_percent = error * 100

        return reconstructed, compression_ratio, error_percent


image_path = Path("data/raw/image/2.jpeg")

image = Image.open(image_path).convert("RGB")
image_array = np.asarray(image, dtype=float)

R = image_array[:, :, 0]
G = image_array[:, :, 1]
B = image_array[:, :, 2]

R_SVD = SVD(R)
G_SVD = SVD(G)
B_SVD = SVD(B)

R_Compression = []
G_Compression = []
B_Compression = []

R_Error = []
G_Error = []
B_Error = []

for k in range(0, 420, 20):
    _, compression, error = R_SVD.approximate(k)
    R_Compression.append(compression)
    R_Error.append(error)

    _, compression, error = G_SVD.approximate(k)
    G_Compression.append(compression)
    G_Error.append(error)

    _, compression, error = B_SVD.approximate(k)
    B_Compression.append(compression)
    B_Error.append(error)

plt.plot(R_Compression, R_Error, label="Red")
plt.plot(G_Compression, G_Error, label="Green")
plt.plot(B_Compression, B_Error, label="Blue")

plt.xlabel("Storage Ratio")
plt.ylabel("Error (%)")
plt.title("SVD Compression vs Reconstruction Error")
plt.legend()
plt.grid()
plt.show()

for k in [40, 80, 120, 160, 200, 300, 400]:
    R_reconstructed, _, _ = R_SVD.approximate(k)
    G_reconstructed, _, _ = G_SVD.approximate(k)
    B_reconstructed, _, _ = B_SVD.approximate(k)

    reconstructed_image = np.stack(
        [R_reconstructed, G_reconstructed, B_reconstructed], axis=2
    )

    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    output_image = Image.fromarray(reconstructed_image)

    output_path = Path(f"data/generated/image/reconstructed_k{k}.png")

    output_image.save(output_path)
    print(f"k={k}: {output_path}")
