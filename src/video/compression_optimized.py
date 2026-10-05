from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
video_path = PROJECT_ROOT / "data" / "raw" / "video" / "1.mp4"

output_dir = PROJECT_ROOT / "data" / "generated1" / "video"
output_dir.mkdir(parents=True, exist_ok=True)


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


for k in [100, 200, 300, 400]:
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = output_dir / f"reconstructed_k{k}.mp4"

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    output_video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        R = frame[:, :, 0]
        G = frame[:, :, 1]
        B = frame[:, :, 2]

        R_SVD = SVD(R)
        G_SVD = SVD(G)
        B_SVD = SVD(B)

        R_reconstructed, _, _ = R_SVD.approximate(k)
        G_reconstructed, _, _ = G_SVD.approximate(k)
        B_reconstructed, _, _ = B_SVD.approximate(k)

        reconstructed_frame = np.stack(
            [R_reconstructed, G_reconstructed, B_reconstructed], axis=2
        )

        reconstructed_frame = np.clip(reconstructed_frame, 0, 255).astype(np.uint8)

        reconstructed_frame = cv2.cvtColor(reconstructed_frame, cv2.COLOR_RGB2BGR)

        output_video.write(reconstructed_frame)

        frame_number += 1
        print(f"k={k}: frame {frame_number}", end="\r")

    video.release()
    output_video.release()

    print(f"\nCreated: {output_path}")
