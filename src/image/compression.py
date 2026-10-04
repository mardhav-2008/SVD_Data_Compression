from pathlib import Path

import numpy as np
from PIL import Image

from core.low_rank import Approximate

image1_path = Path("data/raw/image/1.jpeg")

image = Image.open(image1_path).convert("RGB")
image_array = np.asarray(image, dtype=float)

R = image_array[:, :, 0]
G = image_array[:, :, 1]
B = image_array[:, :, 2]

print(f"Image shape: {image_array.shape}")
print(f"R shape: {R.shape}")
print(f"G shape: {G.shape}")
print(f"B shape: {B.shape}")

appr_R = Approximate(R)
