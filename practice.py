import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# Example 10x10 tensor with random noise
np.random.seed(42)
input_tensor = np.random.rand(12, 12)

# Smoothing the tensor using a uniform filter
smoothed_tensor = uniform_filter(input_tensor, size=5)

# Plot original and smoothed tensors as heatmaps
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(input_tensor, cmap="viridis")
ax[0].set_title("Original Tensor")
ax[0].axis("off")

ax[1].imshow(smoothed_tensor, cmap="viridis")
ax[1].set_title("Smoothed Tensor")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("ss.png")
plt.show()
