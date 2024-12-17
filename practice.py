import torch
from torch.distributions import Normal, Multinomial

# Define a Normal distribution
normal_dist = Normal(
    torch.tensor([[0.0, 0.0], [0.0, 0.0]]), torch.tensor([[1.0, 1.0], [1.0, 1.0]])
)

# Shapes
print("Batch shape:", normal_dist.batch_shape)  # Independent variables
print("Event shape:", normal_dist.event_shape)  # Single variable's shape

# Sample and shape
samples = normal_dist.sample((4,))
print("Sample shape:", samples.shape)  # (4, batch_shape)
