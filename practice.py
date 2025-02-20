import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time  # Import time module
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate 3D state data
def generate_4d_data(n_samples=2000):
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0])
    cov = torch.eye(4) * 0.5
    data = torch.distributions.MultivariateNormal(mean, cov).sample((n_samples,))
    return data

# Step 3: Define a quadratic base distribution
def quadratic_base_distribution(n_samples):
    xy = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((n_samples,))
    z = xy[:, 0]**2 + xy[:, 1]**2  # Quadratic height
    latent_samples = torch.cat([xy, z.unsqueeze(1)], dim=1)
    latent_samples = latent_samples[latent_samples[:, 2].argsort(descending=True)]  # Sort by descending z
    # visualize_mapping(latent_samples.numpy(), latent_samples.numpy())
    return latent_samples

# Step 5: Visualization
def visualize_mapping(original, transformed):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    colors = original[:, -1]  # Use 4d values for coloring
    scatter1 = ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c=colors, cmap='coolwarm')
    ax1.set_title("Original Data")
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")
    # fig.colorbar(scatter1, ax=ax1, label='Y Value')
    
    scatter2 = ax2.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=colors, cmap='coolwarm')  # Keep same colors
    ax2.set_title("Transformed Data (Latent Space)")
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")
    # fig.colorbar(scatter2, ax=ax2, label='Y Value')
    
    plt.show()

# Step 2: Define an affine normalizing flow transformation
class AffineLayer(nn.Module):
    """Single affine transformation layer with input-dependent scale and shift."""
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        # MLP for scale computation
        self.scale_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh() 
        )
        
        # MLP for shift computation
        self.shift_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        scale = self.scale_mlp(x)  # Compute input-dependent scale
        shift = self.shift_mlp(x)  # Compute input-dependent shift
        return x * scale + shift


class MultiAffineFlow(nn.Module):
    """Multiple affine transformations applied sequentially."""
    def __init__(self, num_layers, input_dim=3, hidden_dim=32):
        super().__init__()
        self.layers = nn.ModuleList([AffineLayer(input_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Apply each affine transformation sequentially
        return x

# Step 4: Train the normalizing flow using quadratic prior with minibatch and time tracking
def train_flow(model, data, batch_size=512, epochs=500, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    xyz_data = data[:, :-1]  # Extract (x, y, z)
    c_data = data[:, -1:]    # Extract quadratic prior values

    # Create a DataLoader for minibatching
    dataset = TensorDataset(xyz_data, c_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()  # Start tracking time

    for epoch in range(epochs):
        for xyz_batch, c_batch in dataloader:
            optimizer.zero_grad()
            transformed_x = model(xyz_batch)
            quadratic_r_pred = (torch.norm(transformed_x, p=2, dim=-1)**2).unsqueeze(-1)

            loss = nn.functional.mse_loss(c_batch, quadratic_r_pred)
            loss.backward()
            optimizer.step()

        # Print loss and elapsed time every 2500 epochs
        if epoch % 50 == 0:
            elapsed_time = time.time() - start_time  # Compute elapsed time
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f} sec")
            start_time = time.time()  # Reset start time for next interval


# Run the pipeline
data = generate_4d_data()
# flow_model = MultiAffineFlow(5)
# train_flow(flow_model, data)
# transformed_data = flow_model(data[:, :-1])
# print(data[0,:-1])
# print(transformed_data[0])
# visualize_mapping(data.numpy(), transformed_data.detach().numpy())

import itertools

# Define hyperparameter grid
lr_options = [0.0001, 0.0005, 0.001]
num_layers_options = [3, 5, 7]
hidden_dim_options = [16, 32, 64]
batch_size_options = [128, 256, 512]

best_loss = float('inf')
best_params = None

# Iterate through all combinations
for lr, num_layers, hidden_dim, batch_size in itertools.product(lr_options, num_layers_options, hidden_dim_options, batch_size_options):
    print(f"Testing: lr={lr}, num_layers={num_layers}, hidden_dim={hidden_dim}, batch_size={batch_size}")
    
    # Create model and train
    model = MultiAffineFlow(num_layers, input_dim=3, hidden_dim=hidden_dim)
    train_flow(model, data, batch_size=batch_size, epochs=500, lr=lr)

    # Evaluate performance (use final loss)
    transformed_data = model(data[:, :-1])
    loss = nn.functional.mse_loss(data[:, -1:], (torch.norm(transformed_data, p=2, dim=-1) ** 2).unsqueeze(-1))

    print(f"Loss: {loss.item():.4f}")

    # Track best parameters
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_params = (lr, num_layers, hidden_dim, batch_size)

print(f"Best parameters: lr={best_params[0]}, num_layers={best_params[1]}, hidden_dim={best_params[2]}, batch_size={best_params[3]}, loss={best_loss:.4f}")

