import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import normflows as nf  # Assuming you are using a normalizing flow library
import matplotlib.pyplot as plt
import imageio
import os

from scipy.integrate import solve_ivp

def generate_cartpole_dataset(num_samples=1000, seed=42, dt=0.001):
    np.random.seed(seed)
    
    # Define state and action ranges
    theta_range = (-np.pi, np.pi)
    theta_dot_range = (-6, 6)
    action_range = (-0.225, 0.225)
    
    # Sample random states and actions
    theta = np.random.uniform(*theta_range, num_samples)
    theta_dot = np.random.uniform(*theta_dot_range, num_samples)
    action = np.random.uniform(*action_range, num_samples)
    
    # Define the system dynamics (ODE system)
    def cartpole_dynamics(t, state, action):
        theta, theta_dot = state
        dtheta_dt = theta_dot
        dtheta_dot_dt = np.sin(theta) + action  # Simplified dynamics
        return [dtheta_dt, dtheta_dot_dt]

    # Solve the ODE to find next_theta for each sample
    next_theta = np.zeros(num_samples)
    next_theta_dot = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Initial state: [theta, theta_dot]
        initial_state = [theta[i], theta_dot[i]]
        
        # Use an ODE solver (solve_ivp) to compute the next state over the given time span
        sol = solve_ivp(cartpole_dynamics, [0, dt], initial_state, args=(action[i],), t_eval=[dt])
        
        # The next state (after delta t)
        next_theta[i] = sol.y[0, -1]  # theta after delta t time step
        next_theta_dot[i] = sol.y[1, -1]  # theta_dot after delta t time step

    # Compute cost function: +1 cost if |theta_dot| > 4, otherwise 0
    cost = (np.abs(theta_dot) > 4).astype(int)
    
    # Concatenate theta and theta_dot to make a Nx2 matrix
    state = np.column_stack((theta, theta_dot)) 
    next_state = np.column_stack((next_theta, next_theta_dot))
    
    # Store data as dictionary
    data_dict = {
        'state': state.astype(np.float64),  # Nx2 matrix
        'action': action.astype(np.float64),
        'next_state': next_state.astype(np.float64),
        'cost': cost.astype(np.float64)
    }
    
    # Visualization: Show how state changes with action and next state    
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].plot(data_dict["state"][:20, 1])
    ax[0].grid()
    ax[1].plot(data_dict["action"][:20])
    ax[1].grid()
    ax[2].plot(data_dict["next_state"][:20, 1])
    ax[2].grid()

    plt.title("State, Action, and Next State")
    plt.savefig("dynamics.png")
    plt.close()
    
    return data_dict


# Define base distribution
num_samples = 1000
base = nf.distributions.base.DiagGaussian(2)  # 2D latent space
data = generate_cartpole_dataset(num_samples=num_samples)

# Define flow model
num_layers = 8
flows = []
for _ in range(num_layers):
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode='swap'))  # Improve expressivity

# Construct normalizing flow model
model = nf.NormalizingFlow(base, flows)
model = model.to(torch.float64)  

# Define learnable coefficients for convexity
a = nn.Parameter(torch.ones(1))  # If 2D: [z1] → z2

# Optimizer
optimizer = optim.Adam(list(model.parameters()) + [a], lr=5e-4, weight_decay=1e-5)

# Custom convexity loss
def convexity_loss(z, cost):
    z1, z2 = z[:, 0], z[:, 1]

    loss1 = torch.mean((z2 - a[0] * z1**2)**2)
    loss2 = nn.functional.mse_loss(cost, z2)
    return loss1 + loss2

# Create folders for saving frames
os.makedirs("frames_xzf", exist_ok=True)
os.makedirs("frames_z", exist_ok=True)

# Training loop
loss_hist, convex_hist = [], []

for epoch in range(1000):
    indices = torch.randperm(num_samples)  # Generates a tensor of shuffled indices
    state = data["state"]
    cost = data["cost"]

    x = state[indices[:128]]  # Sample data 
    cost = cost[indices[:128]]  # Sample data 

    x = torch.from_numpy(x).to(torch.float64)
    cost = torch.from_numpy(cost).to(torch.float64)


    z, log_det = model.inverse_and_log_det(x)  # Transform to latent space

    recon_loss = torch.mean((x - model.forward(z))**2)
    convex_loss = 0.01 * convexity_loss(z, cost)

    loss = recon_loss + convex_loss
    loss_hist.append(recon_loss.item())
    convex_hist.append(convex_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Visualization every 100 epochs
    if epoch % 100 == 0:
        with torch.no_grad():
            z_vis, _ = model.inverse_and_log_det(x)  # Get transformed latent space
            x_recon = model.forward(z_vis)  # Reconstruct x from z

        plt.figure(figsize=(18, 5))

        # Plot original data (x-space)
        plt.subplot(1, 3, 1)
        plt.scatter(x[:, 0], x[:, 1], s=5, alpha=0.5, color='blue')
        plt.title(f"Original Data (x-Space) - Epoch {epoch}")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        # Plot transformed latent representation (z-space)
        plt.subplot(1, 3, 2)
        plt.scatter(z_vis[:, 0], z_vis[:, 1], s=5, alpha=0.5, c=cost, cmap='coolwarm')
        plt.title(f"Transformed Data (z-Space) - Epoch {epoch}")
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")

        # Plot reconstructed data (f(z)-space)
        plt.subplot(1, 3, 3)
        plt.scatter(x_recon[:, 0], x_recon[:, 1], s=5, alpha=0.5, color='green')
        plt.title(f"Reconstructed Data (f(z)-Space) - Epoch {epoch}")
        plt.xlabel("$f(z_1)$")
        plt.ylabel("$f(z_2)$")

        plt.savefig(f"frames_xzf/frame_{epoch:04d}.png")
        plt.close()

        # Separate GIF for just z-space
        plt.figure(figsize=(6, 5))
        plt.scatter(z_vis[:, 0], z_vis[:, 1], s=5, alpha=0.5, color='red')
        plt.title(f"Transformed Data (z-Space) - Epoch {epoch}")
        plt.xlabel("$z_1$")
        plt.ylabel("$z_2$")
        plt.savefig(f"frames_z/frame_{epoch:04d}.png")
        plt.close()

# Create GIFs
xzf_frames = sorted(os.listdir("frames_xzf"))
z_frames = sorted(os.listdir("frames_z"))

imageio.mimsave("xzf_transformation.gif", [imageio.imread(f"frames_xzf/{f}") for f in xzf_frames], fps=5)
imageio.mimsave("z_transformation.gif", [imageio.imread(f"frames_z/{f}") for f in z_frames], fps=5)

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='Total Loss')
plt.plot(convex_hist, label='Convexity Loss')
plt.legend()
plt.savefig("loss_history.png")
