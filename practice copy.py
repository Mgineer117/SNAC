import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
V = 5  # Total velocity (not directly used, inferred)
omega = 2  # Circling velocity (rad/s)
z_dot = 1  # Upward velocity (m/s)
R = V / omega  # Radius from V = sqrt(V_c^2 + z_dot^2), assuming V_c = omega * R
T = 10  # Total duration (s)
t = np.linspace(0, T, 500)

# Cartesian Coordinates
x_cart = R * np.cos(omega * t)
y_cart = R * np.sin(omega * t)
z_cart = z_dot * t

# Cylindrical Coordinates
r_cyl = R  # Constant radius
theta_cyl = omega * t  # Angular motion
z_cyl = z_dot * t  # Same as Cartesian

# Convert Cylindrical to Cartesian for plotting
x_cyl = r_cyl * np.cos(theta_cyl)
y_cyl = r_cyl * np.sin(theta_cyl)

# Plot in 3D for both representations
fig = plt.figure(figsize=(12, 8))

# Cartesian plot
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(x_cart, y_cart, z_cart, label="Cartesian Spiral", color="b")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("3D Spiral Motion (Cartesian)")
ax1.legend()

# Cylindrical plot
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot(x_cyl, y_cyl, z_cyl, label="Cylindrical Spiral", color="r", linestyle="dashed")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title("3D Spiral Motion (Cylindrical)")
ax2.legend()

plt.tight_layout()
plt.show()
