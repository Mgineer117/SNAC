import torch
import torch.nn as nn

# Create an example nn.Parameter
param = nn.Parameter(torch.randn(3, 4))
print(param)
param2 = param.cpu()
# Access the elements like a tensor
tensor_values = param.data  # or just use .detach() for similar result

# Convert to NumPy array if needed
numpy_array = tensor_values.cpu().numpy()

# Example: print the values
print(tensor_values)
