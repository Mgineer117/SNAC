import torch

# Example 2D tensor of shape (n, b)
tensor = torch.tensor([[2, 4], [6, 8]])

# Number of classes
num_classes = 12

# Create one-hot encodings for the tensor along the last dimension
one_hot = torch.nn.functional.one_hot(tensor, num_classes=num_classes)

# Reshape the result to concatenate along dim=-1
# Resulting shape will be (n, b * num_classes)
concatenated = one_hot.view(tensor.size(0), -1)

print("Original tensor:")
print(tensor)
print("\nOne-hot encoded tensor:")
print(one_hot)
print("\nConcatenated one-hot encoding along dim=-1:")
print(concatenated)
