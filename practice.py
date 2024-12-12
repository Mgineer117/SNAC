import numpy as np
from scipy.stats import entropy


def compute_categorical_entropy(indices, num_categories=8):
    """
    Computes the categorical distribution and its entropy for a given list of indices.

    Args:
        indices (list or np.ndarray): List of category indices (0 to num_categories-1).
        num_categories (int): Total number of categories (default is 8).

    Returns:
        tuple: A tuple (distribution, entropy_value), where
            - distribution (np.ndarray): The normalized distribution over categories.
            - entropy_value (float): The entropy of the distribution.
    """
    # Count occurrences of each category
    counts = np.bincount(indices, minlength=num_categories)

    # Normalize to get the categorical distribution
    distribution = counts / counts.sum()

    # Compute the entropy
    entropy_value = entropy(distribution, base=2)  # Use base-2 for bits

    return distribution, entropy_value


# Example usage
indices = [0, 1, 1, 2, 3, 1, 7, 7, 7, 0, 2]  # Example indices
distribution, entropy_value = compute_categorical_entropy(indices)

print("Categorical Distribution:", distribution)
print("Entropy:", entropy_value)
