import numpy as np

# Assuming psi is your 30000 x 128 matrix
psi = np.random.randn(30000, 128)

# Perform SVD without computing U (by setting full_matrices=False)
_, S_S, V_S = np.linalg.svd(psi, full_matrices=False)

# Now S_S contains the singular values and V_S contains the right singular vectors (V^T)
subtask_vectors = {"rewards": None, "states": V_S}
