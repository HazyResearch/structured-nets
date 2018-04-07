"""
Computes projections onto various classes.
"""
import numpy as np
from scipy.linalg import toeplitz

def kth_diag_indices(A, k):
    rows, cols = np.diag_indices_from(A)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

# Projects onto Toeplitz matrices, under Frobenius norm.
def toeplitz_project_frob(A):
	assert A.shape[0] == A.shape[1]

	A_proj = np.zeros(A.shape)

	# Get indices of each diagonal
	for diag_idx in np.arange(-(A.shape[0]-1),A.shape[0]):
		this_idx = kth_diag_indices(A, diag_idx)
		# Get average
		avg = np.mean(A[this_idx])
		A_proj[this_idx] = avg

	return A_proj

# Projects onto Hankel matrices, under Frobenius norm.
def hankel_project_frob(A):
	A_flip = np.flipud(A)

	A_flip_proj = toeplitz_project_frob(A_flip)

	return np.flipud(A_flip_proj)

if __name__ == '__main__':
	A = np.random.randint(5, size=(3,3))
	print(A)
	#print kth_diag_indices(A, -4)

	print(hankel_project_frob(A))