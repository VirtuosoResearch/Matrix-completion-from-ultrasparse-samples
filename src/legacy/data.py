import numpy as np
import random as random
from scipy.sparse import csr_matrix

def get_random_matrix(d1, d2, r):
	X = np.random.normal(2, 1, size = (d1, d2))

	U, D, Vt = np.linalg.svd(X, full_matrices=False)
	D[r:] = 0
	X = U @ np.diag(D) @ Vt
	return X


def get_uniformly_random_samples(M, p):
	M_shape = M.shape
	masks = np.random.rand(M_shape[0], M_shape[1]) <= p
	#observed_M = np.multiply(M, masks)
	observed_M = M * masks
	# observed_M = csr_array((data, (row, col)), shape=M_shape)
	same_nonzero_locations = np.array_equal((observed_M != 0), (masks != 0))
	if same_nonzero_locations:
		print("Non-zero elements are correctly aligned with the mask.")
	else:
		print("Non-zero elements do not align correctly with the mask.")
	return observed_M, masks

def get_random_samples_per_row(M, entries_per_row):
	row = []
	col = []
	data = []

	M_shape = M.shape
	masked_matrix = np.zeros(M_shape)
	for i in range(M_shape[0]):
		js = np.random.permutation(M_shape[1])
		for j in range(entries_per_row):
			row.append(i)
			col.append(js[j])
			data.append(M[i, js[j]])
			masked_matrix[i, js[j]] = 1

	observed_M = csr_matrix((data, (row, col)), shape=M_shape)
	return observed_M.toarray(), masked_matrix

def get_reconstruction_error(M_true, M_test, masks=np.array([])):
	assert(M_true.shape == M_test.shape)
	err = 0.0
	if masks.size == 0:
		err = np.linalg.norm(M_true - M_test, 'fro')
	else:
		err_matrix = np.multiply(M_true - M_test, masks)
		err = np.linalg.norm(err_matrix, 'fro')	
	
	return err

def get_normalized_error(M_true, M_test, masks=np.array([])):
	normalized_err = 0.0
	if masks.size == 0:
		err = np.linalg.norm(M_true - M_test, 'fro')

		normalized_err = err / np.linalg.norm(M_true, 'fro')
	else:
		err_matrix = np.multiply(M_true - M_test, masks)

		err = np.linalg.norm(err_matrix, 'fro')

		normalized_err = err / np.linalg.norm(np.multiply(M_true, masks), 'fro')

	return normalized_err

def transform_one_sided(M):
	matrix = M.T @ M
	# need to regenerate the masks
	masks = (abs(matrix) > 0).astype(int)
	return matrix, masks
	