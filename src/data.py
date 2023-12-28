import numpy as np
import random as random
from scipy.sparse import csr_array

def get_random_matrix(d1, d2, r):
	X = np.random.normal(0, 1, size = (d1, d2))

	U, D, Vt = np.linalg.svd(X, full_matrices=False)
	D[r:] = 0
	X = U @ np.diag(D) @ Vt
	return X

def get_uniformly_random_samples(M, m):
	# list_of_indices = set([])
	# vals = dict()
	# return a list of pairs of indices
	row = []
	col = []
	data = []

	M_shape = M.shape
	masked_matrix = np.zeros(M_shape)
	for i in range(m):
		x = np.random.randint(0, M_shape[0] - 1)
		y = np.random.randint(0, M_shape[1] - 1)
		# if (x, y) not in list_of_indices:
		#	list_of_indices.add((x, y))
		#	vals[(x, y)] = M[x, y]
		row.append(x)
		col.append(y)
		data.append(M[x, y])
		masked_matrix[x, y] = 1

	observed_M = csr_array((data, (row, col)), shape=M_shape)
	return observed_M, masked_matrix

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

	observed_M = csr_array((data, (row, col)), shape=M_shape)
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
	masks = (matrix > 0).astype(int)
	return matrix, masks
	