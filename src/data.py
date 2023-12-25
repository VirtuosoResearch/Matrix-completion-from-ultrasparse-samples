import numpy as np
import random as random
from scipy.sparse import csr_array

def get_random_matrix(d1, d2, r):
	X = np.random.normal(0, 1, size = (d1, d2))

	U, D, Vt = np.linalg.svd(X, full_matrices=False)
	D[r:] = 0
	X = U @ np.diag(D) @ Vt
	return X

def get_random_samples(M, m):
	# list_of_indices = set([])
	# vals = dict()
	# return a list of pairs of indices
	row = np.array([])
	col = np.array([])
	data = np.array([])

	M_shape = M.shape
	for i in range(m):
		x = np.random.randint(0, M_shape[0] - 1)
		y = np.random.randint(0, M_shape[1] - 1)
		# if (x, y) not in list_of_indices:
		#	list_of_indices.add((x, y))
		#	vals[(x, y)] = M[x, y]
		np.append(row, x)
		np.append(col, y)
		np.append(data, M[x, y])

	observed_M = csr_array((data, (row, col)), shape=M_shape)
	return observed_M

def get_reconstruction_error(M_true, M_test, list_of_indices=[]):
	assert(M_true.shape == M_test.shape)
	m = len(list_of_indices)
	err = 0.0
	if not list_of_indices:
		return np.linalg.norm(M_true - M_test, 'fro')
	else:
		for (x, y) in list_of_indices:
			err += (M_true[(x, y)] - M_test[(x, y)]) ** 2 / m
	
	return err
