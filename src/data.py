import numpy as np
import random as random

def get_random_matrix(d1, d2, r):
	X = np.random.normal(0, 1, size = (d1, d2))

	U, D, Vt = np.linalg.svd(X, full_matrices=False)
	D[r:] = 0
	X = U @ np.diag(D) @ Vt
	return X

def get_random_samples(M, m):
	list_of_indices = set([])
	# return a list of pairs of indices
	shape = M.shape
	for i in range(m):
		x = np.random.randint(0, shape[0] - 1)
		y = np.random.randint(0, shape[1] - 1)
		if (x, y) not in list_of_indices:
			list_of_indices.add((x, y))

	return list(list_of_indices)

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
