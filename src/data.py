import numpy as np
import random as random

def get_random_matrix(d1, d2, r):
	X = np.random.normal(0, 1, size = (d1, d2))

	U, D, Vt = np.linalg.svd(X, full_matrices=False)
	D[r:] = 0
	X = U @ np.diag(D) @ Vt
	return X

def get_random_samples(M, m):
	indices = []
	# return a list of pairs of indices
	shape = M.shape
	list = []

	return indices
