import numpy as np
import random
from data import *

def factorized_gradient_descent_one_step(observed_M, masks, U, V, eta):
	X = np.multiply(U @ V.T, masks) - observed_M
	grad_U = X @ V
	grad_V = X.T @ U

	updated_U = U - eta * grad_U
	updated_V = V - eta * grad_V
	return updated_U, updated_V

def gradient_desent(observed_M, masks, rank, eta, num_of_epochs, gap=1):
	init_scale = 0.1
	d1, d2 = observed_M.shape 

	U = np.random.normal(0, 1, (d1, rank)) * init_scale
	V = np.random.normal(0, 1, (d2, rank)) * init_scale

	for i in range(num_of_epochs):
		U, V = factorized_gradient_descent_one_step(observed_M, masks, U, V, eta)

		if i % gap == 0:
			err = get_normalized_error(observed_M, U @ V.T, masks)
			print(i, err)

	return U, V

def noisy_gradient_descent(M, eta, num_of_epochs, noise_var):
	return
