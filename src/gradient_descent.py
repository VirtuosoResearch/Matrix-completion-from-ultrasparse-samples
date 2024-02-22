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

def gradient_descent(observed_M, masks, rank, eta, num_of_epochs, gap=1):
	init_scale = 0.001
	d1, d2 = observed_M.shape 

	U = np.random.normal(0, 1, (d1, rank)) * init_scale
	V = np.random.normal(0, 1, (d2, rank)) * init_scale

	for i in range(num_of_epochs):
		U, V = factorized_gradient_descent_one_step(observed_M, masks, U, V, eta)

		if i % gap == 0:
			err = get_normalized_error(observed_M, U @ V.T, masks)
			print(i, err)

	return U, V

def noisy_gradient_descent(observed_M, masks, rank, eta, num_of_epochs, noise_var, gap=10):
	init_scale = 0.001
	d1, d2 = observed_M.shape

	U = np.random.normal(0, 1, (d1, rank)) * init_scale
	V = np.random.normal(0, 1, (d2, rank)) * init_scale

	for i in range(num_of_epochs):
		# first compute gradient
		X = np.multiply(U @ V.T, masks) - observed_M
		grad_U = X @ V
		grad_V = X.T @ U

		# add noise
		noise_U = np.random.normal(0, 1, (d1, rank)) * noise_var
		# noise_V = np.random.normal(0, 1, (d2, rank)) * noise_var

		U = U - eta * (grad_U + noise_U)
		V = V - eta * grad_V

		if i % gap == 0:
			#u1, d1, vt1 = np.linalg.svd(U @ V.T)
			#vt1 = vt1[:rank, :]
			err = get_normalized_error(observed_M, U @ V.T, masks)
			print(i, err)

	return U, V


def symmetric_gradient_descent(symmetric_M, masks, rank, eta, num_of_epochs, gap=10):
	init_scale = 0.0001
	d1, d2 = symmetric_M.shape
	assert(d1 == d2)

	U = np.random.normal(0, 1, (d1, rank)) * init_scale

	for i in range(num_of_epochs):
		U, V = factorized_gradient_descent_one_step(symmetric_M, masks, U, U, eta)

		if i % gap == 0:
			err = get_normalized_error(symmetric_M, U @ U.T, masks)
			print(i, err)

	return U


def symmetric_noisy_gradient_descent(observations, masks, rank, eta, epochs, noise_var, reg, gap=10):
	init_scale = 0.0001
	d1, d2 = observations.shape
	assert(d1 == d2)

	U = np.random.normal(0, 1, (d1, rank)) * init_scale

	for i in range(epochs):
		X = np.multiply(U @ U.T, masks) - observations
		grad = X @ U

		noise = np.random.normal(0, 1, (d1, rank)) * noise_var
		U = U - eta * (grad + noise)
		U = U - eta * reg * U

		if i % gap == 0:
			err = get_normalized_error(observations, U @ U.T, masks)
			print(i, err)

	return U


def projection(X, rank):
	U, D, Vt = np.linalg.svd(X)
	U = U[:, :rank]
	# D = D[:rank]
	Vt = Vt[:rank, :]
	return U @ Vt

def projected_gradient_descent(symmetric_M, rank, eta, num_of_epochs, gap = 10):
	init_scale = 0.001
	d1, d2 = symmetric_M.shape
	assert(d1 == d2)

	V = np.random.normal(0, 1, (d1, rank)) * init_scale
	V = projection(V, rank)

	for i in range(num_of_epochs):
		V = V + eta * (symmetric_M @ V)
		V = projection(V, rank)

	return V


