# test the soft imputation algorithm on M^T M

from data import *

# generate data
d1 = 100
d2 = 10
r = 3
m = 2

M = get_random_matrix(d1, d2, r)
observed_M, masks = get_uniformly_random_samples(M, m)

# run soft impute
epochs = 10
eps = 0.01

X_init = observed_M.T @ observed_M

print(X_init)

#for i in range(epochs):
