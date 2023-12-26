from data import *
from gradient_descent import *

# generate data
d1 = 100
d2 = 10
r = 3
m = 2

M = get_random_matrix(d1, d2, r)
observed_M, masks = get_random_samples_per_row(M, m)

# run gradient descent
rank = 3
eta = 0.01
epochs = 10
r_U, r_V = gradient_desent(observed_M, masks, rank, eta, epochs)

# analyze projection of V to the right singular vectors
U, D, Vt = np.linalg.svd(M)
D = D[:r]
Vt = Vt[:r, :]
