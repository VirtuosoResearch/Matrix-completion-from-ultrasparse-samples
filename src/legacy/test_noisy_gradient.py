# run experiments to test the differential privacy rate of noisy gradient descent
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
noise_var = 0.01
rec_U, rec_V = noisy_gradient_descent(observed_M, masks, rank, eta, epochs, noise_var)

# get test error
rec_M = rec_U @ rec_V.T
test_err = get_normalized_error(M, rec_M)
print(test_err)

# analyze projection
u1, d1, vt1 = np.linalg.svd(rec_U @ rec_V.T)
vt1 = vt1[:rank, :]

u2, d2, vt2 = np.linalg.svd(M)
vt2 = vt2[:rank, :]
corr = vt1 @ vt2.T

print(corr)