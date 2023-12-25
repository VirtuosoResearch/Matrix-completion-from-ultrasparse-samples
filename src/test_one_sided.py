from data import *

d1 = 10
d2 = 10
r = 3
m = d1 * r

# generate data
M = get_random_matrix(d1, d2, r)
observed_indices = get_random_samples(M, m)

