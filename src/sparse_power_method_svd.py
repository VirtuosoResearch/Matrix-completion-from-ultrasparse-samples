import numpy as np
import time
import math
import torch
from torch import norm
from random import normalvariate
from math import sqrt
from sparse_utils import *

def randomUnitVector(n, device='cpu'):
    unnormalized = torch.randn(n, device=device)
    theNorm = torch.norm(unnormalized)
    return unnormalized / theNorm

def orthogonalize(vec, vecs):
    for v in vecs:
        vec = vec - torch.dot(vec, v) * v
    return vec

def svd_1d_sparse(A, epsilon=1e-10, max_iterations=1000, us_list=None, vs_list=None):
    ''' Compute the largest singular value and corresponding singular vectors of sparse matrix A '''
    n, m = A.shape
    device = A.device
    v = torch.randn(m, device=device)
    v = v / torch.norm(v)
    u = torch.zeros(n, device=device)

    if vs_list is None:
        vs_list = []
    if us_list is None:
        us_list = []

    for i in range(max_iterations):
        # Orthogonalize v against previous vs
        if vs_list:
            v = orthogonalize(v, vs_list)
            v = v / torch.norm(v)

        u = torch.sparse.mm(A, v.unsqueeze(1)).squeeze(1)
        # Orthogonalize u against previous us
        if us_list:
            u = orthogonalize(u, us_list)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm

        v_new = torch.sparse.mm(A.t(), u.unsqueeze(1)).squeeze(1)
        # Orthogonalize v_new against previous vs
        if vs_list:
            v_new = orthogonalize(v_new, vs_list)
        v_norm = torch.norm(v_new)
        if v_norm == 0:
            break
        v_new = v_new / v_norm

        # Check for convergence
        if torch.norm(v_new - v) < epsilon:
            v = v_new
            break
        v = v_new

    sigma = torch.norm(torch.sparse.mm(A, v.unsqueeze(1)))
    return sigma, u, v

def power_svd_sparse(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a sparse matrix A
        using the power method. A is the input sparse matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    n, m = A.shape
    device = A.device
    us_list = []
    vs_list = []
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        sigma, u, v = svd_1d_sparse(A, epsilon=epsilon, us_list=us_list, vs_list=vs_list)
        svdSoFar.append((sigma, u, v))

        us_list.append(u)
        vs_list.append(v)

    singularValues, us, vs = [x for x in zip(*svdSoFar)]
    s = torch.tensor(singularValues, device=device)
    U = torch.stack(us, dim=1)
    V = torch.stack(vs, dim=1)
    return U, s, V

# Generate a random sparse matrix
def generate_random_sparse_matrix(n_rows, n_cols, sparsity=0.01, device='cpu'):
    indices = torch.nonzero(torch.rand(n_rows, n_cols) < sparsity, as_tuple=False).t().to(device)
    values = torch.randn(indices.size(1), device=device)
    size = torch.Size([n_rows, n_cols])
    sparse_matrix = torch.sparse.FloatTensor(indices, values, size).to(device)
    return sparse_matrix

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_rows, n_cols = 10000, 10000
    sparsity = 0.01  # 1% non-zero entries
    A = generate_random_sparse_matrix(n_rows, n_cols, sparsity=sparsity, device=device)

    # Compute SVD using power method
    k = 5  # Number of singular values to compute
    start_time = time.time()
    U_p, S_p, V_p = power_svd_sparse(A, k=k, epsilon=1e-6)
    end_time = time.time()
    print("Power method SVD took: {:.6f} seconds".format(end_time - start_time))
    print("Singular values from power method:", S_p)

    A_scipy = torch_sparse_to_scipy(A)
    start_time = time.time()
    U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(A_scipy, k=k)
    end_time = time.time()
    print("scipy SVD took: {:.6f} seconds".format(end_time - start_time))
    print("Singular values from torch.linalg.svd:", D_scipy)
    #print("Difference in singular values:", S_p - D_scipy)

    # For verification, convert A to dense and compute SVD using torch.linalg.svd
    A_dense = A.to_dense()
    start_time = time.time()
    U_full, S_full, V_full = torch.linalg.svd(A_dense)
    end_time = time.time()
    print("Torch SVD took: {:.6f} seconds".format(end_time - start_time))
    print("Singular values from torch.linalg.svd:", S_full[:k])

    # Compute the difference between singular values
    print("Difference in singular values:", S_p - S_full[:k])