import numpy as np
import time
import math
import torch
from torch import norm

from random import normalvariate
from math import sqrt

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = torch.tensor(randomUnitVector(min(n,m)), dtype=torch.float).to(A.device).unsqueeze(1)
    lastV = None
    currentV = x
    tol = 5000

    if n > m:
        B = torch.mm(A.T, A)
    else:
        B = torch.mm(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = torch.mm(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(torch.mm(currentV.t(), lastV).item()) > 1 - epsilon:
            #print("converged in {} iterations!".format(iterations))
            return currentV.squeeze(1)
        if iterations > tol:
            #print("achieved {} iterations with {}!".format(iterations, lastV))
            return currentV.squeeze(1)


def power_svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    #A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.clone()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * torch.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = torch.mm(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = torch.mm(A.T, u.unsqueeze(1))
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v.squeeze(1)))

    singularValues, us, vs = [x for x in zip(*svdSoFar)]
    s = torch.stack(singularValues)
    u = torch.stack(us)
    v = torch.stack(vs)
    return u.t(), s, v


if __name__ == "__main__":
    A = torch.rand(10000,10000).to('cuda:1')
    # v1 = svd_1d(movieRatings)
    # print(v1)

    start_time = time.time()
    up, sp, vp = power_svd(A, k=10)
    print(sp)
    end_time = time.time()
    print("Concatenating a tuple of scalars to a tensor took: {:.6f} seconds".format(end_time - start_time))
    #print(sp)
    #print(up)
    #print(vp)
    # Compare with torch.linalg.svd
    start_time = time.time()
    U_full, S_full, V_full = torch.linalg.svd(A)
    print("Torch SVD Singular Values:", S_full[:3])
    end_time = time.time()
    print("Concatenating a tuple of scalars to a tensor took: {:.6f} seconds".format(end_time - start_time))


    