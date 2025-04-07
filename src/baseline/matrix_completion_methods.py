import numpy as np
import torch

from util import from_numpy, to_device, generate_gaussian_factors

def compute_l2_loss(X_est, X, mask = None):
    if mask is not None:
        X, X_est = X[mask==1], X_est[mask==1]
    return torch.mean((X-X_est)**2)

def compute_l2_loss_weighted(X_est, X, counts):
    loss_matrix = (X - X_est)**2 * counts
    return torch.mean(loss_matrix[loss_matrix!=0])

def compute_regularizer(U, cutoff = 0):
    _, k = U.shape
    row_norms = torch.norm(U, dim=-1)
    return torch.mean(torch.maximum(torch.zeros_like(row_norms), row_norms / k**(1/4) - cutoff)**2)

def complete_matrix_grad(X, counts, k, lr = 1e-3, iters = 10000, reg_lambda = 10, cutoff = 0.5, X_gt = None, symm = False, weighted = False, verbose = True, return_UV = False):
    n, d = X.shape
    mask = counts > 0

    U, V = generate_gaussian_factors(n, d, k)
    magnitude = (torch.sum(X[mask]**2) / torch.sum(mask))**(1/2)
    print("Magnitude: {}".format(magnitude)) # do row-by-row instead?
    U, V = U * magnitude**(1/2), V * magnitude**(1/2)
    if symm:
        V = U
    U.requires_grad = True
    V.requires_grad = True
    trainer = torch.optim.Adam([U, V], lr = lr)
    for i in range(iters):
        trainer.zero_grad()
        X_est = U @ V.mT
        if weighted:
            loss = compute_l2_loss_weighted(X_est, X, counts)
        else:
            loss = compute_l2_loss(X_est, X, mask)
        reg = compute_regularizer(U, cutoff) + compute_regularizer(V, cutoff)
        (loss + reg_lambda * reg).backward()
        trainer.step()
        if i % 500 == 0 and verbose:
            with torch.no_grad():
                print("Iter: {}/{} \n"
                      "Loss (masked, full): {:.4f}, {:.4f}, "
                      "Regularizer: {:.4f}".format(
                      i+1, iters,
                      loss, compute_l2_loss(X_est, X),
                      reg))
                if X_gt is not None:
                    print("Ground truth loss (masked/weighted, full): {:.4f}, {:.4f}, "
                          "Observed vs ground truth: {:.4f}".format(
                          compute_l2_loss(X_est, X_gt, mask), compute_l2_loss(X_est, X_gt),
                          compute_l2_loss(X, X_gt, mask)))
    X_est = X_est.detach()
    if X_gt is not None and verbose:
        print("Final l2 dist: {:.4f}".format(compute_l2_loss(X_est, X_gt)))
    if return_UV:
        return X_est, (U.detach(), V.detach())
    else:
        return X_est

def complete_matrix_iter(X, mask, k, X_gt, iters = 100):
    X_est = X * mask
    for i in range(iters):
        X_est[mask == 1] = X[mask==1]
        U, S, Vh = torch.linalg.svd(X_est, full_matrices=False)
        X_est = U[:,:k] @ torch.diag(S[:k]) @ Vh[:k,:]

        if i % 10 == 0:
            print("Iter: {}/{}"
                  "Masked loss: {:.5f}, Full loss: {:.5f}"
                  "Masked noiseless loss: {:.5f}, Full noiseless loss: {:.5f}".format(
                  i+1, iters,
                  compute_l2_loss(X_est, X, mask), compute_l2_loss(X_est, X),
                  compute_l2_loss(X_est, X_gt, mask), compute_l2_loss(X_est, X_gt)))
    return X_est

def convert_to_item_sim(X, mask):
    X_masked = X * mask
    sim = X_masked.mT @ X_masked
    counts = mask.mT @ mask
    return torch.nan_to_num(sim / counts), counts

def estimate_sim_iter(X, mask, k, iters = 100): # TODO: turn this into actual loss - summed over observations, not matrix l2
    """
    X is matrix (m x n), mask is 1 where entry is observed and 0 where entry is unobserved, k is rank
    goal - for X = UV^T, recover X^TX = VU^TUV^T.
    """
    X, mask = from_numpy(X), from_numpy(mask)
    sim, counts = convert_to_item_sim(X, mask)
    sim_mask = counts > 0
    sim_est = complete_matrix_iter(sim, sim_mask, k, X.mT @ X / X.shape[0], iters = iters)
    return sim_est.cpu().numpy()

def estimate_sim_weighted(X, mask, k, lr=1e-3, iters=10000, reg_lambda=0, cutoff=0): # TODO: turn this into actual loss - summed over observations, not matrix l2
    """
    X is matrix (m x n), mask is 1 where entry is observed and 0 where entry is unobserved, k is rank
    goal - for X = UV^T, recover X^TX = VU^TUV^T.
    """
    X, mask = from_numpy(X), from_numpy(mask)
    sim, counts = convert_to_item_sim(X, mask)
    sim_est = complete_matrix_grad(sim, counts, k, symm=True, weighted=True, lr=lr, reg_lambda=reg_lambda, cutoff=cutoff, iters=iters, verbose=True)
    return sim_est.cpu().numpy()
