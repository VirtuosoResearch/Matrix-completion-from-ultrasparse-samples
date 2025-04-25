import torch
import numpy as np
from tqdm import tqdm

def empirical_reweight_estimate(observed_M):
    second_moment_observe_M =  observed_M.T @ observed_M
    second_moment_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
    second_moment_observe_count = second_moment_observe_count + (second_moment_observe_count == 0) * 1
    p_hat_matrix = second_moment_observe_count

    T = second_moment_observe_M / p_hat_matrix
    return T

def probability_reweight_estimate(observed_M, p):
    n, d = observed_M.shape
    second_moment_observe_M =  observed_M.T @ observed_M
    diag = torch.diag( torch.diag(second_moment_observe_M) )
    T_p = ((1.0 / p) * diag + (1.0 / (p**2)) * (second_moment_observe_M - diag))
    T_p = T_p / n

    return T_p


class IIPW:
    def __init__(self, M, observed_M, masks, p, r):
        # restore parameters
        self.n, self.d = M.shape
        self.p = p
        self.r = r
        self.iter_num = 0

        # process matrix
        self.M = M
        self.observed_M = observed_M
        self.masks = masks
        self.T = empirical_reweight_estimate(observed_M)
        self.normalized_MTM = M.T @ M / self.n
    
    def impute(self, n_iter=100, lr=0.5, tol=1e-7, alpha=0, lam=1):
        def R(X, alpha):
            norms = torch.norm(X, dim=1)  # (d,)
            mask = (norms >= alpha).float()
            r_vals = ((norms - alpha)**4) * mask
            return r_vals.sum()
        def grad_R(X, alpha):
            norms = torch.norm(X, dim=1, keepdim=True)  # shape: (d, 1)
            mask = (norms >= alpha).float()
            coeffs = 4 * ((norms - alpha) ** 3) * mask  # shape: (d, 1)
            grad = coeffs * X / norms.clamp(min=1e-8)   # avoid divide-by-zero
            return grad
        # Initialize parameters
        self.iter_num = n_iter
        T = self.T
        T_masks = 1.0 * (self.T!=0)
        m = T_masks.sum().item()
        U = torch.rand(self.d, self.r).to(T.device) * 0.1
        #U.require_grad = True

        # donw weight the diagonal entries
        down_weight_mask = torch.ones(self.d, self.d)
        diag_values = 1 - (1 - self.p**2)**self.d
        down_weight_mask.fill_diagonal_(diag_values)
        #T_masks.fill_diagonal_(diag_values)

        print('Imputing...')
        err_estimates = []
        loop = tqdm(range(n_iter))
        for i in loop:
            # Compute the gradient
            residual = (U @ U.T - T) * T_masks
            grad = (2/m) * (residual @ U)
            reg_mask = (torch.norm(U,dim=1) > alpha)
            reg = (((torch.norm(U,dim=1)-alpha)**4)[reg_mask]).sum()
            reg_grad = grad_R(U, alpha) / self.d
            
            # Update
            U = U - lr * (grad + lam*reg_grad)
            X_imp = U @ U.T
            err = self.normalized_MTM - X_imp
            relative_err = torch.norm(err, 'fro')
            err_estimates.append(relative_err.item())

            # Tolerance check
            if len(err_estimates) > 1:
                if err_estimates[-1] > err_estimates[-2]:
                    break
            if i > 10:
                if (abs(err_estimates[-1] - err_estimates[-2]) < tol) or (relative_err > err_estimates[0]):
                    break

            loop.set_description(f"Error: {relative_err:.7f}, reg: {reg.item()}, grad: {grad.sum().item()}, reg_grad: {reg_grad.sum().item()}")
        estimation_matrix = X_imp

        self.iter_num = i+1
        return U, estimation_matrix, relative_err.item()
    