import torch
import numpy as np
from typing import Tuple
from tqdm import tqdm
import scipy
import scipy.sparse

def sparse_elementwise_divide(A, B, fill_value=0.0):
    A_coo = A.coalesce()
    B_coo = B.coalesce()
    indices_A = A_coo.indices()
    values_A = A_coo.values()
    b_dict = {(i.item(), j.item()): v.item() for (i,j), v in zip(B_coo.indices().T, B_coo.values())}
    result_indices, result_values = [], []
    for idx in tqdm(range(indices_A.shape[1])):
        i, j = indices_A[0, idx].item(), indices_A[1, idx].item()
        a_val = values_A[idx].item()
        b_val = b_dict.get((i,j), 0.0)
        result_indices.append([i,j])
        result_values.append(a_val / b_val if b_val !=0 else fill_value)
    result_indices = torch.tensor(result_indices).T if result_indices else torch.empty((2,0), dtype=torch.long)
    return torch.sparse_coo_tensor(result_indices, torch.tensor(result_values), A.shape).coalesce()

def sparse_empirical_reweight_estimate(observed_M, device='cpu'):
    observed_M = observed_M.coalesce()
    G = torch.sparse.mm(observed_M.T, observed_M)

    indices = observed_M.indices()
    values_I = torch.ones_like(observed_M.values())
    I_sparse = torch.sparse_coo_tensor(
        indices=indices,
        values=values_I,
        size=observed_M.size()
    ).coalesce()

    I_T_I = torch.sparse.mm(I_sparse.T, I_sparse)
    if I_sparse.is_cuda:
        torch.cuda.synchronize()
    
    S = sparse_elementwise_divide(G, I_T_I)
    del G, I_sparse, I_T_I
    return S

class IIPW_sparse:
    def __init__(self, n, d, S, T, r: int, p: float, device='cpu'):
        self.device = device
        self.r = r
        self.p = p
        self.n = n
        self.d = d

        self.S = S
        self.T = T.to(device)

        self.diag_val = 1.0 - (1.0 - p ** 2) ** self.d

    @staticmethod
    def _grad_R(U: torch.Tensor, alpha: float) -> torch.Tensor:
        nrm = torch.norm(U, dim=1, keepdim=True)  # [d,1]
        mask = (nrm >= alpha).float()
        coeff = 4 * (nrm - alpha) ** 3 * mask
        return coeff * U / nrm.clamp_min_(1e-8)
    
    def impute(self, n_iter: int = 100, lr: float = 0.5, tol: float = 1e-12,
               alpha: float = 0, lam: float = 1.0) -> float:
        idx = self.T.indices()
        vals = self.T.values()
        device = self.device

        d, r = self.d, self.r
        is_diag = idx[0] == idx[1]
        scale = torch.where(is_diag,
                            torch.full_like(vals, self.diag_val),
                            torch.full_like(vals, 2.0))
        m_obs = 2 * idx.size(1) - is_diag.sum().item()  # |Ω|

        # Initialise factor U (small random)
        U = 0.1 * torch.randn(d, r, device=device)
        err_hist: list[float] = []

        loop = tqdm(range(n_iter))
        for it in loop:
            ui = U[idx[0]]
            uj = U[idx[1]]
            res = (ui * uj).sum(dim=1) - vals  # [m]
            
            coeff = (2.0 / m_obs) * (scale * res).unsqueeze(1)
            grad_U = torch.zeros_like(U)
            grad_U.index_add_(0, idx[0], coeff * uj)
            grad_U.index_add_(0, idx[1], coeff * ui)

            reg_grad = self._grad_R(U, alpha) / d
            U -= lr * (grad_U + lam * reg_grad)

            avg_err = torch.sqrt((scale * res.pow(2)).sum())/m_obs
            err_hist.append(avg_err.item())

            if it > 5:
                if err_hist[-1] > err_hist[-2] or abs(err_hist[-1] - err_hist[-2]) < tol:
                    break
            loop.set_description(f"err: {avg_err}, grad: {grad_U.sum().item()}, reg_grad: {reg_grad.sum().item()}")
        self.U = U
        self.err_history = err_hist
        del idx, vals
        return U, err_hist[-1]
    
    def error(self):
        """
        Compute the error of the imputed matrix.
        """
        U = self.U
        idx = self.S.indices().to(self.device)
        vals = self.S.values().to(self.device)
        d, r = U.shape
        m = idx.shape[1]

        # Compute the error
        ui = U[idx[0]]
        uj = U[idx[1]]
        res = (ui * uj).sum(dim=1) - vals
        avg_err = torch.sqrt((res.pow(2)).sum())/m
        del idx, vals
        return avg_err