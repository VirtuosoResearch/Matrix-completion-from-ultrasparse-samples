import torch
import numpy as np

import torch

import torch

import torch

# Row-wise normalization to M. Each row of M is normalized to have mean 2/sqrt(d2) and var 1/d2
def rowwise_normalize(M):
    d1, d2 = M.shape
    target_mean = 2 / d2**0.5
    target_std = (1 / d2)**0.5

    # Mask of non-zero values
    mask = M != 0
    mask_f = mask.float()

    # Count of non-zero values per row
    count_nonzero = mask_f.sum(dim=1, keepdim=True)

    # Avoid divide-by-zero: will use this to skip all-zero rows
    valid_rows = count_nonzero > 0

    # Compute masked mean and std only for valid rows
    sum_nonzero = (M * mask_f).sum(dim=1, keepdim=True)
    mean = sum_nonzero / count_nonzero.clamp(min=1)

    var = (((M - mean) * mask_f) ** 2).sum(dim=1, keepdim=True) / count_nonzero.clamp(min=1)
    std = var.sqrt()

    # Normalize non-zero values for valid rows
    M_norm = (M - mean) / std * target_std + target_mean

    # Keep original zeros
    M_out = torch.where(mask, M_norm, M)

    # Keep all-zero rows unchanged
    M_out[~valid_rows.squeeze()] = M[~valid_rows.squeeze()]

    return M_out

def compute_err_tensor(estimation_matrix, groundtruth_matrix, mask=None):
    if mask is not None:
        #return torch.norm((estimation_matrix - groundtruth_matrix)[mask], 'fro')
        A = estimation_matrix - groundtruth_matrix
        mask = estimation_matrix != 0
        return torch.sqrt(torch.sum((A[mask])**2))
    else:
        return torch.norm(estimation_matrix - groundtruth_matrix, 'fro') 


def normalize_second_moment_groundtruth(M, d1, full=True):
    # Use full when the original matrix M is fully observed
    if full:
        MTM = M.T @ M
        return MTM / d1
    # Use full=False when the original matrix M is partially observed, like in the case of MovieLens or Amazon
    else:
        second_moment_M_count = (1 * (M != 0)).float().T @ (1 * (M != 0).float())
        second_moment_M_count = second_moment_M_count + (second_moment_M_count == 0) * 1
        MTM = M.T @ M
        return MTM / second_moment_M_count


def iipw_T(observed_M):
    second_moment_observe_M =  observed_M.T @ observed_M
    second_moment_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
    second_moment_observe_count = second_moment_observe_count + (second_moment_observe_count == 0) * 1
    p_hat_matrix = second_moment_observe_count

    T = second_moment_observe_M / p_hat_matrix
    return T

def iipw_T_sparse(observed_M):
    # Ensure observed_M is in sparse format
    if not observed_M.is_sparse:
        observed_M = observed_M.to_sparse()
    
    # Compute observed_M.T @ observed_M using sparse matrix multiplication
    second_moment_observe_M = torch.sparse.mm(observed_M.transpose(0, 1), observed_M)
    
    # Create a binary version of observed_M (all nonzeros become 1)
    ones = torch.ones_like(observed_M._values())
    binary_observed = torch.sparse_coo_tensor(observed_M._indices(), ones, observed_M.size())
    
    # Compute the count matrix as (binary_observed.T @ binary_observed)
    second_moment_observe_count = torch.sparse.mm(binary_observed.transpose(0, 1), binary_observed)
    
    # Convert sparse results to dense to perform elementwise operations
    dense_M = second_moment_observe_M.to_dense()
    dense_count = second_moment_observe_count.to_dense()
    
    # For any entry where count is 0, add 1 (to avoid division by zero)
    dense_count = dense_count + (dense_count == 0).float()
    
    # Compute the final T matrix elementwise
    T = dense_M / dense_count
    return T

def prob_T(observed_M, p):
    d1, d2 = observed_M.shape
    second_moment_observe_M =  observed_M.T @ observed_M
    diag = torch.diag( torch.diag(second_moment_observe_M) )
    #T_p = (1.0 / p - 1.0 / p**2) * diag + (1.0 / (p**2)) * second_moment_observe_M 
    T_p = ((1.0 / p) * diag + (1.0 / (p**2)) * (second_moment_observe_M - diag))
    T_p = T_p / d1

    return T_p

def prob_T_sparse(observed_M, p):
    d1, d2 = observed_M.shape

    # Ensure observed_M is sparse
    if not observed_M.is_sparse:
        observed_M = observed_M.to_sparse()

    # Compute second moment: observed_M.T @ observed_M
    second_moment_observe_M = torch.sparse.mm(observed_M.transpose(0, 1), observed_M)

    # Extract the diagonal (as a dense vector)
    diag_vals = torch.diag(second_moment_observe_M.to_dense())

    # Construct sparse diagonal matrix
    diag_indices = torch.arange(d2, device=observed_M.device).unsqueeze(0).repeat(2, 1)
    diag_sparse = torch.sparse_coo_tensor(diag_indices, diag_vals, (d2, d2))

    # Compute T_p
    term1 = (1.0 / p) * diag_sparse.to_dense()
    term2 = (1.0 / (p**2)) * (second_moment_observe_M.to_dense() - diag_sparse.to_dense())
    T_p = (term1 + term2) / d1

    return T_p


def iipw_imputation_X_(M, observed_M, mask):
    T = iipw_T(observed_M)

def compute_var(matrices):
    # Stack into a 3D tensor: shape (num_matrices, rows, cols)
    stacked = torch.stack(matrices)

    # Create a mask for non-zero values
    mask = (stacked != 0).float()

    # Compute sum and count for masked elements
    sum_vals = (stacked * mask).sum(dim=0)
    count_vals = mask.sum(dim=0)
    print("=========> Computing variance")
    print(f"average count: {count_vals.mean()}")
    print(f"minimum count: {count_vals.min()}")

    # Compute mean ignoring zeros
    mean_vals = sum_vals / count_vals.clamp(min=1)  # avoid division by zero

    # Compute squared difference from mean, mask out zeros
    squared_diff = ((stacked - mean_vals)**2) * mask

    # Compute variance ignoring zeros
    var_vals = squared_diff.sum(dim=0) / count_vals.clamp(min=1)

    # Optional: Set variance to 0 where count is 0 (no non-zero entries)
    var_vals[count_vals == 0] = 0.0

    #print("Variance matrix:")
    #print(var_vals)
    return var_vals
