import os
import torch
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder


def load_data(dataset):
    data_path = '../data/'
    if dataset == 'ml-32m':
        file_path = data_path+'ml-32m/matrix.pt'
        if os.path.exists(file_path):
            sparse_matrix = torch.load(file_path, weights_only=False)
        else:
            
            data = pd.read_csv(data_path+'ml-32m/ratings.csv')

            if data['userId'].dtype != 'int':
                data['userId'] = data['userId'].astype(int)

            movie_encoder = LabelEncoder()
            data['movieId'] = movie_encoder.fit_transform(data['movieId'])
            data['rating'] = data['rating'].astype(float)

            num_users = data['userId'].nunique()
            num_movies = data['movieId'].nunique()

            # Create row, col, and data arrays for the sparse matrix
            row = data['userId'].values
            col = data['movieId'].values
            data = data['rating'].values
            sparse_matrix = coo_matrix((data, (row, col)))

            print("Sparse matrix shape: ", sparse_matrix.shape)
            print("Non-zero entries: ", sparse_matrix.nnz)
            print("Number of unique users: ", num_users)
            print("Number of unique movies: ", num_movies)

            torch.save(sparse_matrix, file_path)

        return sparse_matrix
    
    elif dataset == 'ml-25m':
        file_path = data_path+'ml-25m/matrix.pt'
        if os.path.exists(file_path):
            sparse_matrix = torch.load(file_path, weights_only=False)
        else:
            
            data = pd.read_csv(data_path+'ml-25m/ratings.csv')

            if data['userId'].dtype != 'int':
                data['userId'] = data['userId'].astype(int)

            movie_encoder = LabelEncoder()
            data['movieId'] = movie_encoder.fit_transform(data['movieId'])
            data['rating'] = data['rating'].astype(float)

            num_users = data['userId'].nunique()
            num_movies = data['movieId'].nunique()

            # Create row, col, and data arrays for the sparse matrix
            row = data['userId'].values
            col = data['movieId'].values
            data = data['rating'].values
            sparse_matrix = coo_matrix((data, (row, col)))

            print("Sparse matrix shape: ", sparse_matrix.shape)
            print("Non-zero entries: ", sparse_matrix.nnz)
            print("Number of unique users: ", num_users)
            print("Number of unique movies: ", num_movies)

            torch.save(sparse_matrix, file_path)

        return sparse_matrix
    
    elif dataset == 'ml-20m':
        file_path = data_path+'ml-20m/matrix.pt'
        if os.path.exists(file_path):
            sparse_matrix = torch.load(file_path, weights_only=False)
        else:
            
            data = pd.read_csv(data_path+'ml-20m/ratings.csv')

            if data['userId'].dtype != 'int':
                data['userId'] = data['userId'].astype(int)

            movie_encoder = LabelEncoder()
            data['movieId'] = movie_encoder.fit_transform(data['movieId'])
            # Ratings should be float, just in case, let's convert
            data['rating'] = data['rating'].astype(float)

            num_users = data['userId'].nunique()
            num_movies = data['movieId'].nunique()

            # Create row, col, and data arrays for the sparse matrix
            row = data['userId'].values
            col = data['movieId'].values
            data = data['rating'].values
            sparse_matrix = coo_matrix((data, (row, col)))

            print("Sparse matrix shape: ", sparse_matrix.shape)
            print("Non-zero entries: ", sparse_matrix.nnz)
            print("Number of unique users: ", num_users)
            print("Number of unique movies: ", num_movies)

            torch.save(sparse_matrix, file_path)

        return sparse_matrix

def get_random_samples_per_row(M, entries_per_row):
	row = []
	col = []
	data = []

	M_shape = M.shape
	masked_matrix = np.zeros(M_shape)
	for i in range(M_shape[0]):
		js = np.random.permutation(M_shape[1])
		for j in range(entries_per_row):
			row.append(i)
			col.append(js[j])
			data.append(M[i, js[j]])
			masked_matrix[i, js[j]] = 1

	observed_M = csr_matrix((data, (row, col)), shape=M_shape)
	return observed_M.toarray(), masked_matrix

import torch

def random_sample_k_per_row(M, k):
    d1, d2 = M.shape

    # For each row, generate a random permutation of column indices
    rand_indices = torch.rand(d1, d2).argsort(dim=1)
    
    # Create a mask with k entries per row set to 1
    mask = torch.zeros_like(M, dtype=torch.bool)
    row_indices = torch.arange(d1).unsqueeze(1)
    col_indices = rand_indices[:, :k]
    mask[row_indices, col_indices] = 1

    # Apply the mask to get the observed matrix
    observed_M = M * mask

    return observed_M, mask


def get_uniform_masks(M, p):
    M_shape = M.shape
    masks = torch.rand(M_shape[0], M_shape[1]).to(M.device) <= p
	#observed_M = np.multiply(M, masks)
    observed_M = M * masks

    return observed_M, masks

def get_sparse_masks(M, p):
    original_indices = M.indices()
    original_values = M.values()
    total_nonzeros = original_values.size(0)
    num_samples = int(total_nonzeros * p)
    permuted_indices = torch.randperm(total_nonzeros)
    sampled_indices = permuted_indices[:num_samples]

    sampled_coords = original_indices[:, sampled_indices]  # Shape: (2, num_samples)
    sampled_values = original_values[sampled_indices]     # Shape: (num_samples,)

    observed_M = torch.sparse_coo_tensor(sampled_coords, sampled_values, M.size(), device=M.device)
    observed_M = observed_M.coalesce()  # Ensure the indices are coalesced

    # Create the mask sparse matrix (with 1s at sampled positions)
    mask_values = torch.ones(num_samples, dtype=original_values.dtype, device=M.device)
    mask = torch.sparse_coo_tensor(sampled_coords, mask_values, M.size(), device=M.device)
    mask = mask.coalesce()

    return observed_M, mask

# top r svd
def top_r_svd(A, r):
    V, S, Vt = torch.linalg.svd(A, full_matrices=True)

    Vr = V[:, :r]  
    Sr = S[:r]   
    Vtr = Vt[:r, :]
    return Vr, Sr, Vtr

def torch_sparse_to_scipy(sparse_tensor):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    shape = sparse_tensor.shape
    indices = indices.cpu().numpy()
    values = values.cpu().numpy()
    return scipy.sparse.coo_matrix((values, indices), shape=shape)

def scipy_sparse_to_torch(scipy_sparse, device='cpu'):
    scipy_sparse = scipy_sparse.tocoo()
    indices = torch.tensor([scipy_sparse.row, scipy_sparse.col], dtype=torch.long, device=device)
    values = torch.tensor(scipy_sparse.data, dtype=torch.float32, device=device)
    shape = scipy_sparse.shape
    return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

def sample_sparse_cols(M, sample_size):
    num_cols = M.size(1)
    sampled_cols = torch.randperm(num_cols)[:sample_size]

    # Step 2: Extract entries where column index is in sampled_cols
    # Get the COO format indices and values
    indices = M.indices()  # shape (2, nnz)
    values = M.values()    # shape (nnz,)

    # Mask for selected columns
    col_indices = indices[1]  # columns of nonzero entries
    mask = (col_indices.unsqueeze(1) == sampled_cols.unsqueeze(0)).any(dim=1)

    # Apply mask
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Step 3: Adjust column indices to new range (0 to sample_size-1)
    # Map old sampled column indices to new indices
    col_mapping = {old_col.item(): new_col for new_col, old_col in enumerate(sampled_cols)}

    # Remap columns
    new_indices[1] = new_indices[1].apply_(lambda x: col_mapping[x])

    # Step 4: Create new sparse matrix
    new_sparse_mat = torch.sparse_coo_tensor(
        new_indices,
        new_values,
        size=(M.size(0), sample_size)
    )

    new_sparse_mat = new_sparse_mat.coalesce()  # ensure it’s properly coalesced

    return new_sparse_mat