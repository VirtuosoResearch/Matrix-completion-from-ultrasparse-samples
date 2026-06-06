import os
import torch
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder

def load_syn_data_low_rank(r=5, d1=5000, d2=2000, device='cpu'):
    mean = 2 / np.sqrt(d2)
    sigma = 1 / d2
    X = torch.normal(mean, sigma, size = (d1, d2)).to(device)

    U, D, Vt = torch.linalg.svd(X, full_matrices=False)
    D[r:] = 0
    X = U @ torch.diag(D) @ Vt

    return X

def load_syn_data_mixture_model(r=5, d1=5000, d2=2000, device='cpu'):
    factors = np.random.normal(0, 1, size=(r, d2))
    weights = np.random.dirichlet(np.ones(r), size=d1)
    X = weights @ factors

    return torch.tensor(X, dtype=torch.float32).to(device)


def load_data(dataset):
    data_path = './data/'
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