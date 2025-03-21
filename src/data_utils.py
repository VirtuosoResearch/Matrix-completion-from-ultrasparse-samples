import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import cyvcf2
from tqdm import tqdm
import h5py
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder

import os
import pickle
import re

data_path = '../data/'
vcf_path = '../data/gene/'

def load_data_all(dataset, s=None):
    if dataset == 'random':
        mat1 = torch.randn(100,1)
        mat2 = torch.randn(1, 5)
        matrix = mat1 @ mat2
        return mat1, mat2, matrix

    elif dataset == 'ml-1m':
        # read table
        ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
        ratings = pd.read_table(data_path+'ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
        ratings = ratings.filter(regex='UserID|MovieID|ratings')

        # sample
        unique_ids = ratings['MovieID'].unique()
        if s is not None:
            selected_ids = np.random.choice(unique_ids, size=s, replace=False)
            selected_data = ratings[ratings['MovieID'].isin(selected_ids)]
        else:
            selected_data = ratings[ratings['MovieID'].isin(unique_ids)]
        #print(selected_data)
        

        #selected_data = torch.tensor(selected_data.values)
        matrix_df = selected_data.pivot_table(index='UserID', columns='MovieID', values='ratings', aggfunc='first')
        matrix_df = matrix_df.fillna(0)
        matrix = torch.tensor(matrix_df.values)

        return matrix
    
    elif dataset == 'ml-10m':
        # read table
        ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
        ratings = pd.read_table(data_path+'ml-10M100K/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
        ratings = ratings.filter(regex='UserID|MovieID|ratings')

        # sample
        unique_ids = ratings['MovieID'].unique()
        if s is not None:
            selected_ids = np.random.choice(unique_ids, size=s, replace=False)
            selected_data = ratings[ratings['MovieID'].isin(selected_ids)]
        else:
            selected_data = ratings[ratings['MovieID'].isin(unique_ids)]
        #print(selected_data)
        

        #selected_data = torch.tensor(selected_data.values)
        matrix_df = selected_data.pivot_table(index='UserID', columns='MovieID', values='ratings', aggfunc='first')
        matrix_df = matrix_df.fillna(0)
        matrix = torch.tensor(matrix_df.values)

        return matrix
    
    elif dataset == 'ml-32m':
        file_path = data_path+'ml-32m/matrix.pt'
        if os.path.exists(file_path):
            sparse_matrix = torch.load(file_path, weights_only=False)
        else:
            
            data = pd.read_csv(data_path+'ml-32m/ratings.csv')

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
    
    elif dataset == 'netflix':
        df = pd.read_csv(
            data_path+"netflixdata.csv",
            names=["movie_id", "user_id", "rating", "date"],
            parse_dates=["date"],
            encoding="ISO-8859-1",
            engine="python",
        )
        data = df[["movie_id", "user_id", "rating"]]
        # sample
        unique_ids = data['movie_id'].unique()
        if s is not None:
            selected_ids = np.random.choice(unique_ids, size=s, replace=False)
            selected_data = data[data['movie_id'].isin(selected_ids)]
        else:
             selected_data = data[data['movie_id'].isin(unique_ids)]

        #selected_data = torch.tensor(selected_data.values)
        matrix_df = selected_data.pivot_table(index='user_id', columns='movie_id', values='rating', aggfunc='first')
        matrix_df = matrix_df.fillna(0)
        matrix = torch.tensor(matrix_df.values)
        print(matrix.shape)

        return matrix
    
    elif dataset == 'gene':
        data = np.load(vcf_path)
        return torch.tensor(data, dtype=float)
    
    elif dataset == 'sweet':
        # read table
        ratings_title = ['product','user', 'value']
        df = pd.read_csv('../data/sweet.csv')
        product_ids = {product: idx for idx, product in enumerate(df['product'].unique())}
        df['product_id'] = df['product'].map(product_ids)

        # 使用pivot_table生成用户-产品评分矩阵
        rating_matrix = df.pivot_table(index='user', columns='product_id', values='value', fill_value=0)

        # 将DataFrame转换为numpy矩阵
        matrix_np = rating_matrix.values
        matrix = torch.tensor(matrix_np)

        return matrix
    
    elif dataset in ['douban', 'flixster', 'yahoo_music']:
        path_file = f'../data/{dataset}.mat'
        db = h5py.File(path_file, 'r')
        ds = db['M']
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir = np.asarray(ds['ir'])
                jc = np.asarray(ds['jc'])
                out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
            out = np.asarray(ds).astype(np.float32).T

        db.close()

        d1, d2 = out.shape
        if s is not None:
            selected_ids = np.random.choice(d2, size=s, replace=False)

            return torch.tensor(out)[:,selected_ids]
        else:
            return torch.tensor(out)
        #return torch.tensor(out)
    
    elif dataset == "All_Beauty":
        file_path = data_path+f'amazon/{dataset}/rating_matrix_sparse.pt'
        print(file_path)
        if os.path.exists(file_path):
            sparse_matrix = torch.load(file_path, weights_only=False)
            sparse_matrix = sparse_matrix.astype(float)
        return sparse_matrix

def load_data_syn(r=5, d1=5000, d2=2000, device='cpu'):
    X = torch.normal(2, 1, size = (d1, d2)).to(device)

    U, D, Vt = torch.linalg.svd(X, full_matrices=False)
    D[r:] = 0
    X = U @ torch.diag(D) @ Vt

    return X

def get_masks(M, p):
    d1, d2 = M.shape
    k = int(d1*d2*p)
    print(k)
    non_zero_indices = torch.nonzero(M, as_tuple=False)
    sampled_indices = non_zero_indices[torch.randperm(non_zero_indices.size(0))[:k]]
    masks = torch.zeros_like(M, dtype=int)

    # Set the sampled indices to 1 in the mask
    for idx in sampled_indices:
        masks[idx[0], idx[1]] = 1
    observed_M = M*masks
    same_nonzero_locations = np.array_equal((observed_M != 0), (masks != 0))
    if same_nonzero_locations:
        print("Non-zero elements are correctly aligned with the mask.")
    else:
        print("Non-zero elements do not align correctly with the mask.")
    return observed_M, masks

def get_uniform_masks(M, p):
    M_shape = M.shape
    masks = torch.rand(M_shape[0], M_shape[1]).to(M.device) <= p
	#observed_M = np.multiply(M, masks)
    observed_M = M * masks

    return observed_M, masks

def set_random_zeros(matrix, ratio):
    """
    Set a certain ratio of elements to 0 randomly in a full matrix.

    Args:
    matrix (torch.Tensor): The full input matrix.
    ratio (float): The ratio of elements to set to 0 (0 <= ratio <= 1).

    Returns:
    torch.Tensor: A new matrix with the specified ratio of elements set to 0.
    """
    # Get the number of elements in the matrix
    num_elements = matrix.numel()

    # Compute the number of elements to be set to zero
    num_zeros = int(num_elements * ratio)

    # Flatten the matrix to 1D
    flattened_matrix = matrix.view(-1)

    # Generate random indices to set to 0
    zero_indices = torch.randperm(num_elements)[:num_zeros]

    # Set the selected indices to 0
    flattened_matrix[zero_indices] = 0

    # Reshape the matrix back to its original shape
    return flattened_matrix.view_as(matrix)