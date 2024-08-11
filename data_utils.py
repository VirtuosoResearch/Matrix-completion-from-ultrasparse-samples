import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import cyvcf2
from tqdm import tqdm

import os
import pickle
import re

data_path = './data/'
vcf_path = './data/gene/'

def load_data_all(dataset, s=200):
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
        selected_ids = np.random.choice(unique_ids, size=s, replace=False)
        selected_data = ratings[ratings['MovieID'].isin(selected_ids)]

        #selected_data = torch.tensor(selected_data.values)
        matrix_df = selected_data.pivot_table(index='UserID', columns='MovieID', values='ratings', aggfunc='first')
        matrix_df = matrix_df.fillna(0)
        matrix = torch.tensor(matrix_df.values)

        return matrix
    
    elif dataset == 'netflix':
        df = pd.read_csv(
            data_path,
            names=["movie_id", "user_id", "rating", "date"],
            parse_dates=["date"],
            encoding="ISO-8859-1",
            engine="python",
        )
        data = df[["movie_id", "user_id", "rating"]]
        # sample
        unique_ids = data['movie_id'].unique()
        selected_ids = np.random.choice(unique_ids, size=s, replace=False)
        selected_data = data[data['movie_id'].isin(selected_ids)]

        #selected_data = torch.tensor(selected_data.values)
        matrix_df = selected_data.pivot_table(index='user_id', columns='movie_id', values='rating', aggfunc='first')
        matrix_df = matrix_df.fillna(0)
        matrix = torch.tensor(matrix_df.values)

        return matrix
    
    elif dataset == 'gene':
        data = np.load(vcf_path)
        return torch.tensor(data, dtype=float)
    
    elif dataset == 'sweet':
        # read table
        ratings_title = ['product','user', 'value']
        df = pd.read_csv('./data/sweet.csv')
        product_ids = {product: idx for idx, product in enumerate(df['product'].unique())}
        df['product_id'] = df['product'].map(product_ids)

        # 使用pivot_table生成用户-产品评分矩阵
        rating_matrix = df.pivot_table(index='user', columns='product_id', values='value', fill_value=0)

        # 将DataFrame转换为numpy矩阵
        matrix_np = rating_matrix.values
        matrix = torch.tensor(matrix_np)

        return matrix

def load_data_syn(r=5, d1=5000, d2=200):
    mat1 = torch.randn(d1,r)
    mat2 = torch.randn(r, d2)
    matrix = mat1 @ mat2
    return mat1, mat2, matrix