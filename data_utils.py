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

data_path = '~/dataset/ml/'

def load_data_all(dataset, s=300):
    print(dataset)
    preprocess_file = './data/'+dataset+'.pkl'
    full_M_file = './data/'+dataset+'/matrix.pt'
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
            "./data/netflix/data.csv",
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
        M = load_data_gene()

def load_data_ml1m():
    """
    Load Dataset from File
    """
    #读取User数据
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table(data_path+'ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    #改变User数据中性别和年龄
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #读取Movie数据集
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(data_path+'ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python', encoding='ISO-8859-1')
    movies_orig = movies.values
    #将Title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    #电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val:ii for ii, val in enumerate(genres_set)}

    #将电影类型转成等长数字列表，长度是18
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    #电影Title转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val:ii for ii, val in enumerate(title_set)}

    #将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    #读取评分数据集
    ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table(data_path+'ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    #合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    
    #将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig

def load_data_netflix():
    df = pd.read_csv(
        "./data/netflix/data.csv",
        names=["movie_id", "user_id", "rating", "date"],
        parse_dates=["date"],
        encoding="ISO-8859-1",
        engine="python",
    )
    data = df[["movie_id", "user_id", "rating"]]
    tensor = torch.tensor(data.values)
    print(tensor.shape)
    return tensor

def load_data_gene():
    data_file = './data/gene/data.pt'

    vcf = cyvcf2.VCF('./data/gene/ALL_GGVP.chr1.shapeit2_integrated_snvindels_v1b_20200120.GRCh38.phased.vcf')
    matrix = []

    if not os.path.exists(data_file):
        for variant in vcf:
            #print(variant.CHROM, variant.POS, variant.REF, variant.ALT, variant.QUAL, variant.FILTER, variant.INFO)
            ref_vector = encode_sequence(variant.REF)
            alt_vector = encode_sequence(variant.ALT[2:-2])

            row = torch.cat((ref_vector, alt_vector), dim=0)
            row = torch.cat((torch.tensor([int(variant.CHROM)]), row), dim=0)

            matrix.append(row.unsqueeze(0))

        matrix = torch.cat(matrix, dim=0)
        torch.save(matrix, data_file)
    else:
        matrix = torch.load(data_file)
        
    return matrix



def encode_sequence(sequence):
    base_vectors = {
        'A': torch.tensor([0.0, 0.0, 0.0, 1.0]),
        'C': torch.tensor([0.0, 0.0, 1.0, 0.0]),
        'G': torch.tensor([0.0, 1.0, 0.0, 0.0]),
        'T': torch.tensor([1.0, 0.0, 0.0, 0.0]),
        'N': torch.tensor([0.0, 0.0, 0.0, 0.0]) 
    }

    total_vector = torch.zeros(4)
    
    valid_bases = 0  
    for base in sequence:
        if base in base_vectors:
            total_vector += base_vectors[base]
            valid_bases += 1

    if valid_bases > 0:
        average_vector = total_vector / valid_bases
    else:
        average_vector = total_vector  
    
    return average_vector