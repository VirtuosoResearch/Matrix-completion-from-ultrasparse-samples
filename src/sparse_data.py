import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

from utils import *
from data import *
from data_utils import *
from power_method_svd import power_svd
from sparse_utils import *

device = 'cuda:0'

dataset = 'ml-32m'
p=0.1
r=5
M_all = load_data_all(dataset)
print(M_all.shape)
M_dataset = MatrixDataset(M_all, 110000, 10000)
matrix_dataloader = DataLoader(
    M_dataset,
    batch_size=1,         # You can adjust the batch size as needed
    shuffle=False,
    collate_fn=sparse_collate_fn  # Pass the custom collate function
)
for batch in matrix_dataloader:
    M = batch[0]
    print(M.shape)