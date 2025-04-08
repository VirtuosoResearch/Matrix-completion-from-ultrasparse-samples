from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader

from utils import *
from power_method_svd import power_svd
from src.iipw import IIPW

def sparse_collate_fn(batch):
    return batch

class MatrixDataset(Dataset):
    def __init__(self, matrix, d1, d2):
        """
        Args:
            matrix (numpy array or tensor): The matrix to be split into chunks.
            d1 (int): Number of rows to extract per iteration.
        """
        self.matrix = scipy_sparse_to_torch(matrix)
        if not self.matrix.is_coalesced():
            self.matrix = self.matrix.coalesce()
        self.d1 = d1
        self.d2 = d2

        self.total_rows = self.matrix.shape[0]
        self.total_cols = self.matrix.shape[1]

        self.num_row_blocks = (self.total_rows + d1 - 1) // d1
        self.num_col_blocks = (self.total_cols + d2 - 1) // d2
        self.total_blocks = self.num_row_blocks * self.num_col_blocks

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        """
        Return a sparse submatrix of size d1 x d2.
        """
        # Compute the block's row and column indices
        row_block_idx = idx // self.num_col_blocks
        col_block_idx = idx % self.num_col_blocks

        # Compute row and column ranges
        start_row = row_block_idx * self.d1
        end_row = min(start_row + self.d1, self.total_rows)

        start_col = col_block_idx * self.d2
        end_col = min(start_col + self.d2, self.total_cols)

        # Get the coalesced sparse tensor's indices and values
        sparse_indices = self.matrix._indices()  # Shape: (2, nnz)
        sparse_values = self.matrix._values()    # Shape: (nnz,)

        # Create a mask to filter indices within the block
        mask = (sparse_indices[0] >= start_row) & (sparse_indices[0] < end_row) & \
               (sparse_indices[1] >= start_col) & (sparse_indices[1] < end_col)

        # Filter indices and values
        filtered_indices = sparse_indices[:, mask].clone()
        filtered_values = sparse_values[mask].clone()

        # Adjust indices to be relative to the block
        filtered_indices[0] -= start_row
        filtered_indices[1] -= start_col

        # Create a new sparse tensor for this block
        block_size = (end_row - start_row, end_col - start_col)
        sparse_block = torch.sparse.FloatTensor(
            filtered_indices,
            filtered_values,
            torch.Size(block_size)
        )

        return sparse_block

def lstsq_recovery(estimation_goal, M, masks, r, recovery_masks, use_reg=False, lam=0.001):
    recovery_masks = recovery_masks.bool()
    device = M.device
    error_list = []
    test_num = 0
    total_num = 0
    U = []
    _, S, V = top_r_svd(estimation_goal, r)
    for i in tqdm(range(M.shape[0])):
        """
        M_row: 1*d2
        V: r*d2
        user_vector: 1*r
        """
        # make sure there is non-zero value in original matrix
        M_row = M[i]
        non_zero_indices = M_row.nonzero(as_tuple=True)[0]

        # select non-zero & masked elements as training data, non-zero & non-masked elements as testing data
        # Get the mask values for the non-zero indices
        mask_values = recovery_masks[i][non_zero_indices]

        # Use boolean indexing to separate trian and test indices
        train_idx = non_zero_indices[mask_values]
        test_idx = non_zero_indices[~mask_values]

        M_row = M_row.unsqueeze(0)

        num_train = train_idx.numel()
        num_test = test_idx.numel()

        if num_train < 1 or num_test < 1:
            #skip +=1
            continue
        
        test_num += num_test
        total_num += num_test + num_train

        # formula AX=B
        train_A = V[:, train_idx].t()
        train_B = M_row[:, train_idx].t()
        test_A = V[:, test_idx].t()
        test_B = M_row[:, test_idx].t()

        if num_train == 1:
            train_A.squeeze()
        if use_reg:
            # Ridge regression
            lambda_reg = lam
            I = torch.eye(train_A.shape[1], device=device) * lambda_reg
            u = torch.linalg.lstsq(train_A.t() @ train_A + I, train_A.t() @ train_B).solution
        else:
            # Linear regression
            u = torch.linalg.lstsq(train_A, train_B).solution   
        U.append(u.t())

        check_vec = u
        if torch.any(torch.isnan(check_vec)):
            print(f"NAN in epoch: {i}")
            print(check_vec)
            break

        # stat |AX-B| by individual elements
        error_list.append(torch.sum((test_A @ u - test_B)**2).item())

    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/test_num)

    return rmse_err**2, test_num

if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="movielens")
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=10)
    parser.add_argument("--delta", type=float, default=10e-5)
    parser.add_argument("--use_reg", action="store_true", default=True)
    args = parser.parse_args()

    if torch.cuda.is_available(): 
        device = 'cuda:0'
    else:
        device = 'cpu'

    err_list, rmse_list = [], []

    # dataset
    dataset = args.dataset
    print(dataset)

    M_all = load_data_all(dataset)
    d1_all, d2_all = M_all.shape
    all_size = d1_all * d2_all
    if dataset == 'ml-32m':
        div_d1 = 1
        div_d2 = 30
        p=0.8
    elif dataset == 'ml-20m':
        div_d1 = 1
        div_d2 = 10
        p=0.8
    elif dataset == 'ml-25m':
        div_d1 = 1
        div_d2 = 20
        p=0.8

    # split data into batch
    M_dataset = MatrixDataset(M_all, int(d1_all/div_d1)+1, int(d2_all/div_d2)+1)
    matrix_dataloader = DataLoader(
        M_dataset,
        batch_size=1,         # You can adjust the batch size as needed
        shuffle=False,
        collate_fn=sparse_collate_fn  # Pass the custom collate function
    )
    dataset_content = f'sparse d1 = {d1_all}, d2 = {d2_all}, entries = {M_all.nnz}, p = {args.p}\n'
    label = "iipw_grad_sparse_dataset" + dataset

    # privacy
    epsilon = args.epsilon
    delta = 1/d1_all
    sigma = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/epsilon

    # main part
    for run in range(args.runs):
        torch.manual_seed(run)
        total_err = 0
        total_second_observed = 0
        batch_err_list = []
        batch_rmse_list = []
        time_list = []
        cost_time = 0
        
        r = args.r
        total_test_num = 0
        for batch in matrix_dataloader:
            M = batch[0].coalesce()
            num_entries = M._nnz()
            d1, d2 = M.shape
            print(M.shape)
            batch_size = d1*d2
            observed_M, masks = get_sparse_masks(M, p)
            #recovery_p = 0.8
            #_, recovery_masks = get_sparse_masks(M, recovery_p)
            #recovery_masks = recovery_masks.to_dense().bool().to(device)
            recovery_masks = masks.to_dense().bool().to(device)
            M = M.to_dense()
            M = M.to(device)

            print("Sparse ratio: ", num_entries/batch_size)
            start_time = time.time()

            observed_M = observed_M.to_dense().to(device)
            masks = masks.to_dense().to(device)
            
            epochs = 5000
            tol = 1e-10
            lr = 1
            iipw = IIPW(M=M, observed_M=observed_M, masks=masks, r=r, missing_input=True)
            estimation_matrix, _ = iipw.impute_grad_reg(n_iter=epochs, tol=tol, lr=lr, lam=0.000001)
            S = iipw.normalized_MTM
            S_masks = S != 0
            total_second_observed += torch.sum(S_masks).item()
            err = S - estimation_matrix
            relative_err = torch.norm(err, 'fro')

            total_err += relative_err
            print("Relative error: ", relative_err / S_masks.sum())
            cost_time += time.time() - start_time
            #print(aaa)
            # user-level recovery using least square
            print('User-level recovery...')
            rmse, batch_test_num = lstsq_recovery(estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.001)
            print('batch rmse: ', np.sqrt(rmse))
            print('batch test number: ', batch_test_num)
            batch_rmse_list.append(rmse*batch_test_num)
            total_test_num += batch_test_num
            

        #batch_err = np.sum(batch_err_list) / all_size
        batch_err = total_err.item() / total_second_observed
        batch_rmse = np.sqrt(np.sum(batch_rmse_list) / total_test_num)

        err_list.append(batch_err)
        rmse_list.append(batch_rmse)
        time_list.append(cost_time)


    err_array = np.array(err_list)
    err_mean = np.mean(err_array)
    err_std = np.std(err_array)

    rmse_array = np.array(rmse_list)
    rmse_mean = np.mean(rmse_array)
    rmse_std = np.std(rmse_array)

    time_array = np.array(time_list)
    time_mean = np.mean(time_array)
    time_std = np.std(time_array)

    # Define the content in the desired format
    content = f"Dataset: {dataset}, run times: {args.runs}, cost: {time_mean}+-{time_std}\n"
    content += f"err: {err_mean:.4f}+-{err_std:.4f}\nrmse err: {rmse_mean:.4f}+-{rmse_std:.4f}\n"
    content += '\n'
    print(content)
    with open(f'./results/{label}.txt', 'a') as file:
        file.write(dataset_content)
        file.write(content)

    