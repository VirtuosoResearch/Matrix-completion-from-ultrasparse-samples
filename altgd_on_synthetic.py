from argparse import ArgumentParser
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from utils import *
from src.iipw import *
from src.recovery import vanilla_MC

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

    return rmse_err

if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--d1", type=int, default=10000)
    parser.add_argument("--d2", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--ob", type=int, default=-1)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=10)
    parser.add_argument("--delta", type=float, default=0)
    parser.add_argument("--mark", type=str, default="none")
    parser.add_argument("--use_reg", action="store_true", default=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # dataset
    dataset = args.dataset
    print(dataset)

    d1 = args.d1
    d2 = args.d2
    #M = load_data_syn(args.r, d1, d2, device)
    M = load_normalized_data_syn(args.r, d1, d2, device)

    # privacy
    if args.delta == 0:
        delta = 1/d1
    else:
        delta = args.delta
    sigma = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/args.epsilon

    # main part
    err_list = []
    rmse_list = []    
    for run in range(args.runs):
        # sample observed data
        p = args.p
        r = args.r
        recovery_p = 0.75
        if args.ob > 0:
            observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), args.ob)
            observed_M = torch.from_numpy(observed_M).float().to(device)
            masks = torch.from_numpy(masks).to(device)
        else:
            observed_M, masks = get_uniform_masks(M, p)
        
        _, recovery_masks = get_uniform_masks(M, recovery_p)

        # impute missing values from rank-r SVD corresponding to masks
        X = vanilla_MC(M, masks, int(d2*d2), r, draw=False)
        estimation_matrix = (X.T @ X)/d1
        err = torch.norm((M.T @ M)/d1 - estimation_matrix)
        print(err)
        err_list.append(err.item())
        
        # user-level recovery using least square
        print('User-level recovery...')
        lam = 0.0001
        rmse = lstsq_recovery(estimation_goal=estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
        rmse_list.append(rmse)

    # results of runs
    err_mean = np.mean(err_list)
    err_std = np.std(err_list)
    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    # Define the content in the desired format
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"Time: {time}\n p: {args.p}, ob: {args.ob}\n Synthetic data: d1={d1}, d2={d2}:\n\
 Estimation error: {err_mean:.7f}+-{err_std:.7f}, user-level recovery RMSE: {rmse_mean:.7f}+-{rmse_std:.7f}\n"
    content += '\n'
    print(content)
    label = f"altgd_synthetic_d1_{d1}_d2_{d2}_r{r}_p{p}_ob{args.ob}"
    log_file = f"./logs/{label}.txt"
    with open(log_file, "a") as f:
        f.write(content)

    