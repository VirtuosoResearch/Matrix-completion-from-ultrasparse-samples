import torch
import numpy as np
import gc
from datetime import datetime
from argparse import ArgumentParser

from src.iipw_sparse import IIPW_sparse, sparse_elementwise_divide, sparse_empirical_reweight_estimate
from src.utils import load_data, scipy_sparse_to_torch, torch_sparse_to_scipy
from src.utils import get_uniform_masks, get_random_samples_per_row, get_sparse_masks, sample_sparse_cols
from src.row_recovery import lstsq_recovery, optimize_recovery, optimize_recovery_from_sparse

if __name__ == "__main__":
    parser = ArgumentParser()
    # Experiment parameters
    parser.add_argument("--dataset", type=str, default="ml-20m")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    # matrix parameters
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--d", type=int, default=1000)
    # sample parameters
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--ob", type=int, default=2)
    # gradient descent parameters
    parser.add_argument("--lr", type=float, default=100)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--lam_alpha", type=float, default=1e-4)
    parser.add_argument("--n_iter", type=int, default=1000)
    args = parser.parse_args()

    if torch.cuda.is_available(): 
        device = 'cuda:0'
    else:
        device = 'cpu'

    err_list, rmse_list = [], []

    # dataset
    dataset = args.dataset
    print(dataset)

    M = load_data(dataset)
    M = scipy_sparse_to_torch(M)
    M = M.coalesce()
    M = M.to(device)
    S = sparse_empirical_reweight_estimate(M, device)
    M = M.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    n, d = M.shape
    p = args.p
    r = args.r


    # main part
    for run in range(args.runs):
        observed_M, masks = get_sparse_masks(M, p)
        observed_M = observed_M.coalesce()
        masks = masks.coalesce()
        T = sparse_empirical_reweight_estimate(observed_M, device)
        T = T.coalesce()
        
        iipw = IIPW_sparse(n=n, d=d, S=S, T=T, r=args.r, p=p, device=device)
        del S, T
        U, _ = iipw.impute(n_iter=args.n_iter, lr=args.lr, alpha=args.alpha, lam=args.lam_alpha, tol=1e-9)
        err = iipw.error()
        V = U.detach()
        M = M.to(device)
        masks = masks.to(device)
        rmse = optimize_recovery_from_sparse(M=M, masks=masks, r=r, V=V, lr=1, epochs=2000, tol=1e-11, lam=1e-9)
        #rmse = lstsq_recovery(estimation_goal=None, M=M, r=r, recovery_masks=recovery_masks, V=U.detach().T, use_reg=True, lam=0.0001)

        err_list.append(err.item())
        rmse_list.append(rmse)


    err_array = np.array(err_list)
    err_mean = np.mean(err_array)
    err_std = np.std(err_array)

    rmse_array = np.array(rmse_list)
    rmse_mean = np.mean(rmse_array)
    rmse_std = np.std(rmse_array)

    # Define the content in the desired format
    content = f"Dataset: {dataset}, run times: {args.runs}\n"
    content += f"err: {err_mean:.4f}+-{err_std:.4f}\nrmse err: {rmse_mean:.4f}+-{rmse_std:.4f}\n"
    content += '\n'
    print(content)


    