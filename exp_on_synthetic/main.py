import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from src.iipw import IIPW
from src.utils import load_syn_data_common_means, load_syn_data_noisy_mixtures, load_syn_data_linear_mixtures
from src.utils import get_uniform_masks, get_random_samples_per_row
from src.row_recovery import lstsq_recovery, optimize_recovery


if __name__ == "__main__":
    parser = ArgumentParser()
    # Experiment parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    # matrix parameters
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--d", type=int, default=1000)
    # sample parameters
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--ob", type=int, default=2)
    parser.add_argument(
        "--estimator",
        choices=("hajek", "mixture-hajek"),
        default="hajek",
    )
    # gradient descent parameters
    parser.add_argument("--lr", type=float, default=1e4)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--lam_alpha", type=float, default=1e-4)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--skip_recovery", action="store_true", default=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    # dataset
    dataset = args.dataset
    print(dataset)

    n = args.n
    d = args.d
    M = load_syn_data_linear_mixtures(args.r, n, d, device)

    # main part
    err_list = []
    rmse_list = []
    for run in range(args.runs):
        # sample observed data
        p = args.p
        r = args.r
        recovery_p = 0.8
        # Observe random entries uniformly
        if args.sample == 'uniform':  
            observed_M, masks = get_uniform_masks(M, p)
        # Observe fixed number of entries per row
        elif args.sample == 'fixed':
            p = args.ob / d
            observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), args.ob)
            observed_M = torch.from_numpy(observed_M).float().to(device)
            masks = torch.from_numpy(masks).to(device)
        
        # impute missing values from rank-r SVD corresponding to masks
        iipw = IIPW(
            M=M,
            observed_M=observed_M,
            masks=masks,
            p=p,
            r=r,
            estimator=args.estimator,
        )
        U, estimation_matrix, err = iipw.impute(n_iter=args.n_iter, lr=args.lr, alpha=args.alpha, lam=args.lam_alpha, tol=1e-7)
        err = err / torch.norm(M.T @ M / n, p='fro').item()
        err_list.append(err)

        if not args.skip_recovery:
            _, recovery_masks = get_uniform_masks(M, recovery_p)
            # user-level recovery using least square
            print('User-level recovery...')
            rmse = optimize_recovery(M=M, masks=recovery_masks, r=r, V=U.detach(), lr=0.1, epochs=1000, tol=1e-11, lam=1e-10)
            # optioanlly, use lstsq_recovery
            rmse = lstsq_recovery(estimation_goal=estimation_matrix, M=M, r=r, recovery_masks=recovery_masks, V=U.detach().T, use_reg=True, lam=0.0001)
            ground_truth_rmse = torch.sqrt(torch.mean(M**2)).item()
            rmse_list.append(rmse / ground_truth_rmse)

    # results of runs
    err_mean = np.mean(err_list)
    err_std = np.std(err_list)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.sample == 'uniform':
        sampling_info = f"uniform sampling, p={args.p}"
    elif args.sample == 'fixed':
        sampling_info = f"fixed entries per row, ob={args.ob}"
    if args.skip_recovery:
        content = f"Time: {time}\n {sampling_info}\n Synthetic data: n={n}, d={d}:\n\
 Estimation error: {err_mean:.7f}+-{err_std:.7f}\n"
    else:
        rmse_mean = np.mean(rmse_list)
        rmse_std = np.std(rmse_list)
        content = f"Time: {time}\n {sampling_info}\n Synthetic data: n={n}, d={d}:\n Normalized estimation error: {err_mean:.7f}+-{err_std:.7f}, normalized user-level recovery RMSE: {rmse_mean:.7f}+-{rmse_std:.7f}\n"
    content += '\n'
    print(content)
