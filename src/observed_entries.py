from argparse import ArgumentParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from utils import *
from data import *
from data_utils import *
from power_method_svd import power_svd
from sparse_power_method_svd import power_svd_sparse
from sparse_utils import *
from recovery import lstsq_recovery
from postprocess import soft_impute


if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--sample_entry", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1)
    #parser.add_argument("--delta", type=float, default=10e-5)
    parser.add_argument("--mark", type=str, default="none")
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--use_reg", action="store_true", default=True)
    args = parser.parse_args()

    if args.dataset == 'syn':
        dataset_str = f'{args.dataset}_{args.d1}_{args.d2}'
    else:
        dataset_str = f'{args.dataset}'
    r_str = f'_r{args.r}'
    p_str = f'_p{format(args.p, ".0e")}'

    args.label = "observed_entries_" + dataset_str + r_str + p_str

    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        device = 'cuda:{}'.format(free_gpu)
        #device = 'cuda:0'
    else:
        device = 'cpu'
    
    # dataset
    dataset = args.dataset
    print(dataset)
    if dataset == 'syn':
        d1 = args.d1
        d2 = args.d2
        M = load_data_syn(args.r, d1, d2, device)
    else:
        M = load_data_all(dataset)
        M = M.float().to(device)
        d1, d2 = M.shape
    dataset_content = f'd1 = {d1}, d2 = {d2}, entries = {torch.count_nonzero(M)}\n'
    print(dataset_content)

    # privacy
    delta = 1/d1
    args.tau = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/args.epsilon
    privacy_content = f'epsilon = {args.epsilon}, delta = {delta}, args.tau = {args.tau}\n'
    print(privacy_content)

    #sample_perrow_list = [2,4,8,10,20,50]
    #sample_perrow_list = [2]
    sample_ratio_list = [0.002, 0.006, 0.01, 0.02, 0.06, 0.1]
    #sample_ratio_list = [0.06, 0.1]
    original_err_list, original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_prob_err_list, T_prob_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_freq_err_list, T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    SVD_T_err_list, SVD_T_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_original_err_list, X_original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_T_freq_err_list, X_T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    err_list, rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]

    for sample_ratio in sample_ratio_list:
        for run in range(args.runs):
            # main part
            d1, d2 = M.shape
            p = sample_ratio
            r = args.r
            recovery_p = 0.75
            observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), int(sample_ratio*d2))
            p = args.sample_entry / d2
            observed_M = torch.from_numpy(observed_M).float().to(device)
            masks = torch.from_numpy(masks).to(device)
            _, recovery_masks = get_masks(M, recovery_p)

            non_zero_rows = torch.any(observed_M != 0, dim=1)

            start_time = time.time()

            # observed MTM
            cov_observe_M =  observed_M.T @ observed_M

            # freu reweight
            cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
            cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
            noise_matrix = sym_noise(d2, args.tau).to(device)
            T = cov_observe_M / (cov_observe_count/d1) 
            T_masks = 1 * (T!=0)            
            T =  T+noise_matrix
            # MTM
            MTM = M.T @ M
            # prob reweight
            diag_cov = torch.diag( torch.diag(cov_observe_M) )
            T_p = (1.0 / p) * diag_cov + (1.0 / (p**2)) * (cov_observe_M - diag_cov)
            
            
            U, D, Vt = sparse_svds_for_tensor(T, k=r)
            direct_SVD = U @ torch.diag(D) @ Vt

            # impute missing values from rank-r SVD corresponding to masks

            #T_masks = 1 * (T != 0)
            X_p, _ = soft_impute(cov_observe_M+noise_matrix, T_masks, MTM, r, use_power_method=False, draw=False)
            X_T, err_estimates = soft_impute(T, T_masks, MTM, r, use_power_method=False, draw=False)

            original_err = relative_err(cov_observe_M, MTM)
            T_prob_err = relative_err(T_p, MTM)
            T_freq_err = relative_err(T, MTM)
            direct_SVD_err = relative_err(direct_SVD, MTM)
            X_original_err = relative_err(X_p, MTM)
            X_T_freq_err = relative_err(X_T, MTM)

            estimation_matrix = X_T

            rmse_err = lstsq_recovery(estimation_goal=estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.001)
            
            end_time = time.time()
            cost_time = end_time - start_time

            original_err_list[run].append(original_err)
            T_prob_err_list[run].append(T_prob_err)
            T_freq_err_list[run].append(T_freq_err)
            SVD_T_err_list[run].append(direct_SVD_err)
            X_original_err_list[run].append(X_original_err)
            err_list[run].append(X_T_freq_err)
            rmse_list[run].append(rmse_err)

    original_err_array = np.array(original_err_list)
    original_err_mean = np.mean(original_err_array, axis=0)
    original_err_std = np.std(original_err_array, axis=0)

    T_prob_err_array = np.array(T_prob_err_list)
    T_prob_err_mean = np.mean(T_prob_err_array, axis=0)
    T_prob_err_std = np.std(T_prob_err_array, axis=0)

    T_freq_err_array = np.array(T_freq_err_list)
    T_freq_err_mean = np.mean(T_freq_err_array, axis=0)
    T_freq_err_std = np.std(T_freq_err_array, axis=0)

    SVD_T_err_array = np.array(SVD_T_err_list)
    SVD_T_err_mean = np.mean(SVD_T_err_array, axis=0)
    SVD_T_err_std = np.std(SVD_T_err_array, axis=0)

    X_original_err_array = np.array(X_original_err_list)
    X_original_err_mean = np.mean(X_original_err_array, axis=0)
    X_original_err_std = np.std(X_original_err_array, axis=0)

    err_array = np.array(err_list)
    print(err_array)
    err_mean = np.mean(err_array, axis=0)
    err_std = np.std(err_array, axis=0)

    rmse_array = np.array(rmse_list)
    rmse_mean = np.mean(rmse_array, axis=0)
    rmse_std = np.std(rmse_array, axis=0)

    # Get the local time
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_content = f'local time: {local_time}\n'

    # Save
    results = {
        'args': args,
        'dataset_content': dataset_content,
        'original_err_mean': original_err_mean,
        'original_err_std': original_err_std,
        'T_prob_err_mean': T_prob_err_mean,
        'T_prob_err_std': T_prob_err_std,
        'T_freq_err_mean': T_freq_err_mean,
        'T_freq_err_std': T_freq_err_std,
        'SVD_T_err_mean': SVD_T_err_mean,
        'SVD_T_err_std': SVD_T_err_std,
        'X_original_err_mean': X_original_err_mean,
        'X_original_err_std': X_original_err_std,
        'err_mean': err_mean,
        'err_std': err_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std
    }
    torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}\n"
    for i, sample_ratio in enumerate(sample_ratio_list):
        content += f"sample {sample_ratio}:\n\
 original_err: {original_err_mean[i]:.4f}+-{original_err_std[i]:.4f}\n\
 T_prob_err_err: {T_prob_err_mean[i]:.4f}+-{T_prob_err_std[i]:.4f}\n\
 T_freq_err: {T_freq_err_mean[i]:.4f}+-{T_freq_err_std[i]:.4f}\n\
 SVD_T_err: {SVD_T_err_mean[i]:.4f}+-{SVD_T_err_std[i]:.4f}\n\
 X_original_err: {X_original_err_mean[i]:.4f}+-{X_original_err_std[i]:.4f}\n\
 err: {err_mean[i]:.4f}+-{err_std[i]:.4f}, rmse: {rmse_mean[i]:.4f}+-{rmse_std[i]:.4f}\n"
    content += '\n'
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    