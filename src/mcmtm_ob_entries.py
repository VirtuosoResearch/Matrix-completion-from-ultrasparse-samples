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
from recovery import lstsq_recovery, vanilla_MC
from postprocess import *


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
    parser.add_argument("--sample_entry", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=10)
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

    args.label = "mcmtm_observed_entries_2_" + dataset_str + r_str + p_str

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
    #sample_ratio_list = [0.002, 0.004, 0.006, 0.008, 0.01]
    #sample_ratio_list = [0.01, 0.05, 0.1, 0.5]
    sample_ratio_list = [0.06, 0.01]
    original_err_list, original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_prob_err_list, T_prob_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_freq_err_list, T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    SVD_T_err_list, SVD_T_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    SVD_MTM_err_list, SVD_MTM_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_original_err_list, X_original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_T_freq_err_list, X_T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    ob2_err_list, ob2_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    mcmtm_err_list, mcmtm_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    oracle_err_list, oracle_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
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
            _, recovery_masks = get_uniform_masks(M, recovery_p)

            recovery_masks = recovery_masks*(1-masks)

            non_zero_rows = torch.any(observed_M != 0, dim=1)

            start_time = time.time()

            # observed MTM
            cov_observe_M =  observed_M.T @ observed_M

            # freu reweight
            noise_matrix = sym_noise(d2, args.tau).to(device)
            cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
            cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
            T_masks = 1 * (cov_observe_M!=0)   
            cov_observe_M += noise_matrix
            
            T = cov_observe_M / (cov_observe_count/d1)
            print("test", ((T!=0)^(cov_observe_M!=0)).sum()) 

            # MTM
            MTM = M.T @ M
            # prob reweight
            diag_cov = torch.diag( torch.diag(cov_observe_M) )
            T_p = (1.0 / p) * diag_cov + (1.0 / (p**2)) * (cov_observe_M - diag_cov)

            
            U, D, Vt = top_r_svd(MTM, r=r)
            SVD_MTM = U @ torch.diag(D) @ Vt

            # impute missing values from rank-r SVD corresponding to masks

            #T_masks = 1 * (T != 0)
            #X_p, _ = soft_impute(cov_observe_M+noise_matrix, T_masks, MTM, r, use_power_method=False, draw=False)
            #X_2, _ = alt_min(T, T_masks, MTM, r, draw=False)
            #X_T, err_estimates = soft_impute(T, T_masks, MTM, r, use_power_method=False, draw=False)
            num_entries = d1*d2
            #X_T = vanilla_MC(MTM, None, num_entries, r, draw=True)
            X_T = vanilla_MC(cov_observe_M, None, num_entries, r,epochs=300, draw=True)
            #X_T = vanilla_MC(T, None, num_entries, r,epochs=300, draw=True)
            X_oracle = vanilla_MC(MTM, None, num_entries, r,epochs=600, draw=True)

            original_err = relative_err(cov_observe_M, MTM)
            T_prob_err = relative_err(T_p, MTM)
            T_freq_err = relative_err(T, MTM)
            SVD_MTM_err = relative_err(SVD_MTM, MTM)
            #X_original_err = relative_err(X_p, MTM)
            X_T_freq_err = relative_err(X_T, MTM)
            oracle_err = relative_err(X_oracle, MTM)

            _, _, XV = top_r_svd(X_T, r=r)
            _, _, MV = top_r_svd(MTM, r=r)

            print((torch.norm(XV-MV, 'fro') / torch.norm(MV, 'fro')).item())

            estimation_matrix = X_T
            #estimation_matrix = torch.randn(X_T.shape).to(device)
            lam = 0.0001
            #T_rmse_err = lstsq_recovery(estimation_goal=T, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
            #ob2_rmse_err = lstsq_recovery(estimation_goal=X_2, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
            rmse_err = lstsq_recovery(estimation_goal=estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
            oracle_rmse_err = lstsq_recovery(estimation_goal=X_oracle, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
            SVD_MTM_rmse_err = lstsq_recovery(estimation_goal=SVD_MTM, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=lam)
            
            end_time = time.time()
            cost_time = end_time - start_time

            original_err_list[run].append(original_err)
            T_prob_err_list[run].append(T_prob_err)
            T_freq_err_list[run].append(T_freq_err)
            SVD_MTM_err_list[run].append(SVD_MTM_err)
            SVD_MTM_rmse_list[run].append(SVD_MTM_rmse_err)
            #T_freq_rmse_list[run].append(T_rmse_err)
            #SVD_T_err_list[run].append(direct_SVD_err)
            #SVD_T_rmse_list[run].append(SVD_T_rmse_err)
            #ob2_err_list[run].append(ob2_err)
            #ob2_rmse_list[run].append(ob2_rmse_err)
            oracle_err_list[run].append(oracle_err)
            oracle_rmse_list[run].append(oracle_rmse_err)
            #X_original_err_list[run].append(X_original_err)
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

    SVD_MTM_err_array = np.array(SVD_MTM_err_list)
    SVD_MTM_err_mean = np.mean(SVD_MTM_err_array, axis=0)
    SVD_MTM_err_std = np.std(SVD_MTM_err_array, axis=0)

    SVD_MTM_rmse_array = np.array(SVD_MTM_rmse_list)
    SVD_MTM_rmse_mean = np.mean(SVD_MTM_rmse_array, axis=0)
    SVD_MTM_rmse_std = np.std(SVD_MTM_rmse_array, axis=0)

    T_freq_rmse_array = np.array(T_freq_rmse_list)
    T_freq_rmse_mean = np.mean(T_freq_rmse_array, axis=0)
    T_freq_rmse_std = np.std(T_freq_rmse_array, axis=0)

    X_original_err_array = np.array(X_original_err_list)
    X_original_err_mean = np.mean(X_original_err_array, axis=0)
    X_original_err_std = np.std(X_original_err_array, axis=0)

    ob2_err_array = np.array(ob2_err_list)
    ob2_err_mean = np.mean(ob2_err_array, axis=0)
    ob2_err_std = np.std(ob2_err_array, axis=0)

    ob2_rmse_array = np.array(ob2_rmse_list)
    ob2_rmse_mean = np.mean(ob2_rmse_array, axis=0)
    ob2_rmse_std = np.std(ob2_rmse_array, axis=0)

    oracle_err_array = np.array(oracle_err_list)
    oracle_err_mean = np.mean(oracle_err_array, axis=0)
    oracle_err_std = np.std(oracle_err_array, axis=0)

    oracle_rmse_array = np.array(oracle_rmse_list)
    oracle_rmse_mean = np.mean(oracle_rmse_array, axis=0)
    oracle_rmse_std = np.std(oracle_rmse_array, axis=0)

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
        'p_list': sample_ratio_list,
        'original_err_mean': original_err_mean,
        'original_err_std': original_err_std,
        'T_prob_err_mean': T_prob_err_mean,
        'T_prob_err_std': T_prob_err_std,
        'T_freq_err_mean': T_freq_err_mean,
        'T_freq_err_std': T_freq_err_std,
        'T_freq_rmse_mean': T_freq_rmse_mean,
        'T_freq_rmse_std': T_freq_rmse_std,
        'SVD_T_err_mean': SVD_T_err_mean,
        'SVD_T_err_std': SVD_T_err_std,
        'SVD_MTM_err_mean': SVD_MTM_err_mean,
        'SVD_MTM_err_std': SVD_MTM_err_std,
        'SVD_MTM_rmse_mean': SVD_MTM_rmse_mean,
        'SVD_MTM_rmse_std': SVD_MTM_rmse_std,
        'X_original_err_mean': X_original_err_mean,
        'X_original_err_std': X_original_err_std,
        'ob2_err_mean': ob2_err_mean,
        'ob2_err_std': ob2_err_std,
        'ob2_rmse_mean': ob2_rmse_mean,
        'ob2_rmse_std': ob2_rmse_std,
        'oracle_err_mean': oracle_err_mean,
        'oracle_rmse_mean': oracle_rmse_mean,
        'err_mean': err_mean,
        'err_std': err_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std
    }
    #torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}\n"
    for i, sample_ratio in enumerate(sample_ratio_list):
        content += f"sample {sample_ratio}:\n\
 oracle_err: {oracle_err_mean[i]:.4f}+-{oracle_err_std[i]:.4f}, oracle_rmse: {oracle_rmse_mean[i]:.4f}+-{oracle_rmse_std[i]:.4f}\n\
 SVD_MTM_err: {SVD_MTM_err_mean[i]:.4f}+-{SVD_MTM_err_std[i]:.4f}, SVD_MTM_rmse: {SVD_MTM_rmse_mean[i]:.4f}+-{SVD_MTM_rmse_std[i]:.4f}\n\
 err: {err_mean[i]:.4f}+-{err_std[i]:.4f}, rmse: {rmse_mean[i]:.4f}+-{rmse_std[i]:.4f}\n"
    content += '\n'
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    