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
from postprocess import *


if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=100)
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

    args.label = "privacy_tradeoff_" + dataset_str + r_str + p_str

    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        device = 'cuda:{}'.format(free_gpu)
        #device = 'cuda:0'
    else:
        device = 'cpu'
    
    original_err_list, original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_prob_err_list, T_prob_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    T_freq_err_list, T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    SVD_T_err_list, SVD_T_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_original_err_list, X_original_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    X_T_freq_err_list, X_T_freq_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    err_list, rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]

    # dataset
    dataset = args.dataset
    print(dataset)
    num_entries = 1000000
    if dataset == 'syn':
        d1 = args.d1
        d2 = args.d2
        M = load_sparse_data_syn(args.r, d1, d2, num_entries, device)
    else:
        M = load_data_all(dataset)
        M = M.float().to(device)
        d1, d2 = M.shape
        M = M.to_sparse()
    dataset_content = f'sparse d1 = {d1}, d2 = {d2}, entries = {num_entries}\n'
    print(dataset_content)
    
    #users_list = [1000, 2000, 6000, 10000, 20000, 50000]
    epsilon_list = [0.1]
    for epsilon in epsilon_list:
        # privacy
        delta = 1/d1
        tau = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/epsilon
        privacy_content = f'epsilon = {epsilon}, delta = {delta}, tau = {tau}\n'
        print(privacy_content)

        

        for run in range(args.runs):
            # main part
            d1, d2 = M.shape
            p = args.p
            r = args.r
            recovery_p = 0.75
            observed_M, masks = get_sparse_masks(M, p)
            M = M.to_dense()
            #observed_M, masks = get_masks(M, p)
            value_masks = (M>0).float()
            value_ratio = (M!=0).sum() / M.numel()
            print(value_ratio)
            #observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), int(p*d2))
            #p = args.sample_entry / d2
            #observed_M = torch.from_numpy(observed_M).float().to(device)
            #masks = torch.from_numpy(masks).to(device)
            observed_M = observed_M.to_dense()
            masks = masks.to_dense()
            print(observed_M.shape)
            _, recovery_masks = get_masks(M, recovery_p)

            start_time = time.time()

            # observed MTM
            cov_observe_M =  observed_M.T @ observed_M
            MTM = M.T @ M
            #print(cov_observe_M)

            # freu reweight
            cov_M_count = (1 * (M != 0)).float().T @ (1 * (M != 0).float())
            cov_M_count = cov_M_count + (cov_M_count == 0) * 1
            cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
            cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
            #cov_mask_count = (masks.T @ masks).to_dense()
            #cov_mask_count = cov_mask_count + (cov_mask_count == 0) * 1
            #noise_matrix = sym_noise(d2, tau).to(device)
            #print(noise_matrix)
            #T = cov_observe_M / (cov_observe_count/(d1*value_ratio**2))
            T = cov_observe_M / (cov_observe_count/(d1))
            T_M = MTM / (cov_M_count/(d1))
            #T = torch.sparse_coo_tensor(cov_observe_M,T_values, (d2, d2))
            #T = T.to_dense()
                      
            #T =  T+noise_matrix
            # MTM
            print("MTM: ", MTM)
            print("T: ", T)
            print("T_M: ", T_M)
            T_masks = 1*(T!=0)

            mask_err_all = T*T_masks - T_M*T_masks
            #mask_err_all = cov_observe_M - MTM
            #mask_err_mask = T*ETE - MTM*ETE
            #mask_err = T*missing_mask_MTM - MTM*missing_mask_MTM
            #mask_err = T - MTM

            #print(torch.norm(mask_err_mask, 'fro') / torch.norm(MTM, 'fro'))
            print(torch.norm(mask_err_all, 'fro') / torch.norm(T_M, 'fro'))
            # prob reweight
            
            U, D, Vt = sparse_svds_for_tensor(T, k=r)
            direct_SVD = U @ torch.diag(D) @ Vt
            svd_err = relative_err(direct_SVD, T_M)
            X_T = direct_SVD
            print("svd_err: ", svd_err)

            # impute missing values from rank-r SVD corresponding to masks

            #T_masks = 1 * (T != 0)
            #X_p, _ = soft_impute(cov_observe_M+noise_matrix, T_masks, MTM, r, use_power_method=False, draw=False)
            #X_T, err_estimates = sparse_soft_impute(T, MTM, r, use_power_method=True, draw=False)
            X_T, err_estimates = soft_impute(T, T_masks, T_M, r, use_power_method=True, draw=True)
            #X_T, err_estimates = alt_min(T, T_masks, T_M, r, draw=True)


            #original_err = relative_err(cov_observe_M, MTM)
            #X_T_freq_err = relative_err(X_T, MTM)
            original_err = (torch.norm(cov_observe_M - T_M, 'fro') / torch.norm(T_M, 'fro')).item()
            X_T_freq_err = (torch.norm(X_T - T_M, 'fro') / torch.norm(T_M, 'fro')).item()

            estimation_matrix = X_T

            rmse_err = lstsq_recovery(estimation_goal=estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.001)
            
            end_time = time.time()
            cost_time = end_time - start_time

            original_err_list[run].append(original_err)
            err_list[run].append(X_T_freq_err)
            rmse_list[run].append(rmse_err)

    original_err_array = np.array(original_err_list)
    print(original_err_array)
    original_err_mean = np.mean(original_err_array, axis=0)
    original_err_std = np.std(original_err_array, axis=0)

    err_array = np.array(err_list)
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
        'err_mean': err_mean,
        'err_std': err_std,
    }
    torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}\n"
    for i, epsilon in enumerate(epsilon_list):
        content += f"epsilon {epsilon}:\n\
 original_err: {original_err_mean[i]:.4f}+-{original_err_std[i]:.4f}\n\
 err: {err_mean[i]:.4f}+-{err_std[i]:.4f}\n\
 rmse err: {rmse_mean[i]:.4f}+-{rmse_std[i]:.4f}\n"
    content += '\n'
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    