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
    parser.add_argument("--d1", type=int, default=10000)
    parser.add_argument("--d2", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.01)
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

    args.label = "time_" + dataset_str + r_str + p_str

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
    ob2_err_list, ob2_rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]
    err_list, rmse_list = [[] for i in range(args.runs)], [[] for i in range(args.runs)]

    time_list = [[] for i in range(args.runs)]

    # dataset
    dataset = args.dataset

    #users_list = [1000, 2000, 6000, 10000, 20000, 50000]
    #epsilon_list = [0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.001, 0.0015, 0.002]
    item_list = [500, 1000, 1500, 2000, 2500, 3000]
    #item_list = [500, 1000, 2000, 3000, 5000]
    #users_list = [10000]
    for d2 in item_list:
   # for d1 in users_list:
        if dataset == 'syn':
            d1 = args.d1
            entries = 100000
            #d2 = args.d2
            M = load_sparse_data_syn_2(args.r, d1, d2, num_elements=entries)
        else:
            M = load_data_all(dataset)
            M = M.float().to(device)
            d1, d2 = M.shape
        dataset_content = f'd1 = {d1}, d2 = {d2}, entries = {M._nnz()}, density = {M._nnz()/(d1*d2)}\n'
        print(dataset_content)
        # privacy
        delta = 1/d1
        epsilon = args.epsilon
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
            observed_M = observed_M.to_dense()
            masks = masks.to_dense()
            M = M.to_dense()
            p = args.sample_entry / d2
            #observed_M = torch.from_numpy(observed_M).float().to(device)
            #masks = torch.from_numpy(masks).to(device)
            #_, recovery_masks = get_uniform_masks(M, recovery_p)
          

            # observed MTM
            cov_observe_M =  observed_M.T @ observed_M
            MTM = M.T @ M
            # freu reweight
            cov_M_count = (1 * (M != 0)).float().T @ (1 * (M != 0).float())
            cov_M_count = cov_M_count + (cov_M_count == 0) * 1
            cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
            cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
            noise_matrix = sym_noise(d2, tau).to(device)
            #print(noise_matrix)
            
            T = cov_observe_M / (cov_observe_count/(d1))
            T_M = MTM / (cov_M_count/(d1))
            T_masks = 1*(T!=0)
            TM_masks = 1*(T_M!=0)
            T = T.to_sparse()
            T_M = T_M.to_sparse()
            print('non zero in T: ',T._nnz())
            #T = T+noise_matrix
            # prob reweight
            #diag_cov = torch.diag( torch.diag(cov_observe_M) )
            #T_p = (1.0 / p) * diag_cov + (1.0 / (p**2)) * (cov_observe_M - diag_cov)
            
            
            #U, D, Vt = top_r_svd(T, r=r)
            #direct_SVD = U @ torch.diag(D) @ Vt

            # impute missing values from rank-r SVD corresponding to masks
            start_time = time.time()
            #T_masks = 1 * (T != 0)
            X_T, _ = nuclear_reg(T, T_masks, MTM, r)
            #X_p, _ = soft_impute(cov_observe_M+noise_matrix, T_masks, MTM, r, use_power_method=False, draw=False)
            #X_T, err_estimates = sparse_soft_impute(T, T_masks, T_M, TM_masks, r, draw=False)


            X_T_freq_err = relative_err(X_T, T_M)

            estimation_matrix = X_T

            #rmse_err = lstsq_recovery(estimation_goal=estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.001)
            
            end_time = time.time()
            cost_time = end_time - start_time

            #SVD_T_err_list[run].append(direct_SVD_err)
            #X_original_err_list[run].append(X_original_err)
            #ob2_err_list[run].append(ob2_err)
            err_list[run].append(X_T_freq_err)
            #rmse_list[run].append(rmse_err)
            time_list[run].append(cost_time)


    time_array = np.array(time_list)
    time_mean = np.mean(time_array, axis=0)
    time_std = np.std(time_array, axis=0)

    # Get the local time
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_content = f'local time: {local_time}\n'


    # Define the content in the desired format
    content = f"run times: {args.runs}\n"
    for i, d in enumerate(item_list):
        content += f"d2 {d}:\nTime: {time_mean[i]} +- {time_std[i]}\n"
    content += '\n'
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    