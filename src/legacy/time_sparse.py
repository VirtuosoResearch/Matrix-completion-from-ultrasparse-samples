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
from recovery import lstsq_recovery, mean_recovery, vanilla_MC
from postprocess import *


if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--sample", type=str, default="uniform")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--sample_entry", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=20)
    #parser.add_argument("--delta", type=float, default=10e-5)
    parser.add_argument("--mark", type=str, default="ours")
    parser.add_argument("--save_weights", action="store_true", default=False)
    parser.add_argument("--use_reg", action="store_true", default=True)
    args = parser.parse_args()

    if args.dataset == 'syn':
        dataset_str = f'{args.dataset}_{args.d1}_{args.d2}'
    else:
        dataset_str = f'{args.dataset}'
    r_str = f'_r{args.r}'
    p_str = f'_p{format(args.p, ".0e")}'

    args.label = "real_sparse_" + args.mark + dataset_str + r_str + p_str

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
    num_entries = 100000
    if dataset == 'syn':
        
        d1_all = args.d1
        d2_all = args.d2
        M = load_sparse_data_syn(args.r, d1_all, d2_all, num_entries, device)
        M = torch_sparse_to_scipy(M)
    else:
        M = load_data_all(dataset)
        #M = M.float().to(device)
        d1_all, d2_all = M.shape
        if dataset == 'ml-1m':
            M = M.to_sparse()
            M = torch_sparse_to_scipy(M)
    M_all = M
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
    else:
        p=0.75
        div_d1 = 1
        div_d2 = 1
    if args.mark == 'mc':
        div_d1 = 10
        div_d2 = 6
    M_dataset = MatrixDataset(M_all, int(d1_all/div_d1)+1, int(d2_all/div_d2)+1)
    matrix_dataloader = DataLoader(
        M_dataset,
        batch_size=1,         # You can adjust the batch size as needed
        shuffle=False,
        collate_fn=sparse_collate_fn  # Pass the custom collate function
    )
    dataset_content = f'sparse d1 = {d1_all}, d2 = {d2_all}, entries = {num_entries}, p = {args.p}\n'
    print(dataset_content)

    # privacy
    epsilon = args.epsilon
    delta = 1/d1_all
    tau = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/epsilon
    privacy_content = f'epsilon = {epsilon}, delta = {delta}, tau = {tau}\n'
    print(privacy_content)

    for run in range(args.runs):
        batch_err_list = []
        batch_rmse_list = []
        cost_time = 0
        
        # main part
        r = args.r
        recovery_p = 0.8
        total_test_num = 0
        for batch in matrix_dataloader:
            M = batch[0].coalesce()
            num_entries = M._nnz()
            d1, d2 = M.shape
            print(M.shape)
            batch_size = d1*d2
            observed_M, masks = get_sparse_masks(M, p)
            _, recovery_masks = get_sparse_masks(M, recovery_p)
            recovery_masks = masks.to_dense().bool().to(device)
            M = M.to_dense()
            M = M.to(device)

            print("ratio: ", num_entries/batch_size)
            #observed_M, masks = get_masks(M, p)
            
            #value_indices = M.indices()
            #observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), int(p*d2))
            #observed_M = torch.from_numpy(observed_M).float().to(device)
            #masks = torch.from_numpy(masks).to(device)
            observed_M = observed_M.to_dense().to(device)
            masks = masks.to_dense().to(device)
            
            start_time = time.time()
            if args.mark == 'random':
                batch_err_list.append(0)
                rmse_err, batch_test_num = lstsq_recovery(estimation_goal=torch.rand(d2, d2).to(device), M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.0001, batch=True)
                print('batch rmse: ', rmse_err)
                print('batch test number: ', batch_test_num)
                batch_rmse_list.append(rmse_err*batch_test_num)
                total_test_num += batch_test_num
            elif args.mark == 'mean':
                batch_err_list.append(0)
                rmse_err, batch_test_num = mean_recovery(estimation_goal=None, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.0001, batch=True)
                print('batch rmse: ', rmse_err)
                print('batch test number: ', batch_test_num)
                batch_rmse_list.append(rmse_err*batch_test_num)
                total_test_num += batch_test_num
            elif args.mark == 'mc':
                batch_err_list.append(0)
                entry_masks = M!=0
                X = vanilla_MC(M, entry_masks, num_entries, r, draw=True)
                
                test_masks = entry_masks * masks
                batch_test_num = test_masks.sum()
                rmse_err = (((X-M)*test_masks)**2).sum()
                del entry_masks, test_masks
                torch.cuda.empty_cache()
                #rmse_err, batch_test_num = mean_recovery(estimation_goal=None, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.0001, batch=True)
                print('batch rmse: ', rmse_err)
                print('batch test number: ', batch_test_num)
                #print((rmse_err*batch_test_num).cpu().item())
                batch_rmse_list.append((rmse_err).cpu().item())
                total_test_num += batch_test_num.item()
                end_time = time.time()
                cost_time += end_time - start_time
            else:
                # observed MTM
                cov_observe_M =  observed_M.T @ observed_M
                MTM = M.T @ M
                #print(cov_observe_M)

                # freu reweight
                cov_M_count = (1 * (M != 0)).float().T @ (1 * (M != 0).float())
                cov_M_count = cov_M_count + (cov_M_count == 0) * 1
                cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
                cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
                noise_matrix = sym_noise(d2, tau).to(device)
                #print(noise_matrix)
                T_masks = 1*(cov_observe_M!=0)
                TM_masks = 1*(MTM!=0)
                cov_observe_M += noise_matrix
                T = cov_observe_M / (cov_observe_count/(d1))
                T_M = MTM / (cov_M_count/(d1))


                # MTM
                #print("MTM: ", MTM)
                #print("T: ", T)
                #print("T_M: ", T_M)
                

                

                #mask_err_all = T*T_masks - T_M*T_masks
                #mask_err_all = cov_observe_M - MTM
                #mask_err_mask = T*ETE - MTM*ETE
                #mask_err = T*missing_mask_MTM - MTM*missing_mask_MTM
                #mask_err = T - MTM

                #print(torch.norm(mask_err_mask, 'fro') / torch.norm(MTM, 'fro'))
                #print(torch.norm(mask_err_all, 'fro') / torch.norm(T_M, 'fro'))
                # prob reweight
                
                if args.mark == 'svd':
                    U, D, Vt = top_r_svd(T, r=r)
                    direct_SVD = U @ torch.diag(D) @ Vt
                    svd_err = relative_err(direct_SVD, T_M)
                    X_T = direct_SVD
                    print("svd_err: ", svd_err)
                elif args.mark == 'ob2':
                    #X_T, err_estimates = alt_min(T, T_masks, T_M, r, lr=1, draw=True)
                    X_T, err_estimates = nuclear_reg(T, T_masks, T_M, r, lr=1, draw=True)
                elif args.mark == 'ours':
                    X_T, err_estimates = soft_impute(T, T_masks, T_M, r, use_power_method=False, draw=True)          
                


                #original_err = relative_err(cov_observe_M, MTM)
                #X_T_freq_err = relative_err(X_T, MTM)
                X_T_freq_err = (torch.norm(X_T - T_M, 'fro') / torch.norm(T_M, 'fro')).item()
                batch_err_list.append(X_T_freq_err*batch_size)

                estimation_matrix = X_T

                end_time = time.time()

                rmse_err, batch_test_num = lstsq_recovery(estimation_matrix, M=M, masks=masks, r=r, recovery_masks=recovery_masks, use_reg=True, lam=0.001, batch=True)
                print('batch rmse: ', rmse_err)
                print('batch test number: ', batch_test_num)
                batch_rmse_list.append(rmse_err*batch_test_num)
                total_test_num += batch_test_num
                cost_time += end_time - start_time
            
                


            

        batch_err = np.sum(batch_err_list) / all_size
        batch_rmse = np.sqrt(np.sum(batch_rmse_list) / total_test_num)

        err_list[run].append(batch_err)
        rmse_list[run].append(batch_rmse)

    err_array = np.array(err_list)
    print(err_array)
    err_mean = np.mean(err_array, axis=0)
    print(err_mean)
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
        'err_mean': err_mean,
        'err_std': err_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std
    }
    torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}, cost: {cost_time}\n"
    content += f"err: {err_mean[0]:.4f}+-{err_std[0]:.4f}\nrmse err: {rmse_mean[0]:.4f}+-{rmse_std[0]:.4f}\n"
    content += '\n'
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    