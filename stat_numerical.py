import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

from utils import *
from src.iipw import *


if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=10)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=100)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--ob", type=int, default=-1)
    parser.add_argument("--sample", type=str, default='iid')
    parser.add_argument("--epsilon", type=float, default=10)
    parser.add_argument("--delta", type=float, default=0)
    parser.add_argument("--mark", type=str, default="none")
    parser.add_argument("--use_reg", action="store_true", default=True)

    args = parser.parse_args()
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    # dataset
    d1 = args.d1
    d2 = args.d2
    r = args.r
    p = args.p
    ob = args.ob
    sample = args.sample
    runs = args.runs

    total_t = 1
    var_err_list = []
    mean_err_list = []
    for t in range(total_t):

        M = load_normalized_data_syn(r, d1, d2, device)  
        MTM = M.T @ M
        MTM2 = (M.T**2 @ M**2)

        vtv_diag_list = []
        vtv_off_diag_list = []
        A_diag_list = []
        A_off_diag_list = []
        B_diag_list = []
        B_off_diag_list = []
        T_diag_list = []
        T_off_diag_list = []
        A_list = []
        B_list = []
        T_list = []
        MTM_list = []
        MTM2_list = []
        for run in tqdm(range(runs)):

            if sample == "iid":
                # IID sample
                observed_M, masks = get_uniform_masks(M, p)
            else:
                # few entry sample
                observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), ob)
                p = ob/d2
                observed_M = torch.from_numpy(observed_M).float().to(device)
                masks = torch.from_numpy(masks).to(device)
            
            A =  observed_M.T @ observed_M
            A_mask = A != 0
            #diag_mask = torch.eye(A.shape[0], dtype=torch.bool).to(device)
            #off_diag_mask = ~diag_mask
            #A_mask = A_mask & off_diag_mask
            A_list.append(A[A_mask])

            B = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
            B_list.append(B[A_mask])

            T = A / B
            T_list.append(T[A_mask])

            MTM_list.append(MTM[A_mask])
            MTM2_list.append(MTM2[A_mask])

        # statistics of A
        A_tensor = torch.cat(A_list, dim=0)
        print(A_tensor.shape)

        # statistics of B
        B_tensor = torch.cat(B_list, dim=0)

        # statistics of T
        T_tensor = torch.cat(T_list, dim=0)

        # statistics of MTM
        MTM_tensor = torch.cat(MTM_list, dim=0)
        MTM2_tensor = torch.cat(MTM2_list, dim=0)
        
        # 1st
        A_mean = A_tensor.mean()
        B_mean = B_tensor.mean()
        A_var = A_tensor.var()
        B_var = B_tensor.var()
        AB_cov = torch.mean(A_tensor * B_tensor) - A_mean * B_mean

        A_mu = p**2 * MTM_tensor
        B_mu = d1 * p**2
        A_sigma2 = p**2 * (1-p**2) * MTM2_tensor
        B_sigma2 = d1 * p**2 * (1-p**2)
        cov_AB = (1-p**2) * A_mu

        #item1 = A_sigma2 / B_mu**2
        #item2 = A_mu**2 * B_sigma2 / B_mu**4
        #item3 = 2 * A_mu * cov_AB / B_mu**3
        item1 = A_mean / B_mean
        item2 = (A_tensor - A_mean)/B_mean
        item3 = A_mean*(B_tensor - B_mean)/B_mean**2

        var_item1 = A_var / B_mean**2
        var_item2 = A_mean**2 * B_var / B_mean**4
        var_item3 = 2 * A_mean * AB_cov / B_mean**3

        approx = item1 + item2 - item3
        approx_var = var_item1 + var_item2 - var_item3
        approx_var_err = (approx_var.mean() - T_tensor.var()).abs() / T_tensor.var()
        approx_mean_err = (approx.mean() - T_tensor.mean()).abs() / T_tensor.mean()
        var_err_list.append(approx_var_err.item())
        mean_err_list.append(approx_mean_err.item())
        print("avg Estimation of eq7: ", approx.mean())
        print("avg Mean of T: ", T_tensor.mean())
        print("avg Var of estimation: ", approx_var.mean())
        print("avg Var of T: ", T_tensor.var())
        print("error: ", (approx_var.mean() - T_tensor.var()).abs() / T_tensor.var())

        log_file = f"first_order_numerical.txt"
        with open(log_file, "a") as f:
            f.write(f"d1: {d1} d2: {d2} r: {r} p: {p} ob: {ob} sample: {sample}, runs: {runs}, mean size of B: {B_mean.item()}, total number of samples {A_tensor.shape[0]}\n")
            f.write(f"avg Estimation of eq7: {approx.var()}\n")
            f.write(f"avg Var of T: {T_tensor.var()}\n")
            f.write("\n")
    
    print("avg var error: ", np.mean(var_err_list))
    print("avg mean error: ", np.mean(mean_err_list))
    
    if sample == "iid":
        np.save(f'approximation/first_order_numerical_iid_{p}_var.npy', var_err_list)
        np.save(f'approximation/first_order_numerical_iid_{p}_mean.npy', mean_err_list)
    else:
        np.save(f'approximation/first_order_numerical_fix_{ob}_var.npy', var_err_list)
        np.save(f'approximation/first_order_numerical_fix_{ob}_mean.npy', mean_err_list)
    
        


