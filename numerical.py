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

    M = load_normalized_data_syn(r, d1, d2, device)  
    MTM = M.T @ M

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
        
        
        normalized_MTM = MTM / d1
        A =  observed_M.T @ observed_M

        B = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
        B = B + (B == 0) * 1
        B_list.append(B.unsqueeze(0).cpu())

        A_list.append(A.unsqueeze(0).cpu())
        #T = iipw_T(observed_M)
        T = A / B
        T_list.append(T.unsqueeze(0).cpu())


    diag_mask = ~torch.eye(T.size(0), dtype=bool, device='cpu')
    # statistics of A
    A_mean, A_var = compute_mean_and_var(A_list)
    A_mean = A_mean.to(device)
    A_var = A_var.to(device)
    print(A_var.shape)
    A_mean_diag = torch.diag(A_mean)
    A_mean_off_diag = A_mean[diag_mask]
    A_var_diag = torch.diag(A_var)
    A_var_off_diag = A_var[diag_mask]

    # statistics of B
    B_mean, B_var = compute_mean_and_var(B_list)
    B_mean = B_mean.to(device)
    B_var = B_var.to(device)
    B_mean_diag = torch.diag(B_mean)
    B_mean_off_diag = B_mean[diag_mask]
    B_var_diag = torch.diag(B_var)
    B_var_off_diag = B_var[diag_mask]

    # statistics of T
    T_mean, T_var = compute_mean_and_var(T_list)
    T_mean = T_mean.to(device)
    T_var = T_var.to(device)
    T_mean_diag = torch.diag(T_mean)
    T_mean_off_diag = T_mean[diag_mask]
    T_var_diag = torch.diag(T_var)
    T_var_off_diag = T_var[diag_mask]

    # statistics of MTM
    MTM_diag = torch.diag(MTM)
    MTM_off_diag = MTM[diag_mask]
    MTM_diag_mean = MTM_diag.mean()
    MTM_off_diag_mean = MTM_off_diag.mean()

    # check
    print("avg A diag: ", A_mean_diag.mean())
    print("avg A off diag: ", A_mean_off_diag.mean())
    print("avg B diag: ", B_mean_diag.mean())
    print("avg B off diag: ", B_mean_off_diag.mean())
    print("avg T diag: ", T_mean_diag.mean())
    print("avg T off diag: ", T_mean_off_diag.mean())
    print("avg MTM diag: ", MTM_diag_mean)
    print("avg MTM off diag: ", MTM_off_diag_mean)

    cov_AB = (p**2-p**4)*MTM

    # Compute the estimate
    item1 = A_var / B_mean**2
    item2 = A_mean**2 * B_var / B_mean**4
    item3 = 2 * A_mean * cov_AB / B_mean**3

    estimation_eq7 = item1 + item2 - item3
    print("avg Estimation of eq7: ", estimation_eq7.mean())
    print("avg Var of T: ", T_var.mean())

    #print(MTM_off_diag)
    #print(MTM_diag)
    estimate_offdiag = A_var_off_diag / (d1**2 * p**4) - ((1-p**2) * (MTM_off_diag**2)) / (d1**3 * p**2)
    estimate_diag = A_var_diag / (d1**2 * p**2) - ((1-p) * (MTM_diag**2)) / (d1**3 * p)


    print("estimate diag", estimate_diag.mean())
    print("estimate off diag", estimate_offdiag.mean())
    print("var T diag", T_var_diag.mean())
    print("var T off diag", T_var_off_diag.mean())

    log_file = f"estimate_numerical.txt"
    with open(log_file, "a") as f:
        f.write(f"d1: {d1}, d2: {d2}, p: {p}, ob: {ob}, sample: {sample}, runs: {runs}\n")
        f.write(f"avg A diag: {A_mean_diag.mean()}\n")
        f.write(f"avg A off diag: {A_mean_off_diag.mean()}\n")
        f.write(f"avg B diag: {B_mean_diag.mean()}\n")
        f.write(f"avg B off diag: {B_mean_off_diag.mean()}\n")
        f.write(f"avg T diag: {T_mean_diag.mean()}\n")
        f.write(f"avg T off diag: {T_mean_off_diag.mean()}\n")
        f.write(f"avg MTM diag: {MTM_diag_mean}\n")
        f.write(f"avg MTM off diag: {MTM_off_diag_mean}\n")
        f.write(f"avg Estimation of eq7: {estimation_eq7.mean()}\n")
        f.write(f"avg Var of T: {T_var.mean()}\n")
        f.write(f"estimate_diag: {estimate_diag.mean()}\n")
        f.write(f"estimate_offdiag: {estimate_offdiag.mean()}\n")
        f.write(f"var T diag: {T_var_diag.mean()}\n")
        f.write(f"var T off diag: {T_var_off_diag.mean()}\n")
        f.write("\n")
        


