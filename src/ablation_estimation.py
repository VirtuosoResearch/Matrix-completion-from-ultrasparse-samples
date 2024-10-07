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

def main(args, M):
    d1, d2 = M.shape
    p = args.p
    r = args.r
    observed_M, masks = get_masks(M, p)

    non_zero_rows = torch.any(observed_M != 0, dim=1)

    observed_M = observed_M[non_zero_rows]
    masks = masks[non_zero_rows]
    M = M[non_zero_rows]
    print(M.shape)

    start_time = time.time()

    cov_observe_M =  observed_M.T @ observed_M
    cov_observe_count = (observed_M == 0).float().t() @ (observed_M == 0).float()
    #diag_cov = torch.diag( torch.diag(cov_observe_M) )

    cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
    cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
    noise_matrix = sym_noise(d2, args.sigma).to(device)
    T = cov_observe_M / (cov_observe_count/d1) + noise_matrix
    #T = cov_observe_M / (cov_observe_count/d1)
    MTM = M.T @ M

    # impute missing values from rank-r SVD corresponding to masks
    use_power_method = False
    train_losses = []
    err_estimates = []

    epochs = 100
    tol = 1e-7
    lr = 0.1
    X = T.clone()
    X_list = []
    X_list.append(X)
    T_masks = 1 * (T!=0)
    loop = tqdm(range(epochs))
    for i in loop:
        if not use_power_method:
            U, D, Vt = torch.linalg.svd(X)
            D[r:] = 0
        else:
            #U, D, Vt = power_svd(X, k=r)
            X_scipy = torch_sparse_to_scipy(X.to_sparse())
            # Perform sparse SVD using scipy
            U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(X_scipy, k=r)
            U = torch.from_numpy(U_scipy.copy()).to(device)
            D = torch.from_numpy(D_scipy.copy()).to(device)
            Vt = torch.from_numpy(Vt_scipy.copy()).to(device)
        X_update = U @ torch.diag(D) @ Vt

        #X = X * T_masks + X_update * (1 - T_masks)
        X = X * (1-lr) + X_update * lr
        X = X * T_masks + X_update * (1 - T_masks)
        err = MTM - X
        loss = (err**2).mean()
        train_losses.append(loss.item())
        relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
        if i > 3:
            if abs(err_estimates[-1] - err_estimates[-2]) < tol:
                break
        loop.set_description(f"relative err: {relative_err:.7f}")
        err_estimates.append(relative_err.item())
        X_list.append(X)
        #print(relative_err)
    
    plt.figure(figsize=(5, 3))
    plt.plot(err_estimates, label='Relative Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Relative Error')
    plt.legend()
    plt.savefig(f'../plots/{args.label}-rel-err.png', dpi=150)
    plt.show()

    sub_opt_X = X_list[1]
    X_T = X

    #print(T)
    #print(sub_opt_X)
    #print(X_T)

    ablation_goals = [torch.rand(X.shape), T, sub_opt_X, X_T, MTM]
    gap1 = torch.norm(X_T - T)
    gap2 = torch.norm(X_T - sub_opt_X)
    print(gap1, gap2)
    rmse_errs = []
    for goals in ablation_goals:
        rmse_errs.append(lstsq_recovery(estimation_goal=goals.to(device), M=M, masks=masks, r=r))
    
    return rmse_errs

    end_time = time.time()
    cost_time = end_time - start_time

    curve = {
        'train_losses': train_losses,
        'err_estmates': err_estimates,
    }

    return rmse_err, err_estimates[-1], cost_time, curve



if __name__ == "__main__":
    parser = ArgumentParser()
    # Controled by parameters
    parser.add_argument("--dataset", type=str, default="syn")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.01)
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

    args.label = dataset_str + r_str + p_str

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
        _, _, M = load_sparse_data_syn(args.r, d1, d2, device)
    else:
        M = load_data_all(dataset)
        M = M.float().to(device)
        d1, d2 = M.shape
    dataset_content = f'd1 = {d1}, d2 = {d2}, entries = {torch.count_nonzero(M)}\n'
    print(dataset_content)

    # privacy
    delta = 1/d1
    args.sigma = torch.sqrt(2*torch.log(torch.tensor(1.25/delta)))/args.epsilon
    privacy_content = f'epsilon = {args.epsilon}, delta = {delta}, args.sigma = {args.sigma}\n'
    print(privacy_content)

    rmse_errs_list = []
    for run in range(args.runs):
        rmse_errs = main(args, M)
        rmse_errs_list.append(rmse_errs)
    
    array = np.array(rmse_errs_list)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    print(array)

    # Get the local time
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_content = f'local time: {local_time}\n'

    # Define the content in the desired format
    content = f"run times: {args.runs}\nrand: {mean[0]:.4f} +- {std[0]:.4f}\n\
T: {mean[1]:.4f} +- {std[1]:.4f}\n\
sub-opt: {mean[2]:.4f} +- {std[2]:.4f}\n\
X_T: {mean[3]:.4f} +- {std[3]:.4f}\n\
MTM: {mean[4]:.4f} +- {std[4]:.4f}\n\n"
    print(content)
    # Write the content to a file
    with open(f'../results/ablation_estimation_{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    