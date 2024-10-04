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
    noise_matrix = sym_noise(d2, args.tau).to(device)
    T = cov_observe_M / (cov_observe_count/d1) + noise_matrix
    MTM = M.T @ M
    T_masks = 1 * (T!=0)

    # impute missing values from rank-r SVD corresponding to masks
    use_power_method = True
    train_losses = []
    err_estimates = []

    epochs = 100
    tol = 1e-7
    lr = 0.1
    X = T
    #T_masks = 1 * (T != 0)
    print(T_masks.sum())
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
        err = MTM - X
        loss = (err**2).mean()
        train_losses.append(loss.item())
        relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
        if i > 3:
            if abs(err_estimates[-1] - err_estimates[-2]) < tol:
                break
        loop.set_description(f"relative err: {relative_err:.7f}")
        err_estimates.append(relative_err.item())
        #print(relative_err)

    plt.figure(figsize=(5, 3))
    plt.plot(err_estimates, label='Relative Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Relative Error')
    plt.legend()
    plt.savefig(f'../plots/{args.label}-rel-err.png', dpi=150)
    plt.show()

    error_list = []
    missing_num = 0
    total_num = 0
    skip = 0
    result_list_missing = []
    result_list_total = []
    use_reg = True
    U = []

    #estimation_goal = torch.rand(X.shape).to(device)
    estimation_goal = X

    #_, S, V = top_r_svd(estimation_goal, r)
    #_, _, V = power_svd(estimation_goal, k=r)
    _, _, V = sparse_svds_for_tensor(estimation_goal, k=r)
    #print(V)
    for i in tqdm(range(M.shape[0])):
        """
        M_row: 1*d2
        V: r*d2
        user_vector: 1*r
        """

        # make sure there is non-zero value in original matrix
        M_row = M[i]
        non_zero_indices = M_row.nonzero(as_tuple=True)[0]

        #non_zero_indices = torch.tensor(range(M_row.shape[0]))

        # select non-zero & masked elements as training data, non-zero & non-masked elements as testing data
        """
        observed_idx = []
        missing_idx = []
        for non_zero_idx in non_zero_indices:
            if masks[i][non_zero_idx]:
                observed_idx.append(non_zero_idx.item())
            else:
                missing_idx.append(non_zero_idx.item())
        """

        # Get the mask values for the non-zero indices
        mask_values = masks[i][non_zero_indices]

        # Use boolean indexing to separate observed and missing indices
        observed_idx = non_zero_indices[mask_values]
        missing_idx = non_zero_indices[~mask_values]

        M_row = M_row.unsqueeze(0)
        n = non_zero_indices.shape[0]

        #if len(observed_idx)==0 or len(missing_idx) == 0:
        #    skip +=1
        #    continue
        
        missing_num += len(missing_idx)
        total_num += len(missing_idx) + len(observed_idx)

        observed_idx = torch.tensor(observed_idx)
        missing_idx = torch.tensor(missing_idx)

        # formula AX=B
        observed_A = V[:, observed_idx].t()
        observed_B = M_row[:, observed_idx].t()
        missing_A = V[:, missing_idx].t()
        missing_B = M_row[:, missing_idx].t()

        if use_reg:
            # Ridge regression
            lambda_reg = 0.001
            I = torch.eye(observed_A.shape[1], device=device) * lambda_reg
            u = torch.linalg.lstsq(observed_A.t() @ observed_A + I, observed_A.t() @ observed_B).solution
        else:
            # Linear regression
            u = torch.linalg.lstsq(observed_A, observed_B).solution   
        U.append(u.t())

        check_vec = u
        if torch.any(torch.isnan(check_vec)):
            print(f"NAN in epoch: {i}")
            print(check_vec)
            break

        # stat |AX-B| by individual elements
        if not (len(observed_idx)==0 or len(missing_idx)) == 0:
            error_list.append(torch.sum((missing_A @ u - missing_B)**2).item())
    
    end_time = time.time()
    cost_time = end_time - start_time

    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/missing_num)

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
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--d1", type=int, default=1000)
    parser.add_argument("--d2", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.1)
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
        _, _, M = load_data_syn(args.r, d1, d2, device)
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

    rmse_err_list, estimate_err_list, cost_time_list = [], [], []
    for run in range(args.runs):
        rmse_err, estimate_err, cost_time, curve = main(args, M)
        rmse_err_list.append(rmse_err)
        estimate_err_list.append(estimate_err)
        cost_time_list.append(cost_time)

    # Get the local time
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Save
    results = {
        'args': args,
        'dataset_content': dataset_content,
        'rmse_err_list': rmse_err_list,
        'estimate_err_list': estimate_err_list,
        'cost_time_list': cost_time_list,
        'curve': curve
    }
    torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}\nrmse error: {np.mean(rmse_err_list):.4f} +- {np.std(rmse_err_list)}\nestimate error: {np.mean(estimate_err_list):.4f} +- {np.std(estimate_err_list)}\ncost time: {np.mean(cost_time_list):.4f} +- {np.std(cost_time_list)}\nlocal time: {local_time}\n\n"
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    