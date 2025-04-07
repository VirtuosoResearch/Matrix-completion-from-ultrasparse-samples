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

def main(args, M):
    d1, d2 = M.shape
    p = args.p
    r = args.r
    recovery_p = 0.75
    if args.sample == 'uniform':
        observed_M, masks = get_masks(M, p)
    #observed_M, masks = get_uniformly_random_samples(M.cpu().numpy(), p)
    elif args.sample == 'perrow':
        observed_M, masks = get_random_samples_per_row(M.cpu().numpy(), args.sample_entry)
        p = args.sample_entry / d2
        observed_M = torch.from_numpy(observed_M).float().to(device)
        masks = torch.from_numpy(masks).to(device)
    #_, recovery_masks = get_masks(M, recovery_p)
    #print(recovery_masks)

    non_zero_rows = torch.any(observed_M != 0, dim=1)

    #observed_M = observed_M[non_zero_rows]
    #masks = masks[non_zero_rows]
    #M = M[non_zero_rows]
    print(M.shape)

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
    #T += noise_matrix
    MTM = M.T @ M

    mask_err_all = T - MTM
    mask_err_mask = T*T_masks - MTM*T_masks
    diag_cov = torch.diag( torch.diag(cov_observe_M) )
    T_p = (1.0 / p) * diag_cov + (1.0 / (p**2)) * (cov_observe_M - diag_cov)
    #mask_err = T*missing_mask_MTM - MTM*missing_mask_MTM
    #mask_err = T - MTM

    print('original')
    print(torch.norm(cov_observe_M*T_masks - MTM*T_masks, 'fro') / torch.norm(MTM*T_masks, 'fro'))
    print(torch.norm(cov_observe_M - MTM, 'fro') / torch.norm(MTM, 'fro'))
    print("prob")
    print(torch.norm(T_p*T_masks - MTM*T_masks, 'fro') / torch.norm(MTM*T_masks, 'fro'))
    print(torch.norm(T_p - MTM, 'fro') / torch.norm(MTM, 'fro'))
    print("freq")
    print(torch.norm(mask_err_mask, 'fro') / torch.norm(MTM*T_masks, 'fro'))
    print(torch.norm(mask_err_all, 'fro') / torch.norm(MTM, 'fro'))

    # impute missing values from rank-r SVD corresponding to masks
    use_power_method = True
    train_losses = []
    err_estimates = []

    epochs = 30
    tol = 1e-7
    lr = 0.1
    X = T
    
    #T_masks = 1 * (T != 0)
    print(T_masks.sum())
    loop = tqdm(range(epochs))
    """
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

        X = X * T_masks + X_update * (1 - T_masks)
        #X = X * (1-lr) + X_update * lr
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
    print("X")
    print(torch.norm(X*T_masks-MTM*T_masks, 'fro') / torch.norm(MTM, 'fro'))
    print(torch.norm(X-MTM, 'fro') / torch.norm(MTM, 'fro'))

    """
    X = nuclear_reg(T, T_masks, MTM, r)

    plt.figure(figsize=(5, 3))
    plt.plot(err_estimates, label='Relative Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Relative Error')
    plt.legend()
    plt.savefig(f'../plots/{args.label}-rel-err.png', dpi=150)
    plt.show()
    end_time = time.time()
    #rmse_err = lstsq_recovery(estimation_goal=X, M=M, masks=masks, r=r, recovery_masks=recovery_masks)
    
    
    cost_time = end_time - start_time


    curve = {
        'train_losses': train_losses,
        'err_estmates': err_estimates,
    }

    return 0, 0, cost_time, curve
    return rmse_err, err_estimates[-1], cost_time, curve



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
    parser.add_argument("--epsilon", type=float, default=20)
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

    rmse_err_list, estimate_err_list, cost_time_list = [], [], []
    for run in range(args.runs):
        rmse_err, estimate_err, cost_time, curve = main(args, M)
        rmse_err_list.append(rmse_err)
        estimate_err_list.append(estimate_err)
        cost_time_list.append(cost_time)

    # Get the local time
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_content = f'local time: {local_time}\n'

    # Save
    results = {
        'args': args,
        'dataset_content': dataset_content,
        'rmse_err_list': rmse_err_list,
        'estimate_err_list': estimate_err_list,
        'cost_time_list': cost_time_list,
        'curve': curve
    }
    #torch.save(results, f'../results/results_data/{args.label}.pt')

    # Define the content in the desired format
    content = f"run times: {args.runs}\nrmse error: {np.mean(rmse_err_list):.4f} +- {np.std(rmse_err_list)}\nestimate error: {np.mean(estimate_err_list):.4f} +- {np.std(estimate_err_list)}\ncost time: {np.mean(cost_time_list):.4f} +- {np.std(cost_time_list)}\n\n"
    print(content)
    # Write the content to a file
    with open(f'../results/{args.label}.txt', 'a') as file:
        file.write(time_content)
        file.write(dataset_content)
        file.write(privacy_content)
        file.write(content)
    