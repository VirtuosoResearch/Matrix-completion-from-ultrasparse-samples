import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sparse_utils import sparse_svds_for_tensor
from power_method_svd import power_svd
from utils import top_r_svd
import matplotlib.pyplot as plt

def lstsq_recovery(estimation_goal, M, masks, r, recovery_masks, use_reg=False, lam=0.001, batch=False):
    recovery_masks = recovery_masks.bool()
    device = M.device
    error_list = []
    test_num = 0
    total_num = 0
    U = []
    _, S, V = top_r_svd(estimation_goal, r)
    #_, _, V = power_svd(estimation_goal, k=r)
    #_, _, V = sparse_svds_for_tensor(estimation_goal, k=r)
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
        train_idx = []
        test_idx = []
        for non_zero_idx in non_zero_indices:
            if masks[i][non_zero_idx]:
                train_idx.append(non_zero_idx.item())
            else:
                test_idx.append(non_zero_idx.item())
        """

        # Get the mask values for the non-zero indices
        #print(M_row)
        mask_values = recovery_masks[i][non_zero_indices]
        #print('non zero: ', non_zero_indices)
        #print('mask: ', mask_values)
        # Use boolean indexing to separate trian and test indices
        train_idx = non_zero_indices[mask_values]
        test_idx = non_zero_indices[~mask_values]

        M_row = M_row.unsqueeze(0)
        n = non_zero_indices.shape[0]
        #print('train_idx: ', train_idx)

        num_train = train_idx.numel()
        num_test = test_idx.numel()

        if num_train < 1 or num_test < 1:
            #skip +=1
            continue
        
        
        test_num += num_test
        total_num += num_test + num_train

        #train_idx = torch.nonzero(torch.tensor(train_idx)).squeeze()
        #test_idx = torch.nonzero(torch.tensor(test_idx)).squeeze()

        # formula AX=B

        train_A = V[:, train_idx].t()
        train_B = M_row[:, train_idx].t()
        test_A = V[:, test_idx].t()
        test_B = M_row[:, test_idx].t()
        """
        print("train_A shape:", train_A.shape)
        print("train_B shape:", train_B.shape)
        print("train_A contains NaN:", torch.isnan(train_A).any())
        print("train_A contains Inf:", torch.isinf(train_A).any())
        print("train_B contains NaN:", torch.isnan(train_B).any())
        print("train_B contains Inf:", torch.isinf(train_B).any())
        """
        if num_train == 1:
            train_A.squeeze()
        #print(train_A.shape)
        #print(train_B)
        if use_reg:
            # Ridge regression
            lambda_reg = lam
            I = torch.eye(train_A.shape[1], device=device) * lambda_reg
            u = torch.linalg.lstsq(train_A.t() @ train_A + I, train_A.t() @ train_B).solution
        else:
            # Linear regression
            u = torch.linalg.lstsq(train_A, train_B).solution   
        U.append(u.t())

        check_vec = u
        if torch.any(torch.isnan(check_vec)):
            print(f"NAN in epoch: {i}")
            print(check_vec)
            break

        # stat |AX-B| by individual elements
        #if not (num_train < 1 or num_test < 1) == 0:
        error_list.append(torch.sum((test_A @ u - test_B)**2).item())
        #break
    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/test_num)

    if batch:
        return rmse_err**2, test_num
    return rmse_err

def mean_recovery(estimation_goal, M, masks, r, recovery_masks, use_reg=False, lam=0.001, batch=False):
    recovery_masks = recovery_masks.bool()
    device = M.device
    error_list = []
    test_num = 0
    total_num = 0
    U = []
    #_, _, V = power_svd(estimation_goal, k=r)
    #_, _, V = sparse_svds_for_tensor(estimation_goal, k=r)
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
        mask_values = recovery_masks[i][non_zero_indices]
        #print('non zero: ', non_zero_indices)
        #print('mask: ', mask_values)
        # Use boolean indexing to separate trian and test indices
        train_idx = non_zero_indices[mask_values]
        test_idx = non_zero_indices[~mask_values]

        mean = torch.mean(M_row[train_idx])

        test_B = M_row[test_idx].t()
        test_B = M_row[test_idx].t()

        num_train = train_idx.numel()
        num_test = test_idx.numel()

        if num_train < 1 or num_test < 1:
            #skip +=1
            continue
        
        
        test_num += num_test
        total_num += num_test + num_train

        

        # stat |AX-B| by individual elements
        #if not (num_train < 1 or num_test < 1) == 0:
        error_list.append(torch.sum((mean - test_B)**2).item())
        #break
    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/test_num)

    if batch:
        return rmse_err**2, test_num
    return rmse_err

def vanilla_MC(M, masks, num_entries, r, epochs=200, lr=0.1, draw=False):
    device = M.device
    # impute missing values from rank-r SVD corresponding to masks
    train_losses = []
    d1, d2 = M.shape

    U = torch.randn(d1, r, device=device, requires_grad=True)
    V = torch.randn(d2, r, device=device, requires_grad=True)
    optimizer = optim.Adam([U, U], lr=lr)
    tol = 1e-9
    #lam = 0.00001
    lam = 0
    
    #T_masks = 1 * (T != 0)
    loop = tqdm(range(epochs))
    for i in loop:
        optimizer.zero_grad()
        X = U @ U.T
        if masks is not None:
            loss = torch.norm((X-M)*masks, 'fro')/num_entries + lam*(torch.norm(U, p=2)**2/d1 + torch.norm(V, p=2)**2/d2)
        else:
            loss = torch.norm((X-M), 'fro')/num_entries + lam*(torch.norm(U, p=2)**2/d1 + torch.norm(V, p=2)**2/d2)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        #if i > 10:
        #    if train_losses[-1] - train_losses[-2] < tol:
        #        break
        loop.set_description(f"loss: {loss:.7f}")
        #print(relative_err)

    if draw:
        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('../results/tmp/loss.png')
        plt.show()


    return X