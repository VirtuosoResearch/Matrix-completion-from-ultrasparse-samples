import torch
import numpy as np
from tqdm import tqdm
from sparse_utils import sparse_svds_for_tensor
from power_method_svd import power_svd
from utils import top_r_svd

def lstsq_recovery(estimation_goal, M, masks, r, recovery_masks, use_reg=False, lam=0.001):
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
        trian_idx = []
        test_idx = []
        for non_zero_idx in non_zero_indices:
            if masks[i][non_zero_idx]:
                trian_idx.append(non_zero_idx.item())
            else:
                test_idx.append(non_zero_idx.item())
        """

        # Get the mask values for the non-zero indices
        mask_values = recovery_masks[i][non_zero_indices]

        # Use boolean indexing to separate trian and test indices
        trian_idx = non_zero_indices[mask_values]
        test_idx = non_zero_indices[~mask_values]

        M_row = M_row.unsqueeze(0)
        n = non_zero_indices.shape[0]

        #if len(trian_idx)==0 or len(test_idx) == 0:
        #    skip +=1
        #    continue
        
        test_num += len(test_idx)
        total_num += len(test_idx) + len(trian_idx)

        trian_idx = torch.tensor(trian_idx)
        test_idx = torch.tensor(test_idx)

        # formula AX=B
        train_A = V[:, trian_idx].t()
        train_B = M_row[:, trian_idx].t()
        test_A = V[:, test_idx].t()
        test_B = M_row[:, test_idx].t()

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
        if not (len(trian_idx)==0 or len(test_idx)) == 0:
            error_list.append(torch.sum((test_A @ u - test_B)**2).item())

    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/test_num)

    return rmse_err