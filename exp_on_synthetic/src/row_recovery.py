import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Solve the linear system Ax = b by optimizing the Frobenius norm ||Ax - b||_F^2
def optimize_recovery(M, masks, r, V=None, epochs=200, lr=0.1, tol=1e-9, lam=0):
    device = M.device
    train_losses = []
    d1, d2 = M.shape

    nz_masks = M != 0
    masks = masks & nz_masks
    test_masks = ~masks & nz_masks

    # Initialize U (and V)
    U = torch.randn(d1, r, device=device, requires_grad=False)
    magnitude = (torch.sum(M[masks.bool()]**2) / torch.sum(masks.bool()))**(1/2)
    U = U * magnitude**(1/2)
    U.requires_grad = True

    if V is None:
        V = torch.randn(d2, r, device=device, requires_grad=False)
        magnitude = (torch.sum(M[masks.bool()]**2) / torch.sum(masks.bool()))**(1/2)
        V = V * magnitude**(1/2)
        V.requires_grad = True
        optimizer = optim.Adam([U, V], lr=lr, weight_decay=lam)
    else:
        optimizer = optim.Adam([U], lr=lr, weight_decay=lam)
    
    num_entries = torch.sum(masks).item()
    loop = tqdm(range(epochs))
    for i in loop:
        optimizer.zero_grad()
        X = U @ V.T
        loss = torch.norm((X-M)[masks], 'fro')/num_entries

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        if i > 10:
            if abs(train_losses[-1] - train_losses[-2]) < tol:
                break
        loop.set_description(f"loss: {loss:.7f}")
    
    rmse = torch.sqrt(torch.sum((X-M)[test_masks]**2)/test_masks.sum())

    return rmse.item()

def top_r_svd(A, r):
    V, S, Vt = torch.linalg.svd(A, full_matrices=True)

    Vr = V[:, :r]  
    Sr = S[:r]   
    Vtr = Vt[:r, :]
    return Vr, Sr, Vtr

# Solve the linear system Ax = b by row-wise least squares
def lstsq_recovery(estimation_goal, M, r, recovery_masks, V=None, use_reg=False, lam=0.001):
    recovery_masks = recovery_masks.bool()
    device = M.device
    error_list = []
    test_num = 0
    total_num = 0
    U = []
    if V is None:
        _, S, V = top_r_svd(estimation_goal, r)
    for i in tqdm(range(M.shape[0])):
        """
        M_row: 1*d2
        V: r*d2
        user_vector: 1*r
        """
        # make sure there is non-zero value in original matrix
        M_row = M[i]
        non_zero_indices = M_row.nonzero(as_tuple=True)[0]

        # select non-zero & masked elements as training data, non-zero & non-masked elements as testing data
        # Get the mask values for the non-zero indices
        mask_values = recovery_masks[i][non_zero_indices]

        # Use boolean indexing to separate trian and test indices
        train_idx = non_zero_indices[mask_values]
        test_idx = non_zero_indices[~mask_values]

        M_row = M_row.unsqueeze(0)

        num_train = train_idx.numel()
        num_test = test_idx.numel()

        if num_train < 1 or num_test < 1:
            #skip +=1
            continue
        
        test_num += num_test
        total_num += num_test + num_train

        # formula AX=B
        train_A = V[:, train_idx].t()
        train_B = M_row[:, train_idx].t()
        test_A = V[:, test_idx].t()
        test_B = M_row[:, test_idx].t()

        if num_train == 1:
            train_A.squeeze()
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
        error_list.append(torch.sum((test_A @ u - test_B)**2).item())

    # stat |AX-B| by individual elements
    rmse_err = np.sqrt(np.sum(error_list)/test_num)

    return rmse_err