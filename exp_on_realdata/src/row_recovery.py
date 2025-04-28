import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def optimize_recovery_from_sparse(
    M: torch.sparse_coo_tensor,
    masks: torch.Tensor,       # dense bool tensor, True = use for training
    r: int,
    V: torch.Tensor = None,   # optional init for V (d2×r)
    epochs: int = 200,
    lr: float = 0.1,
    tol: float = 1e-9,
    lam: float = 0
):
    """
    Low‐rank completion starting from a sparse input.
    Args:
      M : sparse_coo_tensor of shape (n,d) with only the observed entries stored
      masks      : dense bool tensor (n,d), True for TRAIN positions among those in M
      r         : target rank
      V         : optional init factor (d2×r), if provided only U is optimized
      epochs, lr, tol, lam : as before
    Returns:
      M_hat_sparse : sparse_coo_tensor with recovered values at the original nonzero indices
      rmse         : float, RMSE on the held‐out (masks=False) entries
    """
    device = M.device
    n, d = M.shape

    # pull out all stored entries
    Mc = M.coalesce()
    idx = Mc.indices()      # shape (2, nnz)
    vals = Mc.values()      # shape (nnz,)

    flat_idx = idx[0] * d + idx[1]
    # split into train / test by looking up masks at each index
    if masks.is_sparse:
        # masks is a sparse_coo_tensor of training coords
        mco = masks.coalesce()
        midx = mco.indices()                   # (2, n_mask)
        flat_mask = midx[0] * d + midx[1]
        is_train = torch.isin(flat_idx, flat_mask)
    else:
        # masks is a dense bool tensor
        is_train = masks[idx[0], idx[1]]
    train_idx = idx[:, is_train]             # (2, n_train)
    train_vals = vals[is_train]              # (n_train,)
    test_idx  = idx[:, ~is_train]            # (2, n_test)
    test_vals = vals[~is_train]              # (n_test,)

    n_train = train_vals.shape[0]
    n_test  = test_vals.shape[0]

    # initialize U and V
    # scale init by magnitude of observed entries
    scale = (train_vals.pow(2).sum() / n_train).sqrt()
    U = torch.randn(n, r, device=device) * scale.sqrt()
    U.requires_grad_()

    optimizer = optim.Adam([U], lr=lr, weight_decay=lam)

    loop = tqdm(range(epochs), desc="training")
    prev_loss = None

    for _ in loop:
        optimizer.zero_grad()

        # predict only at training indices:
        Ui = U[train_idx[0]]        # (n_train, r)
        Vj = V[train_idx[1]]  # (n_train, r)
        pred_train = (Ui * Vj).sum(dim=1)  # (n_train,)

        # Fro‐norm of residuals, divided by n_train (to match original)
        loss = pred_train.sub(train_vals).norm(p='fro') / n_train
        loss.backward()
        optimizer.step()

        loop.set_description(f"loss: {loss:.7f}")
        if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
            break
        prev_loss = loss.item()

    # final predictions
    Ui = U[train_idx[0]];  Vj = V[train_idx[1]]
    pred_train = (Ui * Vj).sum(dim=1)

    Ui_test = U[test_idx[0]]; Vj_test = V[test_idx[1]]
    pred_test = (Ui_test * Vj_test).sum(dim=1)

    # test RMSE
    rmse = torch.sqrt((pred_test.sub(test_vals).pow(2).sum()) / n_test)

    return rmse.item()

# Solve the linear system Ax = b by optimizing the Frobenius norm ||Ax - b||_F^2
def optimize_recovery(M, masks, r, V=None, epochs=200, lr=0.1, tol=1e-9, lam=0):
    device = M.device
    train_losses = []
    n, d = M.shape

    nz_masks = M != 0
    masks = masks & nz_masks
    test_masks = ~masks & nz_masks

    # Initialize U (and V)
    U = torch.randn(n, r, device=device, requires_grad=False)
    magnitude = (torch.sum(M[masks.bool()]**2) / torch.sum(masks.bool()))**(1/2)
    U = U * magnitude**(1/2)
    U.requires_grad = True

    if V is None:
        V = torch.randn(d, r, device=device, requires_grad=False)
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
        M_row: 1*d
        V: r*d
        user_vector: 1*r
        """
        # make sure there is non-zero value in original matrix
        M_row = M[i]
        non_zero_indices = M_row.nonzero(as_tuple=True)[0]

        # select non-zero & masked elements as training data, non-zero & non-masked elements as testing data
        # Get the masks values for the non-zero indices
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