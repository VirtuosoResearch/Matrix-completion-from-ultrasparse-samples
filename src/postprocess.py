import torch
import numpy as np

from utils import *
from data import *
from data_utils import *
from power_method_svd import power_svd
from sparse_utils import *
from recovery import lstsq_recovery

def soft_impute(T, T_masks, MTM, r, use_power_method=False, draw=False):
    device = T.device
    train_losses = []
    err_estimates = []

    epochs = 100
    tol = 1e-7
    lr = 0.1
    X = T
    #X = T * (~missing_mask_MTM) + T.mean() * missing_mask_MTM
    #U, D, Vt = sparse_svds_for_tensor(T, k=r)
    #UV = U @ torch.diag(D) @ Vt
    #print(torch.norm(MTM-UV, 'fro') / torch.norm(MTM, 'fro'))

    #print("sparse ratio of T: ", torch.sum(T_masks) / T.numel())
    loop = tqdm(range(epochs))
    r = r
    for i in loop:
        if not use_power_method:
            U, D, Vt = torch.linalg.svd(X)
            D[r:] = 0
        else:
            U, D, Vt = power_svd(X, k=r)
            #U, D, Vt = sparse_svds_for_tensor(X, k=r)
        """
        threshold = D[0] / 1.5
        value_mask = (D - threshold)
        indices = (value_mask <0).float()
        first_index = torch.nonzero(indices, as_tuple=False)[0].item()
        print(first_index)
        D[first_index+1:] = 0
        """
        X_update = U @ torch.diag(D) @ Vt
        #X = X * (1-lr) + X_update * lr
        X = T * T_masks + X_update * (1 - T_masks)
        #X = T * (~missing_mask_MTM) + X_update * missing_mask_MTM
        err = MTM - X
        #loss = (err**2).mean()
        #train_losses.append(loss.item())
        relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
        if len(err_estimates) > 1:
            if err_estimates[-1] > err_estimates[-2]:
                break
        if i > 10:
            if (abs(err_estimates[-1] - err_estimates[-2]) < tol) or (relative_err > err_estimates[0]):
                break
        last_err = relative_err
        loop.set_description(f"relative err: {relative_err:.7f}")
        err_estimates.append(relative_err.item())
        #print(relative_err)
    #X = T * missing_mask_MTM + X * (~missing_mask_MTM)
    replace_err = torch.norm(MTM-X, 'fro') / torch.norm(MTM, 'fro')
    print('last err: ', replace_err)
    if draw:
        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('../results/tmp/loss.png')
        plt.show()

        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Err Estimation')
        plt.xlabel('Epoch')
        plt.ylabel('Err')
        plt.title('Err Estimation')
        plt.savefig('../results/tmp/err.png')
        plt.legend()
    
    return X, err_estimates

def sparse_soft_impute(T, T_masks, T_M, TM_masks, r, draw=False):
    print(T)
    print(T_M)
    device = T.device
    train_losses = []
    err_estimates = []

    epochs = 20
    tol = 1e-7
    lr = 0.1
    X = T.coalesce()
    
    loop = tqdm(range(epochs))
    r = r
    for i in loop:
        X_scipy = torch_sparse_to_scipy(X)
        U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(X_scipy, k=r)
        U = torch.from_numpy(U_scipy[::-1].copy()).to(device).to_sparse()
        D = torch.from_numpy(D_scipy[::-1].copy()).to(device)
        D = torch.diag(D).to_sparse()
        Vt = torch.from_numpy(Vt_scipy[::-1].copy()).to(device).to_sparse()
        
        X_update = U @ D @ Vt
        #print(X_update)
        #X = X * (1-lr) + X_update * lr
        #X = T * T_masks + X_update * (1 - T_masks)
        X = T + X_update 
        #print(X)
        #X = T * (~missing_mask_MTM) + X_update * missing_mask_MTM
        err = T_M - X
        #print(err * err)
        relative_err = torch.norm(err, 'fro') / torch.norm(T_M, 'fro')
        if i > 10:
            if relative_err - last_err < tol:
                break
        last_err = relative_err
        loop.set_description(f"relative err: {relative_err:.7f}")
        err_estimates.append(relative_err.item())
        #print(relative_err)
    #X = T * missing_mask_MTM + X * (~missing_mask_MTM)

    if draw:
        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('../results/tmp/loss.png')
        plt.show()

        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Err Estimation')
        plt.xlabel('Epoch')
        plt.ylabel('Err')
        plt.title('Err Estimation')
        plt.savefig('../results/tmp/err.png')
        plt.legend()
    
    return X, err_estimates

def alt_min(T, T_masks, MTM, r, epochs=10000, lr=1, use_power_method=False, draw=False):
    device = T.device
    # impute missing values from rank-r SVD corresponding to masks
    train_losses = []
    err_estimates = []
    d = T.shape[0]

    U = torch.randn(d, r, device=device, requires_grad=True)
    #V = torch.randn(d, r, device=device, requires_grad=True)
    optimizer = optim.Adam([U, U], lr=lr)
    tol = 1e-5
    lam = 0.001
    #T_masks = 1 * (T != 0)
    print(T_masks.sum())
    loop = tqdm(range(epochs))
    for i in loop:
        optimizer.zero_grad()
        X = U @ U.t()
        loss = ((T-X)**2).mean() + lam*torch.norm(X, 'nuc')
        loss.backward()
        optimizer.step()

        #X = X * T_masks + X_update * (1 - T_masks)
        err = MTM - X
        train_losses.append(loss.item())
        relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
        if i > 19:
            if last_err - relative_err < tol:
                break
        last_err = relative_err
        loop.set_description(f"relative err: {relative_err:.7f}")
        err_estimates.append(relative_err.item())
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

        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Err Estimation')
        plt.xlabel('Epoch')
        plt.ylabel('Err')
        plt.title('Err Estimation')
        plt.legend()
        plt.savefig('../results/tmp/err.png')

    return X, err_estimates

def nuclear_reg(T, T_masks, MTM, r, epochs=100, lr=1, use_power_method=False, draw=False):
    d1, d2 = T.shape
    device = T.device
    # impute missing values from rank-r SVD corresponding to masks
    train_losses = []
    err_estimates = []
    epochs = 100
    lr = 1
    alpha = 0.01
    d = T.shape[0]
    #ETE = T_masks.T.float() @ T_masks.float()

    X = torch.rand(d2, d2, requires_grad=True, device=device)
    #X = T_emp
    X.requires_grad = True
    optimizer = optim.Adam([X], lr=lr)
    tol = 1e-5
    #T_masks = 1 * (T != 0)
    print(T_masks.sum())
    loop = tqdm(range(epochs))
    for i in loop:
        optimizer.zero_grad()
        with torch.no_grad():
            U, D, Vt = torch.linalg.svd(X.detach(), full_matrices=False)
        loss = torch.trace(X.T @ T)
        loss.backward()
        X.grad = X.grad - alpha * U @ Vt
        optimizer.step()
        X.grad.zero_()

        #X = X * T_masks + X_update * (1 - T_masks)
        #err = MTM - X
        train_losses.append(loss.item())
        #relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
        #if i > 19:
        #    if relative_err - last_err < tol:
        #        break
        #last_err = relative_err
        #loop.set_description(f"relative err: {relative_err:.7f}")
        #err_estimates.append(relative_err.item())
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

        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Err Estimation')
        plt.xlabel('Epoch')
        plt.ylabel('Err')
        plt.title('Err Estimation')
        plt.legend()
        plt.savefig('../results/tmp/err.png')

    return X, err_estimates

def direct_svd(T, T_masks, MTM, r, use_power_method=False, draw=False):
    if not use_power_method:
        U, D, Vt = torch.linalg.svd(T)
        D[r:] = 0
    else:
        #U, D, Vt = power_svd(X, k=r)
        U, D, Vt = sparse_svds_for_tensor(T, k=r)

    X = U @ torch.diag(D) @ Vt
    relative_err = torch.norm(X-MTM, 'fro') / torch.norm(MTM, 'fro')

    return X, relative_err