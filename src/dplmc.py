import argparse
import os

import pandas as pd
import pickle
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import sys
from data_utils import load_data_all


dataset = 'yahoo_music'
D = load_data_all(dataset, s=300)

print(f"original user: {D.shape[0]} item: {D.shape[1]}")
#D = D.numpy()

def differential_private_initialization(Y, G, tau, sigma0, p):
    m, n = Y.shape

    def projection(V, alpha2):
        V_projected = V.clone()
        for i in range(V_projected.shape[0]):
            row_norm = torch.linalg.norm(V_projected[i, :], 2)
            if row_norm > alpha2:
                V_projected[i, :] = (alpha2 / row_norm) * V_projected[i, :]
        return V_projected

    # Step 1: Projection
    Y = projection(Y, G)

    # Step 2: Communicate (receive)
    Y_i = Y  # All users send their data to the server

    # Step 4: Server calculates private covariance matrix A
    
    A = tau**2 *sum([torch.outer(Y_i[i], Y_i[i]) for i in range(m)]) / p**2 + torch.normal(0, sigma0, size=(n, n))
    #A = (A + A.T) / 2  # Ensure A is symmetric

    # Step 5: Obtain (V^0, Σ) by performing the rank r SVD of A
    _, S, Vt = torch.linalg.svd(A, full_matrices=False)

    V0 = Vt[:r, :].T
    Sigma = torch.diag(S[:r])
    V0_tilde = V0 * torch.sqrt(S[:r])

    U0 = [(tau * Y[i].T @ V0 * torch.sqrt(S[:r])/p) for i in range(m)]
    U0 = torch.stack(U0)
    return U0.T, V0_tilde

# 初始化参数



#U0, V0_tilde = differential_private_initialization(Y, G, tau, sigma0, p)
#print("Initialized U0^T:\n", U0)
#U0 = U0.T

import numpy as np

p = 0.75

m, n = D.shape
runs = 1
ret_list = []
XTX_list = []
for run in range(runs):
    # 示例初始化
    m, n = D.shape
    Omega = torch.rand(m, n) <= p 

    Omega = Omega * (D>0)
    non_zero_rows = torch.any(D * Omega != 0, axis=1)
    M = D[non_zero_rows]
    Omega = Omega[non_zero_rows]
    m, n = M.shape
    mask = Omega
    Y = M * mask   # 示例数据
    """
    M = D
    m, n = M.shape
    Omega = torch.random.rand(m, n) <= p 
    mask = Omega
    Y = M * mask
    """

    def P_omega(X):
        return X * mask

    def P_omegai(X, i):
        return X * mask[i]

    def differential_private_low_rank_matrix_completion(U0, V0_tilde, Y, T, alpha1, alpha2, G, eta, sigma1, sigma2):
        """
        Ut -> U
        """

        U = U0
        V = V0_tilde
        Y_hat = torch.stack([V @ U[i] for i in range(m)])

        
        def projection(X, alpha):
            norm = torch.linalg.norm(X, ord=2, axis=None)
            return X if norm <= alpha else (alpha / norm) * X
        
        def projection_(V, alpha):

            V_projected = V.clone()
            for i in range(V_projected.shape[0]):
                row_norm = torch.linalg.norm(V_projected[i, :], 2)
                if row_norm > alpha2:
                    V_projected[i, :] = (alpha2 / row_norm) * V_projected[i, :]
            return V_projected
        
        def projection_1d(V, alpha2):

            row_norm = torch.linalg.norm(V, 2)
            if row_norm > alpha2:
                V = (alpha2 / row_norm) * V
            return V

        for t in range(T):
            if t == 0:            
                # 用户端接收
                P_omega_Y_hat_minus_Y = P_omega(Y_hat - Y)

            # 服务器端计算
            P_omega_Y_hat_minus_Y = projection(P_omega_Y_hat_minus_Y, G)

            R_tilde = sum([U[i] @ U[i].T for i in range(m)]) - V.T  @ V + torch.normal(0, sigma1, size=(r, r))
            V = V - (eta / p) * (P_omega_Y_hat_minus_Y.T @ U + torch.normal(0, sigma2, size=(n, r))) + (eta / 2 ) * V @ R_tilde
            V = projection(V, alpha2)

            # 用户端更新
            for i in range(m):
                U[i] = U[i] - (eta / p) * P_omegai(Y_hat[i].T - Y[i].T, i) @ V - (eta / 2) * U[i].T @ R_tilde
                U[i] = projection_1d(U[i], alpha1)
                Y_hat[i] = V @ U[i].T


        return U, V

    # 初始化参数
    print(M.shape)
    T = 5
    alpha1 = 3
    alpha2 = 4
    G = alpha1 * alpha2
    eta = 0.3
    sigma1 = 0.1
    sigma2 = 0.1
    tau = 1
    sigma0 = 0.0
    min_rmse = 1000
    
    """
    for alpha1 in [1,2,3,4,5]:
        for alpha2 in [1,2,3,4,5]:
            for eta in [0.05,0.1, 0.3,0.5,1]:
                for tau in [0.1, 0.5, 1]:
    """                
    
    U0, V0_tilde = differential_private_initialization(Y, G, tau, sigma0, p)
    U0 = U0.T


    #print(U0.shape)
    #print(V0_tilde.shape)

    #print("Initialized U0^T:\n", U0)

    U_final, V_final = differential_private_low_rank_matrix_completion(U0, V0_tilde, Y, T, alpha1, alpha2, G, eta, sigma1, sigma2)

    X_estimation = U_final @ V_final.T
    mask_test = M>0.1
    mask_test = mask
    #print(np.sum(mask_test))
    mask_test = mask_test.float()
    rand_mat = torch.rand(m,n)
    #print((torch.norm(P_omega(M))**2)/(m*n))
    #print((torch.norm((M - rand_mat)*mask_test)**2)/np.sum(mask_test))
    #rmse = np.sqrt(np.sum(P_omega(M - X_estimation)**2/np.sum(mask)))
    #rmse = (torch.norm(P_omega(M - X_estimation))**2)/(np.sum(mask))
    rmse = (torch.norm(M - X_estimation)**2)/(m*n)
    #rmse2 = (torch.norm((M - X_estimation)*mask_test)**2)/np.sum(mask_test)
    ret_list.append(rmse)
    #print(rmse)
    #print(np.sqrt(rmse2))

    MTM = M.T @ M
    XTX = X_estimation.T @ X_estimation
    err = torch.norm(XTX-MTM) / torch.norm(MTM)
    #print(torch.norm(XTX-MTM) / torch.norm(MTM))
    XTX_list.append(err)
    #print("end iter")
    
    if rmse < min_rmse:
        min_rmse = rmse
        alpha1_best = alpha1
        alpha2_best = alpha2
        eta_best = eta
        tau_best = tau
                    
#print("Final U:\n", U_final)
#print("Final V:\n", V_final)
print(np.mean(ret_list))
print(np.std(ret_list))
print(np.mean(XTX_list))
print(np.std(XTX_list))
print("grid search")

print(min_rmse)
print(alpha1_best)
print(alpha2_best)
print(eta_best)
print(tau_best)
