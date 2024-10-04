import numpy as np
from tqdm import tqdm

from power_method_svd import power_svd

from data_utils import load_data_all, load_data_syn
from utils import *
import torch
import time

if torch.cuda.is_available():
    free_gpu = get_free_gpu()
    device = 'cuda:{}'.format(free_gpu)
    #device = 'cuda:0'
else:
    device = 'cpu'

dataset = 'syn'
d1 = 1000
d2 = 100
r = 5
if dataset == 'syn':
    _, _, M = load_data_syn(r, d1, d2, device)
else:
    M = load_data_all(dataset)
    M = M.float().to(device)
    d1, d2 = M.shape
dataset_content = f'd1 = {d1}, d2 = {d2}, entries = {torch.count_nonzero(M)}\n'
print(dataset_content)
p = 0.1

def compute_matrix_adjustment(V):
    # 计算 V^T * V
    #print(V)
    VTV = torch.mm(V.T, V)
    
    # 特征值分解 V^T * V
    eigvals, eigvecs = torch.linalg.eigh(VTV)
    
    # 计算 (V^T V)^-1/2
    # 我们只取正特征值的逆平方根
    inv_sqrt_eigvals = torch.diag(torch.stack([1/torch.sqrt(val) if val > 0 else 0 for val in eigvals]))
    
    # 重构 (V^T V)^-1/2
    VTV_inv_sqrt = eigvecs @ inv_sqrt_eigvals @ eigvecs.T
    
    # 计算 V * (V^T V)^-1/2
    result = V @ VTV_inv_sqrt
    
    return result

def compute_V_tilde(V_tilde):
    """
    使用 SVD 计算矩阵的平方根。
    
    参数：
    V_tilde -- 输入矩阵，形状为 (n, r)
    
    返回：
    计算后的矩阵 V
    """
    # 计算 V_tilde^T V_tilde
    VtV = V_tilde.T @ V_tilde
    
    # 对 VtV 进行 SVD 分解
    #U, S, VT = torch.linalg.svd(VtV)
    U, S, VT = power_svd(VtV, k=r)
    
    # 计算奇异值的平方根的逆
    S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S))
    
    # 计算 V
    V = V_tilde @ U @ S_inv_sqrt @ VT
    
    return V

def matrix_negative_half_power(matrix):
    # 特征值分解
    eigvals, eigvecs = torch.linalg.eig(matrix)
    
    # 修改特征值：负特征值设为零，正特征值取-1/2次幂
    eigvals[eigvals < 0] = 0
    eigvals = eigvals ** (-1/2)

    matrix_neg_half = eigvecs @ torch.diag(eigvals) @ torch.linalg.inv(eigvecs)
    
    return matrix_neg_half

def generate_symmetric_gaussian_noise(r, Gamma_u, sigma):
    # 生成随机矩阵
    A = torch.rand(r, r)
    
    # 使其对称
    A = (A + A.T) / 2
    
    # 调整标准差
    G_j = A * (Gamma_u**4 * sigma)
    
    return G_j

def sample_Omega(Omega, k):
    n, m = Omega.shape
    sampled_indices = []
    
    for i in range(n):
        # 获取当前行的非零列索引
        non_zero_indices = torch.where(Omega[i] != 0)[0].cpu().numpy()
        # 如果非零列索引数量少于 k，则全选，否则随机选择 k 个
        if len(non_zero_indices) <= k:
            selected_indices = non_zero_indices
        else:
            selected_indices = np.random.choice(non_zero_indices, k, replace=False)
        
        sampled_indices.append(selected_indices)
    
    return sampled_indices

def project_to_psd_cone(A):
    # 对矩阵 A 进行特征值分解
    eigvals, eigvecs = torch.linalg.eigh(A)
    
    # 将所有负的特征值设为零
    eigvals[eigvals < 0] = 0
    
    # 重新构造正半定矩阵
    A_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    
    return A_psd

def clip_matrix(M, Gamma_M):
    M = M * max(1, 1/torch.norm(M))
    return M
    #return torch.clip(M, -Gamma_M, Gamma_M)

def P_omega(M, Omega):
    P = torch.zeros_like(M)
    P[Omega] = M[Omega]
    return P

def Aitem(U, Omega, P_Omega_M, k, lambda_, Gamma_u, Gamma_M):
    n, r = U.shape
    m = P_Omega_M.shape[1]
    V = torch.rand(m, r).to(device)
    Omega_prime = sample_Omega(Omega.T, k)
    
    for j in range(m):
        #G_j = torch.random.normal(0, Gamma_u**4 * sigma**2, (r, r))
        G_j = generate_symmetric_gaussian_noise(r, Gamma_u, sigma).to(device)
        g_j = np.random.normal(0, Gamma_u**2 * Gamma_M**2 * sigma**2, r)
        g_j = torch.from_numpy(g_j).float().to(device)
        X_j = lambda_ * torch.eye(r).to(device) + sum([torch.outer(U[i],U[i]).to(device) for i in Omega_prime[j]]) + G_j
        #V_j = torch.linalg.lstsq(project_to_psd_cone(X_j), (sum([P_Omega_M[i, j].to(device) * U[i].to(device) for i in Omega_prime[j]])+g_j)).solution
        V_j = torch.linalg.pinv(project_to_psd_cone(X_j)) @ (sum([P_Omega_M[i, j].to(device) * U[i].to(device) for i in Omega_prime[j]])+g_j)
        #V_j = project_to_psd_cone(torch.linalg.pinv(X_j) @ (sum([P_Omega_M[i, j] * U[i] for i in Omega_prime[j]])+g_j))
        V[j] = V_j
    #print(V)
    #V_tilde = torch.stack(V)
    V_tilde = V
    #VV = matrix_negative_half_power(torch.linalg.pinv(V_tilde.T @ V_tilde))
    VV = torch.linalg.pinv(project_to_psd_cone(V_tilde.T @ V_tilde))
    #print(VV)
    #print(torch.linalg.pinv(VV))
    V = compute_V_tilde(V_tilde)
    return compute_matrix_adjustment(V_tilde)
    #return V
    #return V_tilde @ torch.linalg.pinv(V_tilde.T @ V_tilde) ** 0.5
    #return V_tilde @ VV ** 0.5

def Auser(V, Omega_i, P_Omega_M, T, lambda_, Gamma_u):
    def update_u(V, Omega_i_prime, M, lambda_):
        r = V.shape[1]
        I = torch.eye(r)
        
        # 计算 lambda I + sum(V_j V_j^T)
        A = lambda_ * I
        for j in Omega_i_prime:
            V_j = V[j]
            A += torch.outer(V_j, V_j)
        
        # 计算 sum(M_ij V_j)
        b = torch.zeros(r)
        for j in Omega_i_prime:
            M_ij = M[j]
            V_j = V[j]
            b += M_ij * V_j
        
        # 计算 u
        u = torch.linalg.solve(A, b)
        
        return u
    m, r = V.shape
    U = torch.zeros((n, r)).to(device)

    nonzero_indices = [torch.where(Omega_i[i] != 0)[0].cpu().numpy() for i in range(n)]
    Omega_prime_i = [np.random.choice(nonzero_indices[i], size=max(1, len(nonzero_indices[i]) // T), replace=False) for i in range(n)]

    for i in range(n):
        #print((sum([ V[j] for j in Omega_prime_i[i]])).shape)
        X_i = lambda_ * torch.eye(r).to(device) + sum(torch.outer(V[j], V[j]) for j in Omega_prime_i[i])
        u_i = torch.linalg.lstsq(X_i, sum([P_Omega_M[i, j].to(device) * V[j].to(device) for j in Omega_prime_i[i]])).solution
        #u_i = torch.linalg.pinv(X_i) @ sum([P_Omega_M[i, j].to(device) * V[j].to(device) for j in Omega_prime_i[i]])
        #U[i] = torch.clip(u_i, -Gamma_u, Gamma_u)
        U[i] = clip_matrix(u_i, Gamma_u)

    return U

def DPALS(P_Omega_M, Omega, sigma, Gamma_u, Gamma_M, T, lambda_, r, k, V0):
    n, m = P_Omega_M.shape
    V_t = V0
    U_t = torch.rand(n, r).to(device)
    
    for t in tqdm(range(T)):
        U_t = Auser(V_t, Omega, P_Omega_M, T, lambda_, Gamma_u)
        V_t = Aitem(U_t, Omega, P_Omega_M, k, lambda_, Gamma_u, Gamma_M)
    
    return U_t.T, V_t.T

# 示例初始化
runs = 1
rmse_list = []
G_list = []
time_list = []
"""
min_ret = 1000
for sigma in [0.1, 0.01]:
    for Gamma_u in [2,3,4]:
        for Gamma_M in [2,3,4]:
            for T in [5, 10 ,20]:
"""
D = M
for run in range(runs):
    n, m = D.shape
    # 示例初始化
    Omega = (torch.rand(n, m) <= p).to(device)
    Omega = Omega * (D!=0)
    non_zero_rows = torch.any(D * Omega != 0, axis=1)
    M = D[non_zero_rows]
    Omega = Omega[non_zero_rows]
    n, m = M.shape
    
    sigma = 0.1
    Gamma_u = 4
    Gamma_M = 2
    T = 50
    
    lambda_ = 0.01
    k = 40
    V0 = torch.rand(m, r).to(device)  # 初始 V

    start_time = time.time()

    P_Omega_M = clip_matrix(M*Omega, Gamma_M)
    U_final, V_final = DPALS(P_Omega_M, Omega, sigma, Gamma_u, Gamma_M, T, lambda_, r, k, V0)
    #mask = M>0.1
    mask = M!=0
    num = torch.sum(mask)
    #print(num)
    def P_omega_test(X):
        return X * (~Omega * mask)
    def P(X):
        return X * mask
    X_estimation = U_final.T @ V_final
    rand_mat = torch.rand(n,m)
    rmse_list.append(((torch.norm(P_omega_test(M - X_estimation)))/torch.sqrt(torch.sum((~Omega * mask)))).item())

    end_time = time.time()
    cost_time = end_time - start_time
    time_list.append(cost_time)

    #MTM = M.T @ M
    #XTX = X_estimation.T @ X_estimation
    #G_list.append(torch.norm(XTX-MTM) / torch.norm(MTM))
    #print((torch.linalg.norm(P(M - X_estimation)))/torch.sqrt(num))
    #print((torch.linalg.norm(P_omega_test(M - X_estimation)))/torch.sqrt(torch.sum((~Omega * mask))))

    """
    if rmse_list[-1] < min_ret:
        min_ret = rmse_list[-1]
        sigma_best = sigma
        Gamma_u_best = Gamma_u
        Gamma_M_best = Gamma_M
        T_best = T
    """
    
#print("Final U^T:\n", U_final)
#print("Final V^T:\n", V_final)
print(np.mean(rmse_list))
print(np.std(rmse_list))
print(np.mean(time_list))
print(np.std(time_list))
#print(np.mean(G_list))
#print(np.std(G_list))
"""
print("grid search")
print(min_ret)
print(sigma_best)
print(Gamma_u_best)
print(Gamma_M_best)
print(T_best)
"""