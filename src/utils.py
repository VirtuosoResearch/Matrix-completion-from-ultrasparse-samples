import os
import numpy as np
import yaml
import logging
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

from sparse_utils import sparse_svds_for_tensor

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def load_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    logging.info("Using best configs")

    for k, v in configs.items():
        if "lr" in k or "decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

# sym noise matrix
def sym_noise(d, std):
    noise = torch.normal(mean=0, std=std, size=(d,d))
    # Make the matrix symmetric: A = (noise + noise^T) / 2
    symmetric_noise = (noise + noise.T) / 2

    # Adjust the standard deviation of the symmetric matrix to match the desired std (tau)
    adjusted_symmetric_noise = symmetric_noise * torch.sqrt(torch.tensor(2.0))

    return adjusted_symmetric_noise

# top r svd
def top_r_svd(A, r):
    V, S, Vt = torch.linalg.svd(A, full_matrices=True)

    Vr = V[:, :r]  
    Sr = S[:r]   
    Vtr = Vt[:r, :]
    return Vr, Sr, Vtr

# norm by dividing the max singular value
def svd_norm(A):
    U, S, V = torch.svd(A)

    S_normalized = S / S.max()
    scale = S.max()

    A_normalized = U @ torch.diag(S_normalized) @ V.t()

    return A_normalized, scale

# norm by UV
def uv_norm(A):
    U, S, V = torch.svd(A)

    A_normalized = U @ V.t()

    return A_normalized

# reverse norm by multipying the max singular value
def svd_norm_reverse(A, scale):
    U, S, V = torch.svd(A)

    S_reverse = S * scale

    A_normalized = U @ torch.diag(S_reverse) @ V.t()

    return A_normalized

def l2_dis(a, b, dim=0):
    a_normalized = F.normalize(a, p=2, dim=dim)
    b_normalized = F.normalize(b, p=2, dim=dim)

    l2_distance = torch.norm(a_normalized - b_normalized, p=2, dim=dim)
    return l2_distance.mean()

def check_rank(M, draw=False, max_k=30):
    #U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U, S, Vh = sparse_svds_for_tensor(M, k=max_k)
    singular_values = S.cpu().numpy()

    cumulative_variance = np.cumsum(singular_values**2) / np.sum(singular_values**2)

    threshold = 0.9
    r = np.argmax(cumulative_variance >= threshold) + 1
    print(f"useful rank cum contribution ≥ {int(threshold*100)}%: {r}")

    if draw:
        plt.figure(figsize=(8, 5))
        plt.plot(singular_values, 'o-', linewidth=2)
        plt.title('specturm')
        plt.xlabel('num')
        plt.ylabel('value')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8, 5))
        plt.plot(cumulative_variance, 'o-', linewidth=2)
        plt.axhline(y=0.9, color='r', linestyle='--', label='90\% contribution')
        plt.title('cumulative comtribution')
        plt.xlabel('num')
        plt.ylabel('cumulative contribution')
        plt.legend()
        plt.grid(True)
        plt.show()

def relative_err(X, MTM):
    return (torch.norm(X-MTM, 'fro') / torch.norm(MTM, 'fro')).item()