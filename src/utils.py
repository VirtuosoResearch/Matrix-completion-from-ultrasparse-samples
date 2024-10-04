import os
import numpy as np
import yaml
import logging
import torch
from torch import optim

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