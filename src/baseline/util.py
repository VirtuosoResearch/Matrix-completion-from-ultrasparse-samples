import torch
import numpy as np

device = 'cuda'
if device in ['cuda', 'mps']:
    print("Using {}".format(device))
    def from_numpy(arr):
        return torch.from_numpy(arr).to(dtype=torch.float32, device=torch.device(device), non_blocking=True)
    def to_device(obj):
        return obj.to(dtype=torch.float32, device=torch.device(device), non_blocking=True)
else:
    print("Using cpu")
    from torch import from_numpy
    to_device = lambda x: x

def generate_gaussian_factors(n, d, k):
    U = from_numpy(np.random.normal(size = (n, k)))
    V = from_numpy(np.random.normal(size = (d, k)))
    U = U / k**(1/4)
    V = V / k**(1/4) # u^Tv scales as k**(1/2)
    return U, V
