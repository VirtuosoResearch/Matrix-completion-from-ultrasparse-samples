import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import torch
import scipy

from datasets import load_1000genomes_matrix, load_random_matrix, load_1000genomes_large, load_1000genomes_demographics
from matrix_completion_methods import estimate_sim_weighted, estimate_sim_iter, complete_matrix_iter, complete_matrix_grad
from util import from_numpy, to_device, generate_gaussian_factors

def sample_mask(n, d, n_obs):
    mask = np.zeros((n, d))
    for i in range(n):
        idx = np.random.choice(np.arange(d), size = n_obs, replace = False)
        mask[i,idx] = 1
    return mask

def visualize_factors(v, labels, markers, colors, title = ''):
    """
    v: (d, k) array
    labels: (d,) array of labels
    """
    v_2d = TSNE(n_components=2).fit_transform(v)
    labels = np.array(labels)
    label_set = np.unique(labels)
    for l in label_set:
        plt.scatter(v_2d[labels==l,0], v_2d[labels==l,1], marker = markers[l], c = colors[l], label = l)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show()

def factorize_matrix_and_visualize(matrix, labels, k, markers, colors, take_sqrt = False, approximate = False):
    """
    matrix: (m, n) array
    """
    if not approximate:
        u, s, vh = np.linalg.svd(matrix)
        if take_sqrt:
            s = s**(1/2)
        visualize_factors(vh.T @ np.diag(s), labels, markers, colors)
        visualize_factors(vh.T[:,:k] @ np.diag(s[:k]), labels, markers, colors)
        plt.plot(np.arange(len(s)), s)
    else:
        U, V = factorize_matrix_grad(matrix) # (n,k), (d,k)
        visualize_factors(V, labels, markers, colors)

def factorize_matrix_grad(X, k = None, lr = 100, iters = 100, print_every = 1, verbose = True):
    X = from_numpy(X)
    n, d = X.shape
    if k is None:
        k = min(n,d)
    #U, V = torch.zeros((n,k)), torch.zeros((d,k))
    U, V = generate_gaussian_factors(n, d, k)
    magnitude = torch.mean(torch.abs(X))
    U, V = U * magnitude**(1/2), V * magnitude**(1/2)
    #U, V = U / 100, V / 100 # small init
    U.requires_grad = True
    V.requires_grad = True
    trainer = torch.optim.Adam([U, V], lr = lr)
    for i in range(iters):
        trainer.zero_grad()
        loss = torch.mean((U @ V.mT - X)**2)
        loss.backward()
        trainer.step()
        if i % print_every == 0 and verbose:
            print("Iter: {}/{}, loss={:.4f}".format(i+1,iters,loss))
    return U.detach().cpu().numpy(), V.detach().cpu().numpy()

def compute_v_l2(estimate_v, gt_v):
    """
    compute \min_R || VR - V* ||_2^2
    """
    u, s, vt = np.linalg.svd(estimate_v.T @ gt_v)
    R = u @ vt
    return np.sum((estimate_v @ R - gt_v)**2)

def compute_v_l2_torch(estimate_v, gt_v):
    """
    compute \min_R || VR - V* ||_2^2
    """
    estimate_v, gt_v = from_numpy(estimate_v), from_numpy(gt_v)
    u, s, v = torch.svd(estimate_v.mT @ gt_v)
    R = u @ v.mT
    return torch.sum((estimate_v @ R - gt_v)**2)

def eval_methods_random_matrix():
    n, d, k, n_groups, n_obs = 10000, 100, 20, 5, 2
    matrix, labels = load_random_matrix(n, d, k, n_groups)
    markers = ('o', 'v', 'X', 's', '^', '.', '<', '>')
    colors = ('r', 'g', 'b', 'y', 'm', 'b', 'c', 'w')

    factorize_matrix_and_visualize(matrix, labels, k, markers, colors)
    factorize_matrix_and_visualize(matrix, labels, k, markers, colors, approximate = True)
    factorize_matrix_and_visualize(matrix.T @ matrix, labels, k, markers, colors, take_sqrt = True)

    mask = sample_mask(n, d, n_obs)
    matrix_masked = matrix * mask

    factorize_matrix_and_visualize(matrix_masked, labels, k, markers, colors)
    factorize_matrix_and_visualize(matrix_masked.T @ matrix_masked, labels, k, markers, colors, take_sqrt = True)

    x_estimate = complete_matrix_iter(from_numpy(matrix), from_numpy(mask), k, from_numpy(matrix), iters = 100)
    factorize_matrix_and_visualize(x_estimate, labels, k, markers, colors)

    diagonal_mask = np.eye(d)
    diagonal_mask = -1 * (1 - diagonal_mask)
    factorize_matrix_and_visualize((matrix_masked.T @ matrix_masked) * diagonal_mask, labels, k, markers, colors, take_sqrt = True)

    xtx_estimate = estimate_sim_weighted(matrix, mask, k, lr=1e-2)
    factorize_matrix_and_visualize(xtx_estimate, labels, k, markers, colors, take_sqrt = True)

def eval_methods_1000genomes_large():
    demographics = load_1000genomes_demographics() # dict of {subj_id:[pop, superpop, sex]}
    matrix, ids = load_1000genomes_large() # (genome_length, n_people), (n_people,) array of subject ids
    excluded_ids = np.array([id not in demographics for id in ids])
    ids = np.array(ids)
    matrix, ids = matrix[:,excluded_ids==0], ids[excluded_ids==0]
    labels = np.array(['_'.join(demographics[id][1:]) for id in ids]) # superpop_sex

    np.save('../data/chr1_100000_matrix.npy', matrix)
    np.save('../data/chr1_100000_ids.npy', ids)
    np.save('../data/chr1_100000_labels.npy', labels)


    n, d = matrix.shape
    k, n_obs = 100, 100
    matrix = matrix - np.mean(matrix, axis = 0, keepdims = True)

    markers = {'AFR_male':'o', 'AMR_male':'o', 'EAS_male':'o', 'EUR_male':'o', 'SAS_male':'o', 'AFR_female':'X', 'AMR_female':'X', 'EAS_female':'X', 'EUR_female':'X', 'SAS_female':'X'}
    colors = {'AFR_male':'r', 'AMR_male':'g', 'EAS_male':'b', 'EUR_male':'y', 'SAS_male':'m', 'AFR_female':'r', 'AMR_female':'g', 'EAS_female':'b', 'EUR_female':'y', 'SAS_female':'m'}

    U, V = factorize_matrix_grad(matrix, lr = 1e-2, iters = 20)

    visualize_factors(V, labels, markers, colors)

    factorize_matrix_and_visualize(matrix, labels, k, markers, colors, approximate = True)
    factorize_matrix_and_visualize(matrix.T @ matrix, labels, k, markers, colors, take_sqrt = True, approximate = True)

    mask = sample_mask(n, d, n_obs)
    matrix_masked = matrix * mask

    factorize_matrix_and_visualize(matrix_masked, labels, k, markers, colors, approximate = True)
    factorize_matrix_and_visualize(matrix_masked.T @ matrix_masked, labels, k, markers, colors, take_sqrt = True, approximate = True)

    x_estimate = complete_matrix_iter(from_numpy(matrix), from_numpy(mask), k, from_numpy(matrix), iters = 500)
    factorize_matrix_and_visualize(x_estimate, labels, k, markers, colors, approximate = True)

    diagonal_mask = np.eye(d)
    diagonal_mask = -1 * (1 - diagonal_mask)
    factorize_matrix_and_visualize((matrix_masked.T @ matrix_masked) * diagonal_mask, labels, label_names, k, markers, colors, take_sqrt = True, approximate = True)

    xtx_estimate = estimate_sim_iter(matrix, mask, k, iters = 500)
    factorize_matrix_and_visualize(xtx_estimate, labels, k, markers, colors, take_sqrt = True, approximate = True)
