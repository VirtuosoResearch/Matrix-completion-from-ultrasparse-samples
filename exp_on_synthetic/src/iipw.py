import torch
from tqdm import tqdm

def empirical_reweight_estimate(observed_M, masks=None, return_counts=False):
    if masks is None:
        masks = observed_M != 0

    mask_values = masks.to(device=observed_M.device, dtype=observed_M.dtype)
    second_moment_observe_M = observed_M.T @ observed_M
    second_moment_observe_count = mask_values.T @ mask_values
    T = torch.where(
        second_moment_observe_count > 0,
        second_moment_observe_M / second_moment_observe_count.clamp_min(1),
        torch.zeros_like(second_moment_observe_M),
    )

    if return_counts:
        return T, second_moment_observe_count
    return T

def probability_reweight_estimate(observed_M, p):
    n, d = observed_M.shape
    second_moment_observe_M =  observed_M.T @ observed_M
    diag = torch.diag( torch.diag(second_moment_observe_M) )
    T_p = ((1.0 / p) * diag + (1.0 / (p**2)) * (second_moment_observe_M - diag))
    T_p = T_p / n

    return T_p


class IIPW:
    def __init__(self, M, observed_M, masks, p, r):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if M.shape != observed_M.shape or M.shape != masks.shape:
            raise ValueError("M, observed_M, and masks must have the same shape")

        # restore parameters
        self.n, self.d = M.shape
        self.p = p
        self.r = r
        self.iter_num = 0

        # process matrix
        self.M = M
        self.observed_M = observed_M
        self.masks = masks
        self.T, self.observation_counts = empirical_reweight_estimate(
            observed_M, masks, return_counts=True
        )
        self.normalized_MTM = M.T @ M / self.n

    def impute(self, n_iter=100, lr=0.1, tol=1e-7, alpha=0, lam=1):
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")

        def rank_r_svd_factor(X):
            left_vectors, singular_values, _ = torch.linalg.svd(
                X, full_matrices=False
            )
            return (
                left_vectors[:, :self.r]
                * singular_values[:self.r].sqrt().unsqueeze(0)
            )

        def regularizer(X, alpha):
            norms = torch.norm(X, dim=1)
            return torch.clamp(norms - alpha, min=0).pow(4).sum()

        def grad_R(X, alpha):
            norms = torch.norm(X, dim=1, keepdim=True)  # shape: (d, 1)
            mask = (norms >= alpha).float()
            coeffs = 4 * ((norms - alpha) ** 3) * mask  # shape: (d, 1)
            grad = coeffs * X / norms.clamp(min=1e-8)   # avoid divide-by-zero
            return grad

        def observed_fit(X):
            residual = X @ X.T - T
            return (residual.square() * T_masks).sum() / (2 * m)

        def objective(X):
            return observed_fit(X) + lam * regularizer(X, alpha) / self.d

        # Initialize parameters
        self.iter_num = n_iter
        T = self.T
        T_masks = (self.observation_counts > 0).to(dtype=T.dtype)

        # Diagonal moments are observed more often than off-diagonal moments.
        diag_values = 1 - (1 - self.p**2)**self.n
        diag_idx = torch.arange(self.d, device=T.device)
        T_masks[diag_idx, diag_idx] *= diag_values
        m = T_masks.sum().item()

        if m == 0:
            U = torch.zeros(self.d, self.r, device=T.device, dtype=T.dtype)
            estimation_matrix = U @ U.T
            error = torch.norm(self.normalized_MTM - estimation_matrix, p='fro')
            self.iter_num = 0
            return U, estimation_matrix, error.item()

        U = rank_r_svd_factor(T)
        if observed_fit(U).item() <= tol:
            estimation_matrix = U @ U.T
            error = torch.norm(self.normalized_MTM - estimation_matrix, p='fro')
            self.iter_num = 0
            return U, estimation_matrix, error.item()

        print('Imputing...')
        previous_loss = objective(U).item()
        loop = tqdm(range(n_iter))
        for i in loop:
            # Compute the gradient
            residual = (U @ U.T - T) * T_masks
            grad = (2/m) * (residual @ U)
            reg_grad = grad_R(U, alpha) / self.d

            # Update
            previous_U = U
            U = U - lr * (grad + lam*reg_grad)
            observed_loss = objective(U)

            if not torch.isfinite(observed_loss) or observed_loss.item() > previous_loss:
                U = previous_U
                break
            if i > 10 and abs(previous_loss - observed_loss.item()) < tol:
                break
            previous_loss = observed_loss.item()

            loop.set_description(f"Observed loss: {observed_loss:.7f}")

        X_imp = U @ U.T
        relative_err = torch.norm(self.normalized_MTM - X_imp, p='fro')
        estimation_matrix = X_imp

        self.iter_num = i+1
        return U, estimation_matrix, relative_err.item()
