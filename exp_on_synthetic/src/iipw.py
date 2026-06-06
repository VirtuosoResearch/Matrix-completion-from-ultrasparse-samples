import numpy as np
import torch
from tqdm import tqdm


def _validate_fixed_row_mixture_support(masks, num_components):
    row_counts = masks.sum(dim=1)
    if (
        row_counts.numel() == 0
        or int(row_counts[0].item()) < 2
        or not torch.all(row_counts == row_counts[0])
    ):
        return

    n, d = masks.shape
    observations_per_row = int(row_counts[0].item())
    observations_per_coordinate_component = (
        n * observations_per_row / (d * num_components)
    )
    component_link_degree = (
        n
        * observations_per_row
        * (observations_per_row - 1)
        / (d * num_components)
    )
    required_link_degree = 0.75 * np.log(max(d, 2))
    if (
        observations_per_coordinate_component >= 5
        and component_link_degree >= required_link_degree
    ):
        return

    required_observations = observations_per_row
    while (
        n * required_observations / (d * num_components) < 5
        or n
        * required_observations
        * (required_observations - 1)
        / (d * num_components)
        < required_link_degree
    ):
        required_observations += 1

    max_components = max(
        1,
        int(
            min(
                n * observations_per_row / (5 * d),
                n
                * observations_per_row
                * (observations_per_row - 1)
                / (d * required_link_degree),
            )
        ),
    )
    raise ValueError(
        "fixed-row mixture-hajek is undersampled for "
        f"{num_components} components: only "
        f"{observations_per_coordinate_component:.2f} observations per "
        "coordinate/component and component-link degree "
        f"{component_link_degree:.2f}. Increase observations per row to at "
        f"least {required_observations}, reduce num_components to at most "
        f"{max_components}, or increase the number of rows."
    )


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


def _fixed_row_mixture_labels(observed_M, masks, num_components):
    row_counts = masks.sum(dim=1)
    if (
        row_counts.numel() == 0
        or int(row_counts[0].item()) < 2
        or not torch.all(row_counts == row_counts[0])
    ):
        return None

    values = observed_M.detach().cpu().numpy()
    observed = masks.detach().cpu().numpy().astype(bool)
    n, d = values.shape
    local_labels = np.full((n, d), -1, dtype=np.int16)
    reliable = np.zeros(d, dtype=bool)

    # Recover the unordered scalar mixture means at each coordinate.
    for coordinate in range(d):
        rows = np.flatnonzero(observed[:, coordinate])
        coordinate_values = values[rows, coordinate]
        if coordinate_values.size < num_components:
            continue

        order = np.argsort(coordinate_values)
        sorted_values = coordinate_values[order]
        gaps = np.diff(sorted_values)
        cuts = np.sort(np.argsort(gaps)[-(num_components - 1):] + 1)
        groups = np.split(np.arange(coordinate_values.size), cuts)
        if any(group.size == 0 for group in groups):
            continue

        centers = []
        squared_residual = 0.0
        for component, group in enumerate(groups):
            original_positions = order[group]
            group_values = coordinate_values[original_positions]
            center = group_values.mean()
            centers.append(center)
            squared_residual += np.square(group_values - center).sum()
            local_labels[rows[original_positions], coordinate] = component

        noise_scale = np.sqrt(squared_residual / coordinate_values.size)
        center_gap = np.diff(np.asarray(centers)).min()
        numerical_scale = np.finfo(values.dtype).eps * max(
            1.0, np.max(np.abs(coordinate_values))
        )
        reliable[coordinate] = center_gap > 10 * max(
            noise_scale, numerical_scale
        )

    # A row links the local component labels of its observed coordinates.
    parent = np.arange(d * num_components)
    sizes = np.ones(d * num_components, dtype=np.int64)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first, second):
        first = find(first)
        second = find(second)
        if first == second:
            return
        if sizes[first] < sizes[second]:
            first, second = second, first
        parent[second] = first
        sizes[first] += sizes[second]

    for row in range(n):
        coordinates = np.flatnonzero(observed[row] & reliable)
        if coordinates.size < 2:
            continue
        nodes = coordinates * num_components + local_labels[row, coordinates]
        for node in nodes[1:]:
            union(nodes[0], node)

    roots = np.asarray([find(node) for node in range(d * num_components)])
    unique_roots, component_sizes = np.unique(roots, return_counts=True)
    if unique_roots.size < num_components:
        return None
    largest_indices = np.argsort(component_sizes)[-num_components:]
    if np.any(component_sizes[largest_indices] <= 1):
        return None
    largest = unique_roots[largest_indices]
    global_labels = np.full(d * num_components, -1, dtype=np.int64)
    for component, root in enumerate(largest):
        global_labels[roots == root] = component

    labels = np.full(n, -1, dtype=np.int64)
    for row in range(n):
        coordinates = np.flatnonzero(observed[row])
        coordinates = coordinates[local_labels[row, coordinates] >= 0]
        if coordinates.size == 0:
            continue
        nodes = coordinates * num_components + local_labels[row, coordinates]
        row_labels = global_labels[nodes]
        row_labels = row_labels[row_labels >= 0]
        if row_labels.size and np.all(row_labels == row_labels[0]):
            labels[row] = row_labels[0]

    missing = np.flatnonzero(labels < 0)
    labels[missing] = np.arange(missing.size) % num_components
    return torch.as_tensor(labels, device=observed_M.device)


def mixture_hajek_estimate(observed_M, masks, num_components=2, restarts=5, max_iter=100):
    if num_components < 2:
        raise ValueError("num_components must be at least 2")

    _validate_fixed_row_mixture_support(masks, num_components)

    device = observed_M.device
    mask_values = masks.to(device=device, dtype=observed_M.dtype)
    row_counts = mask_values.sum(dim=1)
    valid_rows = row_counts > 0
    valid_count = int(valid_rows.sum().item())
    if valid_count < num_components:
        raise ValueError("not enough observed rows to estimate mixture components")

    initial_labels = _fixed_row_mixture_labels(
        observed_M, masks, num_components
    )
    initializations = [initial_labels] if initial_labels is not None else []
    initializations.extend([None] * restarts)

    best = None
    for restart, initialization in enumerate(initializations):
        labels = torch.full(
            (observed_M.shape[0],), -1, dtype=torch.long, device=device
        )
        if initialization is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(restart)
            labels[valid_rows] = (
                torch.randperm(valid_count, generator=generator, device=device)
                % num_components
            )
        else:
            labels[valid_rows] = initialization[valid_rows]

        for iteration in range(max_iter):
            component_means = []
            component_counts = []
            for component in range(num_components):
                selected = (labels == component).to(observed_M.dtype).unsqueeze(1)
                counts = (selected * mask_values).sum(dim=0)
                means = (selected * observed_M).sum(dim=0) / counts.clamp_min(1)
                component_means.append(means)
                component_counts.append(counts)

            component_means = torch.stack(component_means)
            component_counts = torch.stack(component_counts)
            losses = torch.stack(
                [
                    (
                        (observed_M - component_means[component]).square()
                        * mask_values
                    ).sum(dim=1)
                    / row_counts.clamp_min(1)
                    for component in range(num_components)
                ],
                dim=1,
            )
            updated_labels = losses.argmin(dim=1)
            updated_labels[~valid_rows] = -1
            if torch.equal(updated_labels, labels):
                break
            labels = updated_labels

        component_sizes = torch.stack(
            [(labels == component).sum() for component in range(num_components)]
        )
        if torch.any(component_sizes == 0):
            continue
        clustering_loss = (
            losses[valid_rows]
            .gather(1, labels[valid_rows].unsqueeze(1))
            .mean()
            .item()
        )
        if best is None or clustering_loss < best[0]:
            best = (
                clustering_loss,
                labels.clone(),
                component_means.clone(),
                component_counts.clone(),
                component_sizes.clone(),
            )

    if best is None:
        raise RuntimeError("masked mixture clustering produced an empty component")

    _, labels, component_means, component_counts, component_sizes = best
    mixture_weights = component_sizes.to(observed_M.dtype) / component_sizes.sum()
    T = (component_means.T * mixture_weights) @ component_means

    pooled_T, pooled_counts = empirical_reweight_estimate(
        observed_M, masks, return_counts=True
    )
    diagonal = torch.arange(T.shape[0], device=device)
    T[diagonal, diagonal] = pooled_T[diagonal, diagonal]

    coordinate_support = torch.all(component_counts > 0, dim=0)
    pair_support = coordinate_support.unsqueeze(1) & coordinate_support.unsqueeze(0)
    observation_counts = pair_support.to(observed_M.dtype)
    observation_counts[diagonal, diagonal] = pooled_counts[diagonal, diagonal]

    return (
        T,
        observation_counts,
        labels,
        component_means,
        mixture_weights,
    )


def probability_reweight_estimate(observed_M, p):
    n, d = observed_M.shape
    second_moment_observe_M =  observed_M.T @ observed_M
    diag = torch.diag( torch.diag(second_moment_observe_M) )
    T_p = ((1.0 / p) * diag + (1.0 / (p**2)) * (second_moment_observe_M - diag))
    T_p = T_p / n

    return T_p


class IIPW:
    def __init__(self, M, observed_M, masks, p, r, estimator="hajek",mixture_restarts=10):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if M.shape != observed_M.shape or M.shape != masks.shape:
            raise ValueError("M, observed_M, and masks must have the same shape")

        # restore parameters
        self.n, self.d = M.shape
        self.p = p
        self.r = r
        self.iter_num = 0
        self.estimator = estimator

        # process matrix
        self.M = M
        self.observed_M = observed_M
        self.masks = masks

        nonzero_count = torch.count_nonzero(self.observed_M.T @ self.observed_M).item()
        total_count = self.d * self.d
        sparsity = 1 - nonzero_count / total_count
        print(
            f"T sparsity: {sparsity:.7f} "
            f"({nonzero_count}/{total_count} nonzero entries)"
        )

        if estimator == "hajek":
            self.T, self.observation_counts = empirical_reweight_estimate(
                observed_M, masks, return_counts=True
            )
        elif estimator == "mixture-hajek":
            (
                self.T,
                self.observation_counts,
                self.mixture_labels,
                self.component_means,
                self.mixture_weights,
            ) = mixture_hajek_estimate(
                observed_M,
                masks,
                num_components=r,
                restarts=mixture_restarts,
            )
            print(f"Estimated mixture weights: {self.mixture_weights.tolist()}")
        else:
            raise ValueError(f"unknown estimator: {estimator}")

        self.normalized_MTM = M.T @ M / self.n
        nonzero_mask = self.T != 0
        target_norm = torch.norm(self.normalized_MTM[nonzero_mask], p='fro')
        if nonzero_count == 0 or target_norm.item() == 0:
            relative_error = float('nan')
        else:
            relative_error = (
                torch.norm(
                    (self.T - self.normalized_MTM)[nonzero_mask], p='fro'
                )
                / target_norm
            ).item()
        print(f"T relative error on nonzero entries: {relative_error:.7f}")

    def impute(
        self,
        n_iter=100,
        lr=0.1,
        tol=1e-14,
        alpha=0,
        lam=0.01,
        max_backtracking=30,
    ):
        if n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {n_iter}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")

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

        def observed_relative_error(X):
            residual_norm = torch.sqrt(
                ((X @ X.T - T).square() * T_masks).sum()
            )
            target_norm = torch.sqrt((T.square() * T_masks).sum())
            return residual_norm / target_norm.clamp_min(
                torch.finfo(T.dtype).tiny
            )

        def objective(X):
            return observed_fit(X) + lam * regularizer(X, alpha) / self.d

        # Initialize parameters
        self.iter_num = n_iter
        T = self.T
        T_masks = (self.observation_counts > 0).to(dtype=T.dtype)

        # Diagonal moments are observed more often than off-diagonal moments.
        diag_values = 1 - (1 - self.p)**self.n
        diag_idx = torch.arange(self.d, device=T.device)
        T_masks[diag_idx, diag_idx] *= diag_values
        m = T_masks.sum().item()

        if m == 0:
            U = torch.zeros(self.d, self.r, device=T.device, dtype=T.dtype)
            estimation_matrix = U @ U.T
            error = torch.norm(self.normalized_MTM - estimation_matrix, p='fro')
            self.iter_num = 0
            self.loss_history = []
            self.learning_rate = lr
            return U, estimation_matrix, error.item()

        U = rank_r_svd_factor(T)
        numerical_tol = 10 * torch.finfo(T.dtype).eps
        if observed_relative_error(U).item() <= max(tol, numerical_tol):
            estimation_matrix = U @ U.T
            error = torch.norm(self.normalized_MTM - estimation_matrix, p='fro')
            self.iter_num = 0
            self.loss_history = [objective(U).item()]
            self.learning_rate = lr
            return U, estimation_matrix, error.item()

        print('Imputing...')
        previous_loss = objective(U).item()
        loss_history = [previous_loss]
        step_size = lr
        loop = tqdm(range(n_iter))
        for i in loop:
            # Compute the gradient
            residual = (U @ U.T - T) * T_masks
            grad = (2/m) * (residual @ U)
            reg_grad = grad_R(U, alpha) / self.d
            direction = grad + lam * reg_grad

            accepted = False
            candidate_step = step_size
            for _ in range(max_backtracking):
                candidate_U = U - candidate_step * direction
                candidate_loss = objective(candidate_U)
                if (
                    torch.isfinite(candidate_loss)
                    and candidate_loss.item() <= previous_loss
                ):
                    accepted = True
                    break
                candidate_step *= 0.5

            if not accepted:
                break

            U = candidate_U
            current_loss = candidate_loss.item()
            loss_history.append(current_loss)
            step_size = candidate_step

            relative_change = abs(previous_loss - current_loss) / max(
                abs(previous_loss), torch.finfo(T.dtype).tiny
            )
            if i > 10 and relative_change < tol:
                previous_loss = current_loss
                break
            previous_loss = current_loss

            loop.set_description(
                f"Observed loss: {current_loss:.7f}, lr: {step_size:.3g}"
            )

        X_imp = U @ U.T
        relative_err = torch.norm(self.normalized_MTM - X_imp, p='fro')
        estimation_matrix = X_imp

        self.iter_num = i+1
        self.loss_history = loss_history
        self.learning_rate = step_size
        return U, estimation_matrix, relative_err.item()
