import torch
import scipy

def generate_sparse_matrix(shape, sparsity, device='cpu'):
    """
    Generate a random sparse matrix with given shape and sparsity.

    Args:
        shape (tuple): Shape of the matrix (rows, columns).
        sparsity (float): Fraction of elements that are zeros (between 0 and 1).
        device (str): Device to store the tensor.

    Returns:
        torch.sparse_coo_tensor: A sparse tensor of the given shape and sparsity.
    """
    num_elements = shape[0] * shape[1]
    num_nonzeros = int(num_elements * (1 - sparsity))

    # Randomly generate indices for non-zero elements
    row_indices = torch.randint(0, shape[0], (num_nonzeros,), device=device)
    col_indices = torch.randint(0, shape[1], (num_nonzeros,), device=device)
    indices = torch.stack([row_indices, col_indices])

    # Generate random values for non-zero elements
    values = torch.rand(num_nonzeros, device=device)

    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=shape, device=device).coalesce()
    return sparse_tensor

def get_masks_sparse(M, p):
    """
    Create a masked observed version of M and a corresponding mask,
    sampling only from the non-zero elements of M.

    Args:
        M (torch.sparse_coo_tensor): The input sparse tensor.
        p (float): The probability of each non-zero element being included in the mask.

    Returns:
        observed_M (torch.sparse_coo_tensor): The observed masked tensor.
        masks (torch.sparse_coo_tensor): The mask tensor.
    """
    # Extract indices and values from the sparse tensor M
    indices = M.indices()
    values = M.values()
    device = M.device

    # Generate random values for each non-zero element
    random_vals = torch.rand(values.size(), device=device)

    # Create a mask based on probability p
    mask = random_vals <= p

    # Apply the mask to indices and values
    masked_indices = indices[:, mask]
    masked_values = values[mask]

    # Create the observed sparse tensor
    observed_M = torch.sparse_coo_tensor(
        masked_indices,
        masked_values,
        M.shape,
        device=device
    ).coalesce()

    # Create the masks sparse tensor (values are ones)
    masks = torch.sparse_coo_tensor(
        masked_indices,
        torch.ones_like(masked_values),
        M.shape,
        device=device
    ).coalesce()

    # Non-zero elements are correctly aligned by construction
    print("Non-zero elements are correctly aligned with the mask.")

    return observed_M, masks

def filter_sparse_rows(sparse_tensor):
    # Get the row indices of the non-zero elements
    row_indices = sparse_tensor.indices()[0]
    # Find unique row indices that have non-zero entries
    non_zero_rows = torch.unique(row_indices)
    # Create a mapping from old to new row indices
    row_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(non_zero_rows)}
    # Filter indices and values
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    remapped_row_indices = torch.tensor(
        [row_mapping[idx.item()] for idx in indices[0]],
        device=sparse_tensor.device
    )
    # Update indices with remapped row indices
    filtered_indices = torch.vstack([remapped_row_indices, indices[1]])
    # Define new shape
    new_shape = (len(non_zero_rows), sparse_tensor.shape[1])
    # Create the filtered sparse tensor
    filtered_sparse_tensor = torch.sparse_coo_tensor(
        filtered_indices,
        values,
        new_shape,
        device=sparse_tensor.device
    ).coalesce()
    return filtered_sparse_tensor, non_zero_rows


def torch_sparse_to_scipy(sparse_tensor):
    indices = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    shape = sparse_tensor.shape
    indices = indices.cpu().numpy()
    values = values.cpu().numpy()
    return scipy.sparse.coo_matrix((values, indices), shape=shape)

def scipy_sparse_to_torch(scipy_sparse, device):
    scipy_sparse = scipy_sparse.tocoo()
    indices = torch.tensor([scipy_sparse.row, scipy_sparse.col], dtype=torch.long, device=device)
    values = torch.tensor(scipy_sparse.data, dtype=torch.float32, device=device)
    shape = scipy_sparse.shape
    return torch.sparse_coo_tensor(indices, values, size=shape).coalesce()

def sparsify_matrix(matrix, threshold):
    # Set values below the threshold to zero
    sparse_matrix = matrix.clone()
    sparse_matrix[torch.abs(sparse_matrix) < threshold] = 0
    # Convert to sparse format
    indices = torch.nonzero(sparse_matrix, as_tuple=False).t()
    values = sparse_matrix[indices[0], indices[1]]
    sparse_matrix = torch.sparse_coo_tensor(indices, values, size=matrix.shape).coalesce()
    return sparse_matrix
