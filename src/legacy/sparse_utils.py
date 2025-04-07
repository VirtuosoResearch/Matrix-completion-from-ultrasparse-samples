import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy
from power_method_svd import *

def sparse_collate_fn(batch):
    """
    Custom collate function to handle sparse tensors.
    
    Args:
        batch: A list of sparse tensors.
        
    Returns:
        The batch of sparse tensors as-is.
    """
    return batch  # Return the list of sparse tensors as-is

class MatrixDataset(Dataset):
    def __init__(self, matrix, d1, d2):
        """
        Args:
            matrix (numpy array or tensor): The matrix to be split into chunks.
            d1 (int): Number of rows to extract per iteration.
        """
        """
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        if matrix.is_sparse:
            self.matrix = matrix
        else:
            self.matrix = matrix.to_sparse()
        """
        self.matrix = scipy_sparse_to_torch(matrix)
        if not self.matrix.is_coalesced():
            self.matrix = self.matrix.coalesce()
        self.d1 = d1
        self.d2 = d2

        self.total_rows = self.matrix.shape[0]
        self.total_cols = self.matrix.shape[1]

        self.num_row_blocks = (self.total_rows + d1 - 1) // d1
        self.num_col_blocks = (self.total_cols + d2 - 1) // d2
        self.total_blocks = self.num_row_blocks * self.num_col_blocks

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        """
        Return a sparse submatrix of size d1 x d2.
        """
        # Compute the block's row and column indices
        row_block_idx = idx // self.num_col_blocks
        col_block_idx = idx % self.num_col_blocks

        # Compute row and column ranges
        start_row = row_block_idx * self.d1
        end_row = min(start_row + self.d1, self.total_rows)

        start_col = col_block_idx * self.d2
        end_col = min(start_col + self.d2, self.total_cols)

        # Get the coalesced sparse tensor's indices and values
        sparse_indices = self.matrix._indices()  # Shape: (2, nnz)
        sparse_values = self.matrix._values()    # Shape: (nnz,)

        # Create a mask to filter indices within the block
        mask = (sparse_indices[0] >= start_row) & (sparse_indices[0] < end_row) & \
               (sparse_indices[1] >= start_col) & (sparse_indices[1] < end_col)

        # Filter indices and values
        filtered_indices = sparse_indices[:, mask].clone()
        filtered_values = sparse_values[mask].clone()

        # Adjust indices to be relative to the block
        filtered_indices[0] -= start_row
        filtered_indices[1] -= start_col

        # Create a new sparse tensor for this block
        block_size = (end_row - start_row, end_col - start_col)
        sparse_block = torch.sparse.FloatTensor(
            filtered_indices,
            filtered_values,
            torch.Size(block_size)
        )

        return sparse_block

class MatrixDataset_col(Dataset):
    def __init__(self, matrix, sub_d2):
        """
        Args:
            matrix (numpy array or tensor): The matrix to be split into chunks.
            d1 (int): Number of rows to extract per iteration.
        """
        """
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        if matrix.is_sparse:
            self.matrix = matrix
        else:
            self.matrix = matrix.to_sparse()
        """
        self.matrix = scipy_sparse_to_torch(matrix)
        if not self.matrix.is_coalesced():
            self.matrix = self.matrix.coalesce()
        self.d2 = sub_d2
        self.total_rows = self.matrix.shape[0]
        self.total_cols = self.matrix.shape[1]

    def __len__(self):
        # Calculate the number of full d1-sized chunks in the matrix
        return (self.total_cols + self.d2 - 1) // self.d2

    def __getitem__(self, idx):
        """
        Return a sparse slice of the matrix along columns.
        """
        start_idx = idx * self.d2
        end_idx = min(start_idx + self.d2, self.total_cols)

        # Get the coalesced sparse tensor's indices and values
        sparse_indices = self.matrix._indices()  # Shape: (2, nnz)
        sparse_values = self.matrix._values()    # Shape: (nnz,)

        # Mask where column indices are in [start_idx, end_idx)
        mask = (sparse_indices[1] >= start_idx) & (sparse_indices[1] < end_idx)

        # Filter indices and values
        filtered_indices = sparse_indices[:, mask].clone()
        filtered_values = sparse_values[mask].clone()

        # Adjust column indices to start from 0
        filtered_indices[1] -= start_idx

        # Create a new sparse tensor of size [D, end_idx - start_idx]
        sparse_batch = torch.sparse.FloatTensor(
            filtered_indices,
            filtered_values,
            torch.Size([self.matrix.shape[0], end_idx - start_idx])
        )

        return sparse_batch

class MatrixDataset_row(Dataset):
    def __init__(self, matrix, sub_d1):
        """
        Args:
            matrix (numpy array or tensor): The matrix to be split into chunks.
            d1 (int): Number of rows to extract per iteration.
        """
        """
        if isinstance(matrix, np.ndarray):
            matrix = torch.tensor(matrix, dtype=torch.float32)
        if matrix.is_sparse:
            self.matrix = matrix
        else:
            self.matrix = matrix.to_sparse()
        """
        self.matrix = scipy_sparse_to_torch(matrix)
        if not self.matrix.is_coalesced():
            self.matrix = self.matrix.coalesce()
        self.d1 = sub_d1
        self.total_rows = self.matrix.shape[0]
        self.total_cols = self.matrix.shape[1]

    def __len__(self):
        # Calculate the number of full d1-sized chunks in the matrix
        return (self.total_rows + self.d1 - 1) // self.d1

    def __getitem__(self, idx):
        """
        Return a sparse slice of the matrix.
        """
        start_idx = idx * self.d1
        end_idx = min(start_idx + self.d1, self.total_rows)
        
        # Get the sparse tensor's indices and values
        sparse_indices = self.matrix._indices()  # 2D tensor with shape (2, number_of_non_zero_elements)
        sparse_values = self.matrix._values()    # 1D tensor with shape (number_of_non_zero_elements)

        # Filter the indices that correspond to the rows in the range [start_idx, end_idx]
        mask = (sparse_indices[0] >= start_idx) & (sparse_indices[0] < end_idx)

        # Get the relevant indices and values
        filtered_indices = sparse_indices[:, mask].clone()
        filtered_values = sparse_values[mask].clone()

        # Adjust the row indices to be relative to the start_idx (to match the new matrix size)
        filtered_indices[0] -= start_idx

        # Create a new sparse matrix for this batch
        sparse_batch = torch.sparse.FloatTensor(
            filtered_indices,
            filtered_values,
            torch.Size([end_idx - start_idx, self.matrix.shape[1]])
        )
        
        return sparse_batch

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

def scipy_sparse_to_torch(scipy_sparse, device='cpu'):
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

def sparse_svds_for_tensor(matrix, k):
    if k > min(matrix.shape):
        k = min(matrix.shape)
    device = matrix.device
    matrix = matrix.cpu().to_sparse()
    #matrix = matrix.cpu()
    matrix_scipy = torch_sparse_to_scipy(matrix)
    U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(matrix_scipy, k=k)
    U = torch.from_numpy(U_scipy[::-1].copy()).to(device)
    D = torch.from_numpy(D_scipy[::-1].copy()).to(device)
    Vt = torch.from_numpy(Vt_scipy[::-1].copy()).to(device)
    return U, D, Vt

def nonzero_ratio(M):
    return (torch.count_nonzero(M)/M.numel()).item()

def load_sparse_data_syn(r=5, d1=5000, d2=2000, num_elements=1000, device='cpu'):
    # Randomly select indices for the non-zero elements
    indices = torch.randint(0, min(d1, d2), size=(2, num_elements))  # 2 rows (for row and column indices)
    
    # Generate random values for the selected indices
    values = torch.normal(2, 1, (num_elements,))  # Random values for the sparse matrix
    
    # Create the sparse matrix with the given shape (d1, d2)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (d1, d2))

    matrix_scipy = torch_sparse_to_scipy(sparse_matrix)
    U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(matrix_scipy, k=r)
    U = torch.from_numpy(U_scipy[::-1].copy()).to(device).to_sparse()
    D = torch.from_numpy(D_scipy[::-1].copy()).to(device)
    D = torch.diag(D).to_sparse()
    Vt = torch.from_numpy(Vt_scipy[::-1].copy()).to(device).to_sparse()
    
    return U @ D @ Vt

def load_sparse_data_syn_2(r=5, d1=5000, d2=2000, num_elements=1000, device='cpu'):
    X = torch.normal(2, 1, size = (d1, d2)).to(device)

    U, D, Vt = torch.linalg.svd(X, full_matrices=False)
    D[r:] = 0
    X = U @ torch.diag(D) @ Vt
    matrix = U @ torch.diag(D) @ Vt
    indices = torch.randperm(matrix.numel())[:num_elements]

    # Create a mask with all zeros
    mask = torch.zeros_like(matrix).view(-1)

    # Set chosen locations in the mask to 1
    mask[indices] = 1

    # Reshape mask back to the matrix size
    mask = mask.view(d1, d2)

    # Apply mask to the matrix
    sparse_matrix = matrix * mask
    sparse_matrix = sparse_matrix.to_sparse()
    
    return sparse_matrix

def get_sparse_masks(M, p):
    original_indices = M.indices()
    original_values = M.values()
    total_nonzeros = original_values.size(0)
    num_samples = int(total_nonzeros * p)
    permuted_indices = torch.randperm(total_nonzeros)
    sampled_indices = permuted_indices[:num_samples]

    sampled_coords = original_indices[:, sampled_indices]  # Shape: (2, num_samples)
    sampled_values = original_values[sampled_indices]     # Shape: (num_samples,)

    observed_M = torch.sparse_coo_tensor(sampled_coords, sampled_values, M.size(), device=M.device)
    observed_M = observed_M.coalesce()  # Ensure the indices are coalesced

    # Create the mask sparse matrix (with 1s at sampled positions)
    mask_values = torch.ones(num_samples, dtype=original_values.dtype, device=M.device)
    mask = torch.sparse_coo_tensor(sampled_coords, mask_values, M.size(), device=M.device)
    mask = mask.coalesce()

    return observed_M, mask