import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

from utils import *
from data import *
from data_utils import *
from power_method_svd import power_svd
from sparse_power_method_svd import power_svd_sparse
from sparse_utils import *

device = 'cuda:0'

dataset = 'ml-32m'
p=0.1
r=10
M_all = load_data_all(dataset)
print(M_all.shape)
M_dataset = MatrixDataset(M_all, 44000, 3000)
matrix_dataloader = DataLoader(
    M_dataset,
    batch_size=1,         # You can adjust the batch size as needed
    shuffle=False,
    collate_fn=sparse_collate_fn  # Pass the custom collate function
)

use_dense = False

if use_dense:
    for batch in matrix_dataloader:
        M = batch[0].to_dense()
        M = M.to(device)
        d1, d2 = M.shape
        observed_M, masks = get_masks(M, p)

        #M = torch.from_numpy(M).to(device).float()
        #observed_M = torch.from_numpy(observed_M).to(device).float()
        #masks = torch.from_numpy(masks).to(device).float()

        non_zero_rows = torch.any(observed_M != 0, dim=1)

        observed_M = observed_M[non_zero_rows]
        masks = masks[non_zero_rows]
        M = M[non_zero_rows]
        print(M.shape)

        cov_observe_M =  observed_M.T @ observed_M
        cov_observe_count = (observed_M == 0).float().t() @ (observed_M == 0).float()
        diag_cov = torch.diag( torch.diag(cov_observe_M) )

        torch.count_nonzero(cov_observe_M).item()

        cov_observe_count = (1 * (observed_M != 0)).float().T @ (1 * (observed_M != 0).float())
        cov_observe_count = cov_observe_count + (cov_observe_count == 0) * 1
        T = cov_observe_M / (cov_observe_count/d1)
        MTM = M.T @ M
        print("nonzero: ", torch.count_nonzero(T)/T.numel())
        print("nonzero: ", torch.count_nonzero(M)/M.numel())
        T_masks = 1 * (T!=0)
        print(T_masks.min())

        # impute missing values from rank-r SVD corresponding to masks
        use_power_method = True
        train_losses = []
        err_estimates = []

        epochs = 30
        tol = 1e-5
        lr = 0.1
        X = T
        #T_masks = 1 * (T != 0)
        print(T_masks.sum())
        loop = tqdm(range(epochs))
        for i in loop:
            if not use_power_method:
                U, D, Vt = torch.linalg.svd(X)
                D[r:] = 0
            else:
                U, D, Vt = power_svd(X, k=r)
            X_update = U @ torch.diag(D) @ Vt

            #X = X * T_masks + X_update * (1 - T_masks)
            X = X * (1-lr) + X_update * lr
            err = MTM - X
            loss = (err**2).mean()
            train_losses.append(loss.item())
            relative_err = torch.norm(err, 'fro') / torch.norm(MTM, 'fro')
            #if i > 10:
            #    if relative_err - last_err < tol:
            #        break
            last_err = relative_err
            loop.set_description(f"relative err: {relative_err:.7f}")
            err_estimates.append(relative_err.item())
            #print(relative_err)

        plt.figure(figsize=(5, 3))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()

        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Err Estimation')
        plt.xlabel('Epoch')
        plt.ylabel('Err')
        plt.title('Err Estimation')
        plt.legend()

        error_list = []
        missing_num = 0
        total_num = 0
        skip = 0
        result_list_missing = []
        result_list_total = []
        use_reg = True
        U = []

        #estimation_goal = torch.rand(X.shape).to(device)
        estimation_goal = X

        #_, S, V = top_r_svd(estimation_goal, r)
        _, _, V = power_svd(estimation_goal, k=r)
        for i in tqdm(range(M.shape[0])):
            """
            M_row: 1*d2
            V: r*d2
            user_vector: 1*r
            """

            # make sure there is non-zero value in original matrix
            M_row = M[i]
            non_zero_indices = M_row.nonzero(as_tuple=True)[0]

            #non_zero_indices = torch.tensor(range(M_row.shape[0]))

            # select non-zero & masked elements as training data, non-zero & non-masked elements as testing data
            """
            observed_idx = []
            missing_idx = []
            for non_zero_idx in non_zero_indices:
                if masks[i][non_zero_idx]:
                    observed_idx.append(non_zero_idx.item())
                else:
                    missing_idx.append(non_zero_idx.item())
            """

            # Get the mask values for the non-zero indices
            mask_values = masks[i][non_zero_indices]

            # Use boolean indexing to separate observed and missing indices
            observed_idx = non_zero_indices[mask_values]
            missing_idx = non_zero_indices[~mask_values]

            M_row = M_row.unsqueeze(0)
            n = non_zero_indices.shape[0]

            #if len(observed_idx)==0 or len(missing_idx) == 0:
            #    skip +=1
            #    continue
            
            missing_num += len(missing_idx)
            total_num += len(missing_idx) + len(observed_idx)

            observed_idx = torch.tensor(observed_idx)
            missing_idx = torch.tensor(missing_idx)

            # formula AX=B
            observed_A = V[:, observed_idx].t()
            observed_B = M_row[:, observed_idx].t()
            missing_A = V[:, missing_idx].t()
            missing_B = M_row[:, missing_idx].t()

            if use_reg:
                # Ridge regression
                lambda_reg = 0.001
                I = torch.eye(observed_A.shape[1], device=device) * lambda_reg
                u = torch.linalg.lstsq(observed_A.t() @ observed_A + I, observed_A.t() @ observed_B).solution
            else:
                # Linear regression
                u = torch.linalg.lstsq(observed_A, observed_B).solution   
            U.append(u.t())

            check_vec = u
            if torch.any(torch.isnan(check_vec)):
                print("NAN!")
                print(check_vec)
                break

            # stat |AX-B| by individual elements
            if not (len(observed_idx)==0 or len(missing_idx)) == 0:
                error_list.append(torch.sum((missing_A @ u - missing_B)**2).item())

        # stat |AX-B| by the estimationed matrix
        """
        U = torch.concat(U, dim=0)
        M_ret = U @ V
        err = torch.norm((M-M_ret)*masks, p=2) / np.sqrt(missing_num)
        print(err)
        """
        # stat |AX-B| by individual elements
        ret = np.sqrt(np.sum(error_list)/missing_num)
        result_list_missing.append(np.sqrt(np.sum(error_list)/missing_num))
        #result_list_total.append(np.sqrt(np.sum(error_list))/total_num)
        print(np.mean(result_list_missing))
        print(np.std(result_list_missing))
        #print(np.mean(result_list_total))
        #print(np.std(result_list_total))
else:
    for batch in matrix_dataloader:
        M = batch[0]
        M = M.to(device)
        M = M.coalesce()
        #print(M)
        d1, d2 = M.shape
        print(M.shape)
        
        # Filter out rows with all zeros
        observed_M, masks = get_masks_sparse(M, p)

        
        #observed_M, non_zero_rows = filter_sparse_rows(observed_M)
        #masks, _ = filter_sparse_rows(masks)
        #M, _ = filter_sparse_rows(M)
        
        print("Computing T...")
        # Create indicator tensors
        indicator = torch.sparse_coo_tensor(
            observed_M.indices(),
            torch.ones_like(observed_M.values()),
            observed_M.shape,
            device=observed_M.device
        ).coalesce()
        indicator_T = indicator.transpose(0, 1)

        # Compute cov_observe_M and cov_observe_count
        cov_observe_M = torch.sparse.mm(observed_M.transpose(0, 1), observed_M)
        cov_observe_count = torch.sparse.mm(indicator_T, indicator)

        # Adjust counts to prevent division by zero
        epsilon = 1e-8
        count_indices = cov_observe_count.indices()
        count_values = cov_observe_count.values() + epsilon
        cov_observe_count = torch.sparse_coo_tensor(
            count_indices,
            count_values,
            cov_observe_count.shape,
            device=cov_observe_count.device
        ).coalesce()

        # Compute T
        # ... (code as defined above for computing T)

        print("Computing MTM...")
        # Compute MTM
        MTM = torch.sparse.mm(M.transpose(0, 1), M)

        print('Computing Cov...')
        # Get indices and values from cov_observe_M
        cov_indices = cov_observe_M.indices()
        cov_values = cov_observe_M.values()

        # Create a dictionary for counts for easy lookup
        count_dict = {
            (i.item(), j.item()): val.item() for i, j, val in zip(
                cov_observe_count.indices()[0],
                cov_observe_count.indices()[1],
                cov_observe_count.values()
            )
        }

        print("Computing T...")
        # Compute T values
        T_values = []
        for idx, val in zip(cov_indices.t(), cov_values):
            i, j = idx[0].item(), idx[1].item()
            count = count_dict.get((i, j), epsilon)  # Default to epsilon if count is missing
            T_values.append(val / (count / d1))

        T_values = torch.tensor(T_values, device=cov_observe_M.device)

        # Create sparse T tensor
        T = torch.sparse_coo_tensor(
            cov_indices,
            T_values,
            cov_observe_M.shape,
            device=cov_observe_M.device
        ).coalesce()

        # Create T_masks
        T_masks = torch.sparse_coo_tensor(
            T.indices(),
            torch.ones_like(T.values()),
            T.shape,
            device=T.device
        ).coalesce()

        # Print minimum value of T_masks
        if T_masks._nnz() > 0:
            T_masks_min = T_masks.values().min()
        else:
            T_masks_min = 0
        print(T_masks_min.item())

        print("Prepare estimation...")
        # Assume T is a sparse tensor of shape (d2, d2)
        T = T.coalesce()
        indices = T.indices()  # Shape: (2, nnz)
        values = T.values()    # Shape: (nnz,)
        X = T.clone()

        epochs = 100
        lr = 0.1
        tol = 1e-5
        train_losses = []
        err_estimates = []

        loop = tqdm(range(epochs))
        for epoch in loop:
            # Convert X to scipy sparse matrix
            X_scipy = torch_sparse_to_scipy(X)
            
            # Perform sparse SVD using scipy
            U_scipy, D_scipy, Vt_scipy = scipy.sparse.linalg.svds(X_scipy, k=r)
            U = torch.from_numpy(U_scipy.copy()).to(device)
            D = torch.from_numpy(D_scipy.copy()).to(device)
            Vt = torch.from_numpy(Vt_scipy.copy()).to(device)

            # Compute U_D
            U_D = U * D.unsqueeze(0)  # Shape: (n_rows, r)
            
            # Get the indices of non-zero elements in X
            I = X.indices()[0]
            J = X.indices()[1]

            # Extract the rows and columns corresponding to non-zero entries
            U_D_rows = U_D[I, :]        # Shape: (nnz, r)
            Vt_cols = Vt[:, J]          # Shape: (r, nnz)

            # Compute the values of X_update at non-zero positions
            X_update_values = torch.sum(U_D_rows * Vt_cols.t(), dim=1)  # Shape: (nnz,)

            # Create X_update as a sparse tensor
            indices = torch.stack([I, J])  # Shape: (2, nnz)
            X_update = torch.sparse_coo_tensor(indices, X_update_values, size=X.shape, device=device).coalesce()

            # Update X in sparse form
            X_values_scaled = X.values() * (1 - lr)
            X_update_values_scaled = X_update.values() * lr

            # Combine indices and values
            indices_combined = torch.cat([X.indices(), X_update.indices()], dim=1)
            values_combined = torch.cat([X_values_scaled, X_update_values_scaled])

            # Create the updated X
            X = torch.sparse_coo_tensor(indices_combined, values_combined, X.shape, device=device).coalesce()

            # Compute the error (assuming MTM is a sparse tensor)
            err = (MTM - X).coalesce()  # Both are sparse tensors
            err_values = err.values()
            loss = (err_values ** 2).mean()
            train_losses.append(loss.item())

            # Compute the relative error
            relative_err = torch.norm(err_values) / torch.norm(MTM.values())
            err_estimates.append(relative_err.item())

            # Update loop description
            loop.set_description(f"relative err: {relative_err:.7f}")

            # Optional: Check for convergence
            if epoch > 1:
                if abs(err_estimates[-1] - err_estimates[-2]) < tol:
                    print("Convergence achieved.")
                    break

        # Plotting relative error
        plt.figure(figsize=(5, 3))
        plt.plot(err_estimates, label='Relative Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Relative Error')
        plt.legend()
        plt.savefig('../plots/rel-err.png', dpi=150)
        plt.show()
        error_list = []
        missing_num = 0
        total_num = 0
        skip = 0
        use_reg = True
        U_list = []

        # Assume 'X' is your estimated matrix from previous computations (sparse tensor)
        estimation_goal = X

        # Obtain 'V' from the power method
        #_, _, V = power_svd(estimation_goal_dense, k=r)  # V: (d2, r)
        estimation_goal_scipy = torch_sparse_to_scipy(estimation_goal)
        _, _, V = scipy.sparse.linalg.svds(estimation_goal_scipy, k=r)
        V = torch.tensor(V).to(device)

        # Ensure 'V' is of shape (r, d2)
        #V = V.t()  # Now V: (r, d2)
        print(V.shape)

        M = M.coalesce()
        masks = masks.coalesce()

        device = M.device
        d2 = M.shape[1]

        for i in tqdm(range(M.shape[0])):
            # Extract the i-th row from M and masks
            M_row_sparse = M[i].coalesce()
            mask_row_sparse = masks[i].coalesce()

            # Convert the sparse row to dense
            #M_row_dense = torch.zeros(d2, device=device)
            M_row_dense = M_row_sparse.to_dense()
            mask_row_dense = mask_row_sparse.to_dense()
            
            #if M_row_sparse._nnz() > 0:
            #    M_row_dense[M_row_sparse.indices()[1]] = M_row_sparse.values()

            #mask_row_dense = torch.zeros(d2, dtype=torch.bool, device=device)
            #if mask_row_sparse._nnz() > 0:
            #    mask_row_dense[mask_row_sparse.indices()[1]] = mask_row_sparse.values().bool()
            

            # Get non-zero indices in M_row_dense
            non_zero_indices = M_row_dense.nonzero(as_tuple=True)[0]
            M_row_dense = M_row_dense.unsqueeze(0)

            if non_zero_indices.numel() == 0:
                skip += 1
                continue

            # Separate observed and missing indices
            observed_mask = mask_row_dense[non_zero_indices].bool()
            #print(observed_mask)
            observed_idx = non_zero_indices[observed_mask]
            missing_idx = non_zero_indices[~observed_mask]

            if observed_idx.numel() == 0 or missing_idx.numel() == 0:
                skip += 1
                continue

            missing_num += missing_idx.numel()
            total_num += observed_idx.numel() + missing_idx.numel()

            # Prepare 'observed_A' and 'observed_B'
            observed_A = V[:, observed_idx].t().clone()  # Shape: (num_observed, r)
            observed_B = M_row_dense[:, observed_idx].t().clone()  # Shape: (num_observed, 1)
            missing_A = V[:, missing_idx].t()
            missing_B = M_row_dense[:, missing_idx].t()

            # Solve for user factor 'u' using regression
            if use_reg:
                # Ridge regression
                lambda_reg = 0.001
                I = torch.eye(r, device=device) * lambda_reg
                A_t_A = observed_A.t() @ observed_A + I
                A_t_B = observed_A.t() @ observed_B
                try:
                    u = torch.linalg.solve(A_t_A, A_t_B)  # Shape: (r, 1)
                except RuntimeError as e:
                    print(f"LinAlgError for user {i}: {e}")
                    skip += 1
                    continue
            else:
                # Linear regression
                u = torch.linalg.lstsq(observed_A, observed_B).solution  # Shape: (r, 1)

            # Check for NaN values
            if torch.any(torch.isnan(u)):
                print(f"NAN detected in user {i}!")
                skip += 1
                continue

            # Store the user factor
            U_list.append(u.t())  # Shape: (1, r)


            # Compute the error on missing entries
            predictions = missing_A @ u  # Shape: (num_missing, 1)
            errors = predictions - missing_B  # Shape: (num_missing, 1)
            error_list.append(torch.sum(errors ** 2).item())

            # Release variables to free memory
            del M_row_sparse, mask_row_sparse, M_row_dense, mask_row_dense, observed_A, observed_B, missing_A, missing_B, predictions, errors, u

        # Compute the root mean squared error over missing entries
        rmse_missing = np.sqrt(np.sum(error_list) / missing_num)
        print(f"Skipped users due to lack of data: {skip}")
        print(f"Total missing entries evaluated: {missing_num}")
        print(f"RMSE on missing entries: {rmse_missing:.6f}")
    
