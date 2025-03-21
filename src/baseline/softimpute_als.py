import numpy as np

def frob(U_old, D_old, V_old, U, D, V):
    old_norm_sq = (D_old ** 2).sum()
    new_norm_sq = (D ** 2).sum()

    utu = U_old.T @ U      # 形状 (J, J)
    vtv = V_old.T @ V      # 形状 (J, J)
    
    cross_mat = np.diag(D_old.flatten()) @ utu @ np.diag(D.flatten()) @ vtv
    cross_val = np.trace(cross_mat)  # 标量
    
    diff_sq = old_norm_sq + new_norm_sq - 2.0 * cross_val
    return diff_sq / max(old_norm_sq, 1e-9)


def soft_thresholding(sigma, lambd):
    return np.maximum(sigma - lambd, 0.0)


class SoftImpute:
    def __init__(self, J=2, thresh=1e-5, lambd=0.0, maxit=100, random_state=None, verbose=False):
        """
        :param J: Rank
        :param thresh: convergence threshold
        :param lambd: regularization parameter
        :param maxit: maximum number of iterations
        :param random_state: random seed
        :param verbose: whether to print convergence information
        """
        self.J = J
        self.thresh = thresh
        self.lambd = lambd
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose
        
        self.u = None
        self.d = None
        self.v = None

    def fit(self, X):
        n, m = X.shape
        xnas = np.isnan(X)
        
        # Fill missing values with column means
        xfill = X.copy()
        col_means = np.nanmean(xfill, axis=0)
        if np.any(np.isnan(col_means)):
            col_means[np.isnan(col_means)] = 0

        np.copyto(xfill, col_means, where=np.isnan(xfill))
        
        # Initialize U with random values and do SVD
        U = self.rs.normal(0.0, 1.0, (n, self.J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        
        # Initialize V and D
        V = np.zeros((m, self.J))
        D = np.ones(self.J)
        
        ratio = 1.0
        iters = 0
        
        while ratio > self.thresh and iters < self.maxit:
            iters += 1
            U_old = U.copy()
            V_old = V.copy()
            D_old = D.copy()
            
            # === (Step 1) Fix V, update U and D ===
            # B = U^T X_fill
            B = U.T @ xfill
            Bt = B.T  # (m, J)
            #print(Bt.shape)
            #print(Bt.min(), Bt.max())
            U_svd, s_svd, Vt_svd = np.linalg.svd(Bt, full_matrices=False)
            # U_svd: (m, m), s_svd: (min(m,J), ), Vt_svd: (min(m,J), J)
            
            # soft-thresholding
            s_thr = soft_thresholding(s_svd, self.lambd)
            
            nonzero_idx = s_thr > 1e-12
            r = np.sum(nonzero_idx)
            
            if r == 0:
                V = np.zeros_like(V)
                D = np.zeros_like(D)
            else:
                s_thr = s_thr[:r]
                V = U_svd[:, :r]
                # D
                D = s_thr
                Rmat = Vt_svd[:r, :]  # (r, J)
                U = U @ Rmat.T  # (n, r)
                
                if r < self.J:
                    V = np.hstack([V, np.zeros((m, self.J - r))])
                    D = np.concatenate([D, np.zeros(self.J - r)])
                    U = np.hstack([U, np.zeros((n, self.J - r))])
            
            xhat = U @ (D[:, None] * V.T)  # (n, m)
            xfill[xnas] = xhat[xnas]
            
            # === (Step 2) Fix U, update V and D ===
            A = xfill @ V  # (n, J)
            U_svd2, s_svd2, Vt_svd2 = np.linalg.svd(A, full_matrices=False)
            s_thr2 = soft_thresholding(s_svd2, self.lambd)
            
            nonzero_idx2 = s_thr2 > 1e-12
            r2 = np.sum(nonzero_idx2)
            
            if r2 == 0:
                U = np.zeros_like(U)
                D = np.zeros_like(D)
                V = np.zeros_like(V)
            else:
                s_thr2 = s_thr2[:r2]
                U = U_svd2[:, :r2]
                D = s_thr2
                Rmat2 = Vt_svd2[:r2, :]  # (r2, J)
                V = V @ Rmat2.T  # (m, r2)
                
                if r2 < self.J:
                    U = np.hstack([U, np.zeros((n, self.J - r2))])
                    D = np.concatenate([D, np.zeros(self.J - r2)])
                    V = np.hstack([V, np.zeros((m, self.J - r2))])
            
            xhat = U @ (D[:, None] * V.T)
            xfill[xnas] = xhat[xnas]
            
            # Convergence check
            ratio = frob(U_old, D_old, V_old, U, D, V)
            if self.verbose:
                print(f"iter: {iters:4d}, ratio = {ratio:.6f}")
        
        self.u = U[:, :self.J]
        self.d = D[:self.J]
        self.v = V[:, :self.J]
        return self

    def suv(self, vd):
        return self.u @ vd.T

    def predict(self, X, copyto=False):
        vd = self.v * self.d[None, :]  # (m, J)
        X_imp = self.u @ vd.T
        if copyto:
            np.copyto(X, X_imp, where=np.isnan(X))
        else:
            return X_imp

class SoftImputeSparse:
    def __init__(self, J=2, thresh=1e-5, lambd=0.0, maxit=100, random_state=None, verbose=False):
        """
        Sparse soft-impute that avoids forming a full dense imputed matrix.
        Assumes X is provided as a sparse CSR matrix storing only observed values.
        Missing entries are assumed to be absent.
        """
        self.J = J
        self.thresh = thresh
        self.lambd = lambd
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose

        self.u = None
        self.d = None
        self.v = None

    def _compute_B(self, U, D, V, X, X_csc):
        """
        Compute B = U^T X_fill for each column j without forming X_fill.
        Here X_fill is defined by
            X_fill[i,j] = X[i,j]  if observed, and
                          U[i,:]·(D * V[j,:]) if missing.
        For each column j we use:
            B[:,j] = (U.T @ imputed_column) +
                     sum_{i in obs_j} U[i,:]*(X[i,j] - imputed_value[i])
        Notice that the full default sum can be computed as
            U.T @ imputed_column = Q · (D * V[j,:])
        where Q = U.T U.
        """
        n, J = U.shape
        m = V.shape[0]
        B = np.zeros((J, m))
        Q = U.T @ U  # J x J
        for j in range(m):
            # For column j, let dv = D * V[j,:]
            dv = D * V[j, :]  # shape (J,)
            default = Q.dot(dv)  # equals U.T @ (U @ (D*V[j,:]))
            # Get observed indices and values in column j (using CSC for fast column access)
            col = X_csc.getcol(j)
            obs_idx = col.indices
            obs_vals = col.data
            if obs_idx.size > 0:
                # For observed rows, compute imputed values: U[i,:]·(D*V[j,:])
                imputed_obs = U[obs_idx, :] @ (D * V[j, :])
                correction = U[obs_idx, :].T @ (obs_vals - imputed_obs)
            else:
                correction = 0.0
            B[:, j] = default + correction
        return B

    def _compute_A(self, U, D, V, X):
        """
        Compute A = X_fill V for each row i without forming X_fill.
        For each row i, X_fill[i,j] is defined as above.
        Then:
            A[i,:] = (sum_{j=1}^m V[j,:]*imputed_value[i,j])
                      + sum_{j in obs_i} V[j,:]*(X[i,j]-imputed_value[i,j])
        The default sum (over all j) can be written as:
            default = M · U[i,:],   with M = V.T @ (V * D)
        (Here V * D multiplies each column of V by the corresponding D element.)
        """
        n, _ = X.shape
        J = U.shape[1]
        A = np.zeros((n, J))
        # Precompute M = V.T @ (V * D)
        W = V * D[np.newaxis, :]  # shape (m, J)
        M = V.T @ W  # shape (J, J)
        for i in range(n):
            default = M.dot(U[i, :])
            row = X.getrow(i)  # efficient row slicing from CSR
            obs_idx = row.indices
            obs_vals = row.data
            if obs_idx.size > 0:
                # For each observed column j in row i, compute imputed value:
                # imputed = U[i,:]·(D*V[j,:])
                imputed_obs = np.array([U[i, :].dot(D * V[j, :]) for j in obs_idx])
                correction = V[obs_idx, :].T @ (obs_vals - imputed_obs)
            else:
                correction = 0.0
            A[i, :] = default + correction
        return A

    def fit(self, X):
        """
        Expects X as a sparse CSR matrix containing only observed entries.
        Missing entries are assumed to be absent.
        """
        # Ensure X is in CSR and also have a CSC copy for column access.
        if not hasattr(X, "tocsr"):
            raise ValueError("Input X must be a sparse matrix.")
        X = X.tocsr()
        X_csc = X.tocsc()
        n, m = X.shape

        # Initialize U with random values (n x J) and re-orthogonalize.
        U = self.rs.normal(0.0, 1.0, (n, self.J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        # Initialize V (m x J) and D (vector length J)
        V = np.zeros((m, self.J))
        D = np.ones(self.J)

        ratio = 1.0
        iters = 0

        while ratio > self.thresh and iters < self.maxit:
            iters += 1
            U_old = U.copy()
            V_old = V.copy()
            D_old = D.copy()

            # --- Step 1: Fix V; update U and D using B = U.T X_fill ---
            B = self._compute_B(U, D, V, X, X_csc)  # B has shape (J, m)
            # Note: B.T is (m x J); m is moderate so we use full dense SVD.
            U_svd, s_svd, Vt_svd = np.linalg.svd(B.T, full_matrices=False)
            s_thr = soft_thresholding(s_svd, self.lambd)
            nonzero = s_thr > 1e-12
            r = np.sum(nonzero)
            if r == 0:
                V = np.zeros_like(V)
                D = np.zeros_like(D)
            else:
                s_thr = s_thr[:r]
                V_new = U_svd[:, :r]  # (m x r)
                D_new = s_thr         # (r,)
                Rmat = Vt_svd[:r, :]  # (r x J)
                # Update U (n x r)
                U = U @ Rmat.T
                # Pad with zeros if needed to maintain shape (n x J)
                if r < self.J:
                    pad_width = self.J - r
                    U = np.hstack([U, np.zeros((n, pad_width))])
                    V_new = np.hstack([V_new, np.zeros((m, pad_width))])
                    D_new = np.concatenate([D_new, np.zeros(pad_width)])
                V = V_new.copy()
                D = D_new.copy()

            # --- Step 2: Fix U; update V and D using A = X_fill V ---
            A = self._compute_A(U, D, V, X)  # A has shape (n, J)
            U_svd2, s_svd2, Vt_svd2 = np.linalg.svd(A, full_matrices=False)
            s_thr2 = soft_thresholding(s_svd2, self.lambd)
            nonzero2 = s_thr2 > 1e-12
            r2 = np.sum(nonzero2)
            if r2 == 0:
                U = np.zeros_like(U)
                D = np.zeros_like(D)
                V = np.zeros_like(V)
            else:
                s_thr2 = s_thr2[:r2]
                U_new = U_svd2[:, :r2]  # (n x r2)
                D_new = s_thr2         # (r2,)
                Rmat2 = Vt_svd2[:r2, :]  # (r2 x J)
                # Update V (m x r2)
                V = V @ Rmat2.T
                if r2 < self.J:
                    pad_width = self.J - r2
                    U_new = np.hstack([U_new, np.zeros((n, pad_width))])
                    D_new = np.concatenate([D_new, np.zeros(pad_width)])
                    V = np.hstack([V, np.zeros((m, pad_width))])
                U = U_new.copy()
                D = D_new.copy()

            ratio = frob(U_old, D_old, V_old, U, D, V)
            if self.verbose:
                print(f"Sparse iter: {iters:4d}, ratio = {ratio:.6f}")

        self.u = U[:, :self.J]
        self.d = D[:self.J]
        self.v = V[:, :self.J]
        return self

    def predict(self, X=None, copyto=False):
        """
        Returns the completed (imputed) matrix as a dense array:
             X_imp = U @ (diag(D) @ V^T).
        (X is ignored here since the completion is stored in the model.)
        """
        vd = self.v * self.d[np.newaxis, :]
        return self.u @ vd.T

def main():
    d1 = 10000
    d2 = 1000
    r = 5
    p = 0.008
    X = np.random.normal(2, 1, (d1, d2))
    U, D, Vt = np.linalg.svd(X, full_matrices=False)
    D[r:] = 0
    X = U.dot(np.diag(D)).dot(Vt)
    mask = np.random.rand(*X.shape) < p
    M = X.copy()
    M[~mask] = np.nan
    print(M)

    clf = SoftImpute(J=2)
    fit = clf.fit(M)
    X_test = X.copy()
    X_imp = clf.predict(X_test)
    mse = np.mean((X[~mask] - X_imp[~mask])**2)
    print(mse)

    XTX = X.T @ X
    XTX_imp = X_imp.T @ X_imp
    print(np.linalg.norm(XTX - XTX_imp) / np.linalg.norm(XTX))

if __name__ == '__main__':
    main()