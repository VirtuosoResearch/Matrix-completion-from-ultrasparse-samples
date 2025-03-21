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

def main():
    d1 = 1000
    d2 = 100
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