import numpy as np
from cvxpy import Variable, Minimize, Problem
from cvxpy import norm as cvxnorm
from cvxpy import SCS

# Generate a random matrix
#n = 3
#X0 = np.random.rand(n, n)
#indices = np.nonzero(np.array(X0))

def nuclear_norm_minimization(X0, indices):
    X0_Omega = X0[indices]

    n = X0.shape[0]
    X = Variable((n, n))
    obj = Minimize(cvxnorm(X, "nuc"))
    constraints = [X[indices] == X0_Omega]
    prob = Problem(obj, constraints)
    val = prob.solve(solver=SCS, verbose=False)
    return X.value, val

#X_nuc, nuc_val = nuclear_norm_minimization(X0, indices)
#print(X_nuc)
#print(nuc_val)