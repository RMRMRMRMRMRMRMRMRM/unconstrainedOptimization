import numpy as np
import math as m
from scipy.sparse import csr_matrix


def banded_trig(X):
    """
    Vectorized Banded Trigonometric function with i-multiplier.
    Works for single vector (n,) or multiple points (m, n).
    F(x) = sum_i i * [(1 - cos(x_i)) + sin(x_{i-1}) - sin(x_{i+1})],
    with x_0 = x_{n+1} = 0
    """
    X = np.atleast_2d(X)       # ensure 2D: (m, n)
    m, n = X.shape

    # add boundary zeros
    X_ext = np.hstack([np.zeros((m,1)), X, np.zeros((m,1))])

    # slices for xi, xi-1, xi+1
    xi    = X_ext[:, 1:n+1]
    xi_m1 = X_ext[:, 0:n]
    xi_p1 = X_ext[:, 2:n+2]

    # multipliers
    i_vec = np.arange(1, n+1)  # shape (n,)
    
    # compute all terms at once
    terms = i_vec * ((1 - np.cos(xi)) + np.sin(xi_m1) - np.sin(xi_p1))  # shape (m, n)

    # sum over columns (dimensions) to get scalar per row
    F = np.sum(terms, axis=1)

    # return scalar if single input
    return F[0] if F.size == 1 else F


def grad_banded_trig(x):
    """
    Gradient:
    ∂F/∂x_j = j*sin(x_j) + (j+1)*cos(x_{j}) - (j-1)*cos(x_{j})
    """
    n = len(x)
    grad = np.zeros(n)

    for j in range(0, n):
        if j == n - 1:
            grad[j] = (j + 1) * m.sin(x[j]) - j * m.cos(x[j])

        else:
            grad[j] = (j + 1) * m.sin(x[j]) + 2 * m.cos(x[j])
    return grad

def grad_banded_trig_test(x):
    return np.array([m.sin(x[0]) - 2 * m.cos(x[0]), 2 * m.sin(x[1]) - m.cos(x[1])])

def hess_banded_trig(x):
    """
    Hessian (banded, tridiagonal structure):
        H[j,j]   = (j+1) * cos(x_j)
        H[j,j-1] = -(j) * sin(x_{j-1})
        H[j,j+1] = (j+2) * sin(x_{j+1})
    """
    n = len(x)
    H = np.zeros((n, n))
    for j in range(0, n):
        if j == n - 1:
            H[j, j] = (j + 1) * m.cos(x[j]) + j * m.sin(x[j])

        else:
            H[j, j] = (j + 1) * m.cos(x[j]) - 2 * m.sin(x[j])

    H_sparse = csr_matrix(H)
    return H_sparse

def hess_banded_trig_test(x):
    """
    Hessian (banded, tridiagonal structure):
        H[j,j]   = (j+1) * cos(x_j)
        H[j,j-1] = -(j) * sin(x_{j-1})
        H[j,j+1] = (j+2) * sin(x_{j+1})
    """
    n = len(x)
    H = np.array([[m.cos(x[0]) - 2 * m.sin(x[0]), 0], [0, 2 * m.cos(x[1]) + m.sin(x[1])]])

    H_sparse = csr_matrix(H)
    return H_sparse
