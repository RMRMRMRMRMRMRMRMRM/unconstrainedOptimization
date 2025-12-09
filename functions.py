import numpy as np
import math as m
from scipy.sparse import csr_matrix


def grad_test(x):
    return np.array([-3 * m.sin(3*x[0]) + 2*x[0], 3 * m.cos(3*x[1]) + 2*x[1]])

def hessian_test(x):
    return np.array([[-9 * m.cos(3*x[0]) + 2, 0], [0, -9 * m.sin(3*x[1]) + 2]])

def test_f(x):
    return x[0]^2 + x[1]^2 + m.cos(3*x[0]) + m.sin(3*x[1])


def banded_trig(x):
    """
    Compute the Banded Trigonometric function with i-multiplier:
    F(x) = sum_{i=1}^n i * [(1 - cos(x_i)) + sin(x_{i-1}) - sin(x_{i+1})],
    with x_0 = x_{n+1} = 0
    """
    n = len(x)
    x = np.ravel(x)
    x_ext = np.concatenate(([0], x, [0]))  # add boundaries
    F = 0.0
    for i in range(1, n + 1):
        F += i * ((1 - np.cos(x_ext[i])) + np.sin(x_ext[i - 1]) - np.sin(x_ext[i + 1]))
    return F

def grad_banded_trig(x):
    """
    Gradient:
    ∂F/∂x_j = j*sin(x_j) + (j+1)*cos(x_{j}) - (j-1)*cos(x_{j})
    """
    n = len(x)
    grad = np.zeros(n)

    for j in range(0, n):
        if j == 0:
            grad[j] = (j + 1) * m.sin(x[j]) + (j + 2) * m.cos(x[j])

        elif j == n - 1:
            grad[j] = (j + 1) * m.sin(x[j]) - (j - 2) * m.cos(x[j])

        else:
            grad[j] = (j + 1) * m.sin(x[j]) + (j + 2) * m.cos(x[j]) - (j - 2) * m.cos(x[j])
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
    for j in range(n):
        if j == 0:
            H[j, j] = (j + 1) * m.cos(x[j]) - (j + 2) * m.sin(x[j])

        elif j == n - 1:
            H[j, j] = (j + 1) * m.cos(x[j]) - (j - 2) * m.sin(x[j])

        else:
            H[j, j] = (j + 1) * m.cos(x[j]) - 2 * m.sin(x[j])

    H_sparse = csr_matrix(H)
    return H_sparse
