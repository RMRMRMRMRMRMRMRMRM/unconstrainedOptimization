import numpy as np
import math as m
from typing import Callable
from scipy.sparse.linalg import spsolve
import functions as fn
import scipy.sparse as sp

def modified_newton(f, grad_f, hess_f, x0, finite_diff=True, k=4, tol=1e-6, max_iter=10000, tau_init=1e-4, rho=0.5, c=1e-4):
    """
    Modified Newton method with backtracking line search.
    
    Parameters:
        f           : function returning scalar value (objective function)
        grad_f      : function returning gradient vector at x
        hess_f      : function returning Hessian matrix at x
        x0          : initial guess (numpy array)
        finite_diff : switch for using finite difference (boolean)
        k           : h = 10^-k
        tol         : tolerance for stopping (gradient norm)
        max_iter    : maximum number of iterations
        tau_init    : initial Hessian modification parameter
        rho         : backtracking reduction factor (0 < rho < 1)
        c           : Armijo condition constant (0 < c < 1)
        
    Returns:
        x       : approximate minimizer
        f(x)    : function value at x
        i       : number of iterations performed
        grad_norm : norm of gradient at final x
    """
    x = x0.copy()
    points = [x]
    norms = []
    num_iters = 0
    success = False
    flag = "-"

    for i in range(max_iter):
        if finite_diff:
            g = finite_diff_grad_central(f, x, k)
            H = finite_diff_hessian_diag_from_grad(f, fn.grad_banded_trig, finite_diff_grad_central, x, k)

        else:
            g = grad_f(x)
            H = hess_f(x)
        
        grad_norm = np.linalg.norm(g) # Frobenius norm
        norms.append(grad_norm)
        if grad_norm < tol:
            success = True
            break  # Stop if gradient is small enough

        num_iters += 1

        tau = tau_init
        
        # H is diagonal
        while True:
            if np.all(H.diagonal() > 0):
                break  # Hessian is now positive definite

            else:
                tau = max(2*tau, m.sqrt((H.multiply(H)).sum()))
                H.setdiag(H.diagonal() + tau) # Update Hessian



        # Solve H_mod * p = -g for p
        p = -spsolve(H, g)

        alpha = 1.0  # Start with full step
        mxitr = 5   # Set mx iterations for armijo condition
        x = armijo_condition(f, x, p, g, alpha, rho, c, mxitr)
        x = np.ravel(x)

        points.append(x)

    if not success:
         flag = "max-iterations-hit"
    points = np.array(points)

    if len(norms) > 2:
            ratios = [norms[i+1]/norms[i] for i in range(len(norms)-1) if norms[i] > 0]
            rate = np.mean(ratios[-5:])  # average of last few ratios #TODO look for certain how its supposed to look
    else:
        rate = np.nan
    
    return {
        "minimum_pt": x,
        "minimum": f(x),
        "num_iter": i + 1,
        "grad_norm": grad_norm,
        "success": success,
        "flag": flag,
        "rate": rate,
        "visited_pt": np.array(points),
        "grad_norms": np.array(norms),
    }

def armijo_condition(f:     Callable[[np.ndarray], float],
                     x:     np.array, 
                     p:     np.array, 
                     grad:  np.array, 
                     alpha: float, 
                     rho:   float, 
                     c:     float,
                     mxitr: int):
    """
    Perform a backtracking line search satisfying the Armijo condition.

    Parameters
    ----------
    f : callable
        Objective function that takes a NumPy array and returns a scalar.
    x : np.ndarray
        Current point.
    p : np.ndarray
        Descent direction (usually -grad).
    grad : np.ndarray
        Gradient at point x.
    alpha : float, optional
        Initial step size (default: 1.0).
    rho : float, optional
        Shrinkage factor in (0, 1) (default: 0.5).
    c : float, optional
        Armijo parameter (default: 1e-4).

    Returns
    -------
    x_new : np.ndarray
        Updated point after step satisfying the Armijo condition.
    """
    fx = f(x)
    p = np.ravel(p)
    itr = 0
    while f(x + alpha * p) > fx + c * alpha * np.dot(grad, p) and itr < mxitr:
            alpha *= rho
            itr += 1

    return (x + alpha * p).flatten()

def finite_diff_grad_central(f, x, k=4, relative=True):
    """
    Central finite-difference gradient, vectorized and efficient.

    Parameters
    ----------
    f : callable
        Vectorized function returning scalar outputs for shape (m, n) inputs.
    x : ndarray
        Point where the gradient is evaluated, shape (n,).
    k : int
        Power for h = 10^(-k).  Typical values: 4, 8, 12.
    relative : bool
        If True, use hi = 10^-k * |x_i|, else constant 10^-k.

    Returns
    -------
    grad : ndarray of shape (n,)
        Central-difference approximation of the gradient.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size

    # step sizes per coordinate
    base_h = 10.0 ** (-k)
    h_vec = base_h * (np.abs(x) if relative else np.ones_like(x))
    h_vec[h_vec == 0] = base_h  # avoid zero steps

    # construct all +h and -h perturbations
    X_plus  = x + np.eye(n) * h_vec
    X_minus = x - np.eye(n) * h_vec
    X_all   = np.vstack([X_plus, X_minus])  # shape (2n, n)

    # evaluate function at all points (vectorized)
    f_vals = f(X_all)           # returns array of shape (2n,)
    f_plus, f_minus = f_vals[:n], f_vals[n:]

    # central difference
    grad = (f_plus - f_minus) / (2 * h_vec)
    return grad

def finite_diff_hessian_diag_from_grad(f, grad_exact, grad_fd,
                                       x, k=8, relative=True,
                                       use_fd_grad=False):
    """
    Compute diagonal Hessian using central finite differences of the gradient.

    Parameters
    ----------
    f : callable
        The scalar function f(x) (only used if grad_fd needs it).
    grad_exact : callable
        Function returning the exact gradient vector of f at x.
    grad_fd : callable
        Function returning finite-difference gradient of f at x.
    x : ndarray
        Point of evaluation, shape (n,).
    k : int
        Power for h = 10^-k.
    relative : bool
        If True, use hi = 10^-k * |x_i|; else constant 10^-k.
    use_fd_grad : bool
        If True, use grad_fd; otherwise use grad_exact.

    Returns
    -------
    H : scipy.sparse.csr_matrix
        Sparse diagonal Hessian approximation.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    base_h = 10.0 ** (-k)
    h_vec = base_h * (np.abs(x) if relative else np.ones_like(x))
    h_vec[h_vec == 0] = base_h  # avoid zero step

    # choose gradient function
    grad_func = (lambda y: grad_fd(f, y, k=k, relative=relative)) if use_fd_grad else grad_exact

    # allocate diagonal entries
    H_diag = np.zeros(n)

    for i in range(n):
        xi = x.copy()
        xi[i] += h_vec[i]
        grad_plus = grad_func(xi)

        xi = x.copy()
        xi[i] -= h_vec[i]
        grad_minus = grad_func(xi)

        H_diag[i] = (grad_plus[i] - grad_minus[i]) / (2 * h_vec[i])

    # return as sparse diagonal
    return sp.diags(H_diag, 0, format='csr')
