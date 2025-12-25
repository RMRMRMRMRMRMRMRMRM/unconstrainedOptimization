import numpy as np
import math as m
from typing import Callable
from scipy.sparse.linalg import spsolve
import functions as fn
import scipy.sparse as sp
import time

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

    before_MN = []
    time_MN = []
    time_backtracking = []

    start_time = time.perf_counter()

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


        before_MN.append(start_time - time.perf_counter())

        start_time = time.perf_counter()
        # Solve H_mod * p = -g for p
        p = -spsolve(H, g)
        time_MN.append(start_time - time.perf_counter())

        start_time = time.perf_counter()

        alpha = 1.0  # Start with full step
        mxitr = 5   # Set mx iterations for armijo condition
        x = line_search_armijo_wolfe(f, grad_f, x, p, g, alpha, rho, c, c2=0.7, wolfe = True)
        x = np.ravel(x)
        time_backtracking.append(start_time - time.perf_counter())

        points.append(x)

    if not success:
         flag = "max-iterations-hit"
    points = np.array(points)

    if len(norms) > 2:
            ratios = [norms[i+1]/norms[i] for i in range(len(norms)-1) if norms[i] > 0]
            rate = np.mean(ratios[-5:])  # average of last few ratios #TODO look for certain how its supposed to look
    else:
        rate = np.nan

    print(f"Average time before linear system: {np.mean(before_MN)}")
    print(f"Average time linesr system: {np.mean(time_MN)}")
    print(f"Average time bactracking: {np.mean(time_backtracking)}")

    
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



def newton_conjugate_gradient(
    f, grad_f, hess_f, x0, finite_diff=True, k=4,
    tol=1e-6, max_iter=10000, tau_init=1e-4,
    rho=0.5, c=1e-4, cg_tol=1e-4, cg_max_iter=None
):
    """
    Newton–Conjugate Gradient (Newton–CG) method with backtracking line search.

    Parameters:
        f           : function returning scalar value (objective function)
        grad_f      : function returning gradient vector at x
        hess_f      : function returning Hessian (sparse matrix) at x
        x0          : initial guess (numpy array)
        finite_diff : switch for using finite difference (boolean)
        k           : h = 10^-k (finite difference step)
        tol         : tolerance for stopping (gradient norm)
        max_iter    : maximum number of iterations
        tau_init    : unused here (kept for signature consistency)
        rho         : backtracking reduction factor (0 < rho < 1)
        c           : Armijo condition constant (0 < c < 1)
        cg_tol      : tolerance for inner CG loop
        cg_max_iter : maximum CG iterations (defaults to len(x))

    Returns:
        dict with same keys as modified_newton:
            "minimum_pt", "minimum", "num_iter", "grad_norm",
            "success", "flag", "rate", "visited_pt", "grad_norms"
    """

    x = x0.copy()
    points = [x]
    norms = []
    success = False
    flag = "-"
    num_iters = 0

    before_CG = []
    time_CG = []
    time_backtracking = []
    start_time = time.perf_counter()

    for i in range(max_iter):

        # Compute gradient and Hessian
        if finite_diff:
            g = finite_diff_grad_central(f, x, k)
            H = finite_diff_hessian_diag_from_grad(
                f, fn.grad_banded_trig, finite_diff_grad_central, x, k
            )
        else:
            g = grad_f(x)
            H = hess_f(x)

        grad_norm = np.linalg.norm(g)
        norms.append(grad_norm)
        if grad_norm < tol:
            success = True
            break

        num_iters += 1

        n = len(x)
        if cg_max_iter is None:
            cg_max_iter = n

        before_CG.append(start_time - time.perf_counter())

        start_time = time.perf_counter()
        # --- Begin Conjugate Gradient solver for H p = -g ---
        p = np.zeros_like(g)
        r = g.copy()            # residual = H*p + g
        d = -r.copy()
        delta_new = r @ r

        for j in range(cg_max_iter):
            Hd = H @ d                  # sparse-safe product
            dHd = d @ Hd                # curvature along d

            if dHd <= 0:
                # negative curvature detected
                if j == 0:
                    p = -g              # fallback: steepest descent
                else:
                    p = p
                break

            alpha_cg = delta_new / dHd
            p = p + alpha_cg * d
            r = r + alpha_cg * Hd

            if np.linalg.norm(r) < cg_tol * np.linalg.norm(g):
                break

            delta_old = delta_new
            delta_new = r @ r
            beta_cg = delta_new / delta_old
            d = -r + beta_cg * d #TODO why minus

        time_CG.append(start_time - time.perf_counter())

        # --- Backtracking line search (Armijo condition) ---
        start_time = time.perf_counter()
        alpha = 1.0
        mxitr = 5
        x = line_search_armijo_wolfe(f, grad_f, x, p, g, alpha, rho, c, mxitr, wolfe = True)
        x = np.ravel(x)
        time_backtracking.append(start_time - time.perf_counter())

        points.append(x)

    if not success and flag == "-":
        flag = "max-iterations-hit"

    points = np.array(points)

    # Estimate rate of convergence (same as modified_newton)
    if len(norms) > 2:
        ratios = [norms[i+1]/norms[i] for i in range(len(norms)-1) if norms[i] > 0]
        rate = np.mean(ratios[-5:]) if len(ratios) >= 5 else np.mean(ratios)
    else:
        rate = np.nan

    print(f"Average time before CG: {np.mean(before_CG)}")
    print(f"Average time CG: {np.mean(time_CG)}")
    print(f"Average time bactracking: {np.mean(time_backtracking)}")


    return {
        "minimum_pt": x,
        "minimum": f(x),
        "num_iter": num_iters,
        "grad_norm": grad_norm,
        "success": success,
        "flag": flag,
        "rate": rate,
        "visited_pt": np.array(points),
        "grad_norms": np.array(norms),
    }
    


def line_search_armijo_wolfe(f: Callable[[np.ndarray], float],
                             grad_f: Callable[[np.ndarray], np.ndarray],
                             x: np.ndarray,
                             p: np.ndarray,
                             grad: np.ndarray,
                             alpha: float = 1.0,
                             rho: float = 0.5,
                             c1: float = 1e-4,
                             c2: float = 0.9,
                             mxitr: int = 20,
                             wolfe: bool = False):
    """
    Backtracking / bracketing line search satisfying the Armijo condition,
    optionally enforcing the Wolfe curvature condition.

    Parameters
    ----------
    f : callable
        Objective function returning a scalar.
    grad_f : callable
        Function returning gradient at x.
    x : np.ndarray
        Current point.
    p : np.ndarray
        Descent direction (usually -grad).
    grad : np.ndarray
        Gradient at point x.
    alpha : float, optional
        Initial step size (default: 1.0).
    rho : float, optional
        Shrinkage factor for initial scaling (default: 0.5).
    c1 : float, optional
        Armijo parameter (default: 1e-4).
    c2 : float, optional
        Wolfe curvature parameter (default: 0.9).
    mxitr : int, optional
        Maximum number of function evaluations.
    wolfe : bool, optional
        If True, also enforce the Wolfe curvature condition.

    Returns
    -------
    x_new : np.ndarray
        Updated point satisfying the chosen conditions.
    """

    fx = f(x)
    phi0_prime = np.dot(grad, p)
    alpha_low, alpha_high = 0, None  # bracket endpoints
    itr = 0

    while itr < mxitr:
        fx_new = f(x + alpha * p)

        # Armijo condition
        if fx_new > fx + c1 * alpha * phi0_prime or \
           (alpha_high is not None and fx_new >= f(x + alpha_low * p)):
            alpha_high = alpha
        else:
            # Check curvature condition if requested
            if wolfe:
                g_new = grad_f(x + alpha * p)
                phi_prime = np.dot(g_new, p)
                if phi_prime < c2 * phi0_prime:
                    # slope still too negative: increase alpha
                    alpha_low = alpha
                elif phi_prime > -c2 * phi0_prime:
                    # strong Wolfe variant; slope reversed sign too much
                    alpha_high = alpha
                else:
                    # Wolfe satisfied
                    break
            else:
                # Armijo satisfied and no Wolfe required
                break

        # Update alpha
        if alpha_high is None:
            alpha *= 2.0  # expand until upper bound found
        else:
            alpha = 0.5 * (alpha_low + alpha_high)

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
