import numpy as np
import math as m
from typing import Callable
from scipy.sparse.linalg import spsolve

def modified_newton(f, grad_f, hess_f, x0, tol=1e-3, max_iter=10000, tau_init=1e-4, rho=0.5, c=1e-4):
    """
    Modified Newton method with backtracking line search.
    
    Parameters:
        f       : function returning scalar value (objective function)
        grad_f  : function returning gradient vector at x
        hess_f  : function returning Hessian matrix at x
        x0      : initial guess (numpy array)
        tol     : tolerance for stopping (gradient norm)
        max_iter: maximum number of iterations
        tau_init: initial Hessian modification parameter
        rho     : backtracking reduction factor (0 < rho < 1)
        c       : Armijo condition constant (0 < c < 1)
        
    Returns:
        x       : approximate minimizer
        f(x)    : function value at x
        k       : number of iterations performed
        grad_norm : norm of gradient at final x
    """
    
    x = x0.copy()
    points = [x]
    norms = []
    num_iters = 0

    for k in range(max_iter):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g) # Frobenius norm
        norms.append(grad_norm)
        if grad_norm < tol:
            break  # Stop if gradient is small enough

        num_iters += 1
        H = hess_f(x)

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
        mxitr = 3   # Set mx iterations for armijo condition
        x = armijo_condition(f, x, p, g, alpha, rho, c, mxitr)
        x = np.ravel(x)

        if num_iters < 10:
            print(140*"-")
            print(f"Iteration number {num_iters}")            
            print(f"point: {x}")
            print(f"gradient: {g}")
            print(f"grad norm {grad_norm}")
            print(f"Function value: {f(x)}")

        points.append(x)

    points = np.array(points)

    return {"minimum_pt": x, "minimum": f(x), "num_iter": num_iters, "grad_norms": norms, "visited_pt": points}

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

    print(f"alpha: {alpha}")
    print(f"direction {p}")
    return (x + alpha * p).flatten()