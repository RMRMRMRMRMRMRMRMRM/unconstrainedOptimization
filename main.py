import os
import numpy as np
import algorithms as alg
import functions as fn
import plot  
import utils

# ----------------------------
# Setup and configuration
# ----------------------------
SEED = 350445  # min student ID
np.random.seed(SEED)

dimensions = [100000]
num_random_starts = 5
max_iter = 10000
k_params = [4, 8, 12]
tol = 1e-6

base = "results"
sub_dirs = ["summary_CSV", "plots/table", "plots/path", "plots/convergence"]

for sub in sub_dirs:
    os.makedirs(os.path.join(base, sub), exist_ok=True)

def get_base_point(n):
    """Return base starting point x̄ for the problem."""
    # adjust if your problem specifies other pattern
    return np.ones(n)

# Analytic Hessian
utils.run_optimization_experiment(
    algorithm_fn=alg.modified_newton,
    algorithm_name="MN",
    function=fn.banded_trig,
    grad_f=fn.grad_banded_trig,
    hess_f=fn.hess_banded_trig,
    dimensions=dimensions,
    num_random_starts=num_random_starts,
    tol=tol,
    max_iter=max_iter,
    base_dir=base,
    finite_diff=False,
)

# # Finite difference Hessian
# utils.run_optimization_experiment(
#     algorithm_fn=alg.modified_newton,
#     algorithm_name="MN",
#     function=fn.banded_trig,
#     grad_f=fn.grad_banded_trig,
#     hess_f=fn.hess_banded_trig,
#     dimensions=dimensions,
#     num_random_starts=num_random_starts,
#     tol=tol,
#     max_iter=max_iter,
#     base_dir=base,
#     finite_diff=True,
#     k_params=k_params,
# )

# ----------------------------
# CG Newton
# ----------------------------

# Analytic Hessian
utils.run_optimization_experiment(
    algorithm_fn=alg.newton_conjugate_gradient,
    algorithm_name="CGN",
    function=fn.banded_trig,
    grad_f=fn.grad_banded_trig,
    hess_f=fn.hess_banded_trig,
    dimensions=dimensions,
    num_random_starts=num_random_starts,
    tol=tol,
    max_iter=max_iter,
    base_dir=base,
    finite_diff=False,
)

# # Finite difference Hessian
# utils.run_optimization_experiment(
#     algorithm_fn=alg.newton_conjugate_gradient,
#     algorithm_name="CGN",
#     function=fn.banded_trig,
#     grad_f=fn.grad_banded_trig,
#     hess_f=fn.hess_banded_trig,
#     dimensions=dimensions,
#     num_random_starts=num_random_starts,
#     tol=tol,
#     max_iter=max_iter,
#     base_dir=base,
#     finite_diff=True,
#     k_params=k_params,
# )

plot.generate_pdf_tables_from_csvs(results_folder="results")