import os
import csv
import time
import numpy as np
import plot


def run_optimization_experiment(
    algorithm_fn,
    algorithm_name: str,
    function,
    grad_f,
    hess_f,
    dimensions: list[int],
    num_random_starts: int,
    tol: float,
    max_iter: int,
    base_dir: str,
    finite_diff: bool = False,
    k_params: list[int] | None = None,
):
    """
    Runs optimization experiments for multiple problem dimensions using a given algorithm.

    Parameters
    ----------
    algorithm_fn : callable
        The optimization algorithm (e.g., alg.modified_newton).
    algorithm_name : str
        Short name for the algorithm, used in output files (e.g., "MN", "CGN").
    function, grad_f, hess_f : callable
        Objective function and its derivatives.
    dimensions : list[int]
        List of problem sizes to test.
    num_random_starts : int
        Number of random initial points.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iteration count.
    base_dir : str
        Base results directory (expects prepared subfolders).
    finite_diff : bool, optional
        If True, runs finite difference variant with given k_params.
    k_params : list[int], optional
        Finite difference step parameters.
    """

    def get_base_point(n):
        return np.ones(n)

    base = "results"
    # ==========================================================
    # NORMAL (analytic Hessian) runs
    # ==========================================================
    if not finite_diff:
        for n in dimensions:
            print(f"\n{'='*80}\nRunning {algorithm_name} for n = {n}\n{'='*80}")

            xbar = get_base_point(n)
            starts = [xbar] + [xbar + np.random.uniform(-1, 1, size=n) for _ in range(num_random_starts)]
            csv_path = os.path.join(base_dir, "summary_CSV", f"summary_{algorithm_name}_n{n}.csv")

            results = []

            for i, x0 in enumerate(starts):
                start_id = "x̄" if i == 0 else f"rnd{i}"
                print(f"\n→ Starting run {start_id} (n={n})")

                start_time = time.perf_counter()
                res = algorithm_fn(function, grad_f, hess_f, x0, finite_diff=False, tol=tol, max_iter=max_iter)
                elapsed = time.perf_counter() - start_time

                results.append({
                    "start_id": start_id,
                    "grad.norm": f"{res['grad_norm']:.3e}",
                    "iters/max.iters": f"{res['num_iter']}/{max_iter}",
                    "success": "yes" if res["success"] else "no",
                    "flag": res["flag"],
                    "rate of conv.": f"{res['rate']:.2f}" if not np.isnan(res["rate"]) else "-",
                    "time": f"{elapsed:.2f}s",
                    "path": res["visited_pt"],
                    "convergence_norms": res["grad_norms"],
                })

                print(f"→ done in {elapsed:.2f}s | grad.norm = {res['grad_norm']:.3e}")

            # Write results
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

            print(f"✅ Saved results for n={n} → {csv_path}")

            # Plots
            convergence_paths = {}
            paths = {}
            for dict in results:
                convergence_paths[f"{dict["start_id"]}"] = dict["convergence_norms"]
                paths[f"{dict["start_id"]}"] = dict["path"]
            plot.plot_convergence_rates_nord(convergence_paths, filename=os.path.join(base, "plots/convergence", f"convergence_{algorithm_name}_n{n}.pdf"))
            # Optional plots for n=2 only
            if n == 2:
                plot.plot_paths_2d_nord(function, paths, filename=os.path.join(base, "plots/path", f"paths_{algorithm_name}_n{n}.pdf"))

    # ==========================================================
    # FINITE DIFFERENCE runs
    # ==========================================================
    else:
        if not k_params:
            raise ValueError("Finite difference mode requires k_params list.")

        for n in dimensions:
            combined_csv_path = os.path.join(base_dir, "summary_CSV", f"summary_{algorithm_name}_FD_n{n}.csv")

            # Remove old file if exists
            if os.path.exists(combined_csv_path):
                os.remove(combined_csv_path)

            combined_results = []

            for k in k_params:
                print(f"\n{'='*90}\nRunning {algorithm_name} (FD) for n = {n}, k = {k}\n{'='*90}")

                xbar = get_base_point(n)
                starts = [xbar] + [xbar + np.random.uniform(-1, 1, size=n) for _ in range(num_random_starts)]

                for j, x0 in enumerate(starts):
                    start_id = "x̄" if j == 0 else f"rnd{j}"
                    print(f"\n→ Starting run {start_id} (n={n}, k={k})")

                    start_time = time.perf_counter()
                    res = algorithm_fn(function, grad_f, hess_f, x0, finite_diff=True, k=k, tol=tol, max_iter=max_iter)
                    elapsed = time.perf_counter() - start_time

                    combined_results.append({
                        "start_id": start_id,
                        "k": k,
                        "grad.norm": f"{res['grad_norm']:.3e}",
                        "iters/max.iters": f"{res['num_iter']}/{max_iter}",
                        "success": "yes" if res["success"] else "no",
                        "flag": res["flag"],
                        "rate of conv.": f"{res['rate']:.2f}" if not np.isnan(res["rate"]) else "-",
                        "time": f"{elapsed:.2f}s",
                        "path": res["visited_pt"],
                        "convergence_norms": res["grad_norms"],
                    })

                    print(f"→ done in {elapsed:.2f}s | grad.norm = {res['grad_norm']:.3e}")

                # --- Generate plots for this k ---
                # Plots
                convergence_dict = {}
                paths_dict = {}
                for entry in combined_results:
                    if entry["k"] == k:
                        name = f"{entry['start_id']} (k={k})"
                        convergence_dict[name] = entry["convergence_norms"]
                        paths_dict[name] = entry["path"]
                plot.plot_convergence_rates_nord(
                    convergence_dict,
                    filename=os.path.join(base, "plots/convergence", f"convergence_{algorithm_name}_FD_k{k}_n{n}.pdf")
                )
                if n == 2:
                    plot.plot_paths_2d_nord(
                        function,
                        paths_dict,
                        filename=os.path.join(base, "plots/path", f"paths_{algorithm_name}_FD_k{k}_n{n}.pdf")
                    )

            # --- Write combined CSV ---
            if combined_results:
                results_for_csv = []
                for r in combined_results:
                    r_copy = r.copy()
                    del r_copy["path"]
                    del r_copy["convergence_norms"]
                    results_for_csv.append(r_copy)

                with open(combined_csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results_for_csv[0].keys())
                    writer.writeheader()
                    writer.writerows(results_for_csv)

                print(f"✅ Saved combined FD results for n={n} → {combined_csv_path}")
