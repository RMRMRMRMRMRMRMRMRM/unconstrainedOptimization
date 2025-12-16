import os
import csv
import time
import numpy as np
import algorithms as alg
import functions as fn
import plot  # your plotting module

# ----------------------------
# Setup and configuration
# ----------------------------
SEED = 1234  # min student ID
np.random.seed(SEED)

dimensions = [2]
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

# ----------------------------
# Experiment loop
# ----------------------------
for n in dimensions:
    print(f"\n{'='*80}\nRunning Modified Newton for dimension n = {n}\n{'='*80}")

    xbar = get_base_point(n)
    starts = [xbar] + [xbar + np.random.uniform(-1, 1, size=n) for _ in range(num_random_starts)]
    csv_path = os.path.join(base, "summary_CSV", f"summary_MN_n{n}.csv")

    results = []

    for i, x0 in enumerate(starts):
        start_id = "x̄" if i == 0 else f"rnd{i}"
        print(f"\n→ Starting run {start_id} (n={n})")

        start_time = time.perf_counter()
        res = alg.modified_newton(fn.banded_trig,
                                  fn.grad_banded_trig,
                                  fn.hess_banded_trig,
                                  x0,
                                  finite_diff = False,
                                  tol=tol,
                                  max_iter=max_iter)
        elapsed = time.perf_counter() - start_time

        results.append({
            "start_id": start_id,
            "grad.norm": f"{res['grad_norm']:.3e}",
            "iters/max.iters": f"{res['num_iter']}/{max_iter}",
            "success": "yes" if res["success"] else "no",
            "flag": res["flag"],
            "rate of conv.": f"{res['rate']:.2f}" if not np.isnan(res['rate']) else "-",
            "time": f"{elapsed:.2f}s",
            "path": res["visited_pt"],
            "convergence_norms": res["grad_norms"]
        })

        print(f"→ done in {elapsed:.2f}s | grad.norm = {res['grad_norm']:.3e} | iters = {res['num_iter']}")

    # ----------------------------
    # Write summary CSV
    # ----------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results for n={n} to {csv_path}")

    # Optional plots for n=2 only
    if n == 2:
        convergence_paths = {}
        paths = {}

        for dict in results:
            convergence_paths[f"{dict["start_id"]}"] = dict["convergence_norms"]
            paths[f"{dict["start_id"]}"] = dict["path"]

        plot.plot_convergence_rates_nord(convergence_paths, filename=os.path.join(base, "plots/convergence", f"convergence_MN_n{n}.pdf"))
        plot.plot_paths_2d_nord(fn.banded_trig, paths, filename=os.path.join(base, "plots/path", f"paths_MN_n{n}.pdf"))


for n in dimensions:
    # Combined results CSV for all k for this dimension
    combined_csv_path = os.path.join(base, "summary_CSV", f"summary_MN_FD_n{n}.csv")

    # Always start fresh — overwrite existing combined file
    if os.path.exists(combined_csv_path):
        os.remove(combined_csv_path)

    combined_results = []

    for k in k_params:
        print(f"\n{'='*90}\nRunning Modified Newton (FD) for n = {n}, k = {k}\n{'='*90}")

        xbar = get_base_point(n)
        starts = [xbar] + [xbar + np.random.uniform(-1, 1, size=n) for _ in range(num_random_starts)]

        for j, x0 in enumerate(starts):
            start_id = "x̄" if j == 0 else f"rnd{j}"
            print(f"\n→ Starting run {start_id} (n={n}, k={k})")

            start_time = time.perf_counter()
            res = alg.modified_newton(
                fn.banded_trig,
                fn.grad_banded_trig,
                fn.hess_banded_trig,
                x0,
                finite_diff=True,
                k=k,
                tol=tol,
                max_iter=max_iter,
            )
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
                # keep arrays for plotting only
                "path": res["visited_pt"],
                "convergence_norms": res["grad_norms"],
            })

            print(f"→ done in {elapsed:.2f}s | grad.norm = {res['grad_norm']:.3e} | iters = {res['num_iter']}")

        # -------------------------------------------------------------------
        # Generate plots for n=2 (visual only)
        # -------------------------------------------------------------------
        if n == 2:
            convergence_dict = {}
            paths_dict = {}
            for entry in combined_results:
                if entry["k"] == k:
                    name = f"{entry['start_id']} (k={k})"
                    convergence_dict[name] = entry["convergence_norms"]
                    paths_dict[name] = entry["path"]

            plot.plot_convergence_rates_nord(
                convergence_dict,
                filename=os.path.join(base, "plots/convergence", f"convergence_MN_FD_k{k}_n{n}.pdf")
            )
            plot.plot_paths_2d_nord(
                fn.banded_trig,
                paths_dict,
                filename=os.path.join(base, "plots/path", f"paths_MN_FD_k{k}_n{n}.pdf")
            )

    # -------------------------------------------------------------------
    # Write combined CSV (overwrite every time)
    # -------------------------------------------------------------------
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

        print(f"\n✅ Saved combined results for n={n} → {combined_csv_path}")

plot.generate_pdf_tables_from_csvs(results_folder="results")