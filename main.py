import functions as fn
import plot
import algorithms as alg

result = alg.modified_newton(fn.banded_trig, fn.grad_banded_trig_test, fn.hess_banded_trig, [1, 1])

print(140*"-")
print("RESULTS")
print(f"minimum: {result["minimum"]} at point: {result["minimum_pt"]} was found in {result["num_iter"]} iterations.")

print(f"supposed minimum: {fn.banded_trig([-1.2, 0.5])} at point [-1.2, 0.5]")

plot.plot_convergence_rates(result["grad_norms"])

plot.plot_paths_2d(fn.banded_trig, result["visited_pt"])
