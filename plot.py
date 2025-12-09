import numpy as np
import matplotlib.pyplot as plt



def plot_convergence_rates(grad_norms, filename='convergence_rates.pdf'):
    """
    grad_norms: 1D list or array of gradient norms for a single optimization path
    filename: output PDF filename
    """
    rates = [grad_norms[k+1]/grad_norms[k] for k in range(len(grad_norms)-1)]
    
    plt.figure(figsize=(8,6))
    plt.plot(rates, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Rate')
    plt.yscale('log')
    plt.title('Convergence Rate')
    plt.grid(True)
    
    plt.savefig(filename, bbox_inches='tight')  # save to PDF in current folder
    plt.close()

def plot_paths_2d(F, paths, filename='paths.pdf', x_range=(-2, 2), y_range=(-2, 2), levels=200):
    """
    Plots a filled contour of F(x) with optimization paths as thin black lines
    and black points connecting the sequence of iterates.
    """

    # Create meshgrid for the contour
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 500),
                       np.linspace(y_range[0], y_range[1], 500))
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = F(np.array([X[i, j], Y[i, j]]))

    plt.figure(figsize=(8, 6))

    # Filled contour background
    plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
    plt.colorbar(label='F(x)')

    # Overlay optimization paths
    for path in paths:
        path = np.atleast_2d(np.array(path))  # ensure 2D
        if path.shape[1] < 2:
            raise ValueError("Path has fewer than 2 variables; cannot plot in 2D")
        # draw thin black line + black points
        plt.plot(path[:, 0], path[:, 1], color='black', linewidth=0.8, alpha=0.9)
        plt.scatter(path[:, 0], path[:, 1], color='black', s=10, zorder=3)

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Optimization Path on F(x)')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(filename, bbox_inches='tight')
    plt.close()