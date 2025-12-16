import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_convergence_rates_nord(runs_dict, filename="convergence_rates.pdf"):
    """
    Plot multiple convergence curves (gradient norms vs iterations) in the Nord color scheme.

    Parameters
    ----------
    runs_dict : dict
        Dictionary of { 'Run Name': [grad_norm_list], ... }
    filename : str
        PDF output filename.
    """

    nord_colors = [
        "#607879", "#633874", "#4888C7", "#7BC778",
        "#D64F4F", "#D46ABD", "#EBCB8B", "#A3BE8C"
    ]

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (name, norms) in enumerate(runs_dict.items()):
        norms = np.asarray(norms)
        iterations = np.arange(1, len(norms) + 1)
        color = nord_colors[i % len(nord_colors)]

        # Line + dots on each iteration
        ax.plot(iterations, norms, color=color, lw=1.8, marker='o', markersize=4,
                label=name, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("Iteration", fontsize=11, labelpad=6)
    ax.set_ylabel("‖∇f(x)‖", fontsize=11, labelpad=6)

    # Minimal Nord-style grid
    # ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=9, loc="best")

    # Nord background
    fig.patch.set_facecolor("#ECEFF4")  # nord6
    ax.set_facecolor("#ECEFF4")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_paths_2d_nord(F, runs_dict, filename="paths.pdf",
                       x_range=(-3, 3), y_range=(-3, 3), levels=200):
    """
    Plot filled contour of F(x) with multiple optimization paths (colored by Nord palette).

    Parameters
    ----------
    F : callable
        Function F(x) taking a 1D array of length 2.
    runs_dict : dict
        Dictionary of { 'Run Name': [path_array], ... }.
        Each path_array should be an array of shape (n_iters, 2).
    filename : str
        PDF output filename.
    """

    nord_colors = [
        "#607879", "#633874", "#4888C7", "#7BC778",
        "#D64F4F", "#D46ABD", "#EBCB8B", "#A3BE8C"
    ]

    # Meshgrid for contour
    X, Y = np.meshgrid(np.linspace(x_range[0], x_range[1], 400),
                       np.linspace(y_range[0], y_range[1], 400))
    Z = np.zeros_like(X)

    # Vectorized evaluation for speed
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = F(np.array([X[i, j], Y[i, j]]))

    # Base figure
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="Wistia", alpha=0.50)
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label("F(x)", rotation=270, labelpad=12)

    # Plot optimization paths
    for i, (name, path) in enumerate(runs_dict.items()):
        path = np.atleast_2d(np.array(path))
        if path.shape[1] < 2:
            raise ValueError(f"Path '{name}' has fewer than 2 variables; cannot plot in 2D")

        color = nord_colors[i % len(nord_colors)]
        # ax.plot(path[:, 0], path[:, 1], color=color, lw=1.3, label=name, alpha=0.9)
        ax.scatter(path[:, 0], path[:, 1], color=color, s=30, zorder=3, label=name)

    # Axis labels & style
    ax.set_xlabel("x₁", fontsize=11)
    ax.set_ylabel("x₂", fontsize=11)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_facecolor("#ECEFF4")
    fig.patch.set_facecolor("#ECEFF4")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    

def generate_pdf_tables_from_csvs(results_folder="results"):
    """
    Reads all summary_*.csv files in 'results_folder',
    computes average rows for successful runs.
    - If a 'k' column exists (finite difference results), compute one average row per k.
    - Otherwise, compute one global average for the entire table.
    Creates Nord-styled PDF tables saved to 'results/plots/'.
    """
    plots_folder = os.path.join(results_folder, "plots/table")
    csv_folder = os.path.join(results_folder, "summary_CSV")
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    csv_files = sorted(
        [f for f in os.listdir(csv_folder) if f.startswith("summary_") and f.endswith(".csv")]
    )
    if not csv_files:
        print("⚠️ No summary_*.csv files found in", csv_folder)
        return

    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)

        avg_rows = []

        # ------------------------------------------------------------------
        # If the file has a 'k' column (finite differences) → average per k
        # ------------------------------------------------------------------
        if "k" in df.columns:
            for k_val in sorted(df["k"].unique()):
                df_k = df[(df["success"] == "yes") & (df["k"] == k_val)]
                if len(df_k) == 0:
                    continue

                grad_vals = pd.to_numeric(df_k["grad.norm"], errors="coerce")
                rate_vals = pd.to_numeric(df_k["rate of conv."], errors="coerce")
                time_vals = df_k["time"].str.replace("s", "").astype(float)
                iters = df_k["iters/max.iters"].apply(lambda x: int(x.split("/")[0]))

                avg_row = {
                    "start_id": f"Avg (k={int(k_val)})",
                    "k": k_val,
                    "grad.norm": f"{grad_vals.mean():.3e}",
                    "iters/max.iters": f"{int(np.mean(iters))}/{df['iters/max.iters'].iloc[0].split('/')[1]}",
                    "success": "-",
                    "flag": "-",
                    "rate of conv.": f"{rate_vals.mean():.2f}" if not np.isnan(rate_vals.mean()) else "-",
                    "time": f"{time_vals.mean():.2f}s",
                }
                avg_rows.append(avg_row)

        # ------------------------------------------------------------------
        # Otherwise → compute single average for entire table
        # ------------------------------------------------------------------
        else:
            drop_cols = ["path", "convergence_norms"]
            for col in drop_cols:
                if col in df.columns:
                    df.drop(columns=col, inplace=True)
            df_success = df[df["success"] == "yes"]
            if len(df_success) > 0:
                grad_vals = pd.to_numeric(df_success["grad.norm"], errors="coerce")
                rate_vals = pd.to_numeric(df_success["rate of conv."], errors="coerce")
                time_vals = df_success["time"].str.replace("s", "").astype(float)
                iters = df_success["iters/max.iters"].apply(lambda x: int(x.split("/")[0]))

                avg_row = {
                    "start_id": "Avg (successes)",
                    "grad.norm": f"{grad_vals.mean():.3e}",
                    "iters/max.iters": f"{int(np.mean(iters))}/{df['iters/max.iters'].iloc[0].split('/')[1]}",
                    "success": "-",
                    "flag": "-",
                    "rate of conv.": f"{rate_vals.mean():.2f}" if not np.isnan(rate_vals.mean()) else "-",
                    "time": f"{time_vals.mean():.2f}s",
                }
                avg_rows.append(avg_row)

        # Append computed average rows
        if avg_rows:
            df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)

        # ------------------------------------------------------------------
        # Build the prettified Nord-style PDF table
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, len(df) * 0.5 + 1))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.3)

        # ----------------------------
        # Style adjustments (Nord-like)
        # ----------------------------
        for (row, col), cell in table.get_celld().items():
            # Header
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#507194")  # dark blue-gray
            # Alternate row shading
            elif row % 2 == 0:
                cell.set_facecolor("#F2F2F2")
            else:
                cell.set_facecolor("white")
            # Average rows highlight

            df_index = row - 1  # adjust because table rows start at 1 (header not included)
            if 0 <= df_index < len(df):
                start_label = str(df.iloc[df_index].get("start_id", ""))
                if start_label.startswith("Avg"):
                    cell.set_text_props(weight="bold")
                    cell.set_facecolor("#D5D8DC")

        # ----------------------------
        # Save as PDF
        # ----------------------------
        pdf_name = csv_file.replace(".csv", ".pdf")
        pdf_path = os.path.join(plots_folder, pdf_name)
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved prettified table: {pdf_path}")

    print(f"\nAll PDF tables exported to '{plots_folder}/'")