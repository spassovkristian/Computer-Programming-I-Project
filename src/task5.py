# TASK 5 - completed lazy random walk implementation
"""Task 5: compare the completed ATP walk with a lazy walk.

AI assistance: OpenAI Codex helped prepare code and explanations; the group
must review and explain them. Reuses the group's Task 3 power iteration.
"""

import argparse
from pathlib import Path
import numpy as np
from task3 import (MAX_ITERATIONS, PROJECT_DIR, TOLERANCE, load_network,
                   plot_player_probabilities, run_from_starting_players)


def check_transition(matrix):
    """Validate a nonempty, square, finite row-stochastic matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Transition matrix must be nonempty and square.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not np.allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("Every transition row must sum to one.")
    return matrix


def lazy_transition(transition, beta=0.5):
    """Build (1-beta)*P+beta*I without modifying the original P."""
    transition = check_transition(transition)
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be between zero and one.")
    lazy = (1 - beta) * transition + beta * np.eye(len(transition))
    check_transition(lazy)
    return lazy


def require_convergence(runs):
    """Never present an unconverged final iterate as a stationary ranking."""
    for name, result in runs.items():
        if not result.converged:
            raise RuntimeError(f"The run from {name} reached the iteration cap.")


def stationary_residual(distribution, transition):
    """Return ||pi*M-pi||_1."""
    return float(np.abs(distribution @ transition - distribution).sum())


def print_model_summary(label, matrix, runs):
    """Report stopping evidence, stationarity residuals, and start agreement."""
    print(f"\n{label}")
    print("Row sums equal one:", np.allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=0))
    print(f"Minimum matrix entry: {matrix.min():.9e}")
    for name, result in runs.items():
        print(f"  {name}: {result.iterations} updates; converged={result.converged}")
        print(f"    last L1 change={result.changes[-1]:.6e}; "
              f"stationary residual={stationary_residual(result.final, matrix):.6e}")
    first, second = runs.values()
    print(f"  Final vectors' L1 gap: {np.abs(first.final-second.final).sum():.6e}")


def plot_model_convergence(models, title):
    """Compare successive-vector L1 changes from identical initial states."""
    import matplotlib.pyplot as plt
    starts = list(next(iter(models.values())))
    colors = ("#2459A6", "#BA500A", "#237A58")
    fig, axes = plt.subplots(1, len(starts), figsize=(11, 4.3), sharey=True,
                             squeeze=False, layout="constrained")
    for ax, start in zip(axes[0], starts):
        for color, (label, runs) in zip(colors, models.items()):
            result = runs[start]
            ax.semilogy(np.arange(1, result.iterations + 1), result.changes,
                        label=f"{label}: {result.iterations} updates", color=color, lw=1.8)
        ax.axhline(TOLERANCE, color="#64748b", ls=":", label="Stopping threshold")
        ax.set(title=f"Start: {start}", xlabel="Update number")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("L1 change between successive vectors (log scale)")
    fig.suptitle(title)
    return fig


def print_worked_row(adjacency, transition, lazy, player_ids):
    """Revisit Martin Damm's nontrivial row actually worked out in Task 1."""
    row = player_ids["Martin Damm"]
    degree = int(adjacency[row].sum())
    neighbor = np.flatnonzero(adjacency[row])[0]
    print(f"\nWorked row: Martin Damm, ID {row}; out-degree={degree}")
    print(f"Each outgoing arrow: P[i,j]=1/{degree}; lazy P[i,j]=1/{2*degree}")
    print(f"P[i,i]={transition[row,row]:.9f}; lazy P[i,i]={lazy[row,row]:.9f}")
    print("In general the stay probability is 0.5+0.5*P[i,i], not always exactly 0.5.")
    print(f"Destination ID {neighbor}: {transition[row,neighbor]:.9f} -> {lazy[row,neighbor]:.9f}")
    print(f"Lazy row sum: {lazy[row].sum():.12f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_DIR / "figures")
    args = parser.parse_args()
    adjacency, transition, player_ids = load_network()
    lazy = lazy_transition(transition)
    original_runs = run_from_starting_players(transition, player_ids)
    lazy_runs = run_from_starting_players(lazy, player_ids)
    require_convergence(original_runs)
    require_convergence(lazy_runs)
    print(f"Same Task 3 rule: L1 change < {TOLERANCE}; cap={MAX_ITERATIONS}")
    print_worked_row(adjacency, transition, lazy, player_ids)
    print_model_summary("Original P", transition, original_runs)
    print_model_summary("Lazy P_0.5", lazy, lazy_runs)
    print("\nProof: pi*P_0.5 = 0.5*pi*P + 0.5*pi = pi.")
    for name in original_runs:
        original = original_runs[name].final
        modified = lazy_runs[name].final
        print(f"{name}: original/lazy L1 gap={np.abs(original-modified).sum():.6e}; "
              f"original candidate's lazy residual={stationary_residual(original,lazy):.6e}")
    # Dataset-specific structural evidence established in Task 4.
    from task4 import print_graph_evidence
    print_graph_evidence(adjacency, transition, player_ids)
    print("Laziness adds only self-loops, not links between separate closed classes.")
    print("An already aperiodic walk need not become faster; compare measured counts.")
    import matplotlib.pyplot as plt
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_model_convergence({"Original": original_runs, "Lazy": lazy_runs},
                                 "Task 5: original versus lazy walk")
    fig.savefig(args.figures_dir / "task5_convergence.png", dpi=180)
    plt.close(fig)
    fig = plot_player_probabilities(lazy_runs, player_ids)
    fig.suptitle("Task 5: lazy-walk probabilities from two starts")
    fig.savefig(args.figures_dir / "task5_player_probabilities.png", dpi=180)
    plt.close(fig)
    print(f"Saved Task 5 figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
