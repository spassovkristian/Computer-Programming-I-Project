"""Task 6: PageRank with uniform teleportation, built from original completed P.

AI assistance: OpenAI Codex; review and understand before use. No ready-made
PageRank function is used. Task 7's alpha sweep is outside these two tasks.
"""

import argparse
import csv
from pathlib import Path
import numpy as np
from task3 import PROJECT_DIR, load_network, run_from_starting_players
from task5 import (check_transition, lazy_transition, plot_model_convergence,
                   print_model_summary, require_convergence, stationary_residual)


def pagerank_transition(transition, alpha=0.85):
    """Return alpha*P+(1-alpha)*1*v, where v is uniform over all nodes."""
    transition = check_transition(transition)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one.")
    n = len(transition)
    uniform = np.ones(n) / n
    # Broadcasting this row over P constructs the rank-one matrix 1*v.
    google = alpha * transition + (1 - alpha) * uniform
    check_transition(google)
    if np.any(google < (1 - alpha) / n):
        raise ValueError("PageRank violates its strictly positive lower bound.")
    return google


def ranked_indices(scores):
    """Descending score, with node ID breaking exact ties."""
    return np.argsort(-scores, kind="stable")


def save_scores(path, original, lazy, pagerank, names_by_id):
    """Save all players' probabilities and ranks, ordered by PageRank."""
    ranks = []
    for scores in (original, lazy, pagerank):
        rank = np.empty(len(scores), dtype=int)
        rank[ranked_indices(scores)] = np.arange(1, len(scores) + 1)
        ranks.append(rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "player", "original_probability", "lazy_probability",
                         "pagerank_probability", "original_rank", "lazy_rank", "pagerank_rank"])
        for i in ranked_indices(pagerank):
            writer.writerow([int(i), names_by_id[i], f"{original[i]:.16g}",
                             f"{lazy[i]:.16g}", f"{pagerank[i]:.16g}",
                             *(int(rank[i]) for rank in ranks)])


def plot_rankings(original, lazy, pagerank, names_by_id, count=10):
    """Compare models on the PageRank top ten; all nodes enter the calculation."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    selected = ranked_indices(pagerank)[:count]
    positions = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
    for offset, scores, label, color in (
        (-0.25, original, "Original P", "#2459A6"),
        (0, lazy, "Lazy P (beta=0.5)", "#BA500A"),
        (0.25, pagerank, "PageRank (alpha=0.85)", "#237A58"),
    ):
        ax.barh(positions + offset, scores[selected], height=0.24, label=label, color=color)
    ax.set_yticks(positions, [names_by_id[i] for i in selected])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Stationary probability", title="Task 6: PageRank top ten across three models")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_DIR / "figures")
    parser.add_argument("--scores-file", type=Path,
                        default=PROJECT_DIR / "data" / "task6_pagerank_scores.csv")
    args = parser.parse_args()
    _, transition, player_ids = load_network()
    names_by_id = {index: name for name, index in player_ids.items()}
    # Crucially use original P, not P_0.5.
    google = pagerank_transition(transition, alpha=0.85)
    original_runs = run_from_starting_players(transition, player_ids)
    lazy_runs = run_from_starting_players(lazy_transition(transition), player_ids)
    pagerank_runs = run_from_starting_players(google, player_ids)
    for runs in (original_runs, lazy_runs, pagerank_runs):
        require_convergence(runs)
    print("G = 0.85*P + 0.15*1*v; uniform v; original completed P.")
    print(f"Every G entry >= 0.15/n = {0.15/len(google):.12e} > 0")
    print_model_summary("PageRank G_0.85", google, pagerank_runs)
    for name, result in pagerank_runs.items():
        if stationary_residual(result.final, google) >= 1e-9:
            raise RuntimeError(f"PageRank from {name} failed stationarity verification.")
    results = list(pagerank_runs.values())
    if np.abs(results[0].final-results[1].final).sum() >= 1e-9:
        raise RuntimeError("PageRank runs disagree beyond numerical tolerance.")
    original = next(iter(original_runs.values())).final
    lazy = next(iter(lazy_runs.values())).final
    pagerank = results[0].final
    print(f"\nTop ten PageRank players (computed on all {len(pagerank)} nodes):")
    for rank, i in enumerate(ranked_indices(pagerank)[:10], start=1):
        print(f"{rank:2}. {names_by_id[i]:25} {pagerank[i]:.8%}")
    print(f"\nPageRank/original L1 difference: {np.abs(pagerank-original).sum():.6e}")
    print("Positive stochastic G has a unique positive stationary distribution and")
    print("converges from every initial distribution. Two runs support, not prove, this theorem.")
    print("Teleportation reaches every player; self-loops alone cannot connect closed classes.")
    save_scores(args.scores_file, original, lazy, pagerank, names_by_id)
    import matplotlib.pyplot as plt
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "task6_convergence.png": plot_model_convergence(
            {"Original": original_runs, "Lazy": lazy_runs, "PageRank": pagerank_runs},
            "Task 6: convergence of the three models"),
        "task6_rankings.png": plot_rankings(original, lazy, pagerank, names_by_id),
    }
    for filename, fig in figures.items():
        fig.savefig(args.figures_dir / filename, dpi=180)
        plt.close(fig)
    print(f"All-node probabilities and ranks: {args.scores_file}")
    print(f"Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
