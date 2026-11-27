# ATP Tennis Network Analysis

We model ATP singles results from 2016–2025 as a directed network. Each node is a player, and an arrow goes from the player who lost to the player who beat them. The course dataset contains 1,191 players and 19,986 directed edges. Repeated results in the same direction count as one edge.

## Where to start

| File | What it shows |
| --- | --- |
| [Tasks 1 and 2 notebook](ATP_Tennis_Tasks_1_and_2.ipynb) | Builds the adjacency and transition matrices from the CSVs, shows labelled extracts and a worked row, and explains the graph structure, ranking prediction, and choice of starting players. |
| [Tasks 3 and 4 notebook](ATP_Tennis_Tasks_3_and_4.ipynb) | Runs power iteration from Djokovic and Bopanna, plots both runs, verifies the stationary distribution with an eigenvector, and works out two stationary equations by hand. |
| [Tasks 5 and 6 notebook](ATP_Tennis_Tasks_5_and_6.ipynb) | Compares the original walk, the lazy walk, and uniform PageRank. Includes the worked lazy row, matrix and stationarity checks, rankings, and Matplotlib figures. |
| [Tasks 3 and 4 report section](report_tasks_3_4.tex) | LaTeX source for the methods, figures, worked equations, and interpretation. |

The notebooks contain saved outputs for review on GitHub. We used small cells so imports, calculations, checks, and plots can be run separately. Assertion failures identify the check that failed.

## Run the notebooks in VS Code

Clone or download the full repository and open its root folder in VS Code with the Microsoft Python and Jupyter extensions installed. With Python 3.13 available, create the environment from the repository root. These commands are for macOS and Linux.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.5.5 pandas==3.1.1 matplotlib==3.11.2 networkx==3.7 scipy==1.18.1 ipykernel==7.4.0
```

On Windows, create the environment with `py -3.13 -m venv .venv` and use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

1. Open the Tasks 1 and 2 notebook. In the kernel picker, choose the Python interpreter inside the project's `.venv`.
2. Run the cells from the top with **Shift+Enter**, or use **Run All**. The notebook reads the course CSVs, rebuilds both matrices, and checks them against the saved archive.
3. Open the Tasks 3 and 4 notebook, select the same kernel, and run it from the top. It uses the checked-in matrix archive and imports the walk, plotting, and stationary-distribution functions from the Python scripts.
4. Open the Tasks 5 and 6 notebook and select the same `.venv` kernel. Use **Run All** from a fresh kernel, or run the cells in order with **Shift+Enter**. The notebook reads the checked-in `data/matrices.npz` and imports functions from the scripts in `src/`. You can also open the notebook on GitHub to review the saved outputs.

### Tasks 5 and 6 outputs

Running the notebook writes these files locally.

| Output | What it shows |
| --- | --- |
| `figures/task5_player_probabilities.png` | Representative players’ probabilities over time for both lazy-walk starts. |
| `figures/task5_convergence.png` | Original and lazy convergence histories for both starts. |
| `figures/task6_rankings.png` | Stationary probabilities for the PageRank top ten across all three models. |
| `figures/task6_convergence.png` | Convergence histories for the original walk, lazy walk, and PageRank. |
| `data/task6_pagerank_scores.csv` | Probabilities and ranks for all 1,191 players across the three models. |

## What the Python files do

| File | Purpose |
| --- | --- |
| [`src/task1_2.py`](src/task1_2.py) | Original group script that reads the tennis data and generates the saved matrices, player statistics, and network figures. |
| [`src/task1_2_helpers.py`](src/task1_2_helpers.py) | Functions called by the Tasks 1 and 2 notebook to rebuild the matrices, analyse components, and draw the worked example. |
| [`src/task3.py`](src/task3.py) | Loads the completed transition matrix, runs power iteration from both starting players, and generates the [probability histories](figures/task3_player_probabilities.png) and [convergence plot](figures/task3_convergence.png). |
| [`src/task4.py`](src/task4.py) | Computes and verifies the stationary distribution, expands the two worked equations, and checks the graph structure and eigenvalues. |
| [`src/task5.py`](src/task5.py) | Builds the lazy transition matrix, checks the worked row, and compares the original and lazy walks using the Task 3 iteration function. |
| [`src/task6.py`](src/task6.py) | Builds uniform PageRank from the original completed matrix, checks stationarity, and exports the rankings and figures. |

The scripts can also be run from the repository root.

```bash
.venv/bin/python src/task1_2.py
.venv/bin/python src/task3.py
.venv/bin/python src/task4.py
.venv/bin/python src/task5.py
.venv/bin/python src/task6.py
```

The first command regenerates `data/matrices.npz`, `data/player_statistics.csv`, and the original network figures. Task 3 saves its two figures in `figures/`. Task 4 prints its numerical checks in the terminal. The Tasks 5 and 6 scripts generate the outputs listed above.

## Main results

The original graph has 63 dangling players and 506 strongly connected components. The largest contains 681 players. The completed walk reaches the stopping tolerance after 28 updates from Novak Djokovic and 33 from Rohan Bopanna. Both runs agree with the eigenvector solution within numerical tolerance. Alexander Zverev has the highest stationary probability, approximately 1.1736%.

## AI assistance

We used OpenAI Codex to help implement the Python scripts and notebook helpers, including those for the random walk, lazy walk, and PageRank. It also helped draft explanations, create Matplotlib figures, and verify the numerical results.
