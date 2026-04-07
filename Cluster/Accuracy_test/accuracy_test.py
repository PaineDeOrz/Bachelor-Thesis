"""
Evaluate multiple chess engines on a test set and compare their human-move
prediction accuracy across Elo bins.

Assumptions:
- This script is run from within the project so that __file__ is available.
- The following folders/files exist relative to this script:
  - ../Stockfish/src/stockfish
  - ../Leela_linux/build/release/lc0
  - ../Leela_linux/build/release/weights/leela3400.pb.gz
  - ../Leela_linux/build/release/lc0 (used as Maia engine binary path here)
  - ../Maia/maia_weights/maia-<ELO>.pb.gz
  - ../Maia_dataset/maia-chess-testing-set.csv
- The testing set CSV contains at least the columns:
  board, move, white_elo, black_elo
- Engine binaries are executable on the current system.
- The environment supports asyncio-based engine execution.
- The script may create accuracy_table.csv in the same folder as this file.
"""

import chess.engine
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import asyncio

# Base directory of this script.
dir_path = os.path.dirname(os.path.realpath(__file__))

# Relative paths to external engine binaries and model weights.
stockfish_path = os.path.join(dir_path, '../Stockfish/src/stockfish')
leela_engine_path = os.path.join(dir_path, '../Leela_linux/build/release/lc0')
leela_net_path = os.path.join(dir_path, '../Leela_linux/build/release/weights/leela3400.pb.gz')
maia_engine_path = os.path.join(dir_path, '../Leela_linux/build/release/lc0')
maia_net_path = os.path.join(dir_path, '../Maia/maia_weights')
test_dataset_path = os.path.join(dir_path, '../Maia_dataset/maia-chess-testing-set.csv')

def load_positions_and_moves(csv_path, bins, per_bin=None, max_rows=None):
    """
    Load board positions and human moves from a CSV file and group them by Elo bin.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the test positions.
    bins : array-like
        Elo bin boundaries, e.g. np.arange(1000, 2501, 100).
    per_bin : int or None
        Maximum number of samples to collect per bin. If None, all samples are used.
    max_rows : int or None
        Optional upper limit on the number of CSV rows to read.

    Returns
    -------
    positions : list[str]
        Board positions in FEN format, flattened in bin order.
    moves : list[str]
        Human move labels in UCI format, flattened in bin order.
    ratings : np.ndarray
        Integer rating assigned to each collected sample.
    counts : np.ndarray
        Number of collected samples per bin.
    """
    n_bins = len(bins) - 1
    positions_per_bin = [[] for _ in range(n_bins)]
    moves_per_bin = [[] for _ in range(n_bins)]
    ratings_per_bin = [[] for _ in range(n_bins)]
    counts = np.zeros(n_bins, dtype=int)
    total_needed = None if per_bin is None else (n_bins * per_bin)
    total_collected = 0

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            # Use the average of white and black Elo as the representative rating.
            w = int(row['white_elo'])
            b = int(row['black_elo'])
            rating = int((w + b) / 2) if (w and b) else (w or b)

            # Assign the position to the corresponding Elo bin.
            bin_idx = np.searchsorted(bins, rating, side='right') - 1
            if bin_idx < 0 or bin_idx >= n_bins:
                continue

            # Collect up to per_bin samples per bin, if a limit is requested.
            if per_bin is None or counts[bin_idx] < per_bin:
                positions_per_bin[bin_idx].append(row['board'])
                moves_per_bin[bin_idx].append(row['move'])
                ratings_per_bin[bin_idx].append(rating)
                counts[bin_idx] += 1
                total_collected += 1

                # Stop once the requested total number of samples has been collected.
                if total_needed is not None and total_collected >= total_needed:
                    break

    # Flatten the per-bin lists while preserving bin order.
    positions = []
    moves = []
    ratings = []
    for idx in range(n_bins):
        positions.extend(positions_per_bin[idx])
        moves.extend(moves_per_bin[idx])
        ratings.extend(ratings_per_bin[idx])

    counts_summary = ", ".join(str(int(c)) for c in counts)
    print(f"Collected per-bin counts (left-to-right bins): {counts_summary}")
    return positions, moves, np.array(ratings, dtype=int), counts

def accuracy_by_bins(engine_moves, human_moves, ratings, bins):
    """
    Compute top-1 move prediction accuracy per Elo bin.

    Parameters
    ----------
    engine_moves : array-like
        Engine-predicted moves in UCI notation.
    human_moves : array-like
        Human reference moves in UCI notation.
    ratings : array-like
        Rating associated with each position.
    bins : array-like
        Elo bin boundaries.

    Returns
    -------
    bin_centers : np.ndarray
        Center value of each bin.
    accuracies : np.ndarray
        Accuracy in percent for each bin; NaN for bins with no valid samples.
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = np.full(len(bin_centers), np.nan)
    human_moves = np.array(human_moves, dtype=object)
    engine_moves = np.array(engine_moves, dtype=object)

    for i in range(len(bin_centers)):
        lo, hi = bins[i], bins[i + 1]
        mask = (ratings >= lo) & (ratings < hi)
        if not np.any(mask):
            continue

        # Ignore missing human moves.
        valid = mask & (human_moves != None)
        if not np.any(valid):
            continue

        correct = np.sum(engine_moves[valid] == human_moves[valid])
        accuracies[i] = 100.0 * correct / np.sum(valid)

    return bin_centers, accuracies

async def get_engine_moves_async(engine_path, positions, time_limit, maia_net_ELO=None):
    """
    Evaluate one engine on all given positions asynchronously.

    Parameters
    ----------
    engine_path : str
        Path to the UCI engine binary.
    positions : list[str]
        Board positions in FEN format.
    time_limit : float
        Search time limit per position, used for Stockfish and lc0.
    maia_net_ELO : int or None
        If provided, load the matching Maia network weights and use a
        one-node search configuration instead of a time-based limit.

    Returns
    -------
    np.ndarray
        Predicted engine moves in UCI notation, one per input position.
    """
    transport, engine = await chess.engine.popen_uci(engine_path)

    # Load the correct network weights depending on the engine variant.
    if engine_path == leela_engine_path:
        await engine.configure({"WeightsFile": leela_net_path})
    if maia_net_ELO:
        weights = os.path.join(maia_net_path, f"maia-{maia_net_ELO}.pb.gz")
        await engine.configure({"WeightsFile": weights})

    engine_moves = []
    try:
        for board_fen in positions:
            board = chess.Board(board_fen)

            # Maia is evaluated with a fixed node budget to mimic the original setup.
            if maia_net_ELO:
                limit = chess.engine.Limit(nodes=1)
            else:
                limit = chess.engine.Limit(time=time_limit)

            res = await engine.play(board, limit)
            engine_moves.append(res.move.uci() if res.move is not None else None)
    finally:
        await engine.quit()

    return np.array(engine_moves, dtype=object)

async def evaluate_all_engines_async(positions, moves, ratings, bins):
    """
    Run all engine evaluations, collect predictions, and compute accuracy curves.

    The engines are scheduled concurrently so that the total runtime is reduced.
    Stockfish and lc0 are evaluated with a time limit, while Maia variants are
    loaded separately for each target Elo.

    Returns
    -------
    curves : dict
        Mapping from engine name to (x, y) accuracy curve data.
    engine_order : list[str]
        Engine names in the order they should appear in tables and plots.
    """
    curves = {}
    engine_order = []

    print("Scheduling Stockfish...")
    stockfish_task = asyncio.create_task(
        get_engine_moves_async(stockfish_path, positions, time_limit=0.5)
    )

    print("Scheduling lc0...")
    leela_task = asyncio.create_task(
        get_engine_moves_async(leela_engine_path, positions, time_limit=0.5)
    )

    # Evaluate multiple Maia skill levels to compare rating-aware behavior.
    maia_elos = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    maia_tasks = []
    for elo in maia_elos:
        print(f"Scheduling Maia {elo}...")
        task = asyncio.create_task(
            get_engine_moves_async(maia_engine_path, positions, time_limit=0.1, maia_net_ELO=elo)
        )
        maia_tasks.append((elo, task))

    # Wait for Stockfish and lc0 first because those tasks are started together.
    moves_s, moves_l = await asyncio.gather(stockfish_task, leela_task)

    print("Collecting Stockfish results...")
    x, y = accuracy_by_bins(moves_s, moves, ratings, bins)
    curves['Stockfish'] = (x, y)
    engine_order.append('Stockfish')

    print("Collecting lc0 results...")
    x, y = accuracy_by_bins(moves_l, moves, ratings, bins)
    curves['lc0'] = (x, y)
    engine_order.append('lc0')

    # Collect Maia results one Elo variant at a time in the same order as scheduled.
    for elo, task in maia_tasks:
        name = f"Maia {elo}"
        print(f"Collecting {name} results...")
        moves_m = await task
        x, y = accuracy_by_bins(moves_m, moves, ratings, bins)
        curves[name] = (x, y)
        engine_order.append(name)

    return curves, engine_order

def save_table_csv(path, bin_centers, engine_names, matrix):
    """
    Save the accuracy matrix to CSV in a readable table format.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Rating'] + engine_names
        writer.writerow(header)
        for i, rc in enumerate(bin_centers):
            row = [int(rc)]
            for val in matrix[i]:
                row.append("" if np.isnan(val) else f"{val:.2f}")
            writer.writerow(row)
    print("Saved CSV:", path)

def print_table_console(bin_centers, engine_names, matrix):
    """
    Print the accuracy table in a fixed-width console layout.
    """
    col_width = 12
    header = "Rating".ljust(col_width) + "".join(name[:col_width-1].rjust(col_width) for name in engine_names)
    print(header)
    for i, rc in enumerate(bin_centers):
        row = str(int(rc)).ljust(col_width)
        for val in matrix[i]:
            if np.isnan(val):
                row += "-".rjust(col_width)
            else:
                row += f"{val:6.2f}".rjust(col_width)
        print(row)

async def amain():
    """
    Main asynchronous workflow:
    1. Load the testing positions.
    2. Run all engines on the same set of positions.
    3. Compute accuracy tables.
    4. Save the table to CSV.
    5. Plot the accuracy curves.
    """
    csv_path = test_dataset_path
    bins = np.arange(1000, 2501, 100)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Load a balanced sample from each Elo bin.
    positions, moves, ratings, counts = load_positions_and_moves(
        csv_path, bins, per_bin=500, max_rows=None
    )

    print("Ratings range:", ratings.min(), "-", ratings.max(), " (n=", len(ratings), ")")

    curves, engine_order = await evaluate_all_engines_async(
        positions, moves, ratings, bins
    )

    n_bins = len(bin_centers)
    n_engines = len(engine_order)
    matrix = np.full((n_bins, n_engines), np.nan)
    for j, name in enumerate(engine_order):
        _, yvals = curves[name]
        matrix[:, j] = yvals

    print("\nAccuracy table (percentage) - rows: rating bins, columns: engines")
    print_table_console(bin_centers, engine_order, matrix)

    csv_out = os.path.join(dir_path, 'accuracy_table.csv')
    save_table_csv(csv_out, bin_centers, engine_order, matrix)

    # Plot the accuracy curves. Missing bins are interpolated only for visualization.
    plt.figure(figsize=(10, 6))
    for label in engine_order:
        xvals = bin_centers
        yvals = np.array(curves[label][1], dtype=float)
        valid = ~np.isnan(yvals)
        if np.sum(valid) == 0:
            continue
        if np.sum(valid) == 1:
            plt.plot(xvals[valid], yvals[valid], marker='o', label=label, linewidth=2)
        else:
            y_interp = np.interp(xvals, xvals[valid], yvals[valid])
            plt.plot(xvals, y_interp, label=label, linewidth=2)

    plt.xlabel('Rating')
    plt.ylabel('Accuracy (%)')
    plt.xlim(1000, 2500)
    plt.ylim(20, 80)
    plt.gca().ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.xticks(np.arange(1000, 2501, 100))
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """
    Synchronous entry point that runs the asynchronous workflow.
    """
    asyncio.run(amain())

if __name__ == "__main__":
    main()
