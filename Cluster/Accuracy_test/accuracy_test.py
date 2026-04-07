import chess.engine
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import asyncio

dir_path = os.path.dirname(os.path.realpath(__file__))

stockfish_path = os.path.join(dir_path, '../Stockfish/src/stockfish')
leela_engine_path = os.path.join(dir_path, '../Leela_linux/build/release/lc0')
leela_net_path = os.path.join(dir_path, '../Leela_linux/build/release/weights/leela3400.pb.gz')
maia_engine_path = os.path.join(dir_path, '../Leela_linux/build/release/lc0')
maia_net_path = os.path.join(dir_path, '../Maia/maia_weights')
test_dataset_path = os.path.join(dir_path, '../Maia_dataset/maia-chess-testing-set.csv')

def load_positions_and_moves(csv_path, bins, per_bin=None, max_rows=None):
    """
    Read the CSV and collect positions per rating bin defined by `bins`.
    If per_bin is None -> include all games in each bin.
    Assumes dataset columns: board, move, white_elo, black_elo and that values are valid ints.
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
            w = int(row['white_elo'])
            b = int(row['black_elo'])
            rating = int((w + b) / 2) if (w and b) else (w or b)
            bin_idx = np.searchsorted(bins, rating, side='right') - 1
            if bin_idx < 0 or bin_idx >= n_bins:
                continue
            if per_bin is None or counts[bin_idx] < per_bin:
                positions_per_bin[bin_idx].append(row['board'])
                moves_per_bin[bin_idx].append(row['move'])
                ratings_per_bin[bin_idx].append(rating)
                counts[bin_idx] += 1
                total_collected += 1
                if total_needed is not None and total_collected >= total_needed:
                    break

    # flatten collected lists (keep bin ordering)
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
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = np.full(len(bin_centers), np.nan)
    human_moves = np.array(human_moves, dtype=object)
    engine_moves = np.array(engine_moves, dtype=object)

    for i in range(len(bin_centers)):
        lo, hi = bins[i], bins[i + 1]
        mask = (ratings >= lo) & (ratings < hi)
        if not np.any(mask):
            continue
        valid = mask & (human_moves != None)
        if not np.any(valid):
            continue
        correct = np.sum(engine_moves[valid] == human_moves[valid])
        accuracies[i] = 100.0 * correct / np.sum(valid)

    return bin_centers, accuracies

async def get_engine_moves_async(engine_path, positions, time_limit, maia_net_ELO=None):
    transport, engine = await chess.engine.popen_uci(engine_path)

    if engine_path == leela_engine_path:
        await engine.configure({"WeightsFile": leela_net_path})
    if maia_net_ELO:
        weights = os.path.join(maia_net_path, f"maia-{maia_net_ELO}.pb.gz")
        await engine.configure({"WeightsFile": weights})

    engine_moves = []
    try:
        for board_fen in positions:
            board = chess.Board(board_fen)
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

    maia_elos = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    maia_tasks = []
    for elo in maia_elos:
        print(f"Scheduling Maia {elo}...")
        task = asyncio.create_task(
            get_engine_moves_async(maia_engine_path, positions, time_limit=0.1, maia_net_ELO=elo)
        )
        maia_tasks.append((elo, task))

    # wait for Stockfish and lc0
    moves_s, moves_l = await asyncio.gather(stockfish_task, leela_task)
    print("Collecting Stockfish results...")
    x, y = accuracy_by_bins(moves_s, moves, ratings, bins)
    curves['Stockfish'] = (x, y)
    engine_order.append('Stockfish')

    print("Collecting lc0 results...")
    x, y = accuracy_by_bins(moves_l, moves, ratings, bins)
    curves['lc0'] = (x, y)
    engine_order.append('lc0')

    # wait for all Maia variants
    for elo, task in maia_tasks:
        name = f"Maia {elo}"
        print(f"Collecting {name} results...")
        moves_m = await task
        x, y = accuracy_by_bins(moves_m, moves, ratings, bins)
        curves[name] = (x, y)
        engine_order.append(name)

    return curves, engine_order

def save_table_csv(path, bin_centers, engine_names, matrix):
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
    # simple fixed-width table
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
    csv_path = test_dataset_path
    bins = np.arange(1000, 2501, 100)
    bin_centers = (bins[:-1] + bins[1:]) / 2

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

    # plotting (unchanged except using curves)
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
    asyncio.run(amain())

if __name__ == "__main__":
    main()
