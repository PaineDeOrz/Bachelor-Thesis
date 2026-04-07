#!/usr/bin/env python3
"""
Stockfish Skill Level Accuracy Test
Measures Stockfish accuracy on human moves from Maia CSV dataset across skill levels.
"""

import asyncio
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

# USER CONFIGURATION - EDIT THESE PATHS
STOCKFISH_PATH = Path(os.path.join(dir_path, '../../../../Stockfish/src/stockfish'))# ? CHANGE THIS
DATASET_PATH = Path(os.path.join(dir_path, '../../../../Maia_dataset/maia-chess-testing-set.csv'))# ? CHANGE THIS

def load_positions_and_moves(csv_path, bins, per_bin=500, max_rows=None):
    """Load positions, moves, and ratings from Maia CSV dataset."""
    n_bins = len(bins) - 1
    positions_per_bin = [[] for _ in range(n_bins)]
    moves_per_bin = [[] for _ in range(n_bins)]
    ratings_per_bin = [[] for _ in range(n_bins)]
    counts = np.zeros(n_bins, dtype=int)
    total_needed = n_bins * per_bin
    total_collected = 0

    print(f"Loading dataset from {csv_path}...")
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
                
            # Use average Elo (same logic as your original)
            w_elo = int(row.get('white_elo', 0))
            b_elo = int(row.get('black_elo', 0))
            rating = int((w_elo + b_elo) / 2) if (w_elo and b_elo) else max(w_elo, b_elo)
            
            bin_idx = np.searchsorted(bins, rating, side='right') - 1
            if 0 <= bin_idx < n_bins and counts[bin_idx] < per_bin:
                positions_per_bin[bin_idx].append(row['board'])
                moves_per_bin[bin_idx].append(row['move'])
                ratings_per_bin[bin_idx].append(rating)
                counts[bin_idx] += 1
                total_collected += 1
                
                if total_collected >= total_needed:
                    break

    # Flatten while preserving bin order
    positions = []
    moves = []
    ratings = []
    for idx in range(n_bins):
        positions.extend(positions_per_bin[idx])
        moves.extend(moves_per_bin[idx])
        ratings.extend(ratings_per_bin[idx])

    print(f"Loaded {len(positions)} positions across {len(bins)-1} bins")
    print(f"Rating range: {min(ratings):d}-{max(ratings):d}")
    return positions, moves, np.array(ratings), counts

def accuracy_by_bins(engine_moves, human_moves, ratings, bins):
    """Compute accuracy per rating bin."""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    accuracies = np.full(len(bin_centers), np.nan)
    
    human_moves = np.array(human_moves, dtype=object)
    engine_moves = np.array(engine_moves, dtype=object)

    for i in range(len(bin_centers)):
        mask = (ratings >= bins[i]) & (ratings < bins[i + 1])
        if not np.any(mask):
            continue
        valid_mask = mask & (human_moves != None) & (engine_moves != None)
        if not np.any(valid_mask):
            continue
        correct = np.sum(engine_moves[valid_mask] == human_moves[valid_mask])
        accuracies[i] = 100.0 * correct / np.sum(valid_mask)

    return bin_centers, accuracies

async def get_stockfish_moves(engine_path, positions, skill_level, max_depth=None, time_limit=0.1):
    """Get moves from Stockfish at specific skill level."""
    transport, engine = await chess.engine.popen_uci(engine_path)
    
    try:
        # Configure skill level (0=weakest ~800 Elo, 20=full strength ~3400 Elo)
        if skill_level is not None:
            await engine.configure({"Skill Level": skill_level})
        
        # Optional: limit depth for additional weakening
        if max_depth:
            limit = chess.engine.Limit(depth=max_depth)
        else:
            limit = chess.engine.Limit(time=time_limit)
            
        engine_moves = []
        print(f"  Generating {len(positions)} moves at Skill Level {skill_level}...")
        
        for i, fen in enumerate(positions):
            board = chess.Board(fen)
            result = await engine.play(board, limit)
            move_uci = result.move.uci() if result.move else None
            engine_moves.append(move_uci)
            
            if (i + 1) % 1000 == 0:
                print(f"    Processed {i+1}/{len(positions)} positions")
                
    finally:
        await engine.quit()
    
    return np.array(engine_moves, dtype=object)

async def evaluate_stockfish_levels(positions, human_moves, ratings, bins):
    """Evaluate Stockfish across multiple skill levels."""
    skill_levels = [1, 2, 3, 4, 5, 6, 7]
    curves = {}
    
    tasks = []
    for skill in skill_levels:
        name = f"Stockfish L{skill}"
        task = asyncio.create_task(
            get_stockfish_moves(STOCKFISH_PATH, positions, skill_level=skill)
        )
        tasks.append((name, task))
    
    print(f"Running {len(tasks)} Stockfish skill levels in parallel...")
    results = await asyncio.gather(*[task for _, task in tasks])
    
    for (name, _), moves in zip(tasks, results):
        x, y = accuracy_by_bins(moves, human_moves, ratings, bins)
        curves[name] = (x, y)
        print(f"{name}: avg accuracy = {np.nanmean(y):.1f}%")
    
    return curves

def print_accuracy_table(bin_centers, engine_names, matrix):
    """Print formatted accuracy table."""
    print("\n" + "="*80)
    print("ACCURACY TABLE (% match to human moves)")
    print("="*80)
    
    header = "Rating".ljust(8)
    for name in engine_names:
        short_name = name.replace("Stockfish ", "SF").replace("L", "lv")
        header += short_name[:10].rjust(12)
    print(header)
    
    print("-"*80)
    for i, rating in enumerate(bin_centers):
        row = f"{int(rating):4d}".ljust(8)
        for acc in matrix[i]:
            if np.isnan(acc):
                row += "     ---".rjust(12)
            else:
                row += f"{acc:7.1f}%".rjust(12)
        print(row)

def save_csv_table(path, bin_centers, engine_names, matrix):
    """Save accuracy table as CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rating'] + engine_names)
        for i, rating in enumerate(bin_centers):
            row = [int(rating)]
            for val in matrix[i]:
                row.append(f"{val:.2f}" if not np.isnan(val) else "")
            writer.writerow(row)
    print(f"Saved table: {path}")

async def main():
    # Rating bins matching Lichess rapid/blitz
    bins = np.arange(1000, 2201, 100)
    
    # Load dataset
    positions, human_moves, ratings, bin_counts = load_positions_and_moves(
        DATASET_PATH, bins, per_bin=3000  # ~300 positions per bin
    )
    
    print(f"Bin counts: {bin_counts}")
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Run evaluation
    curves = await evaluate_stockfish_levels(positions, human_moves, ratings, bins)
    
    # Build results matrix
    engine_order = sorted(curves.keys())
    n_bins = len(bin_centers)
    matrix = np.full((n_bins, len(engine_order)), np.nan)
    
    for j, name in enumerate(engine_order):
        _, yvals = curves[name]
        matrix[:, j] = yvals
    
    # Display results
    print_accuracy_table(bin_centers, engine_order, matrix)
    
    # Save CSV
    output_csv = Path(__file__).parent / "stockfish_accuracy_table.csv"
    save_csv_table(output_csv, bin_centers, engine_order, matrix)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(engine_order)))
    
    for i, name in enumerate(engine_order):
        xvals, yvals = curves[name]
        valid = ~np.isnan(yvals)
        if np.sum(valid) > 0:
            plt.plot(xvals[valid], yvals[valid], 'o-', 
                    label=name, color=colors[i], linewidth=2, markersize=4)
    
    plt.xlabel('Average Game Rating', fontsize=12)
    plt.ylabel('Move Accuracy (%)', fontsize=12)
    plt.title('Stockfish Skill Levels vs Human Move Prediction', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('stockfish_skill_levels.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved: stockfish_skill_levels.png")
    print(f"CSV saved: {output_csv}")

if __name__ == "__main__":
    if not STOCKFISH_PATH.exists():
        print(f"ERROR: Stockfish not found at {STOCKFISH_PATH}")
        print("Please update STOCKFISH_PATH variable at the top of this script.")
        exit(1)
    
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("Please update DATASET_PATH variable at the top of this script.")
        exit(1)
    
    # Check python-chess is installed
    try:
        import chess
        import chess.engine
    except ImportError:
        print("ERROR: python-chess library not installed.")
        print("Install with: pip install python-chess")
        exit(1)
    
    print("Stockfish Skill Level Accuracy Test")
    print(f"Stockfish: {STOCKFISH_PATH}")
    print(f"Dataset:   {DATASET_PATH}")
    print("-" * 50)
    
    asyncio.run(main())
