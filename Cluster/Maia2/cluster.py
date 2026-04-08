"""
Analyze a Lichess game dump, extract handcrafted style features per game,
cluster games into play-style groups, and visualize the result with PCA.

Assumptions:
- The file `lichess_db_standard_rated_2019-04.csv.bz2` is present in the
  same working directory as this script, unless FILENAME is changed.
- The CSV contains the columns used below, including:
  game_id, white_elo, black_elo, num_ply, is_capture, is_check,
  is_blunder_cp, low_time, cp_rel, num_legal_moves, active_queen_count,
  active_bishop_count, active_knight_count, active_pawn_count.
- The dataset is a Lichess monthly dump in CSV.BZ2 format.
- The script is intended to run as a standalone analysis script.
- Output files and folders will be created in the current working directory.
"""

import bz2
import csv
import numpy as np
import pandas as pd
from statistics import pstdev, mean
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os

# Input dataset file to analyze.
FILENAME = "lichess_db_standard_rated_2019-04.csv.bz2"

# Optional cap on how many games to process.
# None means the full dataset is analyzed.
MAX_GAMES = None  # None = full analysis

# Elo ranges used to group games by average player strength.
RATING_BINS = [
    (1000, 1100), (1100, 1200), (1200, 1300),
    (1300, 1400), (1400, 1500), (1500, 1600), (1600, 1700),
    (1700, 1800), (1800, 1900), (1900, 2000)
]

def get_rating_bin(white_elo: int, black_elo: int) -> str:
    """
    Map a game's average Elo to a human-readable rating bin.

    Parameters
    ----------
    white_elo : int
        White player's Elo.
    black_elo : int
        Black player's Elo.

    Returns
    -------
    str
        Bin label such as '[1200-1300)' or 'Other' if the value falls outside
        the configured rating ranges.
    """
    avg_elo = (white_elo + black_elo) / 2.0
    for low, high in RATING_BINS:
        if low <= avg_elo < high:
            return f"[{low}-{high})"
    return "Other"


# Informational start message so the user knows the analysis has begun.
print("Starting 3D PCA analysis with WEIGHTS + BOUNDARIES...")


def safe_float(x: str) -> float:
    """
    Convert a string to float safely.

    Returns 0.0 if conversion fails.
    """
    try:
        return float(x)
    except:
        return 0.0


def safe_int(x: str) -> int:
    """
    Convert a string to int safely.

    Returns 0 if conversion fails.
    """
    try:
        return int(x)
    except:
        return 0


def safe_pstdev(values):
    """
    Compute population standard deviation robustly.

    Returns 0.0 for short or invalid sequences.
    """
    if len(values) <= 1:
        return 0.0
    clean_vals = [v for v in values if np.isfinite(v)]
    if len(clean_vals) <= 1:
        return 0.0
    return pstdev(clean_vals)


# Storage for per-game extracted statistics and metadata.
game_stats = []
game_id_list = []
game_metadata = []
current_game = None
game_count = 0

# Read the compressed CSV one row at a time and aggregate rows by game_id.
with bz2.open(FILENAME, mode="rt", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        gid = row["game_id"]

        # Detect when a new game begins so the previous game's stats can be stored.
        if gid != current_game:
            if current_game is not None:
                game_stats.append(current_game_stats)
                game_id_list.append(current_game)
                game_metadata.append(current_metadata)
                game_count += 1
                if game_count % 100000 == 0:
                    print(f"Processed {game_count:,} games...")
                if MAX_GAMES and game_count >= MAX_GAMES:
                    print(f"Reached target: {MAX_GAMES:,} games!")
                    break

            current_game = gid
            current_metadata = {
                'white_elo': safe_int(row["white_elo"]),
                'black_elo': safe_int(row["black_elo"]),
                'rating_bin': get_rating_bin(safe_int(row["white_elo"]), safe_int(row["black_elo"]))
            }
            current_game_stats = {
                'num_ply': safe_int(row["num_ply"]),
                'captures': 0, 'checks': 0, 'blunders_cp': 0,
                'low_time_moves': 0, 'num_moves': 0,
                'cp_rel_vals': [], 'legal_moves': [],
                'queen_vals': [], 'bishop_vals': [], 'knight_vals': [], 'pawn_vals': []
            }
            continue

        # Update per-game aggregate statistics for each move.
        current_game_stats['num_moves'] += 1
        if row["is_capture"] == "True":
            current_game_stats['captures'] += 1
        if row["is_check"] == "1":
            current_game_stats['checks'] += 1
        if row["is_blunder_cp"] == "True":
            current_game_stats['blunders_cp'] += 1
        if row["low_time"] == "True":
            current_game_stats['low_time_moves'] += 1

        cp_val = safe_float(row["cp_rel"])
        if np.isfinite(cp_val):
            current_game_stats['cp_rel_vals'].append(cp_val)

        current_game_stats['legal_moves'].append(safe_int(row["num_legal_moves"]))

        # These "safe" features are used when the columns are available in the CSV.
        if "active_queen_count" in row:
            current_game_stats['queen_vals'].append(safe_int(row["active_queen_count"]))
        if "active_bishop_count" in row:
            current_game_stats['bishop_vals'].append(safe_int(row["active_bishop_count"]))
        if "active_knight_count" in row:
            current_game_stats['knight_vals'].append(safe_int(row["active_knight_count"]))
        if "active_pawn_count" in row:
            current_game_stats['pawn_vals'].append(safe_int(row["active_pawn_count"]))

# Save the final game after the loop ends.
if current_game:
    game_stats.append(current_game_stats)
    game_id_list.append(current_game)
    game_metadata.append(current_metadata)
    game_count += 1

print(f"Extracted {game_count:,} games total")

# The 10 handcrafted features used for clustering and PCA.
feature_names = [
    'captures_per_move', 'checks_per_move', 'vol_cp', 'blunder_rate_cp',
    'low_time_frac', 'avg_legal_moves', 'inv_num_ply',
    'queen_activity', 'minor_piece_loss', 'pawn_structure_vol'
]
features = []

print("Computing 10 enhanced features...")
for i, stats in enumerate(game_stats):
    if i % 500000 == 0 and i > 0:
        print(f"  Feature progress: {i:,}/{len(game_stats):,} games")

    n = max(stats['num_moves'], 1)
    feat_vec = [
        stats['captures'] / n,
        stats['checks'] / n,
        safe_pstdev(stats['cp_rel_vals']),
        stats['blunders_cp'] / n,
        stats['low_time_moves'] / n,
        mean(stats['legal_moves']) if stats['legal_moves'] else 0.0,
        1.0 / (stats['num_ply'] + 1),
        mean(stats['queen_vals']) if stats['queen_vals'] else 0.0,
    ]

    # Minor-piece-loss proxy: changes in bishop+knight count over the game.
    if len(stats['bishop_vals']) > 1 and len(stats['knight_vals']) > 1:
        bishop_knight = np.array(stats['bishop_vals']) + np.array(stats['knight_vals'])
        feat_vec.append(np.abs(np.diff(bishop_knight)).sum())
    else:
        feat_vec.append(0.0)

    # Volatility of pawn count over the course of the game.
    feat_vec.append(safe_pstdev(stats['pawn_vals']))
    features.append(feat_vec)

# Feature matrix used for clustering and PCA.
X = np.array(features)

# Standardize the feature matrix before clustering and PCA.
print(f"Clustering {len(X):,} games...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans is used to group games into three style clusters.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# PCA is used for visualization and for inspecting principal directions.
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Infer which cluster corresponds to tactical vs positional vs mixed.
tactical_cluster = np.argmax([np.mean(X_scaled[labels == i, 0]) for i in range(3)])
positional_cluster = np.argmin([np.mean(X_scaled[labels == i, 0]) for i in range(3)])
mixed_cluster = 3 - tactical_cluster - positional_cluster

# ======================================================
# CRITICAL: WEIGHTS AND BOUNDARIES OUTPUT
# ======================================================

print("\n" + "="*80)
print("CLUSTER CENTROIDS (FEATURE WEIGHTS)")
print("="*80)

cluster_names = {tactical_cluster: 'TACTICAL', positional_cluster: 'POSITIONAL', mixed_cluster: 'MIXED'}
for i in range(3):
    cluster_name = cluster_names.get(i, 'MIXED')
    center = kmeans.cluster_centers_[i]

    print(f"\n{cluster_name:12} (C{i}, {np.sum(labels==i):,} games):")
    formula = " + ".join([f"{w:.3f}*{n}" for w, n in zip(center, feature_names)])
    print(f"score = {formula}")
    print(f"Raw values: captures={center[0]:.3f}, checks={center[1]:.3f}, blunders={center[3]:.3f}")

# Decision boundaries and cluster interpretation from PCA space.
print("\n" + "="*80)
print("CLUSTER BOUNDARIES (for classifying new games)")
print("="*80)

# PC1 is treated as the main aggression axis.
pc1_tactical_mean = np.mean(X_pca[labels == tactical_cluster, 0])
pc1_positional_mean = np.mean(X_pca[labels == positional_cluster, 0])
pc1_mixed_mean = np.mean(X_pca[labels == mixed_cluster, 0])

print(f"PC1 boundaries (Aggression Axis):")
print(f"  Tactical:    > {pc1_tactical_mean:.3f}")
print(f"  Positional:  < {pc1_positional_mean:.3f}")
print(f"  Mixed:       {min(pc1_tactical_mean, pc1_positional_mean):.3f} to {max(pc1_tactical_mean, pc1_positional_mean):.3f}")

# A simple centroid-distance threshold is also printed for reference.
print(f"\nDistance thresholds to centroids:")
dist_tactical = np.mean([np.linalg.norm(center - kmeans.cluster_centers_[tactical_cluster])
                        for center in kmeans.cluster_centers_])
print(f"  Tactical if distance to C{tactical_cluster} < {dist_tactical:.3f}")

# PCA component loadings help interpret which features contribute to each axis.
print("\n" + "="*80)
print("PCA LOADINGS (what each PC measures)")
print("="*80)
loadings = pd.DataFrame(
    pca.components_.T * np.sqrt(pca.explained_variance_),
    columns=['PC1', 'PC2', 'PC3'],
    index=feature_names
)
print(loadings.round(3))

# ======================================================

# Analyze how the inferred styles are distributed across Elo bins.
print("\n" + "="*80)
print("RATING BIN ANALYSIS")
print("="*80)
bin_stats = {bin_name: {'total': 0, 'tactical': 0, 'positional': 0, 'mixed': 0}
             for bin_name in set([m['rating_bin'] for m in game_metadata])}

for i, meta in enumerate(game_metadata):
    bin_name = meta['rating_bin']
    bin_stats[bin_name]['total'] += 1
    if labels[i] == tactical_cluster:
        bin_stats[bin_name]['tactical'] += 1
    elif labels[i] == positional_cluster:
        bin_stats[bin_name]['positional'] += 1
    else:
        bin_stats[bin_name]['mixed'] += 1

print("Rating    | Total | Tactical | Positional | Mixed | %Tactical")
print("-" * 70)
for bin_name in sorted(bin_stats.keys()):
    stats = bin_stats[bin_name]
    pct_tactical = stats['tactical'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"{bin_name:8} | {stats['total']:5,} | {stats['tactical']:8,} | "
          f"{stats['positional']:9,} | {stats['mixed']:5,} | {pct_tactical:7.1f}%")

# Save the per-bin game IDs to text files for later Maia training.
os.makedirs("rating_bins", exist_ok=True)
total_tactical, total_positional, total_mixed = 0, 0, 0

for bin_name in bin_stats.keys():
    tactical_in_bin = [game_id_list[i] for i in range(len(labels))
                      if labels[i] == tactical_cluster and game_metadata[i]['rating_bin'] == bin_name]
    positional_in_bin = [game_id_list[i] for i in range(len(labels))
                        if labels[i] == positional_cluster and game_metadata[i]['rating_bin'] == bin_name]
    mixed_in_bin = [game_id_list[i] for i in range(len(labels))
                   if labels[i] == mixed_cluster and game_metadata[i]['rating_bin'] == bin_name]

    if tactical_in_bin:
        np.savetxt(f"rating_bins/tactical_{bin_name.replace('/','-')}.txt", tactical_in_bin, fmt='%s')
        total_tactical += len(tactical_in_bin)
    if positional_in_bin:
        np.savetxt(f"rating_bins/positional_{bin_name.replace('/','-')}.txt", positional_in_bin, fmt='%s')
        total_positional += len(positional_in_bin)
    if mixed_in_bin:
        np.savetxt(f"rating_bins/mixed_{bin_name.replace('/','-')}.txt", mixed_in_bin, fmt='%s')
        total_mixed += len(mixed_in_bin)

# Save overall style lists without rating-bin splitting.
np.savetxt('tactical_games.txt', [game_id_list[i] for i in range(len(labels)) if labels[i] == tactical_cluster], fmt='%s')
np.savetxt('positional_games.txt', [game_id_list[i] for i in range(len(labels)) if labels[i] == positional_cluster], fmt='%s')
np.savetxt('mixed_games.txt', [game_id_list[i] for i in range(len(labels)) if labels[i] == mixed_cluster], fmt='%s')

print(f"\nSAVED FILES:")
print(f"  rating_bins/ -> {len([f for f in os.listdir('rating_bins') if f.endswith('.txt')])} files")
print(f"  tactical_games.txt:      {total_tactical:,}")
print(f"  positional_games.txt:    {total_positional:,}")
print(f"  mixed_games.txt:         {total_mixed:,}")

# Plot the PCA result in 3D and in the three pairwise projections.
colors = ['red' if l == tactical_cluster else
          'blue' if l == positional_cluster else 'green' for l in labels]

plt.figure(figsize=(24, 6))
plt.subplot(1, 4, 1, projection='3d')
ax = plt.gca()
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, alpha=0.6, s=1)
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2],
           c='black', marker='X', s=300, linewidths=4)
ax.set_title(f'3D PCA\nTotal: {sum(pca.explained_variance_ratio_):.0%}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.0%})')

projections = [(0, 1), (0, 2), (1, 2)]
proj_names = ['PC1-PC2', 'PC1-PC3', 'PC2-PC3']
for idx, (i, j) in enumerate(projections, 2):
    plt.subplot(1, 4, idx)
    plt.scatter(X_pca[:, i], X_pca[:, j], c=colors, alpha=0.6, s=1)
    plt.scatter(centroids_pca[:, i], centroids_pca[:, j],
               c='black', marker='X', s=250, linewidths=4)
    plt.title(f'PC{i+1}-PC{j+1}')
    plt.xlabel(f'PC{i+1}')
    plt.ylabel(f'PC{j+1}')

plt.subplot(1, 4, 4)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
           label=f'Tactical C{tactical_cluster} ({np.sum(labels==tactical_cluster):,})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
           label=f'Positional C{positional_cluster} ({np.sum(labels==positional_cluster):,})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10,
           label=f'Mixed C{mixed_cluster} ({np.sum(labels==mixed_cluster):,})'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='Centroids')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
plt.axis('off')

plt.tight_layout()
plt.savefig('pca_3d_complete.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nCOMPLETE! Total variance: {sum(pca.explained_variance_ratio_):.1%}")
print("Files ready for Maia training!")