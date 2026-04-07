#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as ticker
import numpy as np

csv_file = sys.argv[1]
png_file = sys.argv[2]

# Read CSV
df = pd.read_csv(csv_file)

# Remove NaN and deduplicate by games (keep first occurrence)
df = df.dropna().drop_duplicates(subset=['games'], keep='first').sort_values('games').reset_index(drop=True)

print(f"After deduplication: {len(df)} unique game checkpoints")

# Format x-axis labels (100K, 1M, 10M, etc.)
def format_games(x, pos):
    if x >= 1e9:
        return f"{x/1e9:.1f}B"
    elif x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}K"
    else:
        return f"{int(x)}"

# Plot all losses - PERFECT SMOOTH SOLID LINES (no interpolation needed)
plt.figure(figsize=(14, 8))

losses = ['total_loss', 'maia_loss', 'side_loss', 'value_loss']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
labels = ['Total Loss', 'Maia Loss', 'Side Loss', 'Value Loss']

for i, loss_col in enumerate(losses):
    plt.plot(df['games'], df[loss_col], 
             color=colors[i], 
             linewidth=4,
             solid_capstyle='round',  # Smooth line ends
             label=labels[i])

plt.xlabel('Number of Games', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=14, fontweight='bold')
plt.title('Maia2 Training Losses Across Checkpoints', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)

# Custom x-axis formatter
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_games))
plt.gca().tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(png_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Perfect smooth plot saved: {png_file}")
