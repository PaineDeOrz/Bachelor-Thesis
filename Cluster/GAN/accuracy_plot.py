#!/usr/bin/env python3
"""
Chess Model Accuracy Plotter
Takes 4 CSV files (gan_top1.csv, maia_top1.csv, gan_top3.csv, maia_top3.csv)
Creates 2 plots: Top-1 and Top-3 accuracy comparison (GAN vs Maia2)
X-axis: ELO 1000-1900, only bins with positions > 0
Publication quality (300 DPI)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Input files - adjust paths as needed
FILES = {
    'gan_top1': 'accuracy_by_elo_temp/GAN_top1_by_elo.csv',
    'maia_top1': 'accuracy_by_elo_temp/NORMAL_top1_by_elo.csv',
    'gan_top3': 'accuracy_by_elo_temp/GAN_top3_by_elo.csv',
    'maia_top3': 'accuracy_by_elo_temp/NORMAL_top3_by_elo.csv'
}

OUTPUT_DIR = Path('accuracy_plots')
OUTPUT_DIR.mkdir(exist_ok=True)

def parse_elo_bin(bin_str):
    """Extract midpoint ELO from bin string like '1000-1099' -> 1050"""
    if '-' in bin_str:
        low, high = map(int, bin_str.split('-'))
        return (low + high) // 2
    elif bin_str.startswith('<'):
        return 750  # Below 800
    elif bin_str.startswith('>='):
        return 2350
    else:
        return np.nan

def load_accuracy_df(filename, acc_col):
    """Load CSV and return df with elo_mid, accuracy (*100 for %)"""
    df = pd.read_csv(filename)
    df['elo_mid'] = df['elo_bin'].apply(parse_elo_bin)
    df['accuracy_pct'] = df[acc_col] * 100
    # Filter relevant range and positions > 0
    mask = (df['elo_mid'] >= 1000) & (df['elo_mid'] <= 1900) & (df['positions'] > 0)
    return df.loc[mask, ['elo_mid', 'accuracy_pct', 'positions']].sort_values('elo_mid')

# Load all data
top1_gan = load_accuracy_df(FILES['gan_top1'], 'top1_accuracy')
top1_maia = load_accuracy_df(FILES['maia_top1'], 'top1_accuracy')
top3_gan = load_accuracy_df(FILES['gan_top3'], 'top3_accuracy')
top3_maia = load_accuracy_df(FILES['maia_top3'], 'top3_accuracy')

print(f"Loaded:\nTop1 GAN: {len(top1_gan)} points\nTop1 Maia: {len(top1_maia)} points")
print(f"Top3 GAN: {len(top3_gan)} points\nTop3 Maia: {len(top3_maia)} points")

# Create Top-1 plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(top1_gan['elo_mid'], top1_gan['accuracy_pct'], 'o-', label='GAN Top-1', linewidth=2.5, markersize=6)
ax.plot(top1_maia['elo_mid'], top1_maia['accuracy_pct'], 's-', label='Maia2 Top-1', linewidth=2.5, markersize=6)
ax.set_xlabel('ELO Rating')
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('Top-1 Move Accuracy: GAN vs Maia2')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(950, 1950)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top1_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Create Top-3 plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(top3_gan['elo_mid'], top3_gan['accuracy_pct'], 'o-', label='GAN Top-3', linewidth=2.5, markersize=6)
ax.plot(top3_maia['elo_mid'], top3_maia['accuracy_pct'], 's-', label='Maia2 Top-3', linewidth=2.5, markersize=6)
ax.set_xlabel('ELO Rating')
ax.set_ylabel('Top-3 Accuracy (%)')
ax.set_title('Top-3 Move Accuracy: GAN vs Maia2')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(950, 1950)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'top3_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Plots saved to {OUTPUT_DIR}/")
print("? top1_accuracy.png (40-60%)")
print("? top3_accuracy.png (65-80%)")
