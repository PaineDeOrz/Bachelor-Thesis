#!/usr/bin/env python3
"""
Maia2 Same-ELO Bins: 3 Plots by Game Phase
1. Overall (same ELO bins only)  
2. Opening (same ELO bins only)
3. Endgame (same ELO bins only)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODELS = ['GENERAL', 'TACTICAL', 'POSITIONAL', 'OFFICIAL']
CSV_FILE = "summary_accuracy3.csv"
OUTPUT_DIR = "maia2_plots_3moves"
ELO_MIDPOINTS = {
    '1000-1100': 1050, '1100-1200': 1150, '1200-1300': 1250, '1300-1400': 1350,
    '1400-1500': 1450, '1500-1600': 1550, '1600-1700': 1650, '1700-1800': 1750,
    '1800-1900': 1850, '1900-2000': 1950, '2000-2100': 2050, '2100-2200': 2150,
    '2200-2300': 2250, '>2300': 2350
}

def parse_elo_bin(bin_str):
    return ELO_MIDPOINTS.get(bin_str, 0)

def load_data(csv_file):
    """Load CSV, add ELO midpoints, filter n>=100"""
    df = pd.read_csv(csv_file)
    df['white_elo_mid'] = df['white_bin'].apply(parse_elo_bin)
    df['black_elo_mid'] = df['black_bin'].apply(parse_elo_bin)
    df = df[df['total'] >= 100].copy()
    return df[df['model'].isin(MODELS)]

def plot_same_elo_by_phase(df, phase, plot_num):
    """Plot same-ELO bins for specific phase (overall/opening/endgame)"""
    phase_data = df[(df['scope'] == phase) & (df['white_bin'] == df['black_bin'])].copy()
    
    plt.figure(figsize=(14, 8))
    
    for model in MODELS:
        model_data = phase_data[phase_data['model'] == model].sort_values('white_elo_mid')
        if len(model_data) > 0:
            plt.plot(model_data['white_elo_mid'], model_data['accuracy_pct'],
                    marker='o', linewidth=4, markersize=10, label=model)
    
    plt.xlabel('ELO Rating (Same-ELO Matchups)', fontsize=16, fontweight='bold')
    plt.ylabel('Top-1 Human Move Accuracy (%)', fontsize=16, fontweight='bold')
    
    phase_titles = {
        'overall': 'Overall Game Positions',
        'opening': 'Opening (Ply < 10)',
        'endgame': 'Endgame (Ply = 120)'
    }
    
    plt.title(f'Maia2 Models: Same-ELO Accuracy - {phase_titles[phase]}\n', 
              fontsize=18, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    plt.xticks(sorted(ELO_MIDPOINTS.values()), list(ELO_MIDPOINTS.keys()), rotation=45, ha='right')
    plt.ylim(50, 100)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/{plot_num}_{phase}_same_elo.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    
    print(f"? Saved: {filename}")
    print(f"   {len(phase_data)} data points across {len(phase_data['model'].unique())} models")

def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print("Loading data...")
    df = load_data(CSV_FILE)
    print(f"Loaded {len(df)} reliable positions (n=100)")
    
    print("\nCreating 3 phase-specific plots...")
    
    # Plot 1: Overall
    plot_same_elo_by_phase(df, 'overall', 1)
    
    # Plot 2: Opening  
    plot_same_elo_by_phase(df, 'opening', 2)
    
    # Plot 3: Endgame
    plot_same_elo_by_phase(df, 'endgame', 3)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY BY PHASE (Same-ELO bins only):")
    print("="*60)
    
    for phase in ['overall', 'opening', 'endgame']:
        phase_summary = df[(df['scope'] == phase) & (df['white_bin'] == df['black_bin'])]
        if len(phase_summary) > 0:
            summary_stats = phase_summary.groupby('model')['accuracy_pct'].agg(['mean', 'count']).round(2)
            print(f"\n{phase.upper()}:")
            print(summary_stats)
    
    print(f"\n? All 3 high-res plots saved to {OUTPUT_DIR}/")
    print("   1_overall_same_elo.png")
    print("   2_opening_same_elo.png") 
    print("   3_endgame_same_elo.png")

if __name__ == "__main__":
    main()
