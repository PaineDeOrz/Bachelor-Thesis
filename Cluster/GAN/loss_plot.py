# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_FILES = {
    "GAN": "losses_table_final.csv",
    "NORMAL": "losses_table_normal.csv",
}

SINGLE_MODEL_NAME = "GAN"
GAN_GAMES_CSV = "losses_table_final_games.csv"
NORMAL_GAMES_CSV = "losses_table_normal_games.csv"

GAMES_PER_CHUNK = 5000
EMA_ALPHA = 0.05

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def smooth(series, alpha=EMA_ALPHA):
    return series.ewm(alpha=alpha).mean()

def load_csv(filepath):
    df = pd.read_csv(filepath)
    return df.dropna(subset=["games"]).sort_values("games")

def human_format(x, pos):
    if x >= 1_000_000:
        val = x / 1_000_000
        return f"{val:.1f}M" if val % 1 else f"{int(val)}M"
    elif x >= 1_000:
        val = x / 1_000
        return f"{val:.0f}K"
    else:
        return f"{int(x)}"

# -------------------------------------------------
# STEP 1: CLEAN CSV PROCESSOR (handles embedded headers + duplicates)
# -------------------------------------------------
def create_games_csv(input_path, output_path):
    """Parse CSV line-by-line, increment chunk index at each new header."""
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    all_rows = []
    next_chunk = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        parts = [p.strip() for p in line.split(',')]
        
        if len(parts) < 4:
            i += 1
            continue
            
        # Skip ANY header (chunk OR games)
        if parts[0] in ['chunk', 'games']:
            print(f"New section at line {i}, next chunk starts at {next_chunk}")
            i += 1
            continue
            
        # Data row: use global continuous chunk numbering + original data
        all_rows.append([next_chunk] + parts[1:])
        next_chunk += 1
        i += 1
    
    if not all_rows:
        raise ValueError("No valid data rows found!")
    
    # **CORRECT COLUMN DETECTION** - count TOTAL columns (including chunk)
    num_total_cols = len(all_rows[0])
    print(f"Detected {num_total_cols} total columns")
    
    if num_total_cols == 4:  # GAN: chunk + 3 losses
        df = pd.DataFrame(all_rows, columns=["chunk", "total_loss", "maia_loss", "disc_loss"])
    elif num_total_cols == 5:  # NORMAL: chunk + 4 losses  
        df = pd.DataFrame(all_rows, columns=["chunk", "total_loss", "maia_loss", "side_loss", "value_loss"])
    else:
        raise ValueError(f"Unexpected column count: {num_total_cols}")
    
    # Convert to games
    df["games"] = df["chunk"] * GAMES_PER_CHUNK
    df = df.drop(columns=["chunk"])
    
    # Force numeric
    for col in df.columns:
        if col != "games":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["total_loss"]).sort_values("games")
    
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path} ({len(df)} rows)")
    print("First 3 rows:")
    print(df.head(3))
    print()

# Create clean CSVs for BOTH files
create_games_csv(MODEL_FILES["GAN"], GAN_GAMES_CSV)
create_games_csv(MODEL_FILES["NORMAL"], NORMAL_GAMES_CSV)

# -------------------------------------------------
# STEP 2: PLOTTING
# -------------------------------------------------
PLOT_FILES = {
    "GAN": GAN_GAMES_CSV,
    "NORMAL": NORMAL_GAMES_CSV
}

OUTPUT_TOTAL = "total_losses_plot.png"
OUTPUT_MAIA = "maia_losses_plot.png"
OUTPUT_SINGLE = "single_model_losses.png"

# 1) TOTAL LOSS COMPARISON
plt.figure(figsize=(10, 6))
for name, filepath in PLOT_FILES.items():
    df = load_csv(filepath)
    if "total_loss" in df.columns:
        y = smooth(df["total_loss"])
        plt.plot(df["games"], y, label=name)
        print(f"{name}: total_loss starts at {y.iloc[0]:.3f}")

plt.xlabel("Number of Games"); plt.ylabel("Total Loss")
plt.title("Total Training Loss Comparison")
plt.gca().xaxis.set_major_formatter(FuncFormatter(human_format))
plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig(OUTPUT_TOTAL, dpi=300); plt.close()
print(f"Saved {OUTPUT_TOTAL}\n")

# 2) MAIA LOSS COMPARISON
plt.figure(figsize=(10, 6))
for name, filepath in PLOT_FILES.items():
    df = load_csv(filepath)
    if "maia_loss" in df.columns:
        y = smooth(df["maia_loss"])
        plt.plot(df["games"], y, label=name)
        print(f"{name}: maia_loss starts at {y.iloc[0]:.3f}")

plt.xlabel("Number of Games"); plt.ylabel("MAIA Loss")
plt.title("MAIA Loss Comparison")
plt.gca().xaxis.set_major_formatter(FuncFormatter(human_format))
plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig(OUTPUT_MAIA, dpi=300); plt.close()
print(f"Saved {OUTPUT_MAIA}\n")

# 3) SINGLE MODEL BREAKDOWN (GAN)
plt.figure(figsize=(10, 6))
df = load_csv(GAN_GAMES_CSV)
for col, label in [("total_loss", "Total Loss"), ("maia_loss", "MAIA Loss"), 
                   ("disc_loss", "Discriminator Loss")]:
    if col in df.columns:
        y = smooth(df[col])
        plt.plot(df["games"], y, label=label)
        print(f"{label}: starts at {y.iloc[0]:.3f}")

plt.xlabel("Number of Games"); plt.ylabel("Loss")
plt.title(f"{SINGLE_MODEL_NAME} - Detailed Loss Breakdown")
plt.gca().xaxis.set_major_formatter(FuncFormatter(human_format))
plt.legend(); plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout(); plt.savefig(OUTPUT_SINGLE, dpi=300); plt.close()
print(f"Saved {OUTPUT_SINGLE}")
