#!/bin/bash
# Parse Maia2 losses from 3 hardcoded .out files and plot losses

OUTPUT_CSV="losses_table_normal.csv"
PLOT_PNG="losses_plot.png"

# Hardcoded input files (change these paths as needed)
FILE1="st188776_maia2_train_15688.out"
FILE2="st188776_maia2_train_15715.out" 
FILE3="st188776_maia2_train_15791.out"

# Check if all 3 files exist
for FILE in "$FILE1" "$FILE2" "$FILE3"; do
    if [ ! -f "$FILE" ]; then
        echo "Error: File not found: $FILE"
        exit 1
    fi
done

echo "Parsing Maia2 losses from 3 files..."

# Initialize CSV with headers
echo "games,total_loss,maia_loss,side_loss,value_loss" > "$OUTPUT_CSV"

# Process each file
for LOG_FILE in "$FILE1" "$FILE2" "$FILE3"; do
    echo "Processing $LOG_FILE..."
    
    awk -v out="$OUTPUT_CSV" '
    BEGIN { current_games = 0 }
    
    # Parse Games line: [# Games]: 240.00K
    /^[ \t]*\[# Games\]:/ {
        if (match($0, /\[# Games\]: ([0-9.]+)(K|M|B)?/, g)) {
            games = g[1] + 0  # Force numeric
            multiplier = 1
            if (g[2] == "K") multiplier = 1000
            else if (g[2] == "M") multiplier = 1000000
            else if (g[2] == "B") multiplier = 1000000000
            current_games = games * multiplier
        }
    }
    
    # Parse Loss line: [# Loss]: 3.8560 | [# Loss MAIA]: 3.0560 | ...
    /^[ \t]*\[# Loss\]:/ {
        total = ""; maia = ""; side = ""; value = ""
        if (match($0, /\[# Loss\]: ([0-9.-]+)/, a)) total = a[1]
        if (match($0, /\[# Loss MAIA\]: ([0-9.-]+)/, b)) maia = b[1]
        if (match($0, /\[# Loss Side Info\]: ([0-9.-]+)/, c)) side = c[1]
        if (match($0, /\[# Loss Value\]: ([0-9.-]+)/, d)) value = d[1]
        
        if (total != "" && current_games > 0) {
            printf "%.0f,%s,%s,%s,%s\n", current_games, total, maia, side, value >> out
        }
    }
    ' "$LOG_FILE"
done

echo "CSV saved: $OUTPUT_CSV"

# Count total rows (excluding header)
COUNT=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
echo "Found $COUNT training steps across all files"

# Generate plot using python (matplotlib) - FIXED DUPLICATES + SMOOTH LINES
cat > plot_losses.py << 'EOF'
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
EOF

python3 plot_losses.py "$OUTPUT_CSV" "$PLOT_PNG"

echo "? All done!"
echo "?? CSV: $OUTPUT_CSV ($COUNT rows, $(python3 -c "import pandas as pd; df=pd.read_csv('$OUTPUT_CSV'); print(len(df.dropna().drop_duplicates(subset=['games'], keep='first')))" | tail -1) unique after dedup)"
echo "???  Plot: $PLOT_PNG (300 DPI, SMOOTH SOLID LINES)"
