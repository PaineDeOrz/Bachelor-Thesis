import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Load CSV
df = pd.read_csv('elo_analysis/elo_accuracy_table.csv')

# Convert ELO bins to numeric centers for plotting
elo_centers = []
for bin_label in df['elo_bin']:
    low, high = map(int, bin_label.split('-'))
    elo_centers.append((low + high) / 2)
elo_centers = np.array(elo_centers)

# Interpolation function for smooth lines
elo_smooth = np.linspace(elo_centers.min(), elo_centers.max(), 500)
orig_interp = interp1d(elo_centers, df['original_accuracy'], kind='linear')
cleaned_interp = interp1d(elo_centers, df['cleaned_accuracy'], kind='linear')

# Plot
plt.figure(figsize=(10,6))
plt.plot(elo_smooth, orig_interp(elo_smooth), label='Original Discriminator', color='blue', linewidth=2)
plt.plot(elo_smooth, cleaned_interp(elo_smooth), label='Cleaned Discriminator', color='red', linewidth=2)

# Scatter original points for reference
plt.scatter(elo_centers, df['original_accuracy'], color='blue', s=40)
plt.scatter(elo_centers, df['cleaned_accuracy'], color='red', s=40)

# Labels, title, legend
plt.xlabel('ELO')
plt.ylabel('Accuracy (%)')
plt.title('Discriminator Accuracy vs ELO')
plt.legend()
plt.grid(alpha=0.3)

# Save and show
plt.tight_layout()
plt.savefig('elo_accuracy_lines.pdf')
plt.show()