#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
LOG_FILE_1 = "discriminator_wgan_original_plot"
LOG_FILE_2 = "discriminator_wgan_cleaned_plot"

LABEL_1 = "Original Model"
LABEL_2 = "Cleaned Dataset Model"

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Regex pattern
# -----------------------------
pattern = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+Loss:.*?\|\s+W-dist:\s+([-\d\.]+).*?\|\s+Acc:\s+([\d\.]+)%\s+\|\s+\(H:\s+([\d\.]+)%\s+F:\s+([\d\.]+)%"
)

def parse_log(filepath):
    epochs = []
    wdist = []
    acc = []
    human_acc = []
    fake_acc = []

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                wdist.append(float(match.group(2)))
                acc.append(float(match.group(3)))
                human_acc.append(float(match.group(4)))
                fake_acc.append(float(match.group(5)))

    return epochs, wdist, acc, human_acc, fake_acc

# -----------------------------
# Parse both logs
# -----------------------------
e1, w1, a1, h1, f1 = parse_log(LOG_FILE_1)
e2, w2, a2, h2, f2 = parse_log(LOG_FILE_2)

# -----------------------------
# Plot 1: Wasserstein Distance
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(e1, w1, label=LABEL_1, linewidth=2)
plt.plot(e2, w2, label=LABEL_2, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Wasserstein Distance")
plt.title("Wasserstein Distance During Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wasserstein_comparison.pdf")
plt.close()

# -----------------------------
# Plot 2A: Total Accuracy Only
# -----------------------------
plt.figure(figsize=(8, 6))

# Original model (darker)
plt.plot(e1, a1, color="blue", linewidth=3.0, label=f"{LABEL_1}")

# Cleaned model (lighter)
plt.plot(e2, a2, color="red", linewidth=3.0, label=f"{LABEL_2}")

plt.xlabel("Epoch")
plt.ylabel("Total Accuracy (%)")
plt.title("Total Discriminator Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_total_comparison.pdf")
plt.close()

# -----------------------------
# Plot 2B: Human & Fake Accuracy
# -----------------------------
plt.figure(figsize=(10, 7))

# ---- Original Model (Darker palette) ----
plt.plot(e1, h1, color="purple", linewidth=1.8, label=f"{LABEL_1} - Human")
plt.plot(e1, f1, color="brown", linewidth=1.8, label=f"{LABEL_1} - Fake")

# ---- Cleaned Model (Lighter palette) ----
plt.plot(e2, h2, color="orange", linewidth=1.8, label=f"{LABEL_2} - Human")
plt.plot(e2, f2, color="lightgreen", linewidth=1.8, label=f"{LABEL_2} - Fake")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Human vs Fake Classification Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_human_fake_comparison.pdf")
plt.close()

print("Plots saved in:", OUTPUT_DIR)