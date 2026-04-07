import pandas as pd
import yaml
import torch
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.append(".")

# ---------------------------------------------
# Imports from your project
# ---------------------------------------------
from main import MAIA2Model
from utils import get_all_possible_moves, create_elo_dict
from inference import inference_each, prepare

# ---------------------------------------------
# Config
# ---------------------------------------------
DATASET_CSV = "../../../Maia_dataset/maia-chess-testing-set.csv"
OUTPUT_DIR = Path("accuracy_by_elo")
OUTPUT_DIR.mkdir(exist_ok=True)

GAN_CHECKPOINT = "../saves/0.0001_16001_1e-05/epoch_1_chunk_5.pt"
NORMAL_CHECKPOINT = "../saves/0.0001_16000_1e-05/epoch_1_2022-01.pgn_complete.pt"

SAVE_EVERY = 5000  # save CSV every N processed positions
GAMES_PER_BIN = 100000  # Number of games to process per Elo bin

# ---------------------------------------------
# Load and filter dataset by Elo bins
# ---------------------------------------------
print("Loading and filtering dataset...")
df = pd.read_csv(DATASET_CSV)

# Filter out invalid rows early
df = df.dropna(subset=["board", "move", "active_elo", "opponent_elo"])

# Create Elo bins for filtering
def elo_to_bin(elo):
    elo = int(elo)
    if elo < 800:
        return "<800"
    if elo >= 2300:
        return ">=2300"
    lower = (elo // 100) * 100
    upper = lower + 99
    return f"{lower}-{upper}"

ALL_BINS = ["<800"] + [f"{b}-{b+99}" for b in range(800, 2300, 100)] + [">=2300"]

# Filter dataset: keep only first GAMES_PER_BIN positions per Elo bin where active_elo == opponent_elo
print("Filtering dataset by Elo bins...")
filtered_rows = []
bin_counts = defaultdict(int)

for _, row in df.iterrows():
    active_elo = int(row["active_elo"])
    opponent_elo = int(row["opponent_elo"])
    
    if active_elo == opponent_elo:  # Only same Elo positions
        active_bin = elo_to_bin(active_elo)
        if bin_counts[active_bin] < GAMES_PER_BIN:
            filtered_rows.append(row)
            bin_counts[active_bin] += 1

df_filtered = pd.DataFrame(filtered_rows)
print(f"Filtered to {len(df_filtered)} positions ({dict(bin_counts)})")

# ---------------------------------------------
# Load model config
# ---------------------------------------------
with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

class Cfg:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

cfg = Cfg(cfg_dict)

# ---------------------------------------------
# Shared resources
# ---------------------------------------------
all_moves = get_all_possible_moves()
elo_dict = create_elo_dict()
INFERENCE_PREP = prepare()

# ---------------------------------------------
# Model loader
# ---------------------------------------------
def load_model(checkpoint_path, device):
    model = MAIA2Model(len(all_moves), elo_dict, cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    # Remove DataParallel prefix if needed
    state_dict = {
        k[7:] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------------------------
# CSV Saving
# ---------------------------------------------
def save_csvs(model_name, stats):
    # Top-1
    top1_rows = []
    for b in ALL_BINS:
        total = stats[b]["total"]
        correct = stats[b]["top1_correct"]

        top1_rows.append({
            "elo_bin": b,
            "positions": total,
            "top1_accuracy": (correct / total) if total > 0 else None
        })

    top1_df = pd.DataFrame(top1_rows)
    top1_df.to_csv(OUTPUT_DIR / f"{model_name}_top1_by_elo.csv", index=False)

    # Top-3
    top3_rows = []
    for b in ALL_BINS:
        total = stats[b]["total"]
        correct3 = stats[b]["top3_correct"]

        top3_rows.append({
            "elo_bin": b,
            "positions": total,
            "top3_accuracy": (correct3 / total) if total > 0 else None
        })

    top3_df = pd.DataFrame(top3_rows)
    top3_df.to_csv(OUTPUT_DIR / f"{model_name}_top3_by_elo.csv", index=False)

    print(f"[{model_name}] CSVs updated")

# ---------------------------------------------
# Unified evaluation loop
# ---------------------------------------------
def evaluate_both_models(gan_model, normal_model, device_gan, device_normal, df):
    gan_stats = defaultdict(lambda: {"total": 0, "top1_correct": 0, "top3_correct": 0})
    normal_stats = defaultdict(lambda: {"total": 0, "top1_correct": 0, "top3_correct": 0})

    diff_file = OUTPUT_DIR / "different_predictions.txt"

    positions_done = 0
    differences = 0

    with open(diff_file, "w") as diff_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            fen = row["board"]
            true_move = row["move"]

            active_elo = int(row["active_elo"])
            active_bin = elo_to_bin(active_elo)

            try:
                gan_probs, _ = inference_each(
                    gan_model,
                    INFERENCE_PREP,
                    fen,
                    active_elo,
                    active_elo  # Same elo
                )

                normal_probs, _ = inference_each(
                    normal_model,
                    INFERENCE_PREP,
                    fen,
                    active_elo,
                    active_elo  # Same elo
                )

            except Exception:
                continue

            if not gan_probs or not normal_probs:
                continue

            # -----------------------
            # GAN stats
            # -----------------------
            gan_sorted = list(gan_probs.keys())
            gan_top1 = gan_sorted[0]
            gan_top3 = gan_sorted[:3]

            gan_stats[active_bin]["total"] += 1
            gan_stats[active_bin]["top1_correct"] += int(true_move == gan_top1)
            gan_stats[active_bin]["top3_correct"] += int(true_move in gan_top3)

            # -----------------------
            # NORMAL stats
            # -----------------------
            normal_sorted = list(normal_probs.keys())
            normal_top1 = normal_sorted[0]
            normal_top3 = normal_sorted[:3]

            normal_stats[active_bin]["total"] += 1
            normal_stats[active_bin]["top1_correct"] += int(true_move == normal_top1)
            normal_stats[active_bin]["top3_correct"] += int(true_move in normal_top3)

            # -----------------------
            # Compare predictions
            # -----------------------
            if gan_top1 != normal_top1:
                diff_f.write(fen + "\n")
                diff_f.flush()
                differences += 1

            positions_done += 1

            if positions_done % SAVE_EVERY == 0:
                save_csvs("GAN", gan_stats)
                save_csvs("NORMAL", normal_stats)

    # Final save
    save_csvs("GAN", gan_stats)
    save_csvs("NORMAL", normal_stats)

    print("\nEvaluation complete.")
    print(f"Total evaluated positions: {positions_done}")
    print(f"Different top-1 predictions: {differences}")
    print(f"Saved differing FENs to: {diff_file}")

# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    print("Loading models...")

    gan_device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
    normal_device = "cuda:1" if torch.cuda.device_count() > 1 else gan_device

    gan_model = load_model(GAN_CHECKPOINT, gan_device)
    normal_model = load_model(NORMAL_CHECKPOINT, normal_device)

    evaluate_both_models(
        gan_model,
        normal_model,
        gan_device,
        normal_device,
        df_filtered  # Use filtered dataset
    )

    print("Done!")
