import pandas as pd
import yaml
import torch
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.append(".")

# ---------------------------------------------
# Imports from your project (adjust as needed)
# ---------------------------------------------
from main import MAIA2Model  # Assuming discriminator is same architecture
from utils import get_all_possible_moves, create_elo_dict
from inference import inference_each, prepare

# ---------------------------------------------
# Config
# ---------------------------------------------
DATASET_CSV = "dataset/discriminator_dataset.csv"  # Update this path
OUTPUT_DIR = Path("discriminator_accuracy")
OUTPUT_DIR.mkdir(exist_ok=True)

GAN_CHECKPOINT = "../saves/0.0001_16001_1e-05/lichess_db_standard_rated_2022-09_e1_f9_c165_20260221-082705.pt"
OUTPUT_CSV = OUTPUT_DIR / "discriminator_accuracy.csv"

UPDATE_EVERY = 1000  # Update CSV every N positions
MAX_POSITIONS = None  # Set to limit total positions, None = all

# Force CPU usage
DEVICE = "cpu"

# ---------------------------------------------
# Load and filter dataset
# ---------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATASET_CSV)

# Filter for valid rows with required columns
required_cols = ["board", "move", "fake_move", "active_elo", "opponent_elo"]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = df.dropna(subset=required_cols)
print(f"Loaded {len(df)} valid positions")

if MAX_POSITIONS:
    df = df.head(MAX_POSITIONS)
    print(f"Limited to {len(df)} positions")

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
print("Loading shared resources...")
all_moves = get_all_possible_moves()
elo_dict = create_elo_dict()
INFERENCE_PREP = prepare()

# ---------------------------------------------
# Model loader
# ---------------------------------------------
def load_discriminator(checkpoint_path, device):
    print(f"Loading discriminator from {checkpoint_path} on {device}...")
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
    print("Discriminator loaded successfully!")
    return model

# ---------------------------------------------
# Save results to CSV
# ---------------------------------------------
def save_results(stats):
    results = {
        "total_positions": stats["total"],
        "real_higher_than_fake": stats["real_higher"],
        "real_prob_gt_50": stats["real_gt_50"],
        "real_prob_gt_75": stats["real_gt_75"],
        "real_prob_gt_90": stats["real_gt_90"],
        "fake_prob_lt_50": stats["fake_lt_50"],
        "fake_prob_lt_25": stats["fake_lt_25"],
        "fake_prob_lt_10": stats["fake_lt_10"],
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"?? Updated results: {results}")

# ---------------------------------------------
# Main evaluation
# ---------------------------------------------
def evaluate_discriminator(model, df):
    stats = defaultdict(int)
    
    print("Starting discriminator evaluation...")
    print("Format: [positions_done] real_prob | fake_prob | real>fake?")
    
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Evaluating")):
        fen = row["board"]
        real_move = row["move"]
        fake_move = row["fake_move"]
        active_elo = int(row["active_elo"])
        opponent_elo = int(row["opponent_elo"])

        try:
            # Get discriminator probabilities
            real_probs, _ = inference_each(model, INFERENCE_PREP, fen, active_elo, opponent_elo)
            fake_probs, _ = inference_each(model, INFERENCE_PREP, fen, active_elo, opponent_elo)
            
            if not real_probs or not fake_probs:
                continue
                
            real_prob = real_probs.get(real_move, 0.0)
            fake_prob = fake_probs.get(fake_move, 0.0)
            
            stats["total"] += 1
            
            # Track all metrics
            if real_prob > fake_prob:
                stats["real_higher"] += 1
            if real_prob > 0.50:
                stats["real_gt_50"] += 1
            if real_prob > 0.75:
                stats["real_gt_75"] += 1
            if real_prob > 0.90:
                stats["real_gt_90"] += 1
            if fake_prob < 0.50:
                stats["fake_lt_50"] += 1
            if fake_prob < 0.25:
                stats["fake_lt_25"] += 1
            if fake_prob < 0.10:
                stats["fake_lt_10"] += 1
                
            # Live feedback every 100 positions
            if (i + 1) % 100 == 0:
                print(f"[{i+1}] real={real_prob:.3f} | fake={fake_prob:.3f} | real>fake={real_prob>fake_prob}")
                
        except Exception as e:
            print(f"Error at position {i}: {e}")
            continue

        # Save every UPDATE_EVERY positions
        if (i + 1) % UPDATE_EVERY == 0:
            print(f"\n?? Saving progress after {i+1} positions...")
            save_results(stats)
    
    # Final save
    print("\n?? Final save...")
    save_results(stats)
    return stats

# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    print(f"?? Discriminator Accuracy Test")
    print(f"?? Dataset: {DATASET_CSV}")
    print(f"?? Output: {OUTPUT_CSV}")
    print(f"???  Device: {DEVICE}")
    print(f"?? Update every: {UPDATE_EVERY} positions")
    print("-" * 60)
    
    # Load model
    model = load_discriminator(GAN_CHECKPOINT, DEVICE)
    
    # Run evaluation
    final_stats = evaluate_discriminator(model, df)
    
    print("\n?? Evaluation complete!")
    print("\n?? Final Results:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
