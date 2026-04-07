import torch
import yaml
import chess
from main import MAIA2Model
from discriminator_model import Discriminator
from utils import get_all_possible_moves, create_elo_dict, board_to_tensor
from inference import prepare, inference_each

# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------
# Load config
# -------------------------------------------------
with open("config.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

class Cfg:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            setattr(self, key, value)
cfg = Cfg(cfg_dict)

# -------------------------------------------------
# Load models
# -------------------------------------------------
checkpoint_path = "../saves/0.0001_16001_1e-05/lichess_db_standard_rated_2022-09_e1_f9_c165_20260221-082705.pt"
ckpt = torch.load(checkpoint_path, map_location=device)

all_moves = get_all_possible_moves()
elo_dict = create_elo_dict()

# Generator
generator = MAIA2Model(len(all_moves), elo_dict, cfg).to(device)
generator.load_state_dict(ckpt['model_state_dict'])
generator.eval()

# Discriminator
discriminator = Discriminator(cfg, len(all_moves)).to(device)
discriminator.load_state_dict(ckpt['discriminator_state_dict'])
discriminator.eval()

#Normal model
std_checkpoint_path = "../saves/NORMAL/epoch_1_chunk_5.pt"
std_ckpt = torch.load(std_checkpoint_path, map_location=device)
generator_std = MAIA2Model(len(all_moves), elo_dict, cfg).to(device)
generator_std.load_state_dict(std_ckpt['model_state_dict'])
generator_std.eval()

# Prepare inference helper
prepared = prepare()

# -------------------------------------------------
# Test position
# -------------------------------------------------
fen = "r2qk2r/pQ3ppp/2p1pn2/8/3P2PP/2N5/PP6/R1B1KBNb w Qkq - 1 14"
elo_self = 2000
elo_oppo = 2000

# -------------------------------------------------
# Get top legal moves using inference_each
# -------------------------------------------------
move_probs_gan, win_prob_gan = inference_each(generator, prepared, fen, elo_self, elo_oppo)

move_probs_std, win_prob_std = inference_each(generator_std, prepared, fen, elo_self, elo_oppo)

print("\n====================================================")
print("                 MODEL COMPARISON")
print("====================================================\n")

print(f"GAN Model Win Prob (White):      {win_prob_gan:.2%}")
print(f"Standard Model Win Prob (White): {win_prob_std:.2%}\n")

# Convert FEN to board tensor for discriminator evaluation
board = chess.Board(fen)
board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
elo_self_idx = torch.tensor([elo_dict[f"{elo_self}-{elo_self}"] if f"{elo_self}-{elo_self}" in elo_dict else 0], device=device)
elo_oppo_idx = torch.tensor([elo_dict[f"{elo_oppo}-{elo_oppo}"] if f"{elo_oppo}-{elo_oppo}" in elo_dict else 0], device=device)

for i, (move_uci, prob) in enumerate(list(move_probs_gan.items())[:3]):
    # Convert move UCI to index
    try:
        move_idx = all_moves.index(move_uci)
    except ValueError:
        print(f"{i+1}. {move_uci:6s} | Probability: {prob:.2%} | Discriminator: ? (move not in move list)")
        continue

    move_tensor = torch.tensor([move_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        human_like = torch.sigmoid(discriminator(board_tensor, move_tensor)).item()

    print(f"{i+1}. {move_uci:6s} | Probability: {prob:.2%} | Discriminator: {human_like:.3f}")

print("\nTop 3 Legal Moves (Standard Model):\n")

for i, (move_uci, prob) in enumerate(list(move_probs_std.items())[:3]):
    try:
        move_idx = all_moves.index(move_uci)
    except ValueError:
        print(f"{i+1}. {move_uci:6s} | Probability: {prob:.2%} | Discriminator: ?")
        continue

    move_tensor = torch.tensor([move_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        human_like = torch.sigmoid(
            discriminator(board_tensor, move_tensor)
        ).item()

    print(f"{i+1}. {move_uci:6s} | Probability: {prob:.2%} | Discriminator: {human_like:.3f}")
