# test_both_maia2_after_e4.py - Position AFTER White plays e4
import torch
import yaml
import sys
import chess
sys.path.append('.')

FEN = "r2qk2r/pQ3ppp/2p1pn2/8/3P2PP/2N5/PP6/R1B1KBNb w Qkq - 1 14"
ELO_SELF = 1200  # Black Elo
ELO_OPPO = 1200  # White Elo

# === 1. YOUR CUSTOM MODELS ===
print("\n1?? YOUR MODELS (Black to move)")
try:
    from main import MAIA2Model
    from utils import get_all_possible_moves, create_elo_dict
    from inference import inference_each, prepare

    with open('config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)

    class Cfg:
        def __init__(self, cfg_dict):
            for key, value in cfg_dict.items():
                setattr(self, key, value)
    cfg = Cfg(cfg_dict)

    device = 'cpu'
    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    # List of your 3 models
    CUSTOM_MODELS = [
        {
            "name": "TACTICAL",
            "checkpoint": "../saves/0.0001_16000_1e-05_tactical/epoch_1_2020-10.pgn_chunk_240.pt",
        },
        {
            "name": "POSITIONAL",
            "checkpoint": "../saves/0.0001_16000_1e-05_positional/epoch_1_2021-10.pgn_chunk_240.pt",
        },
        {
            "name": "GENERAL",
            "checkpoint": "../saves/0.0001_16000_1e-05_general/epoch_1_2022-12.pgn_chunk_240.pt",
        },
    ]

    prepared_yours = prepare()

    your_results = {}  # name -> (move_probs, win_prob)

    for m in CUSTOM_MODELS:
        name = m["name"]
        ckpt = m["checkpoint"]
        print(f"\nLoading {name} from {ckpt}")

        model_yours = MAIA2Model(len(all_moves), elo_dict, cfg).to(device)

        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}
        model_yours.load_state_dict(new_state_dict)
        model_yours.eval()

        move_probs, win_prob = inference_each(
            model_yours, prepared_yours, FEN, ELO_SELF, ELO_OPPO
        )

        your_results[name] = (move_probs, win_prob)
        print(f"? {name} loaded:")
        print(f"  Samples: {checkpoint['accumulated_samples']/1e6:.1f}M positions")

except Exception as e:
    print(f"? Error loading your models: {e}")
    your_results = {}

# === 2. OFFICIAL MAIA-2 (CORRECT API) ===
print("\n2?? OFFICIAL MAIA-2")
try:
    from maia2 import model, inference

    official_model = model.from_pretrained(type="rapid", device="cpu")
    prepared_official = inference.prepare()

    official_move_probs, official_win_prob = inference.inference_each(
        official_model, prepared_official, FEN, ELO_SELF, ELO_OPPO
    )

    print("? Official Maia-2 loaded!")

except Exception as e:
    print(f"? Official failed: {e}")
    official_move_probs = official_win_prob = None

# === 3. COMPARISON ===
print("\n" + "="*80)
print("BLACK'S RESPONSE TO 1.e4 at 1900 ELO")
print("="*80)

def print_top_moves(name, move_probs, win_prob):
    if move_probs:
        print(f"\n{name:12s} | Black Win: {1-win_prob:.1%} ({win_prob:.0%} White)")
        print(" " * 15 + "-" * 50)
        for i, (move, prob) in enumerate(list(move_probs.items())[:6]):
            print(f"   {i+1:2d}. {move:6s} {prob:>7.2%}")
    else:
        print(f"{name:12s} | ? FAILED")

# Your 3 models
for name, (move_probs, win_prob) in your_results.items():
    print_top_moves(name, move_probs, win_prob)

# Official Maia-2
if official_move_probs:
    print_top_moves("OFFICIAL", official_move_probs, official_win_prob)
else:
    print("OFFICIAL     | ? FAILED")
