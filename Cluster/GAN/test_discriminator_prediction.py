#!/usr/bin/env python3
"""
Test discriminator on FEN + move. Outputs human-likeness score.
Usage: python test_discriminator_prediction.py "FEN" "move"
"""

import argparse
import torch
import chess
import yaml
from discriminator_model import Discriminator
from main import board_to_tensor
from utils import get_all_possible_moves


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def fen_move_to_discriminator_input(fen: str, move_str: str, all_moves_dict: dict, device):
    """Convert FEN + move into discriminator inputs (board tensor + move index)"""
    board = chess.Board(fen)

    # Parse move (UCI or SAN)
    if len(move_str) == 4:
        move = chess.Move.from_uci(move_str)
    else:
        move = board.parse_san(move_str)

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {move_str} in position:\n{board}")

    board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
    move_idx = torch.tensor([all_moves_dict[move.uci()]], dtype=torch.long, device=device)

    return board_tensor, move_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fen', help="FEN string")
    parser.add_argument('move', help="Move (UCI like 'e7e5' or SAN like 'e5')")
    parser.add_argument('--checkpoint', default='discriminator_wgan_original.pt', help="Path to discriminator checkpoint")
    args = parser.parse_args()

    # -----------------------------
    # Device setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load config
    # -----------------------------
    with open('config.yaml', 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg_dict)

    # -----------------------------
    # Load move dictionary
    # -----------------------------
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    print(f"Loading checkpoint {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Initialize model
    discriminator = Discriminator(cfg, len(all_moves)).to(device)

    # Load the correct state dict
    if 'model_state_dict' in ckpt:
        discriminator.load_state_dict(ckpt['model_state_dict'])
    else:
        discriminator.load_state_dict(ckpt)  # fallback if raw dict

    discriminator.eval()
    print(f"Discriminator loaded ({count_parameters(discriminator)} parameters)")

    # -----------------------------
    # Convert inputs
    # -----------------------------
    print(f"FEN: {args.fen}")
    print(f"Move: {args.move}")
    board_tensor, move_idx = fen_move_to_discriminator_input(args.fen, args.move, all_moves_dict, device)

    # -----------------------------
    # Predict human-likeness
    # -----------------------------
    with torch.no_grad():
        raw_score = discriminator(board_tensor, move_idx).item()
        prob_human = torch.sigmoid(torch.tensor(raw_score)).item()  # rough 0-1 scale for display

    # -----------------------------
    # Display results
    # -----------------------------
    print("\n" + "="*50)
    print(f" RAW SCORE (WGAN output) : {raw_score:.4f}")
    print(f" HUMAN-LIKENESS (0-1)   : {prob_human:.3f}")
    print("="*50)

    if prob_human > 0.6:
        print(" VERY HUMAN-LIKE (likely Maia2 move)")
    elif prob_human > 0.4:
        print(" MODERATELY HUMAN-LIKE")
    else:
        print(" ENGINE-LIKE (Stockfish-like move)")

    # Show UCI for clarity
    move_uci = [k for k, v in all_moves_dict.items() if v == move_idx.item()][0]
    print(f" UCI move: {move_uci}")


if __name__ == "__main__":
    main()