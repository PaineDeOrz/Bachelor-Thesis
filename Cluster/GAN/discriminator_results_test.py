#!/usr/bin/env python3
import torch
import chess
from utils import (
    parse_args,
    get_all_possible_moves,
    create_elo_dict,
    board_to_tensor,
)
from discriminator_model import Discriminator


# ----------------------------
# Model loading
# ----------------------------
def load_discriminator(model_path="discriminator_test_cpu.pt"):
    cfg = parse_args("config.yaml")

    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}

    elo_dict = create_elo_dict()

    model = Discriminator(cfg, len(all_moves_dict))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, all_moves_dict, elo_dict, cfg


# ----------------------------
# Inference
# ----------------------------
def test_position(
    model,
    all_moves_dict,
    fen: str,
    move_input: str,
):
    # Enforce full FEN
    fen_parts = fen.strip().split()
    if len(fen_parts) != 6:
        raise ValueError(
            "FEN must have 6 fields: "
            "board side castling enpassant halfmove fullmove"
        )

    board = chess.Board(fen)

    # Parse move (SAN preferred, fallback to UCI)
    try:
        move = board.parse_san(move_input)
    except ValueError:
        try:
            move = chess.Move.from_uci(move_input)
        except ValueError:
            raise ValueError(f"Invalid move format: {move_input}")

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for this position: {move_input}")

    move_uci = move.uci()

    if move_uci not in all_moves_dict:
        raise ValueError(f"Move not in model vocabulary: {move_uci}")

    board_tensor = (
        torch.as_tensor(board_to_tensor(board), dtype=torch.float32)
        .flatten()
        .unsqueeze(0)
    )
    move_idx = torch.tensor(
        [all_moves_dict[move_uci]], dtype=torch.long
    )

    with torch.no_grad():
        logit = model(board_tensor, move_idx)
        prob_human = torch.sigmoid(logit).item()

    print("\n" + "=" * 60)
    print("FEN:")
    print(board.fen())
    print(f"Move: {move_input} ? {move_uci}")
    print(f"Model output (sigmoid): {prob_human:.4f}")
    print("? Inference completed successfully")
    print("=" * 60)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    print("?? Loading discriminator...")
    model, all_moves_dict, elo_dict, cfg = load_discriminator()
    print("? Ready")

    print("\nEnter a full FEN (6 fields), then a move.")
    print("Example:")
    print("  FEN  r1bq1rk1/pppn1pbp/3p1np1/3Pp3/2P1P1P1/2N1B2P/PP3P2/R2QKBNR b KQ - 0 8")
    print("  MOVE Nxe4")
    print("Type 'quit' to exit.")

    while True:
        try:
            fen = input("\nFEN  > ").strip()
            if fen.lower() == "quit":
                break

            move = input("MOVE > ").strip()
            if move.lower() == "quit":
                break

            test_position(model, all_moves_dict, fen, move)

        except KeyboardInterrupt:
            print("\n?? Bye")
            break
        except Exception as e:
            print(f"? {e}")
