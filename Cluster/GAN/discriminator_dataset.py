import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
from utils import board_to_tensor, map_to_category, create_elo_dict

class DiscriminatorDataset(Dataset):
    """
    Returns (board_tensor, move_idx, label, blunder_flag)
    - board_tensor: tensor representation of the board
    - move_idx: integer index of move
    - label: 1.0 for human, 0.0 for fake
    - blunder_flag: 1 if move is a blunder, 0 otherwise
    """

    def __init__(self, csv_file, all_moves_dict, elo_dict, cfg, max_samples=None):
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.cfg = cfg

        df = pd.read_csv(csv_file, low_memory=False)
        if max_samples:
            df = df[:max_samples]

        self.data = []
        skipped = 0

        print(f"Processing {len(df)} CSV rows...")

        for idx, row in df.iterrows():
            try:
                fen = row['board']
                board = chess.Board(fen)
                elo_self = int(row.get('active_elo', 1500))
                elo_oppo = int(row.get('opponent_elo', 1500))
                active_win = float(row.get('active_won', 0.0))

                # HUMAN MOVE
                human_move = row['move']
                human_move_obj = chess.Move.from_uci(human_move)
                is_blunder = bool(row.get('is_blunder_cp', False))

                move_ply = int(row.get('move_ply', 0))
                #if move_ply < 10:
                #    continue

                #if row.get('low_time', False):
                #    continue

                #if row.get('is_check', 0):
                #    continue

                num_legal_moves = int(row.get('num_legal_moves', 0))
                #if num_legal_moves <= 5:
                #    continue

                cp_loss = float(row.get('cp_loss', 0))
                #if abs(cp_loss) > 300:
                #    continue

                # Prevent contradictory labels
                if row['fake_move'] == row['move']:
                    continue

                if human_move_obj in board.legal_moves and human_move in self.all_moves_dict:
                    self.data.append((
                        fen,
                        human_move,
                        elo_self,
                        elo_oppo,
                        active_win,
                        1.0,          # label
                        is_blunder    # blunder flag
                    ))

                # FAKE MOVE
                fake_move = row['fake_move']
                fake_move_obj = chess.Move.from_uci(fake_move)
                if fake_move_obj in board.legal_moves and fake_move in self.all_moves_dict:
                    self.data.append((
                        fen,
                        fake_move,
                        elo_self,
                        elo_oppo,
                        active_win,
                        0.0,         # label
                        False        # fake moves are not blunders
                    ))

            except Exception:
                skipped += 1
                continue

        human_count = sum(1 for _,_,_,_,_,l,_ in self.data if l == 1.0)
        fake_count = len(self.data) - human_count
        print(f"? Dataset: {len(self.data)} samples ({human_count} human + {fake_count} fake)")
        print(f"? Balance: {100*human_count/len(self.data):.1f}% human" if self.data else "? EMPTY!")
        print(f"? Skipped: {skipped}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move_uci, elo_self, elo_oppo, active_win, label, is_blunder = self.data[idx]
        board = chess.Board(fen)
        board_tensor = board_to_tensor(board)
        move_idx = self.all_moves_dict[move_uci]  # pre-validated

        return (
            board_tensor,
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(is_blunder, dtype=torch.float32)
        )