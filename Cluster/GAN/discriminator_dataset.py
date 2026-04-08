"""
PyTorch dataset for training a discriminator on chess moves.

Assumptions:
- `csv_file` points to a CSV with at least the following columns:
  board, move, fake_move, active_elo, opponent_elo, active_won,
  is_blunder_cp, move_ply, low_time, is_check, num_legal_moves, cp_loss
- `all_moves_dict` maps UCI move strings to integer move indices.
- `board_to_tensor` converts a python-chess Board to the tensor format
  expected by the model.
- `utils.py` provides board_to_tensor, map_to_category, and create_elo_dict.
- The dataset is used for discriminator training, so each valid row may
  contribute both a human-labeled move and a fake-labeled move.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import chess
from utils import board_to_tensor, map_to_category, create_elo_dict


class DiscriminatorDataset(Dataset):
    """
    Dataset wrapper for discriminator training.

    Each sample returns:
    - board_tensor: tensor encoding the chess position
    - move_idx: integer index of the move in the move vocabulary
    - label: 1.0 for human moves, 0.0 for fake moves
    - blunder_flag: 1.0 if the move is marked as a blunder, otherwise 0.0
    """

    def __init__(self, csv_file, all_moves_dict, elo_dict, cfg, max_samples=None):
        """
        Load and filter the raw CSV data, then build the internal sample list.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file containing chess positions and moves.
        all_moves_dict : dict
            Mapping from UCI move strings to integer indices.
        elo_dict : dict
            Mapping used for Elo-based categorization or conditioning.
        cfg : object
            Configuration object passed through from the training pipeline.
        max_samples : int or None
            Optional cap on the number of CSV rows to read.
        """
        self.all_moves_dict = all_moves_dict
        self.elo_dict = elo_dict
        self.cfg = cfg

        # Load the raw CSV data.
        df = pd.read_csv(csv_file, low_memory=False)
        if max_samples:
            df = df[:max_samples]

        # Internal storage of prevalidated training samples.
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

                # HUMAN MOVE:
                # The original human move from the dataset.
                human_move = row['move']
                human_move_obj = chess.Move.from_uci(human_move)
                is_blunder = bool(row.get('is_blunder_cp', False))

                # Skip very early opening moves, which are handled elsewhere
                # or are not useful for this discriminator setup.
                move_ply = int(row.get('move_ply', 0))
                if move_ply < 10:
                    continue

                # Skip low-time positions because those can be noisy and less stable.
                if row.get('low_time', False):
                    continue

                # Skip checking positions to keep the training set cleaner and more controlled.
                if row.get('is_check', 0):
                    continue

                # Skip positions with very few legal moves, since these are often too constrained.
                num_legal_moves = int(row.get('num_legal_moves', 0))
                if num_legal_moves <= 5:
                    continue

                # Skip positions with extremely large centipawn loss values.
                cp_loss = float(row.get('cp_loss', 0))
                if abs(cp_loss) > 300:
                    continue

                # Prevent contradictory labels: if the fake move is identical to the human move,
                # the same position would appear with conflicting supervision.
                if row['fake_move'] == row['move']:
                    continue

                # Add the human move only if it is legal and present in the move dictionary.
                if human_move_obj in board.legal_moves and human_move in self.all_moves_dict:
                    self.data.append((
                        fen,
                        human_move,
                        elo_self,
                        elo_oppo,
                        active_win,
                        1.0,         # label: human
                        is_blunder    # blunder flag from dataset
                    ))

                # FAKE MOVE:
                # Add the engine-generated or synthetic fake move if it is legal and known.
                fake_move = row['fake_move']
                fake_move_obj = chess.Move.from_uci(fake_move)
                if fake_move_obj in board.legal_moves and fake_move in self.all_moves_dict:
                    self.data.append((
                        fen,
                        fake_move,
                        elo_self,
                        elo_oppo,
                        active_win,
                        0.0,         # label: fake
                        False        # fake moves are not treated as blunders here
                    ))

            except Exception:
                # If a row is malformed or cannot be parsed, skip it silently
                # and continue with the next row.
                skipped += 1
                continue

        human_count = sum(1 for _, _, _, _, _, l, _ in self.data if l == 1.0)
        fake_count = len(self.data) - human_count
        print(f"? Dataset: {len(self.data)} samples ({human_count} human + {fake_count} fake)")
        print(f"? Balance: {100*human_count/len(self.data):.1f}% human" if self.data else "? EMPTY!")
        print(f"? Skipped: {skipped}")

    def __len__(self):
        """
        Return the number of usable samples stored in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Convert one stored sample into tensors for model training.

        Returns
        -------
        tuple
            (board_tensor, move_idx, label, blunder_flag)
        """
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