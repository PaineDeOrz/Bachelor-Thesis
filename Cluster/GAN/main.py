"""
maia2_data_and_model.py

Utilities for:
- parsing monthly PGN data in chunks,
- filtering games by rating / result / event metadata,
- extracting per-move training samples,
- defining MAIA-style datasets,
- defining the MAIA2 model architecture,
- and providing training helpers for supervised and adversarial training.

Assumptions:
- The input PGN files exist in the expected monthly naming scheme:
  `lichess_db_standard_rated_YYYY-MM.pgn`
- `cfg` provides the configuration values used throughout the file
  (e.g. `start_year`, `end_year`, `batch_size`, `dim_cnn`, `dim_vit`, etc.).
- Helper functions imported from `utils` exist and behave as expected:
  `map_to_category`, `board_to_tensor`, `mirror_move`, `extract_clock_time`,
  `get_side_info`, etc.
- The code is intended to run in a GPU-friendly training environment.
- The `data_loader_from_data` helper below wraps already-preprocessed chunk
  data into a PyTorch DataLoader for training.
"""

import chess.pgn
import chess
import pdb
from multiprocessing import Pool, cpu_count, Queue, Process
import torch
import tqdm
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib.concurrent import process_map
import os
import pandas as pd
import time
from einops import rearrange


def process_chunks(cfg, pgn_path, pgn_chunks, elo_dict):
    """
    Process one PGN file in chunks, optionally in parallel.

    Parameters
    ----------
    cfg : object
        Configuration object controlling filtering and processing behavior.
    pgn_path : str
        Path to the PGN file.
    pgn_chunks : list[tuple[int, int]]
        List of byte offsets defining chunk boundaries.
    elo_dict : dict
        Mapping used by `map_to_category` for Elo binning.

    Returns
    -------
    ret : list
        Flattened list of processed move samples.
    count : int
        Total number of accepted games.
    len(pgn_chunks) : int
        Number of chunks processed.
    """
    if cfg.verbose:
        results = process_map(
            process_per_chunk,
            [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks],
            max_workers=len(pgn_chunks),
            chunksize=1
        )
    else:
        with Pool(processes=len(pgn_chunks)) as pool:
            results = pool.map(
                process_per_chunk,
                [(start, end, pgn_path, elo_dict, cfg) for start, end in pgn_chunks]
            )

    # Merge chunk outputs and collect per-chunk frequencies.
    ret = []
    count = 0
    list_of_dicts = []
    for result, game_count, frequency in results:
        ret.extend(result)
        count += game_count
        list_of_dicts.append(frequency)

    # Aggregate the frequency dictionaries from each chunk.
    total_counts = {}
    for d in list_of_dicts:
        for key, value in d.items():
            total_counts[key] = total_counts.get(key, 0) + value

    print(total_counts, flush=True)
    return ret, count, len(pgn_chunks)


def process_per_game(game, white_elo, black_elo, white_win, cfg):
    """
    Extract training samples from a single PGN game.

    For each move after the initial opening section, this function stores:
    - a board representation,
    - the move in UCI format,
    - Elo metadata,
    - and a win/loss signal from the active player's perspective.
    """
    ret = []
    board = game.board()
    moves = list(game.mainline_moves())

    for i, node in enumerate(game.mainline()):
        move = moves[i]

        if i >= cfg.first_n_moves:
            comment = node.comment
            clock_info = extract_clock_time(comment)

            # Alternate perspective so the model sees positions from the side to move.
            if i % 2 == 0:
                board_input = board.fen()
                move_input = move.uci()
                elo_self = white_elo
                elo_oppo = black_elo
                active_win = white_win
            else:
                board_input = board.mirror().fen()
                move_input = mirror_move(move.uci())
                elo_self = black_elo
                elo_oppo = white_elo
                active_win = -white_win

            # Keep only moves where the clock information is missing or acceptable.
            if clock_info is None or clock_info > cfg.clock_threshold:
                ret.append((board_input, move_input, elo_self, elo_oppo, active_win))

        board.push(move)
        if i == cfg.max_ply:
            break

    return ret


def game_filter(game):
    """
    Apply basic metadata-based filters to a PGN game.

    Returns
    -------
    tuple or None
        (game, white_elo, black_elo, white_win) if the game is accepted,
        otherwise None.
    """
    white_elo = game.headers.get("WhiteElo", "?")
    black_elo = game.headers.get("BlackElo", "?")
    time_control = game.headers.get("TimeControl", "?")
    result = game.headers.get("Result", "?")
    event = game.headers.get("Event", "?")

    # Skip games with missing metadata.
    if white_elo == "?" or black_elo == "?" or time_control == "?" or result == "?" or event == "?":
        return

    # Keep only rated games.
    if 'Rated' not in event:
        return

    white_elo = int(white_elo)
    black_elo = int(black_elo)

    # Convert the PGN result into a sign from White's perspective.
    if result == '1-0':
        white_win = 1
    elif result == '0-1':
        white_win = -1
    elif result == '1/2-1/2':
        white_win = 0
    else:
        return

    return game, white_elo, black_elo, white_win


def process_per_chunk(args):
    """
    Process a single PGN chunk by reading games between byte offsets.

    The chunk is filtered, Elo-binned, and converted into per-move samples.
    """
    start_pos, end_pos, pgn_path, elo_dict, cfg = args
    ret = []
    game_count = 0
    frequency = {}

    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        pgn_file.seek(start_pos)
        while pgn_file.tell() < end_pos:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            filtered_game = game_filter(game)
            if filtered_game:
                game, white_elo, black_elo, white_win = filtered_game
                white_elo = map_to_category(white_elo, elo_dict)
                black_elo = map_to_category(black_elo, elo_dict)

                # Count games per Elo pairing so the dataset stays balanced.
                if white_elo < black_elo:
                    range_1, range_2 = black_elo, white_elo
                else:
                    range_1, range_2 = white_elo, black_elo

                freq = frequency.get((range_1, range_2), 0)
                if freq >= cfg.max_games_per_elo_range:
                    continue

                ret_per_game = process_per_game(game, white_elo, black_elo, white_win, cfg)
                ret.extend(ret_per_game)
                if len(ret_per_game):
                    frequency[(range_1, range_2)] = frequency.get((range_1, range_2), 0) + 1

                game_count += 1

    return ret, game_count, frequency


class MAIA1Dataset(torch.utils.data.Dataset):
    """
    Dataset for MAIA1-style training.

    Each item returns:
    - board tensor,
    - move index,
    - side-to-move Elo category,
    - opponent Elo category,
    - legal-move information,
    - side information.
    """
    def __init__(self, data, all_moves_dict, elo_dict, cfg):
        self.all_moves_dict = all_moves_dict
        self.cfg = cfg
        self.data = data.values.tolist()
        self.elo_dict = elo_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move, elo_self, elo_oppo, white_active = self.data[idx]

        if white_active:
            board = chess.Board(fen)
        else:
            board = chess.Board(fen).mirror()
            move = mirror_move(move)

        board_input = board_to_tensor(board)
        move_input = self.all_moves_dict[move]

        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)

        legal_moves, side_info = get_side_info(board, move, self.all_moves_dict)

        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info


class MAIA2Dataset(torch.utils.data.Dataset):
    """
    Dataset for MAIA2-style training.

    Each item returns:
    - board tensor,
    - move index,
    - self Elo category,
    - opponent Elo category,
    - legal-move information,
    - side information,
    - win signal.
    """
    def __init__(self, data, all_moves_dict, cfg):
        self.all_moves_dict = all_moves_dict
        self.data = data
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_input, move_uci, elo_self, elo_oppo, active_win = self.data[idx]
        board = chess.Board(board_input)
        board_input = board_to_tensor(board)
        legal_moves, side_info = get_side_info(board, move_uci, self.all_moves_dict)
        move_input = self.all_moves_dict[move_uci]
        return board_input, move_input, elo_self, elo_oppo, legal_moves, side_info, active_win


class BasicBlock(torch.nn.Module):
    """
    Residual CNN block used in the chess feature extractor.

    The block applies two convolutions with batch normalization and ReLU,
    plus dropout between the convolutions.
    """
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        mid_planes = planes
        self.conv1 = torch.nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_planes)
        self.conv2 = torch.nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)
        return out


class ChessResNet(torch.nn.Module):
    """
    CNN backbone used to process the board before the transformer.

    The network projects the board through several residual blocks and then
    into a final feature map with `cfg.vit_length` channels.
    """
    def __init__(self, block, cfg):
        super(ChessResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(cfg.input_channels, cfg.dim_cnn, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(cfg.dim_cnn)
        self.layers = self._make_layer(block, cfg.dim_cnn, cfg.num_blocks_cnn)
        self.conv_last = torch.nn.Conv2d(cfg.dim_cnn, cfg.vit_length, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_last = torch.nn.BatchNorm2d(cfg.vit_length)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        """
        Build a stack of residual blocks.
        """
        layers = []
        for _ in range(num_blocks):
            layers.append(block(planes, planes, stride))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.conv_last(out)
        out = self.bn_last(out)
        return out


class FeedForward(nn.Module):
    """
    Transformer-style feed-forward block with normalization and dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EloAwareAttention(nn.Module):
    """
    Multi-head self-attention with an Elo-dependent query bias.

    The Elo embedding is projected and added to the query branch so that the
    attention pattern can depend on the players' rating context.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., elo_dim=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.elo_query = nn.Linear(elo_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x, elo_emb):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        elo_effect = self.elo_query(elo_emb).view(x.size(0), self.heads, 1, -1)
        q = q + elo_effect
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Stack of Elo-aware attention blocks and feed-forward layers.
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., elo_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.elo_layers = nn.ModuleList([])
        for _ in range(depth):
            self.elo_layers.append(nn.ModuleList([
                EloAwareAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, elo_dim=elo_dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, elo_emb):
        for attn, ff in self.elo_layers:
            x = attn(x, elo_emb) + x
            x = ff(x) + x
        return self.norm(x)


class MAIA2Model(torch.nn.Module):
    """
    MAIA2 chess model with three outputs:
    - move logits,
    - side information logits,
    - value prediction.
    """
    def __init__(self, output_dim, elo_dict, cfg):
        super(MAIA2Model, self).__init__()
        self.cfg = cfg
        self.chess_cnn = ChessResNet(BasicBlock, cfg)
        heads = 16
        dim_head = 64
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(8 * 8, cfg.dim_vit),
            nn.LayerNorm(cfg.dim_vit),
        )
        self.transformer = Transformer(
            cfg.dim_vit,
            cfg.num_blocks_vit,
            heads,
            dim_head,
            mlp_dim=cfg.dim_vit,
            dropout=0.1,
            elo_dim=cfg.elo_dim * 2
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.vit_length, cfg.dim_vit))
        self.fc_1 = nn.Linear(cfg.dim_vit, output_dim)
        self.fc_2 = nn.Linear(cfg.dim_vit, output_dim + 6 + 6 + 1 + 64 + 64)
        self.fc_3_1 = nn.Linear(cfg.dim_vit, 128)
        self.fc_3 = nn.Linear(128, 1)
        self.elo_embedding = torch.nn.Embedding(len(elo_dict), cfg.elo_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.last_ln = nn.LayerNorm(cfg.dim_vit)

    def forward(self, boards, elos_self, elos_oppo):
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.cfg.input_channels, 8, 8)
        embs = self.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        x = self.to_patch_embedding(embs)
        x += self.pos_embedding
        x = self.dropout(x)

        elos_emb_self = self.elo_embedding(elos_self)
        elos_emb_oppo = self.elo_embedding(elos_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)

        x = self.transformer(x, elos_emb).mean(dim=1)
        x = self.last_ln(x)

        logits_maia = self.fc_1(x)
        logits_side_info = self.fc_2(x)
        logits_value = self.fc_3(self.dropout(torch.relu(self.fc_3_1(x)))).squeeze(dim=-1)
        return logits_maia, logits_side_info, logits_value


def read_monthly_data_path(cfg):
    """
    Build the list of monthly PGN files to process from the config range.
    """
    print('Training Data:', flush=True)
    pgn_paths = []

    for year in range(cfg.start_year, cfg.end_year + 1):
        start_month = cfg.start_month if year == cfg.start_year else 1
        end_month = cfg.end_month if year == cfg.end_year else 12

        for month in range(start_month, end_month + 1):
            formatted_month = f"{month:02d}"
            pgn_path = cfg.data_root + f"/lichess_db_standard_rated_{year}-{formatted_month}.pgn"
            # Skip 2019-12 because it is excluded from the intended training range.
            if year == 2019 and month == 12:
                continue
            print(pgn_path, flush=True)
            pgn_paths.append(pgn_path)

    return pgn_paths


def train_chunks(cfg, data, model, opt_g, all_moves_dict, criterion_maia, criterion_side_info, criterion_value):
    """
    Supervised training loop over preprocessed chunk data.
    """
    model.train()
    total_loss = 0
    total_maia = 0
    total_side = 0
    total_value = 0

    for batch in data_loader_from_data(data, all_moves_dict, cfg):  # create a DataLoader wrapper
        boards, moves, elo_self, elo_oppo, legal_moves, side_info, active_win = batch
        boards = boards.cuda()
        moves = moves.cuda()
        elo_self = elo_self.cuda()
        elo_oppo = elo_oppo.cuda()
        side_info = side_info.cuda()
        active_win = active_win.cuda()

        opt_g.zero_grad()
        logits_maia, logits_side, logits_value = model(boards, elo_self, elo_oppo)

        loss_maia = criterion_maia(logits_maia, moves)
        loss_side = criterion_side_info(logits_side, side_info)
        loss_value = criterion_value(logits_value, active_win.float())

        loss = loss_maia + loss_side + loss_value
        loss.backward()
        opt_g.step()

        total_loss += loss.item()
        total_maia += loss_maia.item()
        total_side += loss_side.item()
        total_value += loss_value.item()

    return total_loss/len(data), total_maia/len(data), total_side/len(data), total_value/len(data)


def train_chunks_adversarial(cfg, data, model, discriminator, opt_g, opt_d, all_moves_dict,
                             criterion_maia, criterion_side_info, criterion_value, criterion_d):
    """
    Simple adversarial training loop using a discriminator updated on
    real and generator-produced fake moves.
    """
    model.train()
    discriminator.train()

    total_loss = 0
    total_maia = 0
    total_side = 0
    total_value = 0
    total_d = 0
    total_adv = 0

    for batch in data_loader_from_data(data, all_moves_dict, cfg):
        boards, moves, elo_self, elo_oppo, legal_moves, side_info, active_win = batch
        boards = boards.cuda()
        moves = moves.cuda()
        elo_self = elo_self.cuda()
        elo_oppo = elo_oppo.cuda()
        side_info = side_info.cuda()
        active_win = active_win.cuda()

        # 1) Update discriminator
        opt_d.zero_grad()
        logits_real = discriminator(boards, moves)
        labels_real = torch.ones_like(logits_real)
        loss_d_real = criterion_d(logits_real, labels_real)

        # Generate fake moves without backprop through the generator.
        with torch.no_grad():
            logits_gen, _, _ = model(boards, elo_self, elo_oppo)
            fake_moves = logits_gen.argmax(dim=1)

        logits_fake = discriminator(boards, fake_moves)
        labels_fake = torch.zeros_like(logits_fake)
        loss_d_fake = criterion_d(logits_fake, labels_fake)

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        opt_d.step()

        # 2) Update generator
        opt_g.zero_grad()
        logits_maia, logits_side, logits_value = model(boards, elo_self, elo_oppo)

        # Adversarial loss encourages the generator to produce moves that the
        # discriminator classifies as human-like.
        logits_for_adv = discriminator(boards, logits_maia.argmax(dim=1))
        adv_loss = criterion_d(logits_for_adv, torch.ones_like(logits_for_adv))

        loss_maia = criterion_maia(logits_maia, moves)
        loss_side = criterion_side_info(logits_side, side_info)
        loss_value = criterion_value(logits_value, active_win.float())

        loss_g = loss_maia + loss_side + loss_value + adv_loss
        loss_g.backward()
        opt_g.step()

        total_loss += loss_g.item()
        total_maia += loss_maia.item()
        total_side += loss_side.item()
        total_value += loss_value.item()
        total_d += loss_d.item()
        total_adv += adv_loss.item()

    return total_loss/len(data), total_maia/len(data), total_side/len(data), total_value/len(data), total_d/len(data), total_adv/len(data)


def train_chunks_wgan_adversarial(cfg, data, model, discriminator, opt_g, opt_d, all_moves_dict,
                                  criterion_maia, criterion_side_info, criterion_value, global_chunk_idx=0):
    """
    WGAN-style adversarial training loop with:
    - multiple discriminator steps per generator step,
    - gradient clipping,
    - optional weight clipping,
    - differentiable move sampling via Gumbel-Softmax.
    """
    model.train()
    discriminator.train()

    print(f"[CHUNK {global_chunk_idx}] WGAN adversarial training (D always unfrozen)")

    dataloader = data_loader_from_data(data, all_moves_dict, cfg)
    num_batches = len(dataloader)

    total_g = total_maia = total_side = total_value = total_d = total_adv = 0.0
    style_lambda = cfg.style_lambda
    d_steps = cfg.d_steps_per_g

    for batch_idx, batch in enumerate(dataloader):
        boards, moves_gt, elo_self, elo_oppo, legal_moves, side_info, active_win = [t.cuda() for t in batch]

        B = boards.size(0)

        # Shape checks are helpful when debugging dataset / model mismatches.
        if batch_idx == 0:
            print(f"[DEBUG] boards shape raw: {boards.shape}")
            print(f"[DEBUG] moves_gt shape raw: {moves_gt.shape}")
            print(f"[DEBUG] elo_self shape: {elo_self.shape}")
            print(f"[DEBUG] elo_oppo shape: {elo_oppo.shape}")

        # Ensure boards is [B, C, 8, 8].
        if boards.dim() == 5 and boards.size(1) == 1:
            boards = boards.squeeze(1)
        assert boards.dim() == 4, f"boards dim should be 4, got {boards.dim()} with shape {boards.shape}"

        # Ensure move labels are flat [B].
        moves_gt = moves_gt.view(-1)
        assert moves_gt.shape[0] == B, f"moves_gt batch mismatch: {moves_gt.shape} vs B={B}"

        # ====================================================
        # 1) DISCRIMINATOR (CRITIC) UPDATE
        # ====================================================
        for d_step in range(d_steps):
            opt_d.zero_grad()

            # Real moves.
            logits_real = discriminator(boards, moves_gt)
            loss_real = torch.mean(logits_real)

            # Fake moves sampled from the generator.
            with torch.no_grad():
                logits_maia_d, _, _ = model(boards, elo_self, elo_oppo)
                probs = torch.softmax(logits_maia_d, dim=-1)              # [B, num_moves]
                topk_probs, topk_moves = torch.topk(probs, k=3, dim=-1)   # [B, 3]
                idx = torch.randint(0, 3, (B,), device=boards.device)     # [B]
                fake_moves = topk_moves[torch.arange(B, device=boards.device), idx]  # [B]

            if batch_idx == 0 and d_step == 0:
                print(f"[DEBUG] logits_maia_d shape: {logits_maia_d.shape}")
                print(f"[DEBUG] topk_moves shape: {topk_moves.shape}")
                print(f"[DEBUG] fake_moves shape before view: {fake_moves.shape}")

            fake_moves = fake_moves.view(-1)  # Force [B]
            assert fake_moves.shape[0] == B, f"fake_moves batch mismatch: {fake_moves.shape} vs B={B}"

            logits_fake = discriminator(boards, fake_moves)
            loss_fake = torch.mean(logits_fake)

            loss_d = -loss_real + loss_fake
            loss_d.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), getattr(cfg, "disc_grad_clip", 0.25))
            opt_d.step()

            # Optional WGAN weight clipping.
            if getattr(cfg, "weight_clip", 0.0) > 0:
                for p in discriminator.parameters():
                    p.data.clamp_(-cfg.weight_clip, cfg.weight_clip)

            total_d += loss_d.item()

        # ====================================================
        # 2) GENERATOR UPDATE
        # ====================================================
        opt_g.zero_grad()
        logits_maia, logits_side, logits_value = model(boards, elo_self, elo_oppo)

        loss_maia = criterion_maia(logits_maia, moves_gt)
        loss_side = criterion_side_info(logits_side, side_info) * getattr(cfg, "side_info_coefficient", 1.0)
        loss_value = criterion_value(logits_value, active_win.float()) * getattr(cfg, "value_coefficient", 1.0)

        # Differentiable move sampling through Gumbel-Softmax.
        tau = getattr(cfg, "gumbel_tau", 1.0)
        gumbel_moves = F.gumbel_softmax(
            logits_maia,
            tau=tau,
            hard=True,
            dim=-1
        )  # [B, num_moves]
        if batch_idx == 0:
            print(f"[DEBUG] gumbel_moves shape: {gumbel_moves.shape}")
        logits_adv = discriminator(boards, gumbel_moves)
        loss_adv = -torch.mean(logits_adv)

        loss_g = loss_maia + loss_side + loss_value + style_lambda * loss_adv
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, "gen_grad_clip", 1.0))
        opt_g.step()

        total_g += loss_g.item()
        total_maia += loss_maia.item()
        total_side += loss_side.item()
        total_value += loss_value.item()
        total_adv += loss_adv.item()

        if batch_idx % getattr(cfg, "log_interval", 50) == 0:
            print(
                f"[BATCH {batch_idx}/{num_batches}] "
                f"G:{loss_g:.3f} (M:{loss_maia:.3f}, Side:{loss_side:.3f}, Val:{loss_value:.3f}, Adv:{loss_adv:.3f}) "
                f"D:{loss_d:.3f}"
            )

    denom = max(1, num_batches)
    avg_g = total_g / denom
    avg_maia = total_maia / denom
    avg_side = total_side / denom
    avg_value = total_value / denom
    avg_d = total_d / denom
    avg_adv = total_adv / denom

    print(
        f"[AVG ACTIVE] STGAN:{avg_g:.3f} MAIA:{avg_maia:.3f} "
        f"Side:{avg_side:.3f} D:{avg_d:.3f} Style:{avg_adv:.3f}"
    )
    return avg_g, avg_maia, avg_side, avg_value, avg_d, avg_adv


def preprocess_thread(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict):
    """
    Worker wrapper that preprocesses a subset of chunks and pushes results
    into a multiprocessing queue.
    """
    try:
        data, game_count, chunk_count = process_chunks(cfg, pgn_path, pgn_chunks_sublist, elo_dict)
        queue.put((data, game_count, chunk_count))
        del data
        torch.cuda.empty_cache()
    except Exception as e:
        queue.put(([], 0, 0))
        raise e


def data_loader_from_data(data, all_moves_dict, cfg):
    """
    Wrap preprocessed raw data in a DataLoader for training.

    This uses MAIA2Dataset so the training loop can iterate over boards, moves,
    Elo categories, side info, and value labels.
    """
    dataset = MAIA2Dataset(data, all_moves_dict, cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size if not getattr(cfg, 'disc_batch_size', None) else cfg.disc_batch_size,
        shuffle=True,
        num_workers=0,  # already using multiprocess chunks
        pin_memory=True,
    )
    return loader