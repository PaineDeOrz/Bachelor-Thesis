import argparse
import os
import glob
import time
import traceback
import datetime
import re
from multiprocessing import Process, Queue

import torch
import torch.nn as nn

from utils import (
    seed_everything,
    readable_time,
    readable_num,
    count_parameters,
    get_all_possible_moves,
    create_elo_dict,
    decompress_zst,
    read_or_create_chunks,
)

from main import (
    MAIA2Model,
    preprocess_thread,
    train_chunks,
    train_chunks_wgan_adversarial,
    read_monthly_data_path,
)

from discriminator_model import Discriminator


# ============================================================
# HELPERS
# ============================================================

def parse_ckpt_name(path):
    """
    Parse checkpoint filenames like:
    lichess_db_standard_rated_2022-04_e1_f4_c195_20260218-205333.pt

    Returns (epoch_idx, file_idx, chunk_idx) where:
    - epoch_idx: 0-based (e1 -> 0)
    - file_idx: 1-based (f4 -> 4)
    - chunk_idx: 0-based (c195 -> 195)
    """
    base = os.path.basename(path)
    base = re.sub(r"\.pt$", "", base)
    m = re.search(r"_e(\d+)_f(\d+)_c(\d+)_", base)
    if not m:
        return None, None, None
    return int(m.group(1)) - 1, int(m.group(2)), int(m.group(3))


# ============================================================
# MAIN
# ============================================================

def run(cfg):

    # --------------------------------------------------------
    # CONFIG PRINT
    # --------------------------------------------------------
    print("\nConfigurations:")
    for arg in vars(cfg):
        print(f"\t{arg}: {getattr(cfg, arg)}")

    seed_everything(cfg.seed)

    # --------------------------------------------------------
    # RESUME STATE
    # --------------------------------------------------------
    current_file_idx = 0
    current_chunk_offset = 0
    resume_epoch = 0

    # --------------------------------------------------------
    # WORKER SETUP
    # --------------------------------------------------------
    num_processes = 1
    print(f"[CONFIG] Using {num_processes} worker(s)")

    save_root = f"../saves/{cfg.lr}_{cfg.batch_size}_{cfg.wd}/"
    os.makedirs(save_root, exist_ok=True)

    # --------------------------------------------------------
    # MOVE + ELO DICTS
    # --------------------------------------------------------
    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------
    model = MAIA2Model(len(all_moves), elo_dict, cfg).cuda()
    print(model)
    print(f"Generator params: {count_parameters(model)}")

    # --------------------------------------------------------
    # ADVERSARIAL SETUP
    # --------------------------------------------------------
    discriminator_plain = None
    discriminator = None
    opt_d = None
    criterion_d = None

    if getattr(cfg, "use_adversarial", False):
        print(">> Adversarial training enabled")
        discriminator_plain = Discriminator(cfg, len(all_moves)).cuda()

        # ---- Optional discriminator pretrain load
        if getattr(cfg, "pretrain_discriminator", False) and hasattr(cfg, "discriminator_path"):
            if os.path.exists(cfg.discriminator_path):
                print(f"Loading pretrained discriminator: {cfg.discriminator_path}")
                disc_ckpt = torch.load(cfg.discriminator_path, map_location="cuda")
                if "discriminator_state_dict" in disc_ckpt:
                    state_dict = disc_ckpt["discriminator_state_dict"]
                else:
                    state_dict = disc_ckpt                
                if list(state_dict.keys())[0].startswith("module."):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                try:
                    discriminator_plain.load_state_dict(state_dict)
                    print("Pretrained discriminator loaded.")
                except Exception as e:
                    print(f"Pretrain failed: {e}")

        print(f"Discriminator params: {count_parameters(discriminator_plain)}")

        if not getattr(cfg, "use_wgan_stgan", False):
            criterion_d = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # OPTIMIZERS
    # --------------------------------------------------------
    opt_g = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()

    # --------------------------------------------------------
    # CHECKPOINT LOAD
    # --------------------------------------------------------
    accumulated_samples = 0
    accumulated_games = 0

    if cfg.from_checkpoint:
        checkpoints = glob.glob(f"{save_root}*.pt")
        if checkpoints:
            # Use specific checkpoint if provided, otherwise latest by mtime
            if hasattr(cfg, "checkpoint_name") and cfg.checkpoint_name:
                latest_ckpt = os.path.join(save_root, cfg.checkpoint_name)
                if not os.path.exists(latest_ckpt):
                    raise FileNotFoundError(f"Checkpoint {latest_ckpt} not found")
            else:
                latest_ckpt = max(checkpoints, key=os.path.getmtime)

            print(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location="cuda")

            model.load_state_dict(checkpoint["model_state_dict"])
            opt_g.load_state_dict(
                checkpoint.get("opt_g_state_dict", checkpoint.get("optimizer_state_dict"))
            )

            if cfg.use_adversarial and "discriminator_state_dict" in checkpoint:
                discriminator_plain.load_state_dict(checkpoint["discriminator_state_dict"])
                opt_d = torch.optim.AdamW(
                    discriminator_plain.parameters(),
                    lr=getattr(cfg, "lr_d", cfg.lr * 2),
                    weight_decay=getattr(cfg, "wd_d", cfg.wd),
                )
                if "opt_d_state_dict" in checkpoint:
                    opt_d.load_state_dict(checkpoint["opt_d_state_dict"])

            accumulated_samples = checkpoint.get("accumulated_samples", 0)
            accumulated_games = checkpoint.get("accumulated_games", 0)
            print(f"Resuming from {readable_num(accumulated_samples)} samples")

            # Parse resume position from filename
            parsed_epoch, parsed_file_idx, parsed_chunk_idx = parse_ckpt_name(latest_ckpt)
            if parsed_file_idx is not None:
                resume_epoch = parsed_epoch
                current_file_idx = parsed_file_idx - 1  # 0-based
                current_chunk_offset = parsed_chunk_idx + 1
                print(
                    f"Resume position: epoch={resume_epoch}, "
                    f"file={current_file_idx} (1-based: {parsed_file_idx}), "
                    f"next_chunk={current_chunk_offset}"
                )

    # DataParallel AFTER loading (preserves loaded state dicts)
    model = nn.DataParallel(model)
    if cfg.use_adversarial:
        discriminator = nn.DataParallel(discriminator_plain)
        if opt_d is None:
            opt_d = torch.optim.AdamW(
                discriminator.parameters(),
                lr=getattr(cfg, "lr_d", cfg.lr * 2),
                weight_decay=getattr(cfg, "wd_d", cfg.wd),
            )

    # --------------------------------------------------------
    # MEMORY REPORT
    # --------------------------------------------------------
    torch.cuda.empty_cache()
    print(f"[MEMORY] Start: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    start_epoch = resume_epoch + 1 if resume_epoch > 0 else 0

    for epoch in range(start_epoch, cfg.max_epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.max_epochs} =====")

        pgn_paths = read_monthly_data_path(cfg)

        for file_idx, pgn_path in enumerate(pgn_paths):
            file_start = time.time()

            # -------------------------------
            # RESUME LOGIC: Skip completed files
            # -------------------------------
            if file_idx < current_file_idx:
                print(f"Skipping completed file {file_idx} ({pgn_path})")
                continue

            # Always decompress first
            decompress_zst(pgn_path + ".zst", pgn_path)
            print(f"Decompress: {readable_time(time.time() - file_start)}")

            pgn_chunks = read_or_create_chunks(pgn_path, cfg)
            if not pgn_chunks:
                print(f"No chunks in {pgn_path}")
                try:
                    os.remove(pgn_path)
                except:
                    pass
                continue

            # Skip chunks within resume file
            if file_idx == current_file_idx and current_chunk_offset > 0:
                print(f"Resuming file {file_idx}: skipping first {current_chunk_offset} chunks")
                pgn_chunks = pgn_chunks[current_chunk_offset:]
                current_chunk_offset = 0  # Only skip once

            print(f"Training {pgn_path} ({len(pgn_chunks)} chunks)")

            pgn_base = os.path.splitext(os.path.basename(pgn_path))[0]
            next_chunk_index = 0
            chunks_since_last_save = 0
            save_interval = 5

            # -------------------------------
            # CHUNK TRAINING LOOP
            # -------------------------------
            for chunk_idx, chunk in enumerate(pgn_chunks):

                queue = Queue(maxsize=1)
                worker = Process(
                    target=preprocess_thread,
                    args=(queue, cfg, pgn_path, [chunk], elo_dict),
                )
                worker.start()

                try:
                    data, game_count, chunk_count = queue.get(timeout=120)
                    max_data_size = 50000
                    data_train = data[:max_data_size]

                    torch.cuda.empty_cache()

                    global_chunk_idx = next_chunk_index

                    # TRAINING CALL
                    if cfg.use_adversarial:
                        if getattr(cfg, "use_wgan_stgan", False):
                            losses = train_chunks_wgan_adversarial(
                                cfg, data_train, model, discriminator, opt_g, opt_d,
                                all_moves_dict, criterion_maia, criterion_side_info, criterion_value, global_chunk_idx
                            )
                            loss, loss_maia, loss_side, loss_value, loss_d, loss_adv = losses
                            print(
                                f"[{next_chunk_index}/{len(pgn_chunks)}] "
                                f"STGAN:{loss:.3f} MAIA:{loss_maia:.3f} "
                                f"Side:{loss_side:.3f} D:{loss_d:.3f} Style:{loss_adv:.3f}"
                            )
                        else:
                            # Note: train_chunks_adversarial must be imported/defined
                            losses = train_chunks_adversarial(
                                cfg, data_train, model, discriminator, opt_g, opt_d,
                                all_moves_dict, criterion_maia, criterion_side_info,
                                criterion_value, criterion_d, next_chunk_idx
                            )
                            loss, loss_maia, loss_side, loss_value, loss_d, loss_adv = losses
                            print(
                                f"[{next_chunk_index}/{len(pgn_chunks)}] "
                                f"Loss:{loss:.3f} MAIA:{loss_maia:.3f} "
                                f"Side:{loss_side:.3f} D:{loss_d:.3f} Adv:{loss_adv:.3f}"
                            )
                    else:
                        losses = train_chunks(
                            cfg, data_train, model, opt_g, all_moves_dict,
                            criterion_maia, criterion_side_info, criterion_value
                        )
                        loss, loss_maia, loss_side, loss_value = losses
                        print(f"[{next_chunk_index}/{len(pgn_chunks)}] Loss:{loss:.3f} MAIA:{loss_maia:.3f}")

                    # Accumulate stats
                    next_chunk_index += chunk_count
                    chunks_since_last_save += chunk_count
                    accumulated_samples += len(data_train)
                    accumulated_games += game_count

                    print(f"Total: {readable_num(accumulated_samples)} pos, {readable_num(accumulated_games)} games")

                    # CHECKPOINT
                    if chunks_since_last_save >= save_interval:
                        ckpt = {
                            "model_state_dict": model.module.state_dict(),
                            "opt_g_state_dict": opt_g.state_dict(),
                            "accumulated_samples": accumulated_samples,
                            "accumulated_games": accumulated_games,
                        }
                        if cfg.use_adversarial:
                            ckpt.update({
                                "discriminator_state_dict": discriminator.module.state_dict(),
                                "opt_d_state_dict": opt_d.state_dict(),
                            })

                        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        ckpt_name = (
                            f"{save_root}"
                            f"{pgn_base}_"
                            f"e{epoch+1}_"
                            f"f{file_idx+1}_"
                            f"c{next_chunk_index}_"
                            f"{timestamp}.pt"
                        )
                        torch.save(ckpt, ckpt_name)
                        print(f"Checkpoint saved: {ckpt_name}")
                        chunks_since_last_save = 0

                except Exception as e:
                    print(f"Worker failed: {repr(e)}")
                    traceback.print_exc()
                finally:
                    worker.terminate()
                    worker.join(timeout=10)

            print(f"### File {file_idx+1}/{len(pgn_paths)}: {readable_time(time.time() - file_start)}")

            try:
                os.remove(pgn_path)
            except:
                pass

    print("Training completed!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg)

    run(cfg)