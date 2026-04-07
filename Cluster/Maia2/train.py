import argparse
import os
import glob
from multiprocessing import Process, Queue, cpu_count
import time
from utils import seed_everything, readable_time, readable_num, count_parameters
from utils import get_all_possible_moves, create_elo_dict
from utils import decompress_zst, read_or_create_chunks
from main import MAIA2Model, preprocess_thread, train_chunks, read_monthly_data_path
import torch
import torch.nn as nn
import pdb


def run(cfg):
    
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    num_processes = cpu_count() - cfg.num_cpu_left

    save_root = f'../saves/{cfg.lr}_{cfg.batch_size}_{cfg.wd}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    print(model, flush=True)
    model = model.cuda()
    model = nn.DataParallel(model)
    criterion_maia = nn.CrossEntropyLoss()
    criterion_side_info = nn.BCEWithLogitsLoss()
    criterion_value = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    N_params = count_parameters(model)
    print(f'Trainable Parameters: {N_params}', flush=True)

    accumulated_samples = 0
    accumulated_games = 0
    last_pgn_file_index = 0
    last_chunk_counter = 0  # absolute chunk index to resume from

    if cfg.from_checkpoint:
        all_checkpoints = glob.glob(f'{save_root}*.pt')
        if all_checkpoints:
            latest_ckpt = max(all_checkpoints, key=os.path.getmtime)
            checkpoint = torch.load(latest_ckpt)
            print(f'Loaded most recent checkpoint: {latest_ckpt}', flush=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            accumulated_samples = checkpoint['accumulated_samples']
            accumulated_games = checkpoint['accumulated_games']
            last_pgn_file_index = checkpoint.get('last_pgn_file_index', 0)
            last_chunk_counter = checkpoint.get('last_chunk_counter', 0)
            print(f'Resuming from {readable_num(accumulated_samples)} samples', flush=True)
            print(f'Skipping to PGN file {last_pgn_file_index}, chunk {last_chunk_counter}')
        else:
            print(f'No checkpoints found in {save_root}', flush=True)
            return

    for epoch in range(cfg.max_epochs):
        print(f'Epoch {epoch + 1}', flush=True)
        pgn_paths = read_monthly_data_path(cfg)
        
        num_file = 0
        for pgn_path in pgn_paths:
            if num_file < last_pgn_file_index:
                print(f'Skipping PGN file {num_file + 1}/{len(pgn_paths)}: {pgn_path}', flush=True)
                num_file += 1
                continue
                
            start_time = time.time()
            decompress_zst(pgn_path + '.zst', pgn_path)
            print(f'Decompressing {pgn_path} took {readable_time(time.time() - start_time)}', flush=True)

            pgn_chunks = read_or_create_chunks(pgn_path, cfg)
            print(f'Training {pgn_path} with {len(pgn_chunks)} chunks.', flush=True)
            
            pgn_chunks_sublists = []
            for i in range(0, len(pgn_chunks), num_processes):
                pgn_chunks_sublists.append(pgn_chunks[i:i + num_processes])
            
            # Determine starting point for resume
            if last_chunk_counter > 0:
                skipped_chunks = min(last_chunk_counter, len(pgn_chunks))
                print(f'Skipping {skipped_chunks}/{len(pgn_chunks)} chunks (already processed)', flush=True)
                chunk_start_idx = skipped_chunks
            else:
                chunk_start_idx = 0

            sublist_idx = max(0, (chunk_start_idx // num_processes))
            
            while sublist_idx < len(pgn_chunks_sublists):
                pgn_chunks_sublist = pgn_chunks_sublists[sublist_idx]
                
                # Skip if we've already processed all chunks in this sublist
                sublist_start_chunk = sublist_idx * num_processes
                if chunk_start_idx >= sublist_start_chunk + len(pgn_chunks_sublist):
                    sublist_idx += 1
                    continue
                
                print(f'Processing sublist {sublist_idx+1}/{len(pgn_chunks_sublists)}', flush=True)
                
                queue = Queue(maxsize=cfg.queue_length)
                worker = Process(target=preprocess_thread, args=(queue, cfg, pgn_path, pgn_chunks_sublist, elo_dict))
                worker.start()
                
                # Absolute index of processed chunks for this PGN file
                next_chunk_index = max(chunk_start_idx, sublist_start_chunk)

                # Counts chunks since last save, just for save_interval logic
                chunks_since_last_save = (
                    last_chunk_counter if sublist_idx == (chunk_start_idx // num_processes) else 0
                )

                save_interval = 120
                
                while worker.is_alive() or not queue.empty():
                    try:
                        data, game_count, chunk_count = queue.get(timeout=1.0)
                        loss, loss_maia, loss_side_info, loss_value = train_chunks(
                            cfg, data, model, optimizer, all_moves_dict, 
                            criterion_maia, criterion_side_info, criterion_value
                        )
                        
                        # Advance absolute and relative chunk counters
                        next_chunk_index += chunk_count
                        chunks_since_last_save += chunk_count
                        accumulated_samples += len(data)
                        accumulated_games += game_count

                        if chunks_since_last_save >= save_interval or next_chunk_index >= len(pgn_chunks):
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'accumulated_samples': accumulated_samples,
                                'accumulated_games': accumulated_games,
                                'last_pgn_file_index': num_file,
                                # For resume: absolute index of next chunk to process
                                'last_chunk_counter': next_chunk_index
                            }, f'{save_root}epoch_{epoch + 1}_{pgn_path[-11:]}_chunk_{next_chunk_index}.pt')
                            print(f'Checkpoint saved at chunk {next_chunk_index}/{len(pgn_chunks)}', flush=True)
                            chunks_since_last_save = 0  # reset after saving
                        
                        print(f'[{next_chunk_index}/{len(pgn_chunks)}]', flush=True)
                        print(f'[# Positions]: {readable_num(accumulated_samples)}', flush=True)
                        print(f'[# Games]: {readable_num(accumulated_games)}', flush=True)
                        print(f'[# Loss]: {loss:.4f} | [# Loss MAIA]: {loss_maia:.4f} | [# Loss Side Info]: {loss_side_info:.4f} | [# Loss Value]: {loss_value:.4f}', flush=True)
                        
                    except:
                        if not worker.is_alive():
                            break
                
                worker.join()
                sublist_idx += 1
                chunk_start_idx = 0  # Reset for next sublist
                last_chunk_counter = 0

            num_file += 1
            last_chunk_counter = 0
            print(f'### ({num_file} / {len(pgn_paths)}) Took {readable_time(time.time() - start_time)} to train {pgn_path} with {len(pgn_chunks)} chunks.', flush=True)
            os.remove(pgn_path)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accumulated_samples': accumulated_samples,
                'accumulated_games': accumulated_games,
                'last_pgn_file_index': num_file,
                'last_chunk_counter': 0
            }, f'{save_root}epoch_{epoch + 1}_{pgn_path[-11:]}_complete.pt')
