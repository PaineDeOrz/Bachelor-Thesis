"""
discriminator_train.py

GPU-optimized discriminator pretraining script for the Maia2 GAN setup.

Assumptions:
- `config.yaml` exists and can be parsed by `parse_args`.
- `dataset/discriminator_dataset.csv` exists and has the columns expected by
  `DiscriminatorDataset`.
- The `checkpoints/` directory can be created in the current working directory.
- Weights & Biases is available and the user has logged in or configured it.
- A CUDA-capable GPU may be available; if not, the script falls back to CPU.
- The script is intended to be run as a standalone training job.
- Early stopping is based on reaching `TARGET_ACCURACY`.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import wandb
import time
import os
from datetime import datetime

from discriminator_dataset import DiscriminatorDataset
from discriminator_model import Discriminator
from utils import parse_args, get_all_possible_moves, create_elo_dict, readable_time


# =============================================================================
# CONFIG
# =============================================================================

# Stop training once the discriminator reaches this training accuracy.
TARGET_ACCURACY = 0.80  # Early stop threshold


def setup_device(cfg):
    """
    Select the runtime device and update the config accordingly.

    Returns
    -------
    torch.device
        CUDA if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cfg.device = "cuda"
        # Enable cuDNN benchmark mode for potentially faster convolutions.
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        cfg.device = "cpu"
    return device


def log_device_info(device):
    """
    Print a short summary of the active compute device.
    """
    print("=" * 80)
    print("DEVICE INFO")
    print("=" * 80)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("Running on CPU")
    print("=" * 80)


# =============================================================================
# DATA
# =============================================================================

def create_dataloader(cfg, csv_file):
    """
    Build the training dataloader and the move vocabulary size.

    The move dictionary is built from the full move list so that move indices
    are consistent with the dataset and model embedding table.
    """
    print("\nLoading dataset...")

    all_moves = get_all_possible_moves()
    move_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    dataset = DiscriminatorDataset(
        csv_file=csv_file,
        all_moves_dict=move_dict,
        elo_dict=elo_dict,
        cfg=cfg,
        max_samples=cfg.max_samples_d
    )

    # Use multiple workers only when running on GPU to avoid unnecessary overhead on CPU.
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_d,
        shuffle=True,
        num_workers=8 if cfg.device == "cuda" else 0,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.device == "cuda"),
        drop_last=True
    )

    print(f"Dataset size: {len(dataset):,}")
    print(f"Batches per epoch: {len(loader)}")
    return loader, len(move_dict)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, cfg):
    """
    Run one training epoch for the discriminator.

    This uses automatic mixed precision when running on CUDA, gradient scaling
    for stable AMP training, gradient clipping, and periodic logging to W&B.
    """
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (boards, moves, labels) in enumerate(loader):
        boards = boards.to(device, non_blocking=True)
        moves = moves.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(boards, moves)

            # Use a margin target instead of hard 0/1 targets for stable regression-style training.
            targets = torch.where(
                labels == 1.0,
                torch.full_like(logits, 0.9),
                torch.full_like(logits, -0.9)
            )

            loss = nn.functional.mse_loss(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip gradients to avoid unstable updates.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip_d
        )

        scaler.step(optimizer)
        scaler.update()

        preds = (logits > 0).float()
        correct = (preds == (labels == 1.0).float()).sum().item()

        total_correct += correct
        total_samples += labels.size(0)
        total_loss += loss.item()

        if batch_idx % cfg.log_interval == 0:
            acc = correct / labels.size(0)
            print(
                f"Epoch {epoch+1} | Batch {batch_idx}/{len(loader)} "
                f"| Loss {loss.item():.4f} "
                f"| Acc {acc:.1%} "
                f"| Grad {grad_norm:.3f}"
            )

            wandb.log({
                "batch_loss": loss.item(),
                "batch_acc": acc,
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"]
            })

    scheduler.step()

    epoch_time = time.time() - start_time
    epoch_acc = total_correct / total_samples
    epoch_loss = total_loss / len(loader)

    print("\n" + "=" * 80)
    print(f"EPOCH {epoch+1} COMPLETE")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Accuracy: {epoch_acc:.2%}")
    print(f"Time: {readable_time(epoch_time)}")
    print("=" * 80)

    if device.type == "cuda":
        print(f"GPU Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    wandb.log({
        "epoch_loss": epoch_loss,
        "epoch_accuracy": epoch_acc,
        "epoch_time": epoch_time,
        "epoch": epoch+1
    })

    return epoch_acc


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main training entry point.

    Loads configuration, initializes the device and logging, builds the
    dataset and model, then runs training with checkpointing and early stopping.
    """
    print("Maia2 Discriminator Training (GPU Optimized)")
    print("=" * 80)

    cfg = parse_args("config.yaml")

    # Override discriminator defaults if they are not already set in the config.
    cfg.lr_d = getattr(cfg, "lr_d", 3e-4)
    cfg.weight_decay_d = getattr(cfg, "weight_decay_d", 1e-5)
    cfg.batch_size_d = getattr(cfg, "batch_size_d", 256)
    cfg.epochs_d = getattr(cfg, "epochs_d", 50)
    cfg.grad_clip_d = getattr(cfg, "grad_clip_d", 1.0)
    cfg.log_interval = getattr(cfg, "log_interval", 25)
    cfg.max_samples_d = getattr(cfg, "max_samples_d", None)

    device = setup_device(cfg)
    log_device_info(device)

    wandb.init(
        project="maia2-discriminator",
        name=f"disc-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config=cfg.__dict__,
        save_code=True
    )

    loader, num_moves = create_dataloader(
        cfg,
        "dataset/discriminator_dataset.csv"
    )

    model = Discriminator(cfg, num_moves).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_d,
        weight_decay=cfg.weight_decay_d
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs_d,
        eta_min=cfg.lr_d * 0.1
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)

    print("\nStarting Training...")
    print(f"Early stopping at {TARGET_ACCURACY*100:.0f}% accuracy")

    for epoch in range(cfg.epochs_d):
        acc = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            cfg
        )

        checkpoint_path = f"checkpoints/disc_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        if acc >= TARGET_ACCURACY:
            print(f"\nTarget accuracy {TARGET_ACCURACY*100:.0f}% reached. Stopping early.")
            break

    final_path = "checkpoints/disc_FINAL.pt"
    torch.save(model.state_dict(), final_path)

    print("\nTraining complete.")
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()