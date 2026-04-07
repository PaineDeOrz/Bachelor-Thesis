import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from utils import parse_args, get_all_possible_moves, create_elo_dict
from discriminator_dataset import DiscriminatorDataset
from discriminator_model import Discriminator


def train_wgan_discriminator(
    csv_file,
    max_csv_rows=100000,
    epochs=50,
    batch_size=32,
    lr=5e-5,
    clip_value=0.1,
    blunder_label=0.7
):

    # -----------------------------
    # Setup
    # -----------------------------
    cfg = parse_args('config.yaml')
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    dataset = DiscriminatorDataset(
        csv_file,
        all_moves_dict,
        elo_dict,
        cfg,
        max_samples=max_csv_rows
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = Discriminator(cfg, len(all_moves_dict)).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Epochs: {epochs} | Batch size: {batch_size}")
    print(f"WGAN | RMSprop lr={lr} | clip={clip_value} | blunder_label={blunder_label}")

    optimizer = RMSprop(model.parameters(), lr=lr)
    best_wdist = -1e9

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0
        total_real = 0.0
        total_fake = 0.0
        total_batches = 0

        real_correct = 0
        fake_correct = 0
        real_total = 0
        fake_total = 0

        blunder_correct = 0
        blunder_total = 0

        total_correct = 0
        total_samples = 0

        for boards, moves, labels, blunders in loader:

            boards = boards.to(device)
            moves = moves.to(device)
            labels = labels.to(device)
            blunders = blunders.to(device)  # 1 if blunder, 0 otherwise

            optimizer.zero_grad()
            logits = model(boards, moves)
            logits = logits - logits.mean()

            # Adjust human labels for blunders
            adjusted_labels = labels.clone()
            adjusted_labels[blunders.bool()] = blunder_label

            real_mask = adjusted_labels > 0.5
            fake_mask = adjusted_labels <= 0.5

            if real_mask.sum() == 0 or fake_mask.sum() == 0:
                continue

            real_scores = logits[real_mask]
            fake_scores = logits[fake_mask]

            # -----------------------------
            # Wasserstein loss
            # -----------------------------
            loss = fake_scores.mean() - real_scores.mean()
            loss.backward()
            optimizer.step()

            # -----------------------------
            # Weight Clipping
            # -----------------------------
            for p in model.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # -----------------------------
            # Logging
            # -----------------------------
            with torch.no_grad():
                total_loss += loss.item()
                total_real += real_scores.mean().item()
                total_fake += fake_scores.mean().item()
                total_batches += 1

                # Adaptive threshold: midpoint between average real/fake
                threshold = (total_real / max(total_batches, 1) +
                             total_fake / max(total_batches, 1)) / 2
                preds = (logits > threshold).float()

                real_total += real_mask.sum().item()
                fake_total += fake_mask.sum().item()
                real_correct += (preds[real_mask] == 1).sum().item()
                fake_correct += (preds[fake_mask] == 0).sum().item()
                total_correct += (preds == real_mask.float()).sum().item()
                total_samples += labels.size(0)

                # Accuracy ignoring blunders
                blunder_mask = ~blunders.bool()
                blunder_correct += (preds[blunder_mask] == labels[blunder_mask]).sum().item()
                blunder_total += blunder_mask.sum().item()

        if total_batches == 0:
            continue

        avg_loss = total_loss / total_batches
        avg_real = total_real / total_batches
        avg_fake = total_fake / total_batches
        wasserstein_distance = avg_real - avg_fake

        acc = total_correct / total_samples
        real_acc = real_correct / max(real_total, 1)
        fake_acc = fake_correct / max(fake_total, 1)
        blunder_acc = blunder_correct / max(blunder_total, 1)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"W-dist: {wasserstein_distance:.4f} | "
            f"Real: {avg_real:.4f} | Fake: {avg_fake:.4f} | "
            f"Acc: {acc:.2%} | (H: {real_acc:.2%} F: {fake_acc:.2%} | "
            f"Non-blunder: {blunder_acc:.2%})"
        )

        # Save best model
        if wasserstein_distance > best_wdist:
            best_wdist = wasserstein_distance
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "wasserstein_distance": wasserstein_distance
            }, "discriminator_wgan_original.pt")
            print(f"? New best model saved (W-dist={best_wdist:.4f})")

    print("\nTraining complete.")
    print(f"Best Wasserstein distance: {best_wdist:.4f}")


if __name__ == "__main__":
    train_wgan_discriminator(
        "dataset/discriminator_dataset.csv",
        max_csv_rows=1000000,
        batch_size=32,
        epochs=50,
        lr=5e-5,
        clip_value=0.07,
        blunder_label=0.1
    )