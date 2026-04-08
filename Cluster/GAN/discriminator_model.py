"""
Discriminator model for human-vs-fake chess move classification.

Assumptions:
- `cfg.input_channels` matches the board tensor channel count produced by the dataset.
- `num_moves` is the size of the move vocabulary used by the generator/discriminator.
- `boards` has shape [B, channels, 8, 8].
- `move_indices` is either:
  - a 1D tensor of integer move indices [B], or
  - a 2D soft/one-hot distribution over moves [B, num_moves].
- The model is used as part of a GAN/WGAN-style setup, so the output is a
  raw score (logit-like human-likeness value), not a probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator that scores chess moves for human-likeness.

    The model combines:
    - a CNN backbone to extract spatial board features,
    - a small self-attention block over flattened board patches,
    - a move embedding branch,
    - and a final MLP classifier that outputs one scalar score per sample.

    Input:
        boards: Tensor of shape [B, channels, 8, 8]
        move_indices: Either [B] integer move ids or [B, num_moves] soft move weights

    Output:
        Tensor of shape [B] containing human-likeness scores.
    """
    def __init__(self, cfg, num_moves):
        """
        Initialize the discriminator.

        Parameters
        ----------
        cfg : object
            Configuration object containing at least `input_channels`.
        num_moves : int
            Size of the move vocabulary.
        """
        super().__init__()
        self.cfg = cfg
        self.dim = 128  # Hidden dimension used throughout the attention and MLP blocks.

        # CNN backbone that maps the input board tensor to a learned feature map.
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, self.dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Self-attention over flattened board patches.
        # batch_first=True means tensors are interpreted as [B, seq_len, embed_dim].
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.dim)

        # Move embedding table used when the move is represented as an index.
        self.move_emb = nn.Embedding(num_moves, self.dim)

        # Final classifier that combines board and move features.
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim // 2, 1)
        )

    def forward(self, boards, move_indices):
        """
        Run a forward pass through the discriminator.

        Parameters
        ----------
        boards : torch.Tensor
            Board tensor of shape [B, channels, 8, 8].
        move_indices : torch.Tensor
            Either a 1D tensor of move indices [B], or a soft move distribution
            [B, num_moves].

        Returns
        -------
        torch.Tensor
            Human-likeness score per sample, shape [B].
        """
        B = boards.size(0)

        # Extract spatial features from the board.
        x = self.cnn(boards)                # [B, dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)    # [B, 64, dim] -> 64 board patches

        # Apply self-attention over the patch sequence.
        attn_out, _ = self.attn(x, x, x)    # [B, 64, dim]
        x = self.attn_norm(attn_out + x)
        board_feat = x.mean(dim=1)          # [B, dim]

        # Convert the move input into a dense embedding.
        # This supports both hard indices and soft move distributions.
        if move_indices.dim() == 1:
            # Integer indices.
            move_feat = self.move_emb(move_indices)  # [B, dim]
        else:
            # Soft or one-hot distribution.
            # move_indices: [B, num_moves]
            # move_emb.weight: [num_moves, dim]
            move_feat = move_indices @ self.move_emb.weight  # [B, dim]

        # Combine board context with move context.
        final_feat = torch.cat([board_feat, move_feat], dim=1)  # [B, 2*dim]

        # Produce a single scalar score per sample.
        out = self.mlp(final_feat)

        return out.squeeze(-1)