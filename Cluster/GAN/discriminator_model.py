# discriminator_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Medium-speed discriminator with slightly deeper CNN + light attention.
    Input: [B, channels=12, 8, 8] (same as before)
    Output: [B] human-likeness score (float)
    """
    def __init__(self, cfg, num_moves):
        super().__init__()
        self.cfg = cfg
        self.dim = 128  # hidden dimension for attention

        # --- CNN backbone (slightly deeper) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.input_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, self.dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Light attention over flattened patches ---
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.dim)

        # --- Move embedding ---
        self.move_emb = nn.Embedding(num_moves, self.dim)

        # --- Final classifier ---
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim, self.dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.dim // 2, 1)
        )

    def forward(self, boards, move_indices):
        """
        boards: [B, channels=12, 8, 8]
        move_indices: [B] long tensor
        returns: [B] human-likeness logits
        """
        B = boards.size(0)

        # --- CNN ---
        x = self.cnn(boards)                # [B, dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)    # [B, 64, dim] flatten 8x8 ? 64 patches

        # --- Self-attention ---
        attn_out, _ = self.attn(x, x, x)    # [B, 64, dim]
        x = self.attn_norm(attn_out + x)
        board_feat = x.mean(dim=1)          # [B, dim]

        # --- Move embedding - also supports Gumbel-softmax
        if move_indices.dim() == 1:
            # Integer indices
            move_feat = self.move_emb(move_indices)  # [B, dim]
        else:
            # Soft or one-hot distribution
            # moves: [B, num_moves]
            # move_emb.weight: [num_moves, dim]
            move_feat = move_indices @ self.move_emb.weight  # [B, dim]

        # --- Concatenate board + move ---
        final_feat = torch.cat([board_feat, move_feat], dim=1)  # [B, 2*dim]

        # --- Classifier ---
        out = self.mlp(final_feat)

        return out.squeeze(-1)