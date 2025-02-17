import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Critical for (B, N, D) input format
        )

    def forward(self, x):
        # x shape: [B, N, D]
        attn_output, _ = self.attn(x, x, x)  # Self-attention
        return attn_output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Use explicit hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)       # [B, embed_dim, grid_size, grid_size]
        x = x.flatten(2)       # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x




class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=12,
        num_heads=8,
        hidden_size=1024,  # Changed from mlp_ratio
        dropout=0.1,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks with explicit hidden_size
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_size=hidden_size,  # Changed
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Original class token initialization
        nn.init.normal_(self.cls_token, std=1e-6)
        
        # Initialize positional embeddings (new)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # More ViT-like initialization
        
        # Head initialization
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B, N, D = x.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        
        # Add learned positional embeddings (modified)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # [B, D]
        logits = self.head(cls_token_final)
        return logits


class LitVisionTransformer(pl.LightningModule):
    def __init__(self, 
                 model_type="standard",
                 lr=1e-4,
                 depth=12,
                 weight_decay=0.01,
                 batch_size=None,
                 epochs=None,
                 swap_interval: Optional[float] = None,
                 swap_strategy: Optional[int] = None,
                 **kwargs):
        super().__init__()
        # Save all hyperparameters (including model_type, batch_size, epochs, etc.)
        self.save_hyperparameters()
        
        # Instantiate the VisionTransformer using depth from the arguments
        # and the remaining keyword arguments
        self.model = VisionTransformer(
            depth=depth,
            **kwargs
        )
        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Accuracy calculation (handles both soft/hard labels)
        preds = logits.argmax(dim=1)
        target_classes = labels.argmax(dim=1) if labels.dim() > 1 else labels  # Key change
        acc = (preds == target_classes).float().mean()

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Validation labels are always hard targets - no need for argmax check
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()  # Direct comparison

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Test labels are always hard targets
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()  # Direct comparison

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Just AdamW, no scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

