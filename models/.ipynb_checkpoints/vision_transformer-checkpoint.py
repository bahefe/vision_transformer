import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Single linear layer for Q/K/V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, D = x.shape
        H, h_dim = self.num_heads, self.head_dim

        # Project all at once [3*B, N, (H * h_dim)]
        qkv = self.qkv(x).chunk(3, dim=-1)  # Tuple of [B, N, D] * 3
        
        # Reshape without permute
        q, k, v = [t.view(B, N, H, h_dim).transpose(1, 2) for t in qkv]

        # Use PyTorch's optimized attention (Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale
        )

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        
        return self.out_proj(attn_output)

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


def sinusoidal_positional_encoding(seq_len, embed_dim):
    position = torch.arange(seq_len).unsqueeze(1)  
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

    pos_embed = torch.zeros(seq_len, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)

    return pos_embed.unsqueeze(0)  # [1, seq_len, embed_dim]


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=4,
        hidden_size=1024,  # Changed from mlp_ratio
        dropout=0.0,
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
                 lr=1e-3,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 embed_dim=256,
                 depth=6,
                 num_heads=4,
                 hidden_size=1024,  # Changed
                 dropout=0.1,
                 label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
    
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            hidden_size=hidden_size,  # Changed
            dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        
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
        # Create the optimizer with weight decay.
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay  # <-- weight decay is added here
        )
        # Define the cosine annealing scheduler.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # or any desired period
            eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

