import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from utils.optimizer_params import create_adam_linear_warmup, create_adam_cosine_warmup

# ---------------- Model Components ----------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3) 
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x)                            # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, H, self.head_dim)  # [B, N, 3, H, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)              # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]              # each [B, H, N, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)                       # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v         # [B, H, N, head_dim]
        out = out.transpose(1, 2)      # [B, N, H, head_dim]
        out = out.reshape(B, N, D)     # [B, N, embed_dim]
        out = self.out_proj(out)
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
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
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = sinusoidal_positional_encoding(num_patches + 1, embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, N, embed_dim]
        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, D]
        
        pos_embed = self.pos_embed.to(x.device)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]  # [B, D]
        logits = self.head(cls_token_final)
        return logits


# ---------------- Lightning Module ----------------

class LitVisionTransformer(pl.LightningModule):
    def __init__(self, 
                 lr=1e-3,
                 img_size=32,
                 patch_size=4,
                 in_channels=3,
                 num_classes=10,
                 embed_dim=384,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 optimizer_type="adam_linear_warmup",
                 max_epochs=100,  # pass for schedulers
                 **kwargs):
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
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        def mixup_loss(logits, soft_targets):
            log_probs = F.log_softmax(logits, dim=1)
            return -(soft_targets * log_probs).sum(dim=1).mean()
        
        # If labels are soft (e.g., from mixup), handle them differently
        if labels.ndim > 1 and labels.size(1) > 1:
            loss = mixup_loss(logits, labels)
            target_class = labels.argmax(dim=1)
        else:
            loss = self.criterion(logits, labels)
            target_class = labels

        preds = logits.argmax(dim=1)
        acc = (preds == target_class).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    

    def configure_optimizers(self):
        opt_type = self.hparams.optimizer_type

        if opt_type == "adam_linear_warmup":
            optimizer, scheduler = create_adam_linear_warmup(
                self.parameters(),
                base_lr=self.hparams.lr,
                betas=(0.9, 0.999),
                weight_decay=0.1,
                warmup_epochs=5,
                total_epochs=self.hparams.max_epochs
            )
        elif opt_type == "adam_cosine_warmup":
            optimizer, scheduler = create_adam_cosine_warmup(
                self.parameters(),
                base_lr=self.hparams.lr,
                betas=(0.9, 0.999),
                weight_decay=0.3,
                warmup_epochs=3,
                total_epochs=self.hparams.max_epochs
            )
        else:
            # Fallback / default Adam
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
            scheduler = None

        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer



# ---------------- Optional Callback(s) ----------------

# models/vision_transformer.py (or a separate callbacks.py file)
class PrintMetricsCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        epoch_mins = elapsed / 60.0
        current_epoch = trainer.current_epoch
        
        # Now just print training metrics. For example:
        train_acc = trainer.callback_metrics.get("train_acc")
        if train_acc is not None:
            train_acc = float(train_acc) * 100.0
            print(f"Epoch {current_epoch} finished in {epoch_mins:.2f} min - train_acc: {train_acc:.2f}%")
        else:
            print(f"Epoch {current_epoch} finished in {epoch_mins:.2f} min - no train_acc logged")

