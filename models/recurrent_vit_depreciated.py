import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
from .vision_transformer import MultiHeadSelfAttention, PatchEmbedding

class RecurrentTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        # Single recurrent step
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

class RecurrentVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_steps=12,
        num_heads=8,
        hidden_size=1024,
        dropout=0.1,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Initialize special tokens and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Single shared encoder layer
        self.recurrent_layer = RecurrentTransformerEncoder(
            embed_dim, num_heads, hidden_size, dropout
        )
        
        # Final classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        self.num_steps = num_steps

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        # Initial embeddings
        x = self.patch_embed(x)
        B, N, D = x.shape
        
        # Add class token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Recurrent processing
        for _ in range(self.num_steps):
            x = self.recurrent_layer(x)
            
        # Final classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

class LitRecurrentVisionTransformer(pl.LightningModule):
    def __init__(self, 
                 model_type = 'latent',
                 lr=1e-3,
                 num_steps=12,
                 weight_decay=0.01,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = RecurrentVisionTransformer(
            num_steps=num_steps,
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


