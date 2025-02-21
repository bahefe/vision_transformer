import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .vision_transformer import PatchEmbedding, TransformerEncoderBlock

class LatentSpaceVisionTransformer(nn.Module):
    """
    Vision Transformer with recurrent processing and latent space data injection.
    
    The model first embeds the image into patches, adds a class token and 
    positional embeddings, and then processes them through:
    
    1. An initial transformer block.
    2. Several recurrent steps (default 10) where, in each iteration, the original 
       patch embeddings are injected (added) into the current state (for patch tokens),
       then processed with a recurrent block. The recurrent block uses an inflated 
       MLP hidden dimension (default 14800) so that its parameter count fills the budget.
    3. A final transformer block, followed by layer normalization and a 
       classification head.
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth_recurrent=10,            # Increased to 10 recurrent iterations
        num_heads=8,
        hidden_size=1024,              # For initial and final blocks
        recurrent_hidden_size=1024,   # For the recurrent block (inflated)
        dropout=0.1
    ):
        super().__init__()
        # 1. Embedding components
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 2. Transformer blocks
        self.initial_block = TransformerEncoderBlock(embed_dim, num_heads, hidden_size, dropout)
        self.recurrent_block = TransformerEncoderBlock(embed_dim, num_heads, recurrent_hidden_size, dropout)
        self.final_block = TransformerEncoderBlock(embed_dim, num_heads, hidden_size, dropout)
        
        # 3. Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.depth_recurrent = depth_recurrent
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, x):
        """
        x: image tensor of shape (B, C, H, W)
        """
        # Obtain patch embeddings: [B, num_patches, embed_dim]
        patch_embeddings = self.patch_embed(x)
        B, N, D = patch_embeddings.shape
        
        # Concatenate class token and add positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, patch_embeddings), dim=1)  # [B, num_patches+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Initial processing through a transformer block
        state = self.initial_block(x)
        
        # Recurrent processing with latent space data injection
        for _ in range(self.depth_recurrent):
            # For patch tokens (all tokens except the class token),
            # add the original patch embeddings to the current state.
            state_patches = state[:, 1:, :] + patch_embeddings
            # Keep the class token unchanged.
            state_cls = state[:, :1, :]
            # Concatenate and pass through the recurrent block.
            state = torch.cat([state_cls, state_patches], dim=1)
            state = self.recurrent_block(state)
        
        # Final processing
        state = self.final_block(state)
        
        # Classification head: layer norm then linear classifier using the class token.
        state = self.norm(state)
        cls_token = state[:, 0]  # Use class token
        logits = self.head(cls_token)
        return logits

class LitLatentSpaceVisionTransformer(pl.LightningModule):
    """
    PyTorch Lightning module for the Latent Space Vision Transformer.
    """
    def __init__(self, 
                 model_type="latent_space",
                 lr=1e-4,
                 depth_recurrent=10,
                 weight_decay=0.01,
                 batch_size=None,
                 epochs=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = LatentSpaceVisionTransformer(
            depth_recurrent=depth_recurrent,
            **kwargs
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Accuracy (handles both one-hot and integer labels)
        preds = logits.argmax(dim=1)
        target_classes = labels.argmax(dim=1) if labels.dim() > 1 else labels
        acc = (preds == target_classes).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
