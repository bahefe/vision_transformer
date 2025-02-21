import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .vision_transformer import PatchEmbedding, TransformerEncoderBlock

class LatentSpaceVisionTransformer(nn.Module):
    """
    Vision Transformer with recurrent processing and latent space data injection,
    where the "injection" is the output of the initial block, re-added at each 
    recurrent step (similar to RecurrentVisionTransformerWithState).
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth_recurrent=10,
        num_heads=8,
        hidden_size=1024,
        recurrent_hidden_size=1024,
        dropout=0.1
    ):
        super().__init__()
        # 1) Embedding components
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 2) Transformer blocks
        self.initial_block = TransformerEncoderBlock(embed_dim, num_heads, hidden_size, dropout)
        self.recurrent_block = TransformerEncoderBlock(embed_dim, num_heads, recurrent_hidden_size, dropout)
        self.final_block = TransformerEncoderBlock(embed_dim, num_heads, hidden_size, dropout)
        
        # 3) Classification head
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
        # -- (1) Patch embedding + positional embedding --
        patch_embeddings = self.patch_embed(x)               # [B, num_patches, D]
        B, N, D = patch_embeddings.shape
        
        cls_tokens = self.cls_token.expand(B, -1, -1)        # [B, 1, D]
        x = torch.cat((cls_tokens, patch_embeddings), dim=1) # [B, N+1, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # -- (2) Initial block: produce an initial hidden state "h" --
        h = self.initial_block(x)  # shape [B, N+1, D]

        # Save "constant" injection (the output of the initial block) 
        # to re-add at each step, just like h + x in the recurrent-state model.
        constant_injection = h

        # -- (3) Recurrent processing --
        for _ in range(self.depth_recurrent):
            # Combine old hidden state (h) with the constant output 
            # from the initial block at every timestep
            combined = h + constant_injection
            
            # Pass through the recurrent block
            h = self.recurrent_block(combined)

        # -- (4) Final processing + classification --
        h = self.final_block(h)
        h = self.norm(h)
        cls_token_final = h[:, 0]  # class token
        logits = self.head(cls_token_final)
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
