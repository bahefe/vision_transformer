import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .vision_transformer import MultiHeadSelfAttention, PatchEmbedding

class RecurrentTransformerEncoderWithState(nn.Module):
    """
    A recurrent transformer block that updates a hidden state 'h' given
    a constant input 'x'. This version includes two LayerNorm layers:
    one before the self-attention and one before the MLP.
    """
    def __init__(self, embed_dim, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)  
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, h):
        """
        x: constant input (B, N, D)
        h: previous hidden state (B, N, D)
        """
        
        combined = h + x
        
        attn_out = self.attn(self.norm1(combined))
        h_new = combined + attn_out
        
        mlp_out = self.mlp(self.norm2(h_new))
        h_new = h_new + mlp_out

        return h_new


class RecurrentVisionTransformerWithState(nn.Module):
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
        
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        
        self.recurrent_layer = RecurrentTransformerEncoderWithState(
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
        """
        x: image tensor of shape (B, C, H, W)
        """
        
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        B, N, D = x.shape
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, num_patches+1, D)
        x = x + self.pos_embed

        
        # Here, we initialize the hidden state as the embedded input.
        # Alternatively, you might initialize h as zeros or a learned parameter.
        h = x

       
        for _ in range(self.num_steps):
            h = self.recurrent_layer(x, h)

        
        h = self.norm(h)
        cls_token_final = h[:, 0]  
        logits = self.head(cls_token_final)
        return logits
    

class LitRecurrentVisionTransformerWithState(pl.LightningModule):
    def __init__(self, 
                 model_type="recurrent_state",
                 lr=1e-3,
                 num_steps=12,
                 weight_decay=0.01,
                 batch_size=None,
                 epochs=None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Now hparams includes model_type, batch_size, epochs, etc.
        
        self.model = RecurrentVisionTransformerWithState(
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
