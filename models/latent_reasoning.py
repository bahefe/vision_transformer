import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

###############################################################################
# 1) RMSNorm (instead of LayerNorm) -- a common choice in "sandwich" blocks
###############################################################################
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, N, D] or [*, D]
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / math.sqrt(x.shape[-1]))
        return (x / (norm + self.eps)) * self.weight


###############################################################################
# 2) "Sandwich" Transformer block
#
# The paper's formula is:
#   x_hat_l = n2( x_{l-1} + Attn(n1(x_{l-1})) )
#   x_l     = n4( x_hat_l + MLP(n3(x_hat_l)) )
###############################################################################
class SandwichBlock(nn.Module):
    """
    A "sandwich" variant of the Transformer block with:
    - RMSNorm before each sub-layer
    - immediate re-normalization after each residual
    - Gated SiLU MLP

    For attention, we use standard PyTorch MultiheadAttention. 
    """
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        # Norm layers
        self.n1 = RMSNorm(embed_dim)
        self.n2 = RMSNorm(embed_dim)
        self.n3 = RMSNorm(embed_dim)
        self.n4 = RMSNorm(embed_dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Gated SiLU MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, embed_dim, bias=True),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # 1) Attn sub-layer
        x_attn_in = self.n1(x)                 # n1(x_{l-1})
        attn_out, _ = self.attn(x_attn_in, x_attn_in, x_attn_in)
        x_hat = x + attn_out                   # residual
        x_hat = self.n2(x_hat)                 # n2(...)

        # 2) MLP sub-layer
        x_mlp_in = self.n3(x_hat)              # n3(x_hat_l)
        mlp_out = self.mlp(x_mlp_in)
        mlp_out = self.drop(mlp_out)
        x_out = x_hat + mlp_out                # residual
        x_out = self.n4(x_out)                 # n4(...)

        return x_out


###############################################################################
# 3) Patch Embedding
###############################################################################
class PatchEmbedding(nn.Module):
    """
    Given (B, C, H, W) images, convert into a sequence of patch embeddings (B, N, D).
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # A simple conv-based patch extraction
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=True
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.proj(x)               # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)              # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)         # [B, num_patches, embed_dim]
        return x


###############################################################################
# 4) LatentReasoningVisionTransformer:
#    - Prelude => Recurrent (latent reasoning) => Coda
#    - "Sandwich" blocks
#    - Concat-based injection
#    - Partial no_grad steps for large # of recurrences
###############################################################################
class LatentReasoningVisionTransformer(nn.Module):
    """
    A Vision Transformer that follows a recurrent "core" approach with latent reasoning:

    1) Prelude: l_P layers
    2) Recurrent block: l_R layers repeated 'r' times with latent space reasoning
       - We inject the original patch embeddings each iteration.
         Instead of "add", we do "concat + linear" (the "adapter").
    3) Coda: l_C layers
    4) Final norm + classifier
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        # Architectural hyperparams:
        l_P=1,            # number of "prelude" blocks
        l_R=4,            # number of layers in the recurrent (latent reasoning) block
        l_C=1,            # number of "coda" blocks
        num_heads=8,
        mlp_hidden_dim=1024,
        # Recurrent scaling:
        adapter_injection=True,    # if True, use concat+linear; else do "add"
        recurrent_hidden_dim=2048, # MLP size for the recurrent block
        mean_recurrence=10,        # total # of recurrent steps
        mean_backprop_depth=4,     # # of those steps done with gradient
        dropout=0.1,
    ):
        super().__init__()
        self.mean_recurrence = mean_recurrence
        self.mean_backprop_depth = mean_backprop_depth
        self.adapter_injection = adapter_injection

        # ---------------------------------------------------------------------
        # (A) Patch embedding
        # ---------------------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # ---------------------------------------------------------------------
        # (B) Prelude: l_P sandwich blocks
        # ---------------------------------------------------------------------
        self.prelude = nn.ModuleList([
            SandwichBlock(embed_dim, num_heads, mlp_hidden_dim, dropout=dropout)
            for _ in range(l_P)
        ])

        # ---------------------------------------------------------------------
        # (C) Recurrent block (latent reasoning): l_R sandwich blocks
        # ---------------------------------------------------------------------
        self.core_block = nn.ModuleList([
            SandwichBlock(embed_dim, num_heads, recurrent_hidden_dim, dropout=dropout)
            for _ in range(l_R)
        ])

        # If using "concat + linear" injection:
        if self.adapter_injection:
            self.adapter = nn.Linear(embed_dim * 2, embed_dim, bias=True)

        # ---------------------------------------------------------------------
        # (D) Coda: l_C sandwich blocks
        # ---------------------------------------------------------------------
        self.coda = nn.ModuleList([
            SandwichBlock(embed_dim, num_heads, mlp_hidden_dim, dropout=dropout)
            for _ in range(l_C)
        ])

        # ---------------------------------------------------------------------
        # (E) Final norm + classification
        # ---------------------------------------------------------------------
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes, bias=True)

        # init
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W) image batch
        """
        # 1) Embed patches
        patch_emb = self.patch_embed(x)  # [B, N, D]
        B, N, D = patch_emb.shape

        # 2) Concat class token + add positional embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        state = torch.cat((cls_tokens, patch_emb), dim=1)  # [B, N+1, D]
        state = state + self.pos_embed[:, :N+1, :]
        state = self.pos_drop(state)

        # 3) Prelude
        for blk in self.prelude:
            state = blk(state)

        # 4) Recurrent iteration with partial no_grad
        no_grad_steps, grad_steps = self._decide_recurrence_schedule()

        # 4a) no-grad steps
        with torch.no_grad():
            for _ in range(no_grad_steps):
                state = self._recurrent_step(state, patch_emb)

        # 4b) grad steps
        for _ in range(grad_steps):
            state = self._recurrent_step(state, patch_emb)

        # 5) Coda
        for blk in self.coda:
            state = blk(state)

        # 6) Final RMSNorm + classification on the CLS token
        state = self.norm(state)          # [B, N+1, D]
        cls_out = state[:, 0, :]          # [B, D]
        logits = self.head(cls_out)       # [B, num_classes]
        return logits

    def _recurrent_step(self, state, patch_emb):
        """
        Single iteration of the recurrent block:
          - split off cls token, apply injection
          - pass through the core_block
        """
        B, N_plus_1, D = state.shape
        cls = state[:, :1, :]       # [B, 1, D]
        patches = state[:, 1:, :]   # [B, N, D]

        # injection
        if self.adapter_injection:
            # Concat + linear adapter
            cat_patches = torch.cat([patches, patch_emb], dim=-1)  # (B, N, 2D)
            patches = self.adapter(cat_patches)                    # (B, N, D)
        else:
            # "Add" injection
            patches = patches + patch_emb

        # Reassemble
        state = torch.cat([cls, patches], dim=1)

        # Pass through the recurrent core block
        for blk in self.core_block:
            state = blk(state)

        return state

    def _decide_recurrence_schedule(self):
        """
        Simple fixed schedule:
          no_grad_steps = mean_recurrence - mean_backprop_depth
          grad_steps = mean_backprop_depth
        """
        total = self.mean_recurrence
        backprop = min(self.mean_backprop_depth, total)
        no_grad = total - backprop
        return no_grad, backprop


###############################################################################
# 5) Lightning Module for training (updated model name: latent_reasoning)
###############################################################################
class LitLatentReasoningVisionTransformer(pl.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        weight_decay=1e-2,
        batch_size=None,
        epochs=None,
        **model_kwargs
    ):
        super().__init__()
        # Save hyperparams
        self.save_hyperparameters()
        # Build the latent reasoning ViT
        self.model = LatentReasoningVisionTransformer(**model_kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)

        # If labels are one-hot, convert them to integer class indices
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)

        acc = (preds == labels).float().mean()


        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)

        # If labels are one-hot, convert them to integer class indices
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)

        acc = (preds == labels).float().mean()



        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)

        # If labels are one-hot, convert them to integer class indices
        if labels.dim() > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)

        acc = (preds == labels).float().mean()


        self.log("test_loss", loss, prog_bar=False)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
