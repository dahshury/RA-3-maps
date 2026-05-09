"""DINOv2 reference-patch style encoder.

Frozen ViT-S/14 (~21M params, no gradient). At training time we sample a
224x224 crop from a different region of the same map, render it through the
RGB palette, and pool DINOv2's CLS token to a 384-d vector. A small
trainable projector maps that to `style_dim`. This replaces the 262-id
lookup table — singleton-cluster maps now condition on a continuous,
generalizing feature space.

Per architecture_research_2026-05.md item 5b. Reference: Oquab et al.,
DINOv2, arXiv:2304.07193.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2StyleEncoder(nn.Module):
    """Frozen DINOv2-S/14 + small trainable projector.

    Inputs are (B, 3, 224, 224) RGB in [0, 1]; we apply the standard ImageNet
    normalisation internally. Style vector dim defaults to 256.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, style_dim: int = 256, model_name: str = "facebook/dinov2-small",
                 cfg_dropout: float = 0.1):
        super().__init__()
        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained(model_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()
        # CLS token dim depends on backbone (small=384, base=768).
        cls_dim = self.dino.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, style_dim),
            nn.GELU(),
            nn.Linear(style_dim, style_dim),
        )
        self.style_dim = style_dim
        self.cfg_dropout = cfg_dropout
        self.register_buffer("_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
        # Learned "unconditional" style for classifier-free dropout.
        self.null_style = nn.Parameter(torch.zeros(style_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        # DINOv2 stays in eval mode (frozen BN/LayerNorm semantics fine).
        self.dino.eval()
        return self

    def encode_image(self, rgb01: torch.Tensor) -> torch.Tensor:
        """rgb01: (B, 3, H, W) in [0, 1]. Returns (B, style_dim).

        Auto-resizes to 224x224 (DINOv2 default). The patch size is 14,
        so 224 = 16x16 patches.
        """
        if rgb01.shape[-1] != 224 or rgb01.shape[-2] != 224:
            rgb01 = F.interpolate(rgb01, size=(224, 224), mode="bilinear",
                                  align_corners=False)
        x = (rgb01 - self._mean) / self._std
        with torch.no_grad():
            out = self.dino(pixel_values=x)
        cls = out.last_hidden_state[:, 0]   # (B, hidden)
        return self.proj(cls)

    def forward(self, rgb01: torch.Tensor, *, drop: torch.Tensor | None = None) -> torch.Tensor:
        """rgb01: (B, 3, H, W) in [0, 1]. drop: (B,) bool mask of samples
        whose conditioning should be replaced by the learned null_style.

        At training time call with `drop = torch.rand(B) < cfg_dropout` to
        get classifier-free guidance support.
        """
        s = self.encode_image(rgb01)
        if drop is not None and drop.any():
            null = self.null_style.unsqueeze(0).expand_as(s)
            s = torch.where(drop.unsqueeze(-1), null, s)
        return s

    def unconditional(self, batch: int, device) -> torch.Tensor:
        """Return (B, style_dim) of pure null_style — for inference w/o ref."""
        return self.null_style.unsqueeze(0).expand(batch, -1).to(device)
