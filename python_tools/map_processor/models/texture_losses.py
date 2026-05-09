"""Auxiliary losses for the texture generation U-Net.

Implements:
  - Focal Frequency Loss (FFL) per Jiang et al., ICCV 2021
  - LPIPS perceptual loss (via the lpips package) operating on a learned
    color embedding of the predicted texture-index map

Both losses operate on a learned mapping from texture-vocab indices to
RGB-like vectors, so that frequency and perceptual distances are
meaningful (the raw integer index has no spatial-correlation semantics).

Usage:
    losses = TextureAuxLosses(vocab_size=350, color_dim=3,
                              ffl_weight=0.2, lpips_weight=0.1)
    loss_aux = losses(logits, target_y)  # logits (B,V,H,W); target_y (B,H,W) long, -1 ignore
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureColorEmbedding(nn.Module):
    """Map a 350-class texture index to a small learnable color vector.

    Initialized from family hash so that semantically-close textures
    (e.g., Snow_Solvang01 vs Snow_Solvang02) start with similar colors.
    """

    def __init__(self, vocab_size: int, color_dim: int = 3, vocab: Optional[list[str]] = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, color_dim)
        if vocab is not None and len(vocab) == vocab_size:
            self._init_from_family(vocab)
        else:
            nn.init.uniform_(self.emb.weight, 0.0, 1.0)

    def _init_from_family(self, vocab):
        import hashlib
        with torch.no_grad():
            for i, name in enumerate(vocab):
                family = name.split("_", 1)[0] if "_" in name else name
                h = int(hashlib.md5(family.encode()).hexdigest()[:6], 16)
                r = ((h >> 16) & 0xFF) / 255.0
                g = ((h >> 8) & 0xFF) / 255.0
                b = (h & 0xFF) / 255.0
                # Add a small per-name jitter so siblings differ slightly
                h2 = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
                jr = (((h2 >> 16) & 0xFF) / 255.0 - 0.5) * 0.1
                jg = (((h2 >> 8) & 0xFF) / 255.0 - 0.5) * 0.1
                jb = ((h2 & 0xFF) / 255.0 - 0.5) * 0.1
                self.emb.weight[i, 0] = r + jr
                self.emb.weight[i, 1] = g + jg
                self.emb.weight[i, 2] = b + jb

    def soft_color(self, logits: torch.Tensor) -> torch.Tensor:
        """logits (B,V,H,W) -> (B,3,H,W) via softmax-weighted color sum."""
        probs = F.softmax(logits, dim=1)
        # (B,V,H,W) x (V,3) -> (B,3,H,W)
        color = torch.einsum("bvhw,vc->bchw", probs, self.emb.weight)
        return color

    def hard_color(self, indices: torch.Tensor) -> torch.Tensor:
        """indices (B,H,W) long -> (B,3,H,W). Negative indices -> zero."""
        safe = indices.clamp_min(0)
        out = self.emb(safe).permute(0, 3, 1, 2).contiguous()
        mask = (indices >= 0).float().unsqueeze(1)
        return out * mask


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss (Jiang et al., ICCV 2021).

    Computes per-frequency residuals in the FFT domain, weighted by their
    own magnitude raised to a focal exponent. The model is forced to focus
    on frequencies it currently fails on.
    """

    def __init__(self, alpha: float = 1.0, patch_factor: int = 1):
        super().__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor

    def _freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 2D FFT (real + imag)
        Fp = torch.fft.fft2(pred, norm="ortho")
        Ft = torch.fft.fft2(target, norm="ortho")
        diff = Fp - Ft
        # spectrum weight matrix: |Fp - Ft|^alpha (re-detached so it doesn't
        # add a second-order gradient path; this matches the paper)
        with torch.no_grad():
            w = (diff.abs() ** self.alpha)
            w = w / (w.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-8))
        # Per-element squared error in frequency domain
        diff2 = diff.real ** 2 + diff.imag ** 2
        return (w * diff2).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W). Optional patching.
        if self.patch_factor == 1:
            return self._freq_loss(pred, target)
        B, C, H, W = pred.shape
        ph, pw = H // self.patch_factor, W // self.patch_factor
        loss = 0.0; n = 0
        for i in range(self.patch_factor):
            for j in range(self.patch_factor):
                p = pred[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                t = target[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                loss = loss + self._freq_loss(p, t); n += 1
        return loss / n


class TextureAuxLosses(nn.Module):
    """Combined auxiliary losses (FFL + LPIPS) over texture color embeddings."""

    def __init__(
        self,
        vocab_size: int,
        vocab: Optional[list[str]] = None,
        ffl_weight: float = 0.2,
        lpips_weight: float = 0.1,
        ignore_index: int = -1,
        ffl_alpha: float = 1.0,
        use_lpips: bool = True,
    ):
        super().__init__()
        self.color = TextureColorEmbedding(vocab_size, color_dim=3, vocab=vocab)
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)
        self.ffl_weight = ffl_weight
        self.lpips_weight = lpips_weight
        self.ignore_index = ignore_index
        self.use_lpips = use_lpips and lpips_weight > 0
        if self.use_lpips:
            try:
                import lpips as _lpips
                self.lpips = _lpips.LPIPS(net="vgg")
                for p in self.lpips.parameters():
                    p.requires_grad = False
            except Exception as e:  # noqa: BLE001
                print(f"[TextureAuxLosses] LPIPS init failed ({e}); disabling.")
                self.lpips = None
                self.use_lpips = False
        else:
            self.lpips = None

    def forward(self, logits: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
        # Soft predicted color (differentiable through softmax)
        pred_color = self.color.soft_color(logits)
        # Hard ground-truth color (via indices)
        gt_color = self.color.hard_color(target_y)

        # Mask invalid pixels (target == ignore_index) per-channel
        valid = (target_y != self.ignore_index).float().unsqueeze(1)
        pred_m = pred_color * valid
        gt_m = gt_color * valid

        # FFL (over the masked color maps)
        loss_ffl = self.ffl(pred_m, gt_m) if self.ffl_weight > 0 else pred_m.sum() * 0.0
        loss_lp = pred_m.sum() * 0.0
        if self.use_lpips and self.lpips is not None:
            # LPIPS expects [-1, 1] range, 3 channels
            p = (pred_m * 2.0 - 1.0).clamp(-1.0, 1.0)
            g = (gt_m * 2.0 - 1.0).clamp(-1.0, 1.0)
            loss_lp = self.lpips(p, g).mean()

        return self.ffl_weight * loss_ffl + self.lpips_weight * loss_lp
