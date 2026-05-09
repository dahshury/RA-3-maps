"""RGB palette + differentiable renderer for v2 architecture.

Builds a fixed (palette_size, 3) lookup of RGB colors per texture name using
the project's existing `MapVisualizer._get_texture_color` rule set so the
discriminator/FFL/DINOv2 losses see colors that match the canonical render.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rgb_palette(texture_names: List[str]) -> torch.Tensor:
    """Return (V, 3) float32 in [0, 1] from MapVisualizer._get_texture_color."""
    from ..utils.map_visualizer import MapVisualizer
    rgb = torch.zeros(len(texture_names), 3, dtype=torch.float32)
    for i, name in enumerate(texture_names):
        try:
            r, g, b = MapVisualizer._get_texture_color(name)
        except Exception:
            r, g, b = (160, 140, 120)
        rgb[i, 0] = float(r) / 255.0
        rgb[i, 1] = float(g) / 255.0
        rgb[i, 2] = float(b) / 255.0
    return rgb


class PaletteRenderer(nn.Module):
    """Differentiable tile-class -> RGB renderer.

    - hard_render(idx): non-differentiable, ground-truth path (idx >= 0).
    - soft_render(logits, tau): straight-through Gumbel-softmax;
      forward = one-hot @ palette, backward = softmax @ palette. Allows the
      GAN/FFL/DINO feature losses to flow gradients to the tile head.
    """

    def __init__(self, texture_names: List[str], learnable_residual: bool = True):
        super().__init__()
        rgb = build_rgb_palette(texture_names)
        # Frozen base palette (from name-based lookup).
        self.register_buffer("base_rgb", rgb)
        # Optional learnable residual on top so the model can refine the
        # per-class color if the deterministic mapping is too crude.
        if learnable_residual:
            self.residual = nn.Parameter(torch.zeros_like(rgb))
        else:
            self.register_buffer("residual", torch.zeros_like(rgb))

    @property
    def palette(self) -> torch.Tensor:
        return (self.base_rgb + self.residual).clamp(0.0, 1.0)

    def hard_render(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, H, W) long. Returns (B, 3, H, W). Negative idx -> zero."""
        safe = idx.clamp_min(0)
        rgb = self.palette[safe]                    # (B, H, W, 3)
        out = rgb.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        mask = (idx >= 0).float().unsqueeze(1)
        return out * mask

    def soft_render(self, logits: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """logits: (B, V, H, W). Returns differentiable (B, 3, H, W).

        Uses Gumbel-softmax with hard=True so the forward image is one-hot
        through the palette while gradients use the soft sample.
        """
        B, V, H, W = logits.shape
        flat = logits.permute(0, 2, 3, 1).reshape(-1, V)
        soft = F.gumbel_softmax(flat, tau=tau, hard=True, dim=-1)  # (BHW, V)
        rgb = soft @ self.palette                                  # (BHW, 3)
        return rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
