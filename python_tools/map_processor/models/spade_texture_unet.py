"""SPADE-conditioned U-Net for per-tile texture classification.

Architecture v3: replaces the early-fusion BatchNorm U-Net with a model
where every normalization layer is a SPADE block taking a discretized
segmentation map of the input context (heightmap, slope, masks, object
densities). Style is injected via FiLM at the bottleneck and as an
additional channel in the segmentation map.

Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
Normalization" (CVPR 2019), https://arxiv.org/abs/1903.07291
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Discretization: turn the 10-channel continuous input into a small
# integer segmentation map. Used at training and inference identically.

SEG_N_CLASSES = 16
SEG_NAMES = [
    "deep_water",          # 0
    "shallow_water",       # 1
    "water_edge",          # 2
    "low_buildable_flat",  # 3
    "low_buildable_road",  # 4
    "low_buildable_build", # 5
    "low_buildable_deco",  # 6
    "low_unbuildable",     # 7
    "low_resource_zone",   # 8
    "mid_slope",           # 9
    "steep_slope",         # 10
    "near_cliff",          # 11
    "high_buildable",      # 12
    "high_unbuildable",    # 13
    "ridge",               # 14
    "peak",                # 15
]


def discretize_input(x: torch.Tensor) -> torch.Tensor:
    """(B, 10, H, W) float -> (B, H, W) long in [0, SEG_N_CLASSES).

    Channel order from prepare_texture_transfer_dataset.py:
      0 heightmap, 1 slope, 2 water_mask, 3 buildability, 4 passability,
      5 density_resource, 6 density_building, 7 density_decoration,
      8 density_road, 9 density_cliff
    """
    h = x[:, 0]; slope = x[:, 1]; water = x[:, 2]
    build = x[:, 3]; passable = x[:, 4]
    d_res = x[:, 5]; d_build = x[:, 6]; d_deco = x[:, 7]
    d_road = x[:, 8]; d_cliff = x[:, 9]

    seg = torch.zeros_like(h, dtype=torch.long)

    # Water tiers
    is_water = water > 0.5
    deep = is_water & (h < 0.20)
    shallow = is_water & ~deep
    edge_to_water = (~is_water) & (slope > 0.05) & (h < 0.30) & (build < 0.5)
    seg[deep] = 0
    seg[shallow] = 1
    seg[edge_to_water] = 2

    # Land categorization based on elevation, slope, density
    land = ~is_water & ~edge_to_water
    near_cliff = land & (d_cliff > 0.3)
    seg[near_cliff] = 11
    rem = land & ~near_cliff

    steep = rem & (slope > 0.4)
    mid = rem & (slope > 0.15) & ~steep
    seg[steep] = 10
    seg[mid] = 9

    flat = rem & ~steep & ~mid
    high = flat & (h > 0.7)
    high_b = high & (build > 0.5)
    seg[high & build.gt(0.5)] = 12
    seg[high & build.le(0.5)] = 13
    peak = high & (h > 0.9)
    ridge = high & (h > 0.85) & ~peak
    seg[peak] = 15
    seg[ridge] = 14

    low = flat & ~high
    has_road = low & (d_road > 0.25)
    has_build = low & (d_build > 0.25) & ~has_road
    has_deco = low & (d_deco > 0.25) & ~has_road & ~has_build
    has_res = low & (d_res > 0.25) & ~has_road & ~has_build & ~has_deco
    seg[has_road] = 4
    seg[has_build] = 5
    seg[has_deco] = 6
    seg[has_res] = 8
    other_low = low & ~(has_road | has_build | has_deco | has_res)
    seg[other_low & build.gt(0.5)] = 3
    seg[other_low & build.le(0.5)] = 7

    return seg


# --- SPADE: spatially-adaptive (de)normalization

class SPADE(nn.Module):
    def __init__(self, n_channels: int, label_nc: int, style_dim: int, hidden: int = 64):
        super().__init__()
        # Standard parameter-free normalization (instance norm avoids batch-stat issues at small batch)
        self.norm = nn.InstanceNorm2d(n_channels, affine=False)
        # MLP from segmentation one-hot + style embedding -> per-channel gamma/beta
        in_ch = label_nc + style_dim
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, n_channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden, n_channels, 3, padding=1)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, seg_onehot: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        # seg_onehot: (B, label_nc, H, W); style_map: (B, style_dim, H, W) - already broadcast
        if seg_onehot.shape[2:] != x.shape[2:]:
            seg_onehot = F.interpolate(seg_onehot, size=x.shape[2:], mode="nearest")
            style_map = F.interpolate(style_map, size=x.shape[2:], mode="nearest")
        cond = torch.cat([seg_onehot, style_map], dim=1)
        h = self.shared(cond)
        g = self.gamma(h); b = self.beta(h)
        return self.norm(x) * (1.0 + g) + b


class SPADEResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, label_nc: int, style_dim: int):
        super().__init__()
        mid = min(in_ch, out_ch)
        self.norm1 = SPADE(in_ch, label_nc, style_dim)
        self.conv1 = nn.Conv2d(in_ch, mid, 3, padding=1)
        self.norm2 = SPADE(mid, label_nc, style_dim)
        self.conv2 = nn.Conv2d(mid, out_ch, 3, padding=1)
        self.skip = (in_ch != out_ch)
        if self.skip:
            self.norm_s = SPADE(in_ch, label_nc, style_dim)
            self.conv_s = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x, seg, sty):
        h = self.conv1(F.leaky_relu(self.norm1(x, seg, sty), 0.2))
        h = self.conv2(F.leaky_relu(self.norm2(h, seg, sty), 0.2))
        if self.skip:
            x = self.conv_s(F.leaky_relu(self.norm_s(x, seg, sty), 0.2))
        return h + x


class SPADEUNet(nn.Module):
    """U-Net with SPADE blocks at every conv site.

    Inputs
    ------
    x         : (B, C_in, H, W) float - the 10 context channels
    style_id  : (B,) long - cluster id

    Output
    ------
    logits    : (B, vocab_size, H, W) float
    """

    def __init__(
        self,
        in_channels: int = 10,
        n_styles: int = 8,
        style_dim: int = 16,
        vocab_size: int = 350,
        base: int = 32,
        n_seg_classes: int = SEG_N_CLASSES,
    ):
        super().__init__()
        self.style_emb = nn.Embedding(n_styles, style_dim)
        self.in_channels = in_channels
        self.n_seg_classes = n_seg_classes
        self.style_dim = style_dim

        # Initial projection from raw input to base feature map
        self.in_proj = nn.Conv2d(in_channels, base, 3, padding=1)

        # Encoder
        self.enc1 = SPADEResBlock(base, base * 2, n_seg_classes, style_dim)
        self.enc2 = SPADEResBlock(base * 2, base * 4, n_seg_classes, style_dim)
        self.enc3 = SPADEResBlock(base * 4, base * 8, n_seg_classes, style_dim)
        self.bottleneck = SPADEResBlock(base * 8, base * 8, n_seg_classes, style_dim)

        # Decoder
        self.dec3 = SPADEResBlock(base * 8 + base * 8, base * 4, n_seg_classes, style_dim)
        self.dec2 = SPADEResBlock(base * 4 + base * 4, base * 2, n_seg_classes, style_dim)
        self.dec1 = SPADEResBlock(base * 2 + base * 2, base, n_seg_classes, style_dim)

        self.out_conv = nn.Conv2d(base, vocab_size, 1)

    def forward(self, x: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        seg = discretize_input(x)  # (B, H, W)
        seg_oh = F.one_hot(seg, num_classes=self.n_seg_classes).permute(0, 3, 1, 2).float()
        sty = self.style_emb(style_id)  # (B, style_dim)
        sty_map = sty[:, :, None, None].expand(-1, -1, H, W)

        x0 = self.in_proj(x)               # (B, base, H, W)
        x1 = self.enc1(x0, seg_oh, sty_map)
        x1d = F.avg_pool2d(x1, 2)
        x2 = self.enc2(x1d, seg_oh, sty_map)
        x2d = F.avg_pool2d(x2, 2)
        x3 = self.enc3(x2d, seg_oh, sty_map)
        x3d = F.avg_pool2d(x3, 2)
        b = self.bottleneck(x3d, seg_oh, sty_map)

        u3 = F.interpolate(b, scale_factor=2, mode="nearest")
        u3 = self.dec3(torch.cat([u3, x3], dim=1), seg_oh, sty_map)
        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.dec2(torch.cat([u2, x2], dim=1), seg_oh, sty_map)
        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.dec1(torch.cat([u1, x1], dim=1), seg_oh, sty_map)

        return self.out_conv(u1)
