"""
Patch-based multi-task U-Net for RA3 blend prediction (Phase 3).

Instead of the per-cell Token Transformer (Phase 2) that processes 5x5 windows,
this U-Net operates on large spatial patches (16x16 to 64x64) and predicts
blend properties for ALL cells simultaneously, leveraging spatial context
through encoder-decoder skip connections.

Architecture:
  - Input: texture_embedding (32ch) + normalized_elevation (1ch) = 33ch per cell
  - Encoder: 3 downsampling stages with residual conv blocks
  - Decoder: shared decoder with 3 task-specific heads
      Head 1: blend_present     (1ch, sigmoid, BCE loss)
      Head 2: neighbor_mask     (8ch, sigmoid, ASL loss)
      Head 3: direction         (12ch, softmax, logit-adjusted CE loss)
  - Only predicts BLEND layer (not single_edge), keeping the model focused.

Data pipeline:
  - Loads .map files directly using map_processor
  - Extracts random NxN patches with progressive growing (16->32->64)
  - FlipX augmentation (50% chance, ~2% noise per research findings)

Usage:
  # Training
  python scripts/train_blend_unet.py \\
      --maps_dir "../RA3 Official maps" \\
      --output_dir "../blendinfo dataset/_generated/unet_v1" \\
      --patch_size 64 --epochs 30

  # Inference on a single map
  python scripts/train_blend_unet.py \\
      --predict \\
      --model_path "../blendinfo dataset/_generated/unet_v1/best_model.pt" \\
      --map_path "../RA3 Official maps/2 II/map_mp_2_rao1.map" \\
      --output_dir "../RA3 Official maps/2 II/unet_test"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup so we can import map_processor from the parent package
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ===========================================================================
# Map loading utilities (matching generate_blendinfo_dataset.py / predict_blends.py)
# ===========================================================================

# Neighbor layout: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
# These are (row_offset, col_offset) to match numpy [x, y] indexing.
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# All known BlendDirection values (as raw ints) in sorted order.
# Index 0 is "no direction" (-1 / invalid). Classes 1..16 map to real blend directions.
# This must match the vocabulary used in the per-cell model for consistency.
# Values: -1, 1(Left), 2(Bottom), 4(ExceptTopRight), 8(ExceptTopLeft),
#         17(Right), 18(Top), 20(ExceptBottomRight), 24(ExceptBottomLeft),
#         33, 34, 36(BottomLeft), 40(BottomRight), 49, 50, 52(TopLeft), 56(TopRight)
DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
DIR_VAL_TO_CLASS = {v: i for i, v in enumerate(DIRECTION_VALUES)}
NUM_DIR_CLASSES = len(DIRECTION_VALUES)  # 17

# Default number of direction classes for the head (skip class 0 = invalid).
# We use all 17 including the invalid/-1 class during training but mask it.
# The head outputs 12 "real" directions (classes 1-16, but some like 33,34,49,50 are rare).
# For simplicity we predict all 17 classes and mask class 0 via ignore_index=-100.
# Actually, we will output NUM_DIR_CLASSES and use class labels 0..16.
# Direction class 0 maps to raw value -1 (no blend). We only train direction where present=1.


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """Decode texture ID from tile value. Inverse of BlendTileData.GetTexture."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _get_tile_from_texture(x: int, y: int, texture_id: int) -> int:
    """Encode texture_id + position into secondary_texture_tile format."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return texture_id * 64 + current


def _decode_texture_grid(tiles: np.ndarray) -> np.ndarray:
    """Convert tile values grid to texture id grid. Shape: [W, H] -> [W, H]."""
    w, h = tiles.shape
    tex = np.empty((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


def _decode_texture_grid_vectorized(tiles: np.ndarray) -> np.ndarray:
    """Vectorized version of texture grid decoding. Much faster for large maps."""
    w, h = tiles.shape
    xs = np.arange(w, dtype=np.int32)[:, None]
    ys = np.arange(h, dtype=np.int32)[None, :]
    row_first = (ys % 8) // 2 * 16 + (ys % 2) * 2
    current = (xs % 8) // 2 * 4 + (xs % 2) + row_first
    return (tiles.astype(np.int32) - current) // 64


def load_map_data(map_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a .map file and extract all grids needed for training.

    Returns dict with keys:
        tex_grid:       [W, H] int32 - texture IDs per cell
        elev_grid:      [W, H] float32 - elevation per cell (or None)
        blend_present:  [W, H] uint8 - 1 where blend exists
        blend_mask:     [W, H] uint8 - 8-bit neighbor bitmask (255=ignore)
        blend_dir:      [W, H] int16 - direction class index (0..16), -1 if no blend
        textures:       list of Texture objects
    Returns None if the map cannot be loaded or lacks required data.
    """
    from map_processor.ra3map import Ra3Map
    from map_processor.assets.terrain.blend_tile_data import BlendTileData
    from map_processor.assets.terrain.height_map_data import HeightMapData

    try:
        m = Ra3Map(str(map_path))
        m.parse()
        ctx = m.get_context()
        blend = ctx.get_asset_by_type(BlendTileData)
        height = ctx.get_asset_by_type(HeightMapData)

        if blend is None or blend.tiles is None:
            return None

        tiles = np.asarray(blend.tiles, dtype=np.int32)
        w, h = tiles.shape

        # Texture grid
        tex_grid = _decode_texture_grid_vectorized(tiles)

        # Elevation grid
        elev_grid = None
        if height is not None and height.elevations is not None:
            eg = np.asarray(height.elevations, dtype=np.float32)
            if eg.shape == (w, h):
                elev_grid = eg

        # Blend labels: present, secondary neighbor mask, direction
        blends_arr = np.asarray(blend.blends, dtype=np.int32)
        blend_present = (blends_arr > 0).astype(np.uint8)

        # Build secondary texture grid and direction grid from blend_info
        info = blend.blend_info or []
        sec_tex_grid = np.full((w, h), -1, dtype=np.int32)
        dir_raw_grid = np.full((w, h), -1, dtype=np.int32)

        for x in range(w):
            for y in range(h):
                idx = int(blends_arr[x, y])
                if idx <= 0 or idx > len(info):
                    continue
                bi = info[idx - 1]
                sec_tex_grid[x, y] = _get_texture_from_tile(x, y, int(bi.secondary_texture_tile))
                dir_raw_grid[x, y] = int(bi.blend_direction)

        # Build 8-bit neighbor mask (vectorized)
        tex_pad = np.pad(tex_grid, pad_width=((1, 1), (1, 1)), mode="edge")
        sec_pad = np.pad(sec_tex_grid, pad_width=((1, 1), (1, 1)), mode="edge")
        mask8 = np.zeros((w, h), dtype=np.uint8)

        for ni, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
            neigh = tex_pad[1 + dx: 1 + dx + w, 1 + dy: 1 + dy + h]
            sec = sec_pad[1:1 + w, 1:1 + h]
            hit = (blend_present != 0) & (sec >= 0) & (neigh == sec)
            mask8 = np.bitwise_or(mask8, (hit.astype(np.uint8) << np.uint8(ni)))

        # If present but no neighbor matches, mark as IGNORE (255)
        mask8 = np.where((blend_present != 0) & (mask8 == 0), np.uint8(255), mask8).astype(np.uint8)

        # Direction class index grid
        dir_class_grid = np.full((w, h), -1, dtype=np.int16)
        for raw_val, cls_idx in DIR_VAL_TO_CLASS.items():
            dir_class_grid[dir_raw_grid == raw_val] = cls_idx

        return {
            "tex_grid": tex_grid,
            "elev_grid": elev_grid,
            "blend_present": blend_present,
            "blend_mask": mask8,
            "blend_dir": dir_class_grid,
            "textures": blend.textures or [],
            "map_width": w,
            "map_height": h,
        }

    except Exception as e:
        print(f"  [WARN] Failed to load {map_path}: {e}", flush=True)
        return None


def find_map_files(root: Path) -> List[Path]:
    """Recursively find .map files, excluding blendless/generated/pruned."""
    maps = []
    for p in sorted(root.rglob("*.map")):
        name_lower = p.stem.lower()
        parts_lower = [part.lower() for part in p.parts]
        if "blendless" in name_lower:
            continue
        if any(token in name_lower for token in ("_predicted", "_original", "unet_", "archon_test")):
            continue
        if any(part.startswith("_") for part in p.relative_to(root).parts):
            continue
        maps.append(p)
    return maps


def load_prepared_npz(npz_path: str, with_dist: bool = False,
                      with_pattern: bool = False) -> Optional[Dict[str, np.ndarray]]:
    """Load a per-map .npz produced by prepare_unet_dataset.py.

    Returns the same dict format as load_map_data() so MapPatchDataset is unchanged.
    Much faster than re-parsing .map files.

    If with_dist=True, also computes a Chebyshev distance-to-texture-boundary grid
    ([W, H] float32) and stores it under key "dist_grid". The research found this
    feature has 8x lift over baseline (P(blend|boundary)=55% vs P(blend|interior)=7%).

    If with_pattern=True, also loads/computes pattern_code (int8 [W, H], 0..12)
    -- the deterministic blend pattern bucket. Empirical: argmax(P(dir|pattern_code))
    alone gives ~75% dir_acc vs U-Net's 0.43 raw; adding it as one-hot input is
    expected to close most of the gap to the token model's 0.89.
    """
    try:
        d = np.load(npz_path)
        tex_grid = np.asarray(d["tex_grid"], dtype=np.int32)
        w, h = tex_grid.shape
        out = {
            "tex_grid": tex_grid,
            "elev_grid": np.asarray(d["elev_grid"], dtype=np.float32),
            "blend_present": np.asarray(d["blend_present"], dtype=np.uint8),
            "blend_mask": np.asarray(d["blend_mask"], dtype=np.uint8),
            "blend_dir": np.asarray(d["blend_dir"], dtype=np.int16),
            "textures": [],
            "map_width": w,
            "map_height": h,
        }
        if with_dist:
            if "dist_grid" in d.files:
                out["dist_grid"] = np.asarray(d["dist_grid"], dtype=np.float32)
            else:
                out["dist_grid"] = _compute_distance_to_boundary(tex_grid)
        if with_pattern:
            if "pattern_code" in d.files:
                out["pattern_code"] = np.asarray(d["pattern_code"], dtype=np.int8)
            else:
                # Backward compat: compute on the fly for old prepared dirs.
                out["pattern_code"] = _compute_pattern_code(tex_grid)
        return out
    except Exception as e:
        print(f"  [WARN] Failed to load prepared {npz_path}: {e}", flush=True)
        return None


def _compute_pattern_code(tex_grid: np.ndarray) -> np.ndarray:
    """Per-cell deterministic blend pattern bucket (0..12). See prepare_unet_dataset.py."""
    c = tex_grid
    padded = np.pad(c, 1, mode="edge")
    left   = padded[1:-1,  :-2]
    right  = padded[1:-1, 2:  ]
    top    = padded[ :-2, 1:-1]
    bottom = padded[2:  , 1:-1]
    tl     = padded[ :-2,  :-2]
    tr     = padded[ :-2, 2:  ]
    bl     = padded[2:  ,  :-2]
    br     = padded[2:  , 2:  ]
    pattern_tex = np.full(c.shape, -1, dtype=np.int32)
    pattern_dir = np.zeros(c.shape, dtype=np.int8)

    def _try(mask, tex, code):
        m = mask & (pattern_tex < 0)
        pattern_tex[m] = tex[m]
        pattern_dir[m] = code

    _try((left == top)    & (top != c),    top, 1)
    _try((right == top)   & (top != c),    top, 2)
    _try((right == bottom)& (bottom != c), bottom, 3)
    _try((left == bottom) & (bottom != c), bottom, 4)
    _try(left   != c, left,    5)
    _try(right  != c, right,   6)
    _try(top    != c, top,     7)
    _try(bottom != c, bottom,  8)
    _try(tl != c, tl, 9)
    _try(tr != c, tr, 10)
    _try(br != c, br, 11)
    _try(bl != c, bl, 12)
    return pattern_dir


# Number of pattern_code buckets including 0 (no pattern). One-hot expansion size.
NUM_PATTERN_CODES = 13


def _compute_distance_to_boundary(tex_grid: np.ndarray) -> np.ndarray:
    """Chebyshev distance from each cell to the nearest 4-neighbor texture boundary.

    Returns float32 [W, H]. Cells AT a boundary get 0; deep interior cells get
    larger values. The distance transform uses chessboard metric, so a cell at
    distance d has an L_inf neighborhood of size (2d+1)^2 of identical textures.
    """
    from scipy.ndimage import distance_transform_cdt
    w, h = tex_grid.shape
    # Boundary mask: 1 where this cell differs from any 4-neighbor.
    pad = np.pad(tex_grid, ((1, 1), (1, 1)), mode="edge")
    diff = (
        (pad[1:1 + w, 1:1 + h] != pad[:w, 1:1 + h]) |
        (pad[1:1 + w, 1:1 + h] != pad[2:2 + w, 1:1 + h]) |
        (pad[1:1 + w, 1:1 + h] != pad[1:1 + w, :h]) |
        (pad[1:1 + w, 1:1 + h] != pad[1:1 + w, 2:2 + h])
    )
    # Distance transform: distance to nearest True (boundary) cell.
    # distance_transform_cdt with metric='chessboard' = Chebyshev distance.
    dist = distance_transform_cdt(~diff, metric="chessboard").astype(np.float32)
    return dist


# ===========================================================================
# Asymmetric Loss (ASL) for multi-label classification
# ===========================================================================

def asymmetric_loss(logits: "torch.Tensor", targets: "torch.Tensor",
                    gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                    clip: float = 0.05) -> "torch.Tensor":
    """
    Asymmetric Loss for Multi-Label Classification (ASL).
    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.

    Key idea: apply stronger down-weighting to easy negatives (gamma_neg > gamma_pos)
    and probability shifting (clip) to further reduce easy-negative contribution.

    Args:
        logits: [B, 8, H, W] raw logits
        targets: [B, 8, H, W] binary targets (0 or 1)
        gamma_pos: focusing param for positives (0 = no down-weight)
        gamma_neg: focusing param for negatives (4 = strong down-weight of easy negatives)
        clip: probability margin to shift negatives (reduces gradient from very easy negatives)
    """
    import torch
    import torch.nn.functional as F

    probs = torch.sigmoid(logits)

    # Probability shifting for negatives
    probs_neg = (probs - clip).clamp(min=0.0)

    # Separate positive and negative terms
    loss_pos = targets * torch.log(probs.clamp(min=1e-8))
    loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))

    # Asymmetric focusing
    if gamma_pos > 0:
        loss_pos = loss_pos * ((1 - probs) ** gamma_pos)
    if gamma_neg > 0:
        pt_neg = probs_neg
        loss_neg = loss_neg * (pt_neg ** gamma_neg)

    loss = -(loss_pos + loss_neg)
    return loss.mean()


# ===========================================================================
# U-Net Model
# ===========================================================================

def _make_model(num_textures: int, tex_embed_dim: int, hidden_channels: int,
                extra_input_channels: int = 0,
                dir_head_type: str = "deep",
                map_emb_dim: int = 0,
                map_ds_size: int = 32,
                use_neighbor_tex: bool = False):
    """Build the BlendUNet model. Returns a nn.Module.

    extra_input_channels: number of additional float input channels beyond
    (texture_embedding + elevation). e.g. 1 for distance-to-boundary.

    dir_head_type:
      "linear" - single 1x1 conv (legacy, weak — direction underperforms)
      "deep"   - dedicated 3x3 + 3x3 + 1x1 head (default; explicit neighbor awareness)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        """Two-conv residual block with GroupNorm."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
            self.act = nn.GELU()

        def forward(self, x):
            residual = self.skip(x)
            out = self.act(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return self.act(out + residual)

    class MapEncoder(nn.Module):
        """Tiny CNN that consumes a downsampled full-map view and produces
        a fixed-size style embedding broadcast to all spatial locations.

        Designed to attack the train-val distribution-shift gap: gives the
        U-Net a learned, map-level signature rather than relying on local
        receptive field to infer style implicitly.
        """

        def __init__(self, in_ch: int, emb_dim: int):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False),  # ds/2 -> 16
                nn.GroupNorm(32, 64), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),     # /4 -> 8
                nn.GroupNorm(32, 64), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),     # /8 -> 4
                nn.GroupNorm(32, 64), nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(64, emb_dim)

        def forward(self, x):
            return self.head(self.body(x).flatten(1))  # [B, emb_dim]

    class BlendUNet(nn.Module):
        """
        Multi-task U-Net for blend prediction on map patches.

        Input channels: tex_embed_dim + 1 (elevation) + extra_input_channels
                        + map_emb_dim (broadcast from MapEncoder, if enabled).
        Three task heads produce per-pixel predictions.
        """

        def __init__(self):
            super().__init__()
            # Neighbor-tex embedding adds 4 sides * tex_embed_dim channels.
            self.use_neighbor_tex = bool(use_neighbor_tex)
            neighbor_extra = (4 * tex_embed_dim) if self.use_neighbor_tex else 0
            in_ch = tex_embed_dim + 1 + extra_input_channels + map_emb_dim + neighbor_extra

            # Texture embedding: maps discrete texture ID -> dense vector
            self.tex_embed = nn.Embedding(num_textures, tex_embed_dim, padding_idx=0)

            # Optional map-level encoder. Takes downsampled tex_embed + elev,
            # produces a fixed-size embedding broadcast across all spatial cells.
            self.map_emb_dim = int(map_emb_dim)
            self.map_ds_size = int(map_ds_size)
            if self.map_emb_dim > 0:
                self.map_encoder = MapEncoder(tex_embed_dim + 1, map_emb_dim)
            else:
                self.map_encoder = None

            # Encoder path
            C = hidden_channels  # 64 by default
            self.enc1 = ResBlock(in_ch, C)          # -> C
            self.down1 = nn.MaxPool2d(2)
            self.enc2 = ResBlock(C, C * 2)           # -> 2C
            self.down2 = nn.MaxPool2d(2)
            self.enc3 = ResBlock(C * 2, C * 4)       # -> 4C
            self.down3 = nn.MaxPool2d(2)

            # Bottleneck
            self.bottleneck = ResBlock(C * 4, C * 8)  # -> 8C

            # Decoder path with skip connections
            self.up3 = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
            self.dec3 = ResBlock(C * 8, C * 4)  # concat skip from enc3

            self.up2 = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
            self.dec2 = ResBlock(C * 4, C * 2)  # concat skip from enc2

            self.up1 = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
            self.dec1 = ResBlock(C * 2, C)      # concat skip from enc1

            # Task heads. present + mask use cheap 1x1 convs; direction
            # gets a heavier neighbor-aware stack because it's a 17-way
            # classification that depends on which sides of the cell are
            # blended (the token model used MultiheadAttention over 8
            # neighbors -- this 3x3 stack gives the U-Net the same kind of
            # explicit local mixing).
            self.present_head = nn.Conv2d(C, 1, 1)            # blend present (sigmoid)
            self.mask_head = nn.Conv2d(C, 8, 1)               # neighbor mask (8-bit, sigmoid per bit)
            if dir_head_type == "deep":
                hC = max(C * 2, NUM_DIR_CLASSES * 2)
                self.dir_head = nn.Sequential(
                    nn.Conv2d(C, hC, 3, padding=1, bias=False),
                    nn.GroupNorm(min(32, hC), hC),
                    nn.GELU(),
                    nn.Conv2d(hC, hC, 3, padding=1, bias=False),
                    nn.GroupNorm(min(32, hC), hC),
                    nn.GELU(),
                    nn.Conv2d(hC, NUM_DIR_CLASSES, 1),
                )
            else:
                self.dir_head = nn.Conv2d(C, NUM_DIR_CLASSES, 1)

        def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            """
            Forward pass.
            x: [B, in_ch, H, W] where in_ch = tex_embed_dim + 1
            Returns dict of logits for each head.
            """
            # Encoder
            e1 = self.enc1(x)        # [B, C, H, W]
            e2 = self.enc2(self.down1(e1))  # [B, 2C, H/2, W/2]
            e3 = self.enc3(self.down2(e2))  # [B, 4C, H/4, W/4]

            # Bottleneck
            bn = self.bottleneck(self.down3(e3))  # [B, 8C, H/8, W/8]

            # Decoder
            d3 = self.up3(bn)                     # [B, 4C, H/4, W/4]
            d3 = self.dec3(torch.cat([d3, e3], dim=1))

            d2 = self.up2(d3)                     # [B, 2C, H/2, W/2]
            d2 = self.dec2(torch.cat([d2, e2], dim=1))

            d1 = self.up1(d2)                     # [B, C, H, W]
            d1 = self.dec1(torch.cat([d1, e1], dim=1))

            # Task heads
            present_logits = self.present_head(d1)  # [B, 1, H, W]
            mask_logits = self.mask_head(d1)         # [B, 8, H, W]
            dir_logits = self.dir_head(d1)           # [B, 17, H, W]

            return {
                "present_logits": present_logits,
                "mask_logits": mask_logits,
                "dir_logits": dir_logits,
            }

        def encode_input(self, tex_ids: "torch.Tensor", elevation: "torch.Tensor",
                         extras: Optional["torch.Tensor"] = None,
                         map_tex_ds: Optional["torch.Tensor"] = None,
                         map_elev_ds: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            """
            Build the input tensor from raw data.

            tex_ids:     [B, H, W] long - texture IDs (per-cell of the patch)
            elevation:   [B, H, W] float - normalized elevation (per-cell of the patch)
            extras:      [B, K, H, W] float (optional) - K extra channels
            map_tex_ds:  [B, S, S] long (optional) - downsampled tex IDs for map encoder
            map_elev_ds: [B, S, S] float (optional) - downsampled normalized elevation

            Returns: [B, tex_embed_dim + 1 + K + map_emb_dim, H, W]
            """
            tex_clamped = tex_ids.clamp(0, num_textures - 1)
            emb = self.tex_embed(tex_clamped).permute(0, 3, 1, 2)   # [B, D, H, W]
            elev = elevation.unsqueeze(1)                           # [B, 1, H, W]
            parts = [emb, elev]
            if extras is not None and extra_input_channels > 0:
                parts.append(extras)
            # Optional explicit per-side neighbor texture embeddings (T,R,B,L).
            # Matches the convention used by data prep: patch is [W, H] with
            # axis 1 = X (W), axis 2 = Y (H). Sides correspond to cardinal shifts.
            # Wrap-around at borders contaminates ~3% of cells; the conv layers
            # tolerate this small edge noise.
            if self.use_neighbor_tex:
                top_ids = torch.roll(tex_clamped, shifts=-1, dims=2)  # y+1
                bot_ids = torch.roll(tex_clamped, shifts=+1, dims=2)  # y-1
                rgt_ids = torch.roll(tex_clamped, shifts=-1, dims=1)  # x+1
                lft_ids = torch.roll(tex_clamped, shifts=+1, dims=1)  # x-1
                emb_t = self.tex_embed(top_ids).permute(0, 3, 1, 2)
                emb_r = self.tex_embed(rgt_ids).permute(0, 3, 1, 2)
                emb_b = self.tex_embed(bot_ids).permute(0, 3, 1, 2)
                emb_l = self.tex_embed(lft_ids).permute(0, 3, 1, 2)
                parts.extend([emb_t, emb_r, emb_b, emb_l])
            if self.map_encoder is not None and map_tex_ds is not None and map_elev_ds is not None:
                # Encode the full map at the downsampled resolution.
                ds_t_clamped = map_tex_ds.clamp(0, num_textures - 1)
                ds_emb = self.tex_embed(ds_t_clamped).permute(0, 3, 1, 2)  # [B, D, S, S]
                ds_elev = map_elev_ds.unsqueeze(1)                          # [B, 1, S, S]
                ds_in = torch.cat([ds_emb, ds_elev], dim=1)                 # [B, D+1, S, S]
                map_z = self.map_encoder(ds_in)                             # [B, map_emb_dim]
                # Broadcast across patch spatial dims and append.
                B, _, H, W = emb.shape
                map_z_b = map_z.view(B, self.map_emb_dim, 1, 1).expand(-1, -1, H, W)
                parts.append(map_z_b)
            return torch.cat(parts, dim=1)

    return BlendUNet()


# ===========================================================================
# Dataset
# ===========================================================================

class MapPatchDataset:
    """
    Dataset that loads RA3 maps and yields random NxN patches for U-Net training.

    Each map is loaded once and cached. Patches are extracted randomly with
    optional flipX augmentation.

    Each sample returns:
        tex_ids:       [P, P] int32 - texture IDs
        elevation:     [P, P] float32 - normalized elevation
        blend_present: [P, P] uint8 - binary target
        blend_mask:    [P, P] uint8 - 8-bit neighbor bitmask target (255=ignore)
        blend_dir:     [P, P] int16 - direction class (0..16), -1 where no blend
    """

    def __init__(
        self,
        map_paths: List[Path] = None,
        data_dicts: List[Dict[str, np.ndarray]] = None,
        patch_size: int = 64,
        patches_per_map: int = 200,
        augment_flipx: bool = True,
        elev_mean: float = 168.5,
        elev_std: float = 113.85,
        seed: int = 42,
        use_dist_to_boundary: bool = False,
        boundary_bias: float = 0.0,
        use_pattern_code: bool = False,
        rfs_t: float = 0.0,
        rare_class_bias: float = 0.0,
        rare_classes: Optional[List[int]] = None,
        use_map_style: bool = False,
        style_mode: str = "pat",
        use_map_encoder: bool = False,
        map_ds_size: int = 32,
    ):
        self.map_paths = list(map_paths) if map_paths else []
        self.patch_size = patch_size
        self.patches_per_map = patches_per_map
        self.augment_flipx = augment_flipx
        self.elev_mean = elev_mean
        self.elev_std = elev_std
        self.use_dist_to_boundary = use_dist_to_boundary
        self.use_pattern_code = use_pattern_code
        self.use_map_style = bool(use_map_style)
        self.style_mode = str(style_mode) if use_map_style else "pat"
        self.use_map_encoder = bool(use_map_encoder)
        self.map_ds_size = int(map_ds_size)
        self.boundary_bias = float(boundary_bias)
        self._style_dim = 0
        # Repeat-Factor Sampling (Gupta et al. CVPR 2019). rfs_t > 0 enables it.
        # Per-map weight: r_p = max_c max(1, sqrt(t / f_c)) over rare classes c
        # present in map p, where f_c = fraction of maps containing class c.
        self.rfs_t = float(rfs_t)
        # rare_class_bias: probability of centering a patch on a rare-class cell.
        self.rare_class_bias = float(rare_class_bias)
        self.rare_classes = list(rare_classes) if rare_classes else [9, 10, 13, 14]
        self._seed = int(seed)
        self.rng = np.random.default_rng(seed)

        # Load and cache all maps
        self._maps: List[Dict[str, np.ndarray]] = []
        self._map_weights: List[float] = []  # proportional to map area for balanced sampling

        if data_dicts is not None:
            for data in data_dicts:
                w, h = data["map_width"], data["map_height"]
                if w >= patch_size and h >= patch_size:
                    self._maps.append(data)
                    self._map_weights.append(float(w * h))
        else:
            print(f"Loading {len(self.map_paths)} maps...", flush=True)
            for i, p in enumerate(self.map_paths):
                data = load_map_data(str(p))
                if data is not None:
                    w, h = data["map_width"], data["map_height"]
                    if w >= patch_size and h >= patch_size:
                        self._maps.append(data)
                        self._map_weights.append(float(w * h))
                    else:
                        print(f"  [SKIP] {p.name}: {w}x{h} smaller than patch_size={patch_size}", flush=True)
                if (i + 1) % 20 == 0:
                    print(f"  Loaded {i + 1}/{len(self.map_paths)} maps ({len(self._maps)} valid)", flush=True)

        if not self._maps:
            raise ValueError("No valid maps loaded! Check --maps_dir / --prepared_dir path.")

        # Normalize weights
        total = sum(self._map_weights)
        self._map_weights = [w / total for w in self._map_weights]
        print(f"Loaded {len(self._maps)} valid maps for patch_size={patch_size}", flush=True)

        # All unique textures across maps (for embedding table size)
        all_tex_ids = set()
        for data in self._maps:
            all_tex_ids.update(np.unique(data["tex_grid"]).tolist())
        self.max_tex_id = max(all_tex_ids) + 1 if all_tex_ids else 1
        print(f"Max texture ID across all maps: {self.max_tex_id}", flush=True)

        # Pre-compute distance-to-boundary grids if requested.
        if self.use_dist_to_boundary:
            print("Computing distance-to-boundary grids...", flush=True)
            for data in self._maps:
                if "dist_grid" not in data:
                    data["dist_grid"] = _compute_distance_to_boundary(data["tex_grid"])

        # Pre-compute per-map style vector (13-dim pattern_code histogram) for map-level
        # conditioning. Hypothesis: present_f1 plateau is dominated by per-map author-style
        # variance that local context (even at 128x128 RF) can't disambiguate. A global
        # style fingerprint broadcast to all spatial locations gives the U-Net a cheap way
        # to condition local predictions on "what kind of map this is."
        if self.use_map_style:
            # Determine style-vec composition. Modes:
            #   "pat"  -> 13-dim pattern_code histogram only
            #   "tex"  -> tex histogram (fixed 64 dims, decoupled from per-dataset max_tex_id)
            #   "both" -> concat (13 + 64)
            # tex_dim is FIXED so train_ds and val_ds produce same-sized vectors.
            mode = self.style_mode
            tex_dim = 64
            print(f"Pre-computing per-map style vectors (mode='{mode}', tex_dim={tex_dim})...",
                  flush=True)
            for data in self._maps:
                if "pattern_code" not in data:
                    data["pattern_code"] = _compute_pattern_code(data["tex_grid"])
                parts = []
                if mode in ("pat", "both"):
                    pat = data["pattern_code"]
                    hp = np.bincount(pat.ravel().astype(np.int64),
                                     minlength=NUM_PATTERN_CODES).astype(np.float32)
                    hp = hp / max(1.0, float(hp.sum()))
                    parts.append(hp[:NUM_PATTERN_CODES])
                if mode in ("tex", "both"):
                    tex = data["tex_grid"].astype(np.int64)
                    tex = np.clip(tex, 0, tex_dim - 1)  # tex_grid may carry -1 fill cells
                    ht = np.bincount(tex.ravel(), minlength=tex_dim).astype(np.float32)[:tex_dim]
                    ht = ht / max(1.0, float(ht.sum()))
                    parts.append(ht)
                data["style_vec"] = np.concatenate(parts).astype(np.float32)
            self._style_dim = int(self._maps[0]["style_vec"].shape[0])
            print(f"  style_vec dim = {self._style_dim}", flush=True)

        # Pre-compute downsampled tex/elev arrays for the map encoder.
        # We use stride sampling (nearest) for tex_ids (categorical) and
        # area-style avg for elevation. Result is [S, S] per map.
        if self.use_map_encoder:
            S = self.map_ds_size
            print(f"Pre-computing downsampled map inputs ({S}x{S}) for map encoder...",
                  flush=True)
            for data in self._maps:
                tex = data["tex_grid"]
                elev = data["elev_grid"] if data.get("elev_grid") is not None \
                    else np.full(tex.shape, self.elev_mean, dtype=np.float32)
                w, h = tex.shape
                # Stride-sample for tex (preserve discrete IDs).
                xs = np.linspace(0, w - 1, S).astype(np.int32)
                ys = np.linspace(0, h - 1, S).astype(np.int32)
                tex_ds = tex[xs[:, None], ys[None, :]].astype(np.int64)
                # Avg-pool for elev: reshape to S blocks each ~ (w/S)x(h/S).
                # Cheaper: just stride-sample matching tex, then z-normalize.
                elev_ds = elev[xs[:, None], ys[None, :]].astype(np.float32)
                elev_ds = (elev_ds - self.elev_mean) / max(self.elev_std, 1e-6)
                data["map_tex_ds"] = tex_ds
                data["map_elev_ds"] = elev_ds

        # Pre-compute boundary cell index lists if boundary_bias > 0.
        # A boundary cell is one where the dist_grid is 0 (i.e. at a 4-neighbor texture
        # boundary). Sampling patches centered there focuses learning on the cells
        # where the present-detection decision actually has signal.
        if self.boundary_bias > 0.0:
            print(f"Pre-computing boundary cell lists (bias={self.boundary_bias:.2f})...", flush=True)
            for data in self._maps:
                if "dist_grid" not in data:
                    data["dist_grid"] = _compute_distance_to_boundary(data["tex_grid"])
                # All cells with dist == 0 are boundary cells.
                bcs = np.argwhere(data["dist_grid"] == 0.0)  # [N, 2] (x, y)
                # Restrict to cells that can be centers of patches that fit in-bounds.
                w_, h_ = data["map_width"], data["map_height"]
                half = self.patch_size // 2
                in_bounds = (
                    (bcs[:, 0] >= half) & (bcs[:, 0] < w_ - half + 1) &
                    (bcs[:, 1] >= half) & (bcs[:, 1] < h_ - half + 1)
                )
                data["boundary_cells"] = bcs[in_bounds]

        # Pre-compute rare-class cell index lists if rare_class_bias > 0.
        # Rare classes (default 9,10,13,14) are the minority direction labels stuck
        # near 0% accuracy. Centering patches on them gives the rare-class supervision
        # the loss can actually use.
        if self.rare_class_bias > 0.0:
            print(f"Pre-computing rare-class cell lists (bias={self.rare_class_bias:.2f}, "
                  f"classes={self.rare_classes})...", flush=True)
            half = self.patch_size // 2
            n_with_rare = 0
            total_rare = 0
            for data in self._maps:
                dir_grid = data["blend_dir"]
                rare_mask = np.zeros_like(dir_grid, dtype=bool)
                for c in self.rare_classes:
                    rare_mask |= (dir_grid == c)
                rcs = np.argwhere(rare_mask)
                w_, h_ = data["map_width"], data["map_height"]
                if rcs.size > 0:
                    in_bounds = (
                        (rcs[:, 0] >= half) & (rcs[:, 0] < w_ - half + 1) &
                        (rcs[:, 1] >= half) & (rcs[:, 1] < h_ - half + 1)
                    )
                    data["rare_cells"] = rcs[in_bounds]
                    if len(data["rare_cells"]) > 0:
                        n_with_rare += 1
                        total_rare += int(len(data["rare_cells"]))
                else:
                    data["rare_cells"] = np.zeros((0, 2), dtype=np.int64)
            print(f"  Maps with in-bounds rare cells: {n_with_rare}/{len(self._maps)}, "
                  f"total rare cells: {total_rare}", flush=True)

        # Repeat-Factor Sampling: bias map sampling probability toward maps that
        # contain rare classes. Per Gupta et al. CVPR 2019:
        #   f_c = fraction of maps containing class c
        #   r_c = max(1, sqrt(t / f_c))
        #   r_p = max over c in classes(p) of r_c (1 if no rare class in map)
        # We multiply the area-based weight by r_p so large maps still dominate
        # patch budget proportionally, but rare-containing maps see ~r_p more
        # patches than they otherwise would.
        if self.rfs_t > 0.0:
            n_maps = len(self._maps)
            class_freq = {c: 0 for c in self.rare_classes}
            map_classes: List[set] = []
            for data in self._maps:
                unique_classes = set(int(x) for x in np.unique(data["blend_dir"]).tolist())
                map_classes.append(unique_classes)
                for c in self.rare_classes:
                    if c in unique_classes:
                        class_freq[c] += 1
            rare_r = {}
            for c in self.rare_classes:
                f_c = max(1, class_freq[c]) / max(1, n_maps)
                rare_r[c] = float(max(1.0, np.sqrt(self.rfs_t / f_c)))
            rfs_factors = []
            for unique_classes in map_classes:
                r_p = 1.0
                for c in self.rare_classes:
                    if c in unique_classes:
                        r_p = max(r_p, rare_r[c])
                rfs_factors.append(r_p)
            new_weights = [aw * rf for aw, rf in zip(self._map_weights, rfs_factors)]
            total = sum(new_weights)
            self._map_weights = [w / total for w in new_weights]
            print(f"RFS enabled (t={self.rfs_t:.4f}). Class freqs (n maps containing): "
                  f"{class_freq}. Per-class r: "
                  f"{ {c: round(r, 2) for c, r in rare_r.items()} }. "
                  f"Per-map r_p: min={min(rfs_factors):.2f} median={float(np.median(rfs_factors)):.2f} "
                  f"max={max(rfs_factors):.2f} (1.0 means no rare class)", flush=True)

    def set_patch_size(self, new_size: int):
        """Update patch size (for progressive growing). Re-filters maps."""
        self.patch_size = new_size
        # No need to reload; just skip too-small maps at sampling time

    def reset_rng(self) -> None:
        """Re-seed the sampling RNG to its initial state.

        Call this at the start of each val pass so dir_acc / F1 trajectories
        reflect actual model improvement, not which random patches happen to
        land in the val sample this epoch. (Without this the val sample drifts
        each epoch and direction accuracy oscillates 0.20-0.50 just from sampling.)
        Don't call on the train dataset -- you want different train patches each epoch.
        """
        self.rng = np.random.default_rng(self._seed)

    def __len__(self) -> int:
        return len(self._maps) * self.patches_per_map

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Pick a random map (area-weighted)
        map_idx = self.rng.choice(len(self._maps), p=self._map_weights)
        data = self._maps[map_idx]

        w, h = data["map_width"], data["map_height"]
        ps = self.patch_size

        # Ensure map is large enough (should be guaranteed by loading filter)
        if w < ps or h < ps:
            # Fallback: pick another map
            for mi in range(len(self._maps)):
                d = self._maps[mi]
                if d["map_width"] >= ps and d["map_height"] >= ps:
                    data = d
                    w, h = d["map_width"], d["map_height"]
                    break

        # Crop position. Three sampling regimes, checked in priority order:
        #   1. rare_class_bias: center on a rare-class cell (if any in this map)
        #   2. boundary_bias: center on a 4-neighbor texture boundary
        #   3. uniform random
        sampled = False
        if self.rare_class_bias > 0.0:
            rcs = data.get("rare_cells")
            if rcs is not None and len(rcs) > 0 and self.rng.random() < self.rare_class_bias:
                cx, cy = rcs[self.rng.integers(0, len(rcs))]
                x0 = int(max(0, min(w - ps, int(cx) - ps // 2)))
                y0 = int(max(0, min(h - ps, int(cy) - ps // 2)))
                sampled = True
        if not sampled and self.boundary_bias > 0.0:
            bcs = data.get("boundary_cells")
            if bcs is not None and len(bcs) > 0 and self.rng.random() < self.boundary_bias:
                cx, cy = bcs[self.rng.integers(0, len(bcs))]
                x0 = int(max(0, min(w - ps, int(cx) - ps // 2)))
                y0 = int(max(0, min(h - ps, int(cy) - ps // 2)))
                sampled = True
        if not sampled:
            x0 = int(self.rng.integers(0, w - ps + 1))
            y0 = int(self.rng.integers(0, h - ps + 1))

        tex_patch = data["tex_grid"][x0:x0 + ps, y0:y0 + ps].copy()
        present_patch = data["blend_present"][x0:x0 + ps, y0:y0 + ps].copy()
        mask_patch = data["blend_mask"][x0:x0 + ps, y0:y0 + ps].copy()
        dir_patch = data["blend_dir"][x0:x0 + ps, y0:y0 + ps].copy()

        # Elevation
        if data["elev_grid"] is not None:
            elev_patch = data["elev_grid"][x0:x0 + ps, y0:y0 + ps].copy()
        else:
            elev_patch = np.full((ps, ps), self.elev_mean, dtype=np.float32)

        # Normalize elevation
        elev_patch = (elev_patch.astype(np.float32) - self.elev_mean) / max(self.elev_std, 1e-6)

        # Optional dist-to-boundary patch (already in cells; transform: 1/(1+d) so boundary≈1, deep≈0)
        dist_patch = None
        if self.use_dist_to_boundary and "dist_grid" in data:
            dp = data["dist_grid"][x0:x0 + ps, y0:y0 + ps].copy().astype(np.float32)
            dist_patch = 1.0 / (1.0 + dp)

        # Optional pattern_code patch (int8 0..12; one-hot expanded later in collate)
        pattern_patch = None
        if self.use_pattern_code and "pattern_code" in data:
            pattern_patch = data["pattern_code"][x0:x0 + ps, y0:y0 + ps].copy().astype(np.int64)

        # FlipX augmentation (horizontal flip along axis=1, i.e. flip columns)
        # Research finding: flipX has ~2% noise, acceptable.
        if self.augment_flipx and self.rng.random() < 0.5:
            tex_patch = np.flip(tex_patch, axis=1).copy()
            elev_patch = np.flip(elev_patch, axis=1).copy()
            present_patch = np.flip(present_patch, axis=1).copy()
            mask_patch = _flip_mask_x(np.flip(mask_patch, axis=1).copy())
            dir_patch = _flip_dir_x(np.flip(dir_patch, axis=1).copy())
            if dist_patch is not None:
                dist_patch = np.flip(dist_patch, axis=1).copy()
            if pattern_patch is not None:
                # pattern_code values bake in left/right asymmetry (codes 5/6, 1/2, 9/10, 11/12).
                # Flip the spatial axis AND remap the codes that swap under horizontal flip.
                pattern_patch = _flip_pattern_code_x(np.flip(pattern_patch, axis=1).copy())

        sample = {
            "tex_ids": tex_patch.astype(np.int64),
            "elevation": elev_patch.astype(np.float32),
            "blend_present": present_patch.astype(np.float32),
            "blend_mask": mask_patch,  # uint8, 255=ignore
            "blend_dir": dir_patch.astype(np.int64),
        }
        if dist_patch is not None:
            sample["dist"] = dist_patch.astype(np.float32)
        if pattern_patch is not None:
            sample["pattern_code"] = pattern_patch.astype(np.int64)
        if self.use_map_style and "style_vec" in data:
            sample["style_vec"] = data["style_vec"].astype(np.float32)  # [13]
        if self.use_map_encoder and "map_tex_ds" in data:
            sample["map_tex_ds"] = data["map_tex_ds"]
            sample["map_elev_ds"] = data["map_elev_ds"]
        return sample


def _flip_pattern_code_x(pat: np.ndarray) -> np.ndarray:
    """Remap pattern_code values that flip under horizontal mirror.
    Code legend (see _compute_pattern_code):
      0=none  1=L==T  2=R==T  3=R==B  4=L==B  5=L  6=R  7=T  8=B
      9=TL  10=TR  11=BR  12=BL
    Under flipX (left<->right): 1<->2, 3<->4, 5<->6, 9<->10, 11<->12. 7,8,0 stay.
    """
    swap = np.array([0, 2, 1, 4, 3, 6, 5, 7, 8, 10, 9, 12, 11], dtype=pat.dtype)
    return swap[np.clip(pat, 0, 12)]


def _flip_mask_x(mask8: np.ndarray) -> np.ndarray:
    """
    Flip the 8-bit neighbor mask horizontally (left<->right).
    Bit layout: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
    FlipX swaps: TL<->TR (0<->2), L<->R (3<->4), BL<->BR (5<->7), T and B stay.
    """
    # Ignore cells with mask=255
    ignore = (mask8 == 255)
    m = mask8.astype(np.uint16)

    # Extract bits
    bit = [(m >> i) & 1 for i in range(8)]

    # Rebuild with swapped positions
    out = np.zeros_like(m)
    # TL(0) <-> TR(2)
    out |= bit[2] << 0  # new TL = old TR
    out |= bit[1] << 1  # T stays
    out |= bit[0] << 2  # new TR = old TL
    # L(3) <-> R(4)
    out |= bit[4] << 3  # new L = old R
    out |= bit[3] << 4  # new R = old L
    # BL(5) <-> BR(7)
    out |= bit[7] << 5  # new BL = old BR
    out |= bit[6] << 6  # B stays
    out |= bit[5] << 7  # new BR = old BL

    out = out.astype(np.uint8)
    out[ignore] = 255
    return out


# FlipX direction mapping.
# Directions that swap under horizontal flip:
#   Left(1) <-> Right(17)
#   TopLeft(52) <-> TopRight(56)
#   BottomLeft(36) <-> BottomRight(40)
#   ExceptTopLeft(8) <-> ExceptTopRight(4)
#   ExceptBottomLeft(24) <-> ExceptBottomRight(20)
#   Top(18) and Bottom(2) stay the same.
#   33, 34, 49, 50 are rare combo values; swap pairs: 33<->34, 49<->50
_FLIPX_DIR_MAP = {
    -1: -1,
    1: 17, 17: 1,        # Left <-> Right
    2: 2,                 # Bottom stays
    18: 18,               # Top stays
    4: 8, 8: 4,           # ExceptTopRight <-> ExceptTopLeft
    20: 24, 24: 20,       # ExceptBottomRight <-> ExceptBottomLeft
    36: 40, 40: 36,       # BottomLeft <-> BottomRight
    52: 56, 56: 52,       # TopLeft <-> TopRight
    33: 34, 34: 33,       # rare combos
    49: 50, 50: 49,       # rare combos
}


def _flip_dir_x(dir_class: np.ndarray) -> np.ndarray:
    """
    Flip direction classes horizontally.
    dir_class: [H, W] int array of direction class indices (0..16)
    """
    # Build a class-to-class LUT for flipX
    lut = np.arange(NUM_DIR_CLASSES, dtype=np.int64)
    for raw_from, raw_to in _FLIPX_DIR_MAP.items():
        cls_from = DIR_VAL_TO_CLASS.get(raw_from, -1)
        cls_to = DIR_VAL_TO_CLASS.get(raw_to, -1)
        if cls_from >= 0 and cls_to >= 0:
            lut[cls_from] = cls_to

    # Apply LUT (handle -1 as "no direction")
    valid = (dir_class >= 0) & (dir_class < NUM_DIR_CLASSES)
    out = dir_class.copy()
    out[valid] = lut[dir_class[valid]]
    return out


# ===========================================================================
# Loss functions
# ===========================================================================

def compute_losses(
    outputs: Dict[str, "torch.Tensor"],
    targets: Dict[str, "torch.Tensor"],
    dir_class_freqs: Optional["torch.Tensor"] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    logit_adj_tau: float = 1.0,
    present_pos_weight: float = 1.0,
    present_boundary_weight: float = 0.0,
    dir_loss_type: str = "ce",
    dir_focal_gamma: float = 0.0,
    dir_class_weights: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """
    Compute multi-task loss.

    Args:
        outputs: model outputs dict with present_logits, mask_logits, dir_logits
        targets: dict with blend_present [B,H,W], blend_mask [B,H,W], blend_dir [B,H,W]
        dir_class_freqs: [NUM_DIR_CLASSES] tensor of class frequencies for logit adjustment
        loss_weights: optional dict of loss component weights

    Returns:
        (total_loss, loss_dict) where loss_dict has per-component losses for logging
    """
    import torch
    import torch.nn.functional as F

    w = loss_weights or {}
    w_present = w.get("present", 1.0)
    w_mask = w.get("mask", 1.0)
    w_dir = w.get("dir", 1.0)

    present_logits = outputs["present_logits"]  # [B, 1, H, W]
    mask_logits = outputs["mask_logits"]         # [B, 8, H, W]
    dir_logits = outputs["dir_logits"]           # [B, 17, H, W]

    y_present = targets["blend_present"]  # [B, H, W] float
    y_mask = targets["blend_mask"]        # [B, H, W] uint8 (255=ignore)
    y_dir = targets["blend_dir"]          # [B, H, W] int64

    B, _, H, W = present_logits.shape

    # ---- Head 1: blend_present (BCE, optionally pos-weighted, optionally boundary-weighted) ----
    # Boundary-weighted BCE: cells AT the GT-blend boundary (any 4-neighbor disagrees in
    # blend_present) get extra weight. Targets the dominant failure mode where 82% of FPs
    # are 1-cell extensions of real blends and ~50% of FNs are perimeter cells. Sharpening
    # the boundary in the loss landscape forces the model to commit at exactly these cells.
    if present_boundary_weight > 0.0:
        # 4-neighbor disagreement mask. y_present is float [B,H,W] in {0,1}; pad with edge.
        yp = y_present.unsqueeze(1)  # [B,1,H,W]
        # Use replicate padding so border cells aren't artificially flagged.
        yp_pad = F.pad(yp, (1, 1, 1, 1), mode='replicate')
        diff_top = (yp_pad[:, :, :-2, 1:-1] != yp).float()
        diff_bot = (yp_pad[:, :, 2:,  1:-1] != yp).float()
        diff_lft = (yp_pad[:, :, 1:-1, :-2] != yp).float()
        diff_rgt = (yp_pad[:, :, 1:-1, 2:]  != yp).float()
        boundary_mask = (diff_top + diff_bot + diff_lft + diff_rgt > 0).float().squeeze(1)  # [B,H,W]
        per_cell_w = 1.0 + present_boundary_weight * boundary_mask
    else:
        per_cell_w = None

    if present_pos_weight != 1.0:
        pw = torch.tensor(present_pos_weight, device=present_logits.device, dtype=present_logits.dtype)
        loss_present_raw = F.binary_cross_entropy_with_logits(
            present_logits.squeeze(1), y_present, pos_weight=pw, reduction="none" if per_cell_w is not None else "mean"
        )
    else:
        loss_present_raw = F.binary_cross_entropy_with_logits(
            present_logits.squeeze(1), y_present, reduction="none" if per_cell_w is not None else "mean"
        )

    if per_cell_w is not None:
        loss_present = (loss_present_raw * per_cell_w).sum() / per_cell_w.sum().clamp_min(1e-9)
    else:
        loss_present = loss_present_raw

    # ---- Head 2: neighbor_mask (ASL, Asymmetric Loss) ----
    # Only compute where mask is valid (not 255)
    valid_mask = (y_mask != 255)  # [B, H, W]
    if valid_mask.any():
        # Unpack 8-bit mask to [B, 8, H, W] binary targets
        m = y_mask.long()  # [B, H, W]
        bits = torch.stack([(m >> i) & 1 for i in range(8)], dim=1).float()  # [B, 8, H, W]

        # Create a validity mask expanded to [B, 8, H, W]
        valid_expanded = valid_mask.unsqueeze(1).expand_as(bits)

        # ASL loss only on valid positions
        # We compute full ASL then mask
        probs = torch.sigmoid(mask_logits)
        clip_val = 0.05
        probs_neg = (probs - clip_val).clamp(min=0.0)

        loss_pos = bits * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - bits) * torch.log((1 - probs_neg).clamp(min=1e-8))

        gamma_neg = 4.0
        pt_neg = probs_neg
        loss_neg = loss_neg * (pt_neg ** gamma_neg)

        per_element_loss = -(loss_pos + loss_neg)
        # Zero out invalid positions
        per_element_loss = per_element_loss * valid_expanded.float()
        loss_mask = per_element_loss.sum() / valid_expanded.float().sum().clamp(min=1.0)
    else:
        loss_mask = present_logits.new_zeros(())

    # ---- Head 3: direction (logit-adjusted CE, masked to present cells) ----
    # Only train where blend is actually present
    present_bool = (y_present > 0.5)  # [B, H, W]
    dir_valid = present_bool & (y_dir >= 0) & (y_dir < NUM_DIR_CLASSES)

    if dir_valid.any():
        # Reshape for cross_entropy: [N, C]
        dir_logits_flat = dir_logits.permute(0, 2, 3, 1).reshape(-1, NUM_DIR_CLASSES)  # [B*H*W, 17]
        dir_target_flat = y_dir.reshape(-1)  # [B*H*W]
        dir_valid_flat = dir_valid.reshape(-1)  # [B*H*W]

        # Logit adjustment: subtract log(class_freq) from logits to handle class imbalance
        if dir_class_freqs is not None and logit_adj_tau > 0.0:
            log_freqs = torch.log(dir_class_freqs.clamp(min=1e-8)) * logit_adj_tau
            dir_logits_flat = dir_logits_flat - log_freqs.unsqueeze(0)

        dir_logits_valid = dir_logits_flat[dir_valid_flat]
        dir_target_valid = dir_target_flat[dir_valid_flat]

        if dir_loss_type == "focal" or dir_loss_type == "cb_focal":
            # (Optionally class-balanced) focal cross-entropy.
            # log_softmax + nll, with focal modulation (1-pt)^gamma and optional
            # per-class weights. Reduces dominance of easy/well-classified cells
            # so rare directions stop collapsing to 0% accuracy.
            log_probs = F.log_softmax(dir_logits_valid, dim=-1)
            log_pt = log_probs.gather(1, dir_target_valid.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp().clamp(max=1.0 - 1e-6)
            gamma = dir_focal_gamma
            focal_factor = (1.0 - pt).pow(gamma) if gamma > 0 else 1.0
            per_sample = -focal_factor * log_pt
            if dir_class_weights is not None:
                w_per_sample = dir_class_weights[dir_target_valid]
                loss_dir = (per_sample * w_per_sample).sum() / w_per_sample.sum().clamp(min=1e-8)
            else:
                loss_dir = per_sample.mean()
        else:
            # Standard CE (with optional logit adjustment already applied above).
            loss_dir = F.cross_entropy(
                dir_logits_valid,
                dir_target_valid,
                weight=dir_class_weights,
                reduction="mean",
            )
    else:
        loss_dir = present_logits.new_zeros(())

    total = w_present * loss_present + w_mask * loss_mask + w_dir * loss_dir

    loss_dict = {
        "loss_present": float(loss_present.item()),
        "loss_mask": float(loss_mask.item()),
        "loss_dir": float(loss_dir.item()),
        "loss_total": float(total.item()),
    }
    return total, loss_dict


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(
    outputs: Dict[str, "torch.Tensor"],
    targets: Dict[str, "torch.Tensor"],
    present_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute F1, accuracy, etc. for each head. Must be called inside torch.no_grad()."""
    import torch

    present_logits = outputs["present_logits"].squeeze(1)  # [B, H, W]
    mask_logits = outputs["mask_logits"]                    # [B, 8, H, W]
    dir_logits = outputs["dir_logits"]                      # [B, 17, H, W]

    y_present = targets["blend_present"]
    y_mask = targets["blend_mask"]
    y_dir = targets["blend_dir"]

    metrics = {}

    # -- Present: F1, precision, recall, accuracy --
    pred_present = (torch.sigmoid(present_logits) > present_threshold).float()
    tp = (pred_present * y_present).sum().item()
    fp = (pred_present * (1 - y_present)).sum().item()
    fn = ((1 - pred_present) * y_present).sum().item()
    tn = ((1 - pred_present) * (1 - y_present)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    metrics["present_f1"] = f1
    metrics["present_prec"] = precision
    metrics["present_rec"] = recall
    metrics["present_acc"] = accuracy

    # -- Mask: per-bit accuracy (on valid cells) --
    valid_mask = (y_mask != 255)
    if valid_mask.any():
        m = y_mask.long()
        bits_target = torch.stack([(m >> i) & 1 for i in range(8)], dim=1).float()  # [B,8,H,W]
        bits_pred = (torch.sigmoid(mask_logits) > 0.5).float()
        valid_exp = valid_mask.unsqueeze(1).expand_as(bits_target).float()
        correct = ((bits_pred == bits_target).float() * valid_exp).sum()
        total_valid = valid_exp.sum().clamp(min=1)
        metrics["mask_bit_acc"] = (correct / total_valid).item()

        # Exact match: all 8 bits correct
        exact = ((bits_pred == bits_target).all(dim=1).float() * valid_mask.float())
        metrics["mask_exact_acc"] = (exact.sum() / valid_mask.float().sum().clamp(min=1)).item()
    else:
        metrics["mask_bit_acc"] = 0.0
        metrics["mask_exact_acc"] = 0.0

    # -- Direction: accuracy on present cells --
    present_bool = (y_present > 0.5)
    dir_valid = present_bool & (y_dir >= 0) & (y_dir < NUM_DIR_CLASSES)
    if dir_valid.any():
        dir_pred = dir_logits.argmax(dim=1)  # [B, H, W]
        dir_correct = (dir_pred == y_dir) & dir_valid
        metrics["dir_acc"] = (dir_correct.float().sum() / dir_valid.float().sum().clamp(min=1)).item()
    else:
        metrics["dir_acc"] = 0.0

    return metrics


def accumulate_metric_counters(
    outputs: Dict[str, "torch.Tensor"],
    targets: Dict[str, "torch.Tensor"],
    pool: Dict[str, float],
    present_threshold: float = 0.5,
) -> None:
    """Accumulate raw counts into `pool` so the caller can derive POOLED metrics.

    Per-batch averaging of derived metrics (F1, dir_acc) systematically biases
    upward when small-N batches happen to have higher accuracy than large-N
    batches -- we hit this on dir_acc (per-batch ~0.77, true population ~0.43).
    Use this helper instead of averaging compute_metrics() outputs.
    """
    import torch

    present_logits = outputs["present_logits"].squeeze(1)
    mask_logits = outputs["mask_logits"]
    dir_logits = outputs["dir_logits"]

    y_present = targets["blend_present"]
    y_mask = targets["blend_mask"]
    y_dir = targets["blend_dir"]

    pred_present = (torch.sigmoid(present_logits) > present_threshold).float()
    pool["tp"] += (pred_present * y_present).sum().item()
    pool["fp"] += (pred_present * (1 - y_present)).sum().item()
    pool["fn"] += ((1 - pred_present) * y_present).sum().item()
    pool["tn"] += ((1 - pred_present) * (1 - y_present)).sum().item()

    valid_mask = (y_mask != 255)
    if valid_mask.any():
        m = y_mask.long()
        bits_target = torch.stack([(m >> i) & 1 for i in range(8)], dim=1).float()
        bits_pred = (torch.sigmoid(mask_logits) > 0.5).float()
        valid_exp = valid_mask.unsqueeze(1).expand_as(bits_target).float()
        pool["mask_bit_correct"] += ((bits_pred == bits_target).float() * valid_exp).sum().item()
        pool["mask_bit_total"] += valid_exp.sum().item()
        exact = ((bits_pred == bits_target).all(dim=1).float() * valid_mask.float())
        pool["mask_exact_correct"] += exact.sum().item()
        pool["mask_exact_total"] += valid_mask.float().sum().item()

    present_bool = (y_present > 0.5)
    dir_valid = present_bool & (y_dir >= 0) & (y_dir < NUM_DIR_CLASSES)
    if dir_valid.any():
        dir_pred = dir_logits.argmax(dim=1)
        pool["dir_correct"] += ((dir_pred == y_dir) & dir_valid).float().sum().item()
        pool["dir_total"] += dir_valid.float().sum().item()


def finalize_pool(pool: Dict[str, float]) -> Dict[str, float]:
    """Derive F1, precision, recall, dir_acc, mask_*  from accumulated counters."""
    tp = pool.get("tp", 0.0); fp = pool.get("fp", 0.0)
    fn = pool.get("fn", 0.0); tn = pool.get("tn", 0.0)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {
        "present_f1": float(f1),
        "present_prec": float(prec),
        "present_rec": float(rec),
        "present_acc": float((tp + tn) / max(1.0, tp + fp + fn + tn)),
        "mask_bit_acc": float(pool.get("mask_bit_correct", 0.0) / max(1.0, pool.get("mask_bit_total", 0.0))),
        "mask_exact_acc": float(pool.get("mask_exact_correct", 0.0) / max(1.0, pool.get("mask_exact_total", 0.0))),
        "dir_acc": float(pool.get("dir_correct", 0.0) / max(1.0, pool.get("dir_total", 0.0))),
    }


# ===========================================================================
# Training loop
# ===========================================================================

def train(args):
    """Main training function."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial patch size (for progressive growing).
    initial_patch = args.patch_size
    if args.progressive:
        initial_patch = 16
        print(f"Progressive growing enabled: 16 -> 32 -> {args.patch_size}", flush=True)

    # Source maps: prefer prepared .npz directory if given.
    if args.prepared_dir:
        # Support comma-separated list of dirs to combine multiple datasets.
        # Filename collisions are resolved keep-first (earlier dir wins).
        prep_dirs = [Path(s.strip()).resolve() for s in args.prepared_dir.split(",") if s.strip()]
        seen_names: set = set()
        npz_paths = []
        for d in prep_dirs:
            d_paths = sorted(d.glob("*.npz"))
            d_kept = []
            for p in d_paths:
                if p.name in seen_names:
                    continue
                seen_names.add(p.name)
                d_kept.append(p)
            npz_paths.extend(d_kept)
            print(f"Loaded {len(d_kept)} prepared .npz files from {d}", flush=True)
        if not npz_paths:
            raise SystemExit(f"No .npz files found in any of {prep_dirs}.")
        prep_dir = prep_dirs[0]  # used for downstream messages
        if args.exclude_patterns:
            patterns = [s.strip().lower() for s in args.exclude_patterns.split(",") if s.strip()]
            kept = [p for p in npz_paths if not any(pat in p.name.lower() for pat in patterns)]
            dropped = [p.name for p in npz_paths if p not in kept]
            print(f"Excluded {len(dropped)} files matching {patterns}: {dropped}", flush=True)
            npz_paths = kept

        if args.val_files:
            val_set = set()
            with open(args.val_files) as f:
                for line in f:
                    n = line.strip()
                    if n and not n.startswith("#"):
                        val_set.add(n)
            val_indices = [i for i, p in enumerate(npz_paths) if p.name in val_set]
            train_indices = [i for i, p in enumerate(npz_paths) if p.name not in val_set]
            missing = val_set - {npz_paths[i].name for i in val_indices}
            if missing:
                print(f"WARNING: val_files lists {len(missing)} maps not found: {sorted(missing)}",
                      flush=True)
            print(f"Using explicit val list: {len(val_indices)} val, {len(train_indices)} train",
                  flush=True)
        else:
            rng = np.random.default_rng(args.seed)
            indices = np.arange(len(npz_paths))
            rng.shuffle(indices)
            n_val = max(1, int(len(npz_paths) * args.val_frac))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

        train_dicts = []
        val_dicts = []
        # Parallel load: I/O-bound so ThreadPool is fine.
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.time()
        from functools import partial
        loader = partial(load_prepared_npz,
                         with_dist=bool(args.use_dist_to_boundary),
                         with_pattern=bool(args.use_pattern_code))
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4))) as pool:
            train_dicts = [d for d in pool.map(
                loader,
                [str(npz_paths[i]) for i in train_indices],
            ) if d is not None]
            val_dicts = [d for d in pool.map(
                loader,
                [str(npz_paths[i]) for i in val_indices],
            ) if d is not None]
        print(f"Loaded prepared maps in {time.time() - t0:.1f}s "
              f"(train={len(train_dicts)}, val={len(val_dicts)})", flush=True)

        rare_cls_list = [int(x) for x in str(args.rare_classes).split(",") if x.strip() != ""] \
            if args.rare_classes else None
        train_ds = MapPatchDataset(
            data_dicts=train_dicts,
            patch_size=initial_patch,
            patches_per_map=args.patches_per_map,
            augment_flipx=args.augment_flipx,
            seed=args.seed,
            use_dist_to_boundary=args.use_dist_to_boundary,
            boundary_bias=args.boundary_bias,
            use_pattern_code=args.use_pattern_code,
            rfs_t=args.rfs_t,
            rare_class_bias=args.rare_class_bias,
            rare_classes=rare_cls_list,
            use_map_style=args.use_map_style,
            style_mode=args.style_mode,
            use_map_encoder=(args.map_emb_dim > 0),
            map_ds_size=args.map_ds_size,
        )
        val_ds = MapPatchDataset(
            data_dicts=val_dicts,
            patch_size=initial_patch,
            patches_per_map=max(50, args.patches_per_map // 4),
            augment_flipx=False,
            seed=args.seed + 1,
            use_dist_to_boundary=args.use_dist_to_boundary,
            boundary_bias=0.0,  # eval is uniform
            use_pattern_code=args.use_pattern_code,
            rfs_t=0.0,           # eval uniform
            rare_class_bias=0.0, # eval uniform
            use_map_style=args.use_map_style,
            style_mode=args.style_mode,
            use_map_encoder=(args.map_emb_dim > 0),
            map_ds_size=args.map_ds_size,
        )
    else:
        # Legacy path: parse .map files directly (slow).
        maps_dir = Path(args.maps_dir)
        map_files = find_map_files(maps_dir)
        print(f"Found {len(map_files)} map files in {maps_dir}", flush=True)
        if not map_files:
            raise SystemExit("No map files found. Check --maps_dir.")

        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(map_files))
        rng.shuffle(indices)
        n_val = max(1, int(len(map_files) * args.val_frac))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        train_maps = [map_files[i] for i in train_indices]
        val_maps = [map_files[i] for i in val_indices]
        print(f"Train maps: {len(train_maps)}, Val maps: {len(val_maps)}", flush=True)

        train_ds = MapPatchDataset(
            map_paths=train_maps,
            patch_size=initial_patch,
            patches_per_map=args.patches_per_map,
            augment_flipx=args.augment_flipx,
            seed=args.seed,
        )
        val_ds = MapPatchDataset(
            map_paths=val_maps,
            patch_size=initial_patch,
            patches_per_map=max(50, args.patches_per_map // 4),
            augment_flipx=False,
            seed=args.seed + 1,
        )

    # Determine embedding table size
    num_textures = max(train_ds.max_tex_id, val_ds.max_tex_id) + 1
    num_textures = max(num_textures, 400)  # safety margin
    print(f"Num textures for embedding: {num_textures}", flush=True)

    # Build model
    extra_in_ch = 0
    if args.use_dist_to_boundary:
        extra_in_ch += 1
    if args.use_pattern_code:
        extra_in_ch += NUM_PATTERN_CODES  # one-hot 13 channels
    if args.use_map_style:
        extra_in_ch += int(getattr(train_ds, "_style_dim", NUM_PATTERN_CODES))
    model = _make_model(
        num_textures=num_textures,
        tex_embed_dim=args.tex_embed_dim,
        hidden_channels=args.hidden_channels,
        extra_input_channels=extra_in_ch,
        dir_head_type=args.dir_head_type,
        map_emb_dim=args.map_emb_dim,
        map_ds_size=args.map_ds_size,
        use_neighbor_tex=args.use_neighbor_tex,
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", flush=True)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_ds) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
        eta_min=args.lr * 0.01,
    )

    # Direction class frequency for logit adjustment.
    # Compute ONCE from the full training set before epoch 1 -- avoids the
    # per-epoch oscillation that destabilized direction accuracy in v1.
    # Counts are computed unconditionally (cheap) so cb_focal can use them
    # even when logit adjustment is off.
    print("Computing direction class counts from training set...", flush=True)
    dir_class_counts_init = torch.zeros(NUM_DIR_CLASSES, device=device)
    for data in train_ds._maps:
        yp = data["blend_present"]
        yd = data["blend_dir"]
        valid = (yp > 0) & (yd >= 0) & (yd < NUM_DIR_CLASSES)
        if not valid.any():
            continue
        vals = yd[valid]
        for c in range(NUM_DIR_CLASSES):
            dir_class_counts_init[c] += float((vals == c).sum())
    print(f"  Direction counts: {[int(c) for c in dir_class_counts_init.tolist()]}", flush=True)

    if args.logit_adj_tau > 0.0:
        if dir_class_counts_init.sum() > 0:
            # Smooth with a small Laplace constant so rare classes still appear.
            dir_class_freqs = (dir_class_counts_init + 1.0) / (dir_class_counts_init.sum() + NUM_DIR_CLASSES)
        else:
            dir_class_freqs = torch.ones(NUM_DIR_CLASSES, device=device) / NUM_DIR_CLASSES
        print(f"  Direction prior: {[f'{p:.4f}' for p in dir_class_freqs.tolist()]}", flush=True)
    else:
        dir_class_freqs = None
        print("Logit adjustment disabled (--logit_adj_tau 0).", flush=True)

    # Class-balanced effective-number weights (Cui et al., 2019)
    # weight_c = (1 - beta) / (1 - beta^n_c), then normalized so sum=NUM_CLASSES
    if args.dir_loss_type == "cb_focal":
        beta = float(args.dir_cb_beta)
        n_per = dir_class_counts_init.clamp(min=1.0)
        # Effective number: (1 - beta^n) / (1 - beta)
        eff_num = (1.0 - torch.pow(torch.tensor(beta, device=device), n_per)) / (1.0 - beta)
        cb_weights = 1.0 / eff_num
        # Normalize so mean weight ~= 1 (doesn't change relative scale).
        cb_weights = cb_weights * (NUM_DIR_CLASSES / cb_weights.sum())
        dir_class_weights = cb_weights
        print(f"  Class-balanced dir weights (beta={beta}): "
              f"{[f'{w:.3f}' for w in dir_class_weights.tolist()]}", flush=True)
    else:
        dir_class_weights = None

    # Loss weights
    loss_weights = {
        "present": args.loss_w_present,
        "mask": args.loss_w_mask,
        "dir": args.loss_w_dir,
    }

    # Collate function
    def collate_fn(batch):
        tex = torch.stack([torch.from_numpy(b["tex_ids"]) for b in batch]).long()
        elev = torch.stack([torch.from_numpy(b["elevation"]) for b in batch]).float()
        present = torch.stack([torch.from_numpy(b["blend_present"]) for b in batch]).float()
        mask = torch.stack([torch.from_numpy(b["blend_mask"]) for b in batch]).long()
        dir_t = torch.stack([torch.from_numpy(b["blend_dir"]) for b in batch]).long()
        out = {
            "tex_ids": tex,
            "elevation": elev,
            "blend_present": present,
            "blend_mask": mask,
            "blend_dir": dir_t,
        }
        # Build extras tensor: optional dist (1ch) + optional pattern_code one-hot (13ch)
        # + optional map-style histogram broadcast (13ch).
        extras_list = []
        if "dist" in batch[0]:
            dist = torch.stack([torch.from_numpy(b["dist"]) for b in batch]).float()
            extras_list.append(dist.unsqueeze(1))  # [B, 1, H, W]
        if "pattern_code" in batch[0]:
            pat = torch.stack([torch.from_numpy(b["pattern_code"]) for b in batch]).long()  # [B, H, W]
            pat = pat.clamp(0, NUM_PATTERN_CODES - 1)
            oh = torch.nn.functional.one_hot(pat, NUM_PATTERN_CODES).permute(0, 3, 1, 2).float()
            extras_list.append(oh)  # [B, 13, H, W]
        if "style_vec" in batch[0]:
            sv = torch.stack([torch.from_numpy(b["style_vec"]) for b in batch]).float()  # [B, 13]
            H, W = batch[0]["tex_ids"].shape
            sv_b = sv.view(sv.shape[0], sv.shape[1], 1, 1).expand(-1, -1, H, W).contiguous()
            extras_list.append(sv_b)  # [B, 13, H, W]
        if extras_list:
            out["extras"] = torch.cat(extras_list, dim=1)
        if "map_tex_ds" in batch[0]:
            out["map_tex_ds"] = torch.stack([torch.from_numpy(b["map_tex_ds"]) for b in batch]).long()
            out["map_elev_ds"] = torch.stack([torch.from_numpy(b["map_elev_ds"]) for b in batch]).float()
        return out

    # Progressive growing schedule
    grow_schedule = []
    if args.progressive:
        epochs_per_stage = max(1, args.epochs // 3)
        grow_schedule = [
            (0, 16),
            (epochs_per_stage, 32),
            (epochs_per_stage * 2, args.patch_size),
        ]
    else:
        grow_schedule = [(0, args.patch_size)]

    current_patch_size = grow_schedule[0][1]
    next_grow_idx = 1

    best_val_f1 = 0.0
    train_history = []

    print(f"\n{'='*60}", flush=True)
    print(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}", flush=True)
    print(f"Loss weights: present={loss_weights['present']}, mask={loss_weights['mask']}, dir={loss_weights['dir']}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for epoch in range(args.epochs):
        # Check progressive growing
        if next_grow_idx < len(grow_schedule) and epoch >= grow_schedule[next_grow_idx][0]:
            current_patch_size = grow_schedule[next_grow_idx][1]
            next_grow_idx += 1
            print(f"\n>>> Progressive grow: patch_size -> {current_patch_size}", flush=True)
            train_ds.set_patch_size(current_patch_size)
            val_ds.set_patch_size(current_patch_size)

        # Create dataloaders (recreated each epoch since dataset is random anyway)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

        # ---- Training ----
        model.train()
        epoch_losses = []
        epoch_metrics_list = []
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            tex = batch["tex_ids"].to(device)
            elev = batch["elevation"].to(device)
            targets = {
                "blend_present": batch["blend_present"].to(device),
                "blend_mask": batch["blend_mask"].to(device),
                "blend_dir": batch["blend_dir"].to(device),
            }

            # Build input
            x = model.encode_input(
                tex, elev,
                extras=batch.get("extras", None).to(device) if batch.get("extras", None) is not None else None,
                map_tex_ds=batch.get("map_tex_ds", None).to(device) if batch.get("map_tex_ds", None) is not None else None,
                map_elev_ds=batch.get("map_elev_ds", None).to(device) if batch.get("map_elev_ds", None) is not None else None,
            )

            # Forward
            outputs = model(x)

            # Loss
            loss, loss_dict = compute_losses(
                outputs, targets,
                dir_class_freqs=dir_class_freqs,
                loss_weights=loss_weights,
                logit_adj_tau=args.logit_adj_tau,
                present_pos_weight=args.present_pos_weight,
                present_boundary_weight=args.present_boundary_weight,
                dir_loss_type=args.dir_loss_type,
                dir_focal_gamma=args.dir_focal_gamma,
                dir_class_weights=dir_class_weights,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss_dict)

            # (No per-step dir-class accumulation: priors are fixed before training.)
            pass

            if (step + 1) % args.log_every == 0:
                avg_loss = np.mean([d["loss_total"] for d in epoch_losses[-args.log_every:]])
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch+1} step {step+1}/{len(train_loader)}: "
                      f"loss={avg_loss:.4f} lr={lr:.2e}", flush=True)

        # (Direction priors are fixed; no per-epoch update.)

        train_time = time.time() - t0
        avg_train_loss = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}

        # ---- Validation ----
        model.eval()
        # Re-seed the val dataset RNG so the same patches are evaluated every
        # epoch -- otherwise sampling noise dominates dir_acc trajectories.
        if hasattr(val_ds, "reset_rng"):
            val_ds.reset_rng()
        val_losses = []
        # Pooled counters -- per-batch averaging biases dir_acc/F1 upward
        # because rare-class batches with few valid cells get equal weight.
        val_pool: Dict[str, float] = {
            "tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0,
            "mask_bit_correct": 0.0, "mask_bit_total": 0.0,
            "mask_exact_correct": 0.0, "mask_exact_total": 0.0,
            "dir_correct": 0.0, "dir_total": 0.0,
        }

        with torch.no_grad():
            for batch in val_loader:
                tex = batch["tex_ids"].to(device)
                elev = batch["elevation"].to(device)
                targets = {
                    "blend_present": batch["blend_present"].to(device),
                    "blend_mask": batch["blend_mask"].to(device),
                    "blend_dir": batch["blend_dir"].to(device),
                }

                x = model.encode_input(
                tex, elev,
                extras=batch.get("extras", None).to(device) if batch.get("extras", None) is not None else None,
                map_tex_ds=batch.get("map_tex_ds", None).to(device) if batch.get("map_tex_ds", None) is not None else None,
                map_elev_ds=batch.get("map_elev_ds", None).to(device) if batch.get("map_elev_ds", None) is not None else None,
            )
                outputs = model(x)

                _, loss_dict = compute_losses(outputs, targets, dir_class_freqs=dir_class_freqs,
                                               loss_weights=loss_weights,
                                               logit_adj_tau=args.logit_adj_tau,
                                               present_pos_weight=args.present_pos_weight,
                                               present_boundary_weight=args.present_boundary_weight,
                                               dir_loss_type=args.dir_loss_type,
                                               dir_focal_gamma=args.dir_focal_gamma,
                                               dir_class_weights=dir_class_weights)
                val_losses.append(loss_dict)

                accumulate_metric_counters(outputs, targets, val_pool)

        avg_val_loss = {k: np.mean([d[k] for d in val_losses]) for k in val_losses[0]} if val_losses else {}
        avg_val_metrics = finalize_pool(val_pool)

        val_f1 = avg_val_metrics.get("present_f1", 0.0)

        # Log epoch
        epoch_info = {
            "epoch": epoch + 1,
            "patch_size": current_patch_size,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_metrics": avg_val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_s": train_time,
        }
        train_history.append(epoch_info)

        print(f"\nEpoch {epoch+1}/{args.epochs} (patch={current_patch_size}, {train_time:.0f}s):", flush=True)
        print(f"  Train: loss={avg_train_loss.get('loss_total', 0):.4f} "
              f"(present={avg_train_loss.get('loss_present', 0):.4f}, "
              f"mask={avg_train_loss.get('loss_mask', 0):.4f}, "
              f"dir={avg_train_loss.get('loss_dir', 0):.4f})", flush=True)
        if avg_val_loss:
            print(f"  Val:   loss={avg_val_loss.get('loss_total', 0):.4f} "
                  f"(present={avg_val_loss.get('loss_present', 0):.4f}, "
                  f"mask={avg_val_loss.get('loss_mask', 0):.4f}, "
                  f"dir={avg_val_loss.get('loss_dir', 0):.4f})", flush=True)
        print(f"  Metrics: F1={val_f1:.4f}, "
              f"Prec={avg_val_metrics.get('present_prec', 0):.4f}, "
              f"Rec={avg_val_metrics.get('present_rec', 0):.4f}, "
              f"MaskBitAcc={avg_val_metrics.get('mask_bit_acc', 0):.4f}, "
              f"DirAcc={avg_val_metrics.get('dir_acc', 0):.4f}", flush=True)

        # Save best model by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_metrics": avg_val_metrics,
                "num_textures": num_textures,
                "tex_embed_dim": args.tex_embed_dim,
                "hidden_channels": args.hidden_channels,
                "patch_size": args.patch_size,
                "dir_class_freqs": dir_class_freqs.cpu() if dir_class_freqs is not None else None,
                "extra_input_channels": extra_in_ch,
                "use_dist_to_boundary": bool(args.use_dist_to_boundary),
                "use_pattern_code": bool(args.use_pattern_code),
                "use_map_style": bool(args.use_map_style),
                "style_mode": str(args.style_mode),
                "use_neighbor_tex": bool(args.use_neighbor_tex),
                "map_emb_dim": int(args.map_emb_dim),
                "map_ds_size": int(args.map_ds_size),
                "logit_adj_tau": float(args.logit_adj_tau),
                "present_boundary_weight": float(args.present_boundary_weight),
                "dir_head_type": args.dir_head_type,
                "dir_loss_type": args.dir_loss_type,
                "dir_focal_gamma": float(args.dir_focal_gamma),
                "dir_cb_beta": float(args.dir_cb_beta),
            }, save_path)
            print(f"  >>> New best model saved (F1={val_f1:.4f})", flush=True)

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "num_textures": num_textures,
                "tex_embed_dim": args.tex_embed_dim,
                "hidden_channels": args.hidden_channels,
                "patch_size": args.patch_size,
                "dir_class_freqs": dir_class_freqs.cpu() if dir_class_freqs is not None else None,
                "extra_input_channels": extra_in_ch,
                "use_dist_to_boundary": bool(args.use_dist_to_boundary),
                "use_pattern_code": bool(args.use_pattern_code),
                "use_map_style": bool(args.use_map_style),
                "style_mode": str(args.style_mode),
                "use_neighbor_tex": bool(args.use_neighbor_tex),
                "map_emb_dim": int(args.map_emb_dim),
                "map_ds_size": int(args.map_ds_size),
                "logit_adj_tau": float(args.logit_adj_tau),
                "present_boundary_weight": float(args.present_boundary_weight),
                "dir_head_type": args.dir_head_type,
                "dir_loss_type": args.dir_loss_type,
                "dir_focal_gamma": float(args.dir_focal_gamma),
                "dir_cb_beta": float(args.dir_cb_beta),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}", flush=True)

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=2, default=str)
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}", flush=True)
    print(f"Training history: {history_path}", flush=True)
    print(f"Best model: {output_dir / 'best_model.pt'}", flush=True)


# ===========================================================================
# Inference: sliding window with Gaussian-weighted overlap
# ===========================================================================

def predict(args):
    """Run inference on a full map using sliding window with overlap."""
    import torch
    import torch.nn.functional as F
    from map_processor.ra3map import Ra3Map
    from map_processor.assets.terrain.blend_tile_data import BlendTileData
    from map_processor.assets.terrain.height_map_data import HeightMapData
    from map_processor.assets.terrain.blend_info import BlendInfo
    from map_processor.assets.terrain.blend_direction import BlendDirection

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load model checkpoint
    ckpt_path = Path(args.model_path)
    print(f"Loading model from {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_textures = ckpt["num_textures"]
    tex_embed_dim = ckpt["tex_embed_dim"]
    hidden_channels = ckpt["hidden_channels"]
    extra_in_ch = int(ckpt.get("extra_input_channels", 0))
    use_dist = bool(ckpt.get("use_dist_to_boundary", False))
    use_pattern = bool(ckpt.get("use_pattern_code", False))
    use_neighbor_tex = bool(ckpt.get("use_neighbor_tex", False))
    # Older checkpoints predate the dedicated direction head; default to "linear"
    # so they still load. New runs save "deep".
    dir_head_type = str(ckpt.get("dir_head_type", "linear"))

    model = _make_model(
        num_textures=num_textures,
        tex_embed_dim=tex_embed_dim,
        hidden_channels=hidden_channels,
        extra_input_channels=extra_in_ch,
        dir_head_type=dir_head_type,
        use_neighbor_tex=use_neighbor_tex,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')}, F1={ckpt.get('val_f1', '?')}, "
          f"use_dist={use_dist})", flush=True)

    dir_class_freqs = ckpt.get("dir_class_freqs", None)
    if dir_class_freqs is not None:
        dir_class_freqs = dir_class_freqs.to(device)

    # Load map
    map_path = Path(args.map_path)
    print(f"Loading map: {map_path}", flush=True)
    data = load_map_data(str(map_path))
    if data is None:
        raise SystemExit(f"Failed to load map: {map_path}")

    tex_grid = data["tex_grid"]
    elev_grid = data["elev_grid"]
    w, h = data["map_width"], data["map_height"]
    print(f"  Map size: {w}x{h}", flush=True)

    # Elevation normalization
    elev_mean, elev_std = 168.5, 113.85
    if elev_grid is not None:
        elev_norm = (elev_grid.astype(np.float32) - elev_mean) / max(elev_std, 1e-6)
    else:
        elev_norm = np.zeros((w, h), dtype=np.float32)

    # Sliding window parameters
    patch_size = ckpt.get("patch_size", 64)
    stride = max(1, patch_size // 2)  # 50% overlap

    # Ensure patch_size is power-of-2 friendly for U-Net (divisible by 8)
    assert patch_size % 8 == 0, f"patch_size must be divisible by 8, got {patch_size}"

    # Build 2D Gaussian weighting kernel for overlap blending
    sigma = patch_size / 4.0
    ax = np.arange(patch_size, dtype=np.float32) - patch_size / 2 + 0.5
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gauss = gauss / gauss.max()  # normalize peak to 1
    gauss_t = torch.from_numpy(gauss).to(device)

    # Accumulation buffers
    present_accum = np.zeros((w, h), dtype=np.float64)
    mask_accum = np.zeros((8, w, h), dtype=np.float64)
    dir_accum = np.zeros((NUM_DIR_CLASSES, w, h), dtype=np.float64)
    weight_accum = np.zeros((w, h), dtype=np.float64)

    # Pad the map so that sliding window covers all cells
    pad_w = (patch_size - w % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    tex_padded = np.pad(tex_grid, ((0, pad_w), (0, pad_h)), mode="edge")
    elev_padded = np.pad(elev_norm, ((0, pad_w), (0, pad_h)), mode="edge")

    # Optional dist-to-boundary input.
    dist_padded = None
    if use_dist:
        dist_grid = _compute_distance_to_boundary(tex_grid)
        dist_norm = 1.0 / (1.0 + dist_grid.astype(np.float32))
        dist_padded = np.pad(dist_norm, ((0, pad_w), (0, pad_h)), mode="edge")

    # Optional pattern_code input (one-hot, NUM_PATTERN_CODES channels).
    pattern_padded = None
    if use_pattern:
        pattern_grid = _compute_pattern_code(tex_grid)
        pattern_padded = np.pad(pattern_grid.astype(np.int64),
                                ((0, pad_w), (0, pad_h)), mode="edge")

    W_pad, H_pad = tex_padded.shape

    # Generate all window positions
    positions = []
    for x0 in range(0, W_pad - patch_size + 1, stride):
        for y0 in range(0, H_pad - patch_size + 1, stride):
            positions.append((x0, y0))

    print(f"  Sliding window: patch={patch_size}, stride={stride}, "
          f"positions={len(positions)}", flush=True)

    # Process in batches
    batch_size = args.batch_size
    with torch.no_grad():
        for batch_start in range(0, len(positions), batch_size):
            batch_end = min(batch_start + batch_size, len(positions))
            batch_positions = positions[batch_start:batch_end]

            tex_batch = []
            elev_batch = []
            for x0, y0 in batch_positions:
                tex_batch.append(tex_padded[x0:x0+patch_size, y0:y0+patch_size])
                elev_batch.append(elev_padded[x0:x0+patch_size, y0:y0+patch_size])

            tex_t = torch.from_numpy(np.stack(tex_batch)).long().to(device)
            elev_t = torch.from_numpy(np.stack(elev_batch)).float().to(device)
            extras_parts = []
            if dist_padded is not None:
                dist_batch = [dist_padded[x0:x0+patch_size, y0:y0+patch_size]
                              for x0, y0 in batch_positions]
                extras_parts.append(torch.from_numpy(np.stack(dist_batch)).float().unsqueeze(1).to(device))
            if pattern_padded is not None:
                pat_batch = [pattern_padded[x0:x0+patch_size, y0:y0+patch_size]
                             for x0, y0 in batch_positions]
                pat_t = torch.from_numpy(np.stack(pat_batch)).long().clamp(0, NUM_PATTERN_CODES - 1).to(device)
                pat_oh = F.one_hot(pat_t, NUM_PATTERN_CODES).permute(0, 3, 1, 2).float()
                extras_parts.append(pat_oh)
            extras_t = torch.cat(extras_parts, dim=1) if extras_parts else None

            x_input = model.encode_input(tex_t, elev_t, extras_t)
            outputs = model(x_input)

            present_probs = torch.sigmoid(outputs["present_logits"].squeeze(1))  # [B, P, P]
            mask_probs = torch.sigmoid(outputs["mask_logits"])                   # [B, 8, P, P]
            dir_probs = F.softmax(outputs["dir_logits"], dim=1)                  # [B, 17, P, P]

            # Accumulate with Gaussian weighting
            for i, (x0, y0) in enumerate(batch_positions):
                # Clip to original map bounds
                x_end = min(x0 + patch_size, w)
                y_end = min(y0 + patch_size, h)
                if x0 >= w or y0 >= h:
                    continue
                px = x_end - x0
                py = y_end - y0

                gw = gauss[:px, :py]

                present_accum[x0:x_end, y0:y_end] += present_probs[i, :px, :py].cpu().numpy() * gw
                weight_accum[x0:x_end, y0:y_end] += gw

                for ch in range(8):
                    mask_accum[ch, x0:x_end, y0:y_end] += mask_probs[i, ch, :px, :py].cpu().numpy() * gw
                for ch in range(NUM_DIR_CLASSES):
                    dir_accum[ch, x0:x_end, y0:y_end] += dir_probs[i, ch, :px, :py].cpu().numpy() * gw

            if (batch_end % max(1, len(positions) // 10)) == 0 or batch_end == len(positions):
                pct = 100.0 * batch_end / len(positions)
                print(f"  Processed {batch_end}/{len(positions)} windows ({pct:.0f}%)", flush=True)

    # Normalize by accumulated weights
    weight_accum = np.maximum(weight_accum, 1e-8)
    present_final = present_accum / weight_accum
    mask_final = mask_accum / weight_accum[None, :, :]
    dir_final = dir_accum / weight_accum[None, :, :]

    # Threshold and convert to predictions
    present_threshold = args.present_threshold
    pred_present = (present_final > present_threshold).astype(np.uint8)
    pred_mask_bits = np.zeros((w, h), dtype=np.uint8)
    for ch in range(8):
        pred_mask_bits |= ((mask_final[ch] > 0.5).astype(np.uint8) << ch)
    pred_dir_class = dir_final.argmax(axis=0).astype(np.int16)

    # Stats
    n_present = int(pred_present.sum())
    print(f"\nPrediction stats:", flush=True)
    print(f"  Blend present: {n_present}/{w*h} cells ({100*n_present/(w*h):.1f}%)", flush=True)

    # Compare with ground truth if available
    gt_present = data["blend_present"]
    gt_n = int(gt_present.sum())
    match = (pred_present == gt_present).mean()
    tp = (pred_present & gt_present).sum()
    fp = (pred_present & ~gt_present).sum()
    fn = (~pred_present & gt_present).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"  Ground truth: {gt_n} blend cells", flush=True)
    print(f"  Presence match: {match:.4f} (F1={f1:.4f}, P={prec:.4f}, R={rec:.4f})", flush=True)

    # Reconstruct blend_info and write output map
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original map for writing
    m = Ra3Map(str(map_path))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)

    # Build new blend_info list
    pred_blend_info: List[BlendInfo] = []
    blend_info_lookup: Dict[Tuple[int, int], int] = {}
    new_blends = np.zeros((w, h), dtype=np.uint16)

    for x in range(w):
        for y in range(h):
            if not pred_present[x, y]:
                continue

            # Determine secondary texture from mask prediction
            center_tex = tex_grid[x, y]
            mask_val = int(pred_mask_bits[x, y])

            # Find best neighbor based on mask probabilities
            best_ni = -1
            best_prob = -1.0
            for ni in range(8):
                dx, dy = NEIGHBOR_OFFSETS[ni]
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and tex_grid[nx, ny] != center_tex:
                    prob = mask_final[ni, x, y]
                    if prob > best_prob:
                        best_prob = prob
                        best_ni = ni

            if best_ni < 0:
                # No valid different-texture neighbor; skip
                continue

            dx, dy = NEIGHBOR_OFFSETS[best_ni]
            sec_tex = int(tex_grid[x + dx, y + dy])
            sec_tex_tile = _get_tile_from_texture(x, y, sec_tex)

            # Direction
            dir_cls = int(pred_dir_class[x, y])
            dir_raw = DIRECTION_VALUES[dir_cls] if 0 <= dir_cls < NUM_DIR_CLASSES else 0
            if dir_raw < 0:
                dir_raw = 0  # fallback for invalid

            # Find or create blend_info entry
            key = (sec_tex_tile, dir_raw)
            if key not in blend_info_lookup:
                bi = BlendInfo()
                bi.secondary_texture_tile = sec_tex_tile
                bi.blend_direction = BlendDirection(dir_raw)
                bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
                bi.i3 = 0xFFFFFFFF
                bi.i4 = 2061107200
                pred_blend_info.append(bi)
                blend_info_lookup[key] = len(pred_blend_info)

            new_blends[x, y] = blend_info_lookup[key]

    # Apply predictions
    blend.blends = new_blends.astype(np.uint16)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends, dtype=np.uint16)
    blend.blend_info = pred_blend_info
    blend.blends_count = len(pred_blend_info)

    # Fix raw bytes for all blend_info
    for bi in blend.blend_info:
        bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)

    # Save predicted map
    out_path = output_dir / f"{map_path.stem}_unet_predicted.map"
    m.save(str(out_path), compress=True)
    print(f"\nSaved predicted map: {out_path}", flush=True)
    print(f"  {n_present} blend cells, {len(pred_blend_info)} unique blend_info entries", flush=True)

    # Also save blendless for comparison
    blendless_path = output_dir / f"{map_path.stem}_blendless.map"
    m2 = Ra3Map(str(map_path))
    m2.parse()
    ctx2 = m2.get_context()
    blend2 = ctx2.get_asset_by_type(BlendTileData)
    blend2.blends = np.zeros_like(blend2.blends, dtype=np.uint16)
    blend2.single_edge_blends = np.zeros_like(blend2.single_edge_blends, dtype=np.uint16)
    for bi in blend2.blend_info:
        bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
    m2.save(str(blendless_path), compress=True)
    print(f"Saved blendless map: {blendless_path}", flush=True)

    # Copy original for comparison
    import shutil
    orig_copy = output_dir / f"{map_path.stem}_original.map"
    shutil.copy(str(map_path), str(orig_copy))
    print(f"Copied original: {orig_copy}", flush=True)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Patch-based multi-task U-Net for RA3 blend prediction (Phase 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python scripts/train_blend_unet.py \\
      --maps_dir "../RA3 Official maps" \\
      --output_dir "../blendinfo dataset/_generated/unet_v1" \\
      --patch_size 64 --epochs 30

  # Inference
  python scripts/train_blend_unet.py \\
      --predict \\
      --model_path "../blendinfo dataset/_generated/unet_v1/best_model.pt" \\
      --map_path "../RA3 Official maps/2 II/map_mp_2_rao1.map" \\
      --output_dir "../RA3 Official maps/2 II/unet_test"
""",
    )

    # Mode
    ap.add_argument("--predict", action="store_true",
                    help="Run inference mode instead of training")

    # Data
    ap.add_argument("--maps_dir", type=str, default="",
                    help="Root directory containing .map files for training (slow path)")
    ap.add_argument("--prepared_dir", type=str, default="",
                    help="Directory of pre-extracted .npz files (fast path; "
                         "produced by prepare_unet_dataset.py)")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for model/checkpoints/predictions")

    # Training params
    ap.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    ap.add_argument("--patch_size", type=int, default=64,
                    help="Patch size for training (must be divisible by 8)")
    ap.add_argument("--patches_per_map", type=int, default=200,
                    help="Number of patches sampled per map per epoch")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="Fraction of maps reserved for validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader workers (0 = main thread, recommended on Windows)")

    # Model architecture
    ap.add_argument("--tex_embed_dim", type=int, default=32,
                    help="Texture embedding dimension")
    ap.add_argument("--hidden_channels", type=int, default=64,
                    help="Base hidden channels for U-Net encoder")

    # Progressive growing
    ap.add_argument("--progressive", action="store_true",
                    help="Enable progressive patch growing: 16->32->patch_size")

    # Augmentation
    ap.add_argument("--augment_flipx", action="store_true", default=True,
                    help="Enable horizontal flip augmentation (default: True)")
    ap.add_argument("--no_augment_flipx", action="store_false", dest="augment_flipx",
                    help="Disable horizontal flip augmentation")

    # Extra input features
    ap.add_argument("--use_dist_to_boundary", action="store_true",
                    help="Add distance-to-boundary as an input channel "
                         "(8x signal lift per research, free spatial-context cue)")
    ap.add_argument("--boundary_bias", type=float, default=0.0,
                    help="Probability of sampling patches centered on a texture-boundary "
                         "cell (0=uniform random, 0.7=70%% biased). Focuses learning where "
                         "decisions matter; blends only occur near boundaries.")
    ap.add_argument("--rfs_t", type=float, default=0.0,
                    help="Repeat-Factor Sampling threshold (Gupta et al. CVPR 2019). "
                         "Per-map weight scaled by max(1, sqrt(t/f_c)) for the rarest class "
                         "in the map. Typical t=0.001-0.01. 0 disables.")
    ap.add_argument("--rare_class_bias", type=float, default=0.0,
                    help="Probability of sampling patches centered on a rare-class direction "
                         "cell (default classes 9,10,13,14 -- raw 33,34,49,50). Takes priority "
                         "over boundary_bias when set.")
    ap.add_argument("--rare_classes", type=str, default="9,10,13,14",
                    help="Comma-separated rare direction class indices for RFS / rare_class_bias.")
    ap.add_argument("--use_map_style", action="store_true",
                    help="Add per-map global style vector broadcast to all spatial locations "
                         "as extra input channels. Tests whether per-map style variance is "
                         "the present_f1 ceiling.")
    ap.add_argument("--style_mode", type=str, default="pat", choices=["pat", "tex", "both"],
                    help="Style vector composition. 'pat' = 13-dim pattern_code hist; "
                         "'tex' = top-K texture frequency hist; 'both' = concat. "
                         "Richer signatures may help but cost extra input channels.")
    ap.add_argument("--map_emb_dim", type=int, default=0,
                    help="If >0, enable a small CNN MapEncoder that consumes the full map "
                         "downsampled to map_ds_size and produces an emb_dim style vector "
                         "broadcast to all spatial cells. Direct attack on per-map "
                         "distribution shift. 0 = disabled.")
    ap.add_argument("--map_ds_size", type=int, default=32,
                    help="Downsampled map size fed to the MapEncoder (square; default 32).")

    # Loss weights
    ap.add_argument("--loss_w_present", type=float, default=1.0,
                    help="Weight for blend_present loss")
    ap.add_argument("--loss_w_mask", type=float, default=1.0,
                    help="Weight for neighbor_mask loss (ASL)")
    ap.add_argument("--loss_w_dir", type=float, default=1.0,
                    help="Weight for direction loss (logit-adjusted CE)")
    ap.add_argument("--logit_adj_tau", type=float, default=1.0,
                    help="Direction logit adjustment temperature. 0 disables it.")
    ap.add_argument("--present_pos_weight", type=float, default=1.0,
                    help="pos_weight for blend_present BCE. 1.0 = no reweighting. "
                         "Try ~9.0 for ~10%% positive rate to balance prec/rec.")
    ap.add_argument("--present_boundary_weight", type=float, default=0.0,
                    help="Boundary-weighted BCE: cells at the GT-blend boundary (any 4-neighbor "
                         "disagrees) get weight (1 + alpha). Targets the dominant failure: "
                         "82%% of FPs are 1-cell extensions of real blends, ~50%% of FNs are "
                         "boundary cells. Try 2.0-4.0. 0 disables.")
    ap.add_argument("--dir_head_type", type=str, default="deep",
                    choices=["linear", "deep"],
                    help="Direction head architecture. 'linear' = single 1x1 conv (legacy, "
                         "weak -- direction underperforms). 'deep' = 3x3 + 3x3 + 1x1 stack "
                         "with explicit neighbor mixing (default).")
    ap.add_argument("--use_pattern_code", action="store_true",
                    help="Add deterministic blend pattern_code (one-hot, 13 channels) "
                         "as input. Empirically: argmax(P(dir|pattern_code)) alone "
                         "gives 75% dir_acc -- this is the feature the token model uses "
                         "to leapfrog the U-Net's 0.43 dir_acc to 0.89.")
    ap.add_argument("--use_neighbor_tex", action="store_true",
                    help="Add explicit per-side neighbor texture embeddings (T,R,B,L) "
                         "by reusing the shared tex_embed table on torch.roll'd tex_ids. "
                         "Adds 4*tex_embed_dim input channels. Tests the hypothesis that "
                         "blend probability depends on the specific (center, neighbor) "
                         "texture pair rather than just abstract pattern_code. Bayes F1 "
                         "with explicit pairs = 0.7255 vs pat-only = 0.6800 (+4pt).")
    ap.add_argument("--dir_loss_type", type=str, default="ce",
                    choices=["ce", "focal", "cb_focal"],
                    help="Direction loss. 'ce' = plain cross-entropy. 'focal' = focal CE "
                         "(downweights easy cells). 'cb_focal' = class-balanced focal "
                         "(per-class weight = (1 - beta^n) / (1 - beta), Cui et al. 2019). "
                         "Use cb_focal to lift rare-class accuracy (classes 9,10,13,14 "
                         "are at 0% in v9).")
    ap.add_argument("--dir_focal_gamma", type=float, default=2.0,
                    help="Focal modulation gamma. Only used when --dir_loss_type=focal "
                         "or cb_focal. 0 = no focal, 2.0 = standard.")
    ap.add_argument("--dir_cb_beta", type=float, default=0.999,
                    help="Class-balance beta (effective number). 0.999 is standard; "
                         "closer to 1 gives more weight to rare classes.")
    ap.add_argument("--exclude_patterns", type=str, default="",
                    help="Comma-separated list of substrings; any prepared .npz "
                         "whose filename matches any substring (case-insensitive) "
                         "is dropped before train/val split. Use to dedup duplicate "
                         "maps (e.g. 'archon,ban_3v3,test_archon_is_v3').")
    ap.add_argument("--val_files", type=str, default="",
                    help="Path to a text file listing exact .npz basenames "
                         "(one per line) to use as the val set. Overrides random "
                         "val_frac sampling. Use to keep val deterministic across "
                         "runs that change the train pool (e.g. dedup).")

    # Logging/saving
    ap.add_argument("--log_every", type=int, default=50,
                    help="Log training loss every N steps")
    ap.add_argument("--save_every", type=int, default=5,
                    help="Save checkpoint every N epochs")

    # Device
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for training/inference (cuda/cpu)")

    # Inference-specific
    ap.add_argument("--model_path", type=str, default="",
                    help="Path to trained model checkpoint (for --predict)")
    ap.add_argument("--map_path", type=str, default="",
                    help="Path to .map file for inference (for --predict)")
    ap.add_argument("--present_threshold", type=float, default=0.5,
                    help="Threshold for blend_present in inference")

    args = ap.parse_args()

    if args.predict:
        if not args.model_path:
            ap.error("--model_path required for --predict mode")
        if not args.map_path:
            ap.error("--map_path required for --predict mode")
        predict(args)
    else:
        if not args.maps_dir and not args.prepared_dir:
            ap.error("--maps_dir or --prepared_dir required for training mode")
        train(args)


if __name__ == "__main__":
    main()
