"""End-to-end texture-prediction model with learnable per-object embeddings.

Inputs at forward time:
  elev    : (B, 1, W, H) float
  water   : (B, 1, W, H) float in [0, 1]
  coord   : (B, 2, W, H) — normalised x, y (CoordConv-style positional input)
  style_id: (B,) long
  objects : per-batch list of dicts with 'tile_x', 'tile_y', 'type_id', 'owner_id', 'angle_deg'

The ObjectStamper looks up learnable type/owner embeddings, projects an angle
feature, and accumulates each object's vector at its tile in a (B, D, W, H)
"object stamp grid" via scatter_add. The U-Net then consumes the concatenated
input.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------- Stamper --------------------------

class ObjectStamper(nn.Module):
    def __init__(self, n_types: int, n_owners: int, embed_dim: int = 32):
        super().__init__()
        # +1 for "unknown" / padding bucket at index 0
        self.type_emb = nn.Embedding(n_types + 1, embed_dim, padding_idx=0)
        self.owner_emb = nn.Embedding(n_owners + 1, embed_dim, padding_idx=0)
        self.angle_proj = nn.Linear(2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, batch_objects: List[List[Dict]], W: int, H: int) -> torch.Tensor:
        """batch_objects[b] is a list of dicts; returns (B, D, W, H)."""
        B = len(batch_objects)
        device = self.type_emb.weight.device
        out = torch.zeros(B, self.embed_dim, W * H, device=device)
        for b, objs in enumerate(batch_objects):
            if not objs:
                continue
            type_ids = torch.tensor([o["type_id"] for o in objs], dtype=torch.long, device=device)
            owner_ids = torch.tensor([o["owner_id"] for o in objs], dtype=torch.long, device=device)
            angles = torch.tensor([o["angle_deg"] for o in objs], dtype=torch.float32, device=device)
            xs = torch.tensor([min(max(int(round(o["tile_x"])), 0), W - 1) for o in objs],
                              dtype=torch.long, device=device)
            ys = torch.tensor([min(max(int(round(o["tile_y"])), 0), H - 1) for o in objs],
                              dtype=torch.long, device=device)
            angle_rad = angles * (math.pi / 180.0)
            angle_feat = self.angle_proj(torch.stack([torch.sin(angle_rad), torch.cos(angle_rad)], dim=1))  # (N, D)
            embeds = self.type_emb(type_ids) + self.owner_emb(owner_ids) + angle_feat                       # (N, D)
            flat_idx = xs * H + ys                                                                          # (N,)
            # scatter_add over the spatial flat dim
            out[b].scatter_add_(1, flat_idx.unsqueeze(0).expand(self.embed_dim, -1), embeds.T)
        return out.view(B, self.embed_dim, W, H)


# -------------------------- U-Net --------------------------

class _DoubleConv(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class _UNetTrunk(nn.Module):
    """U-Net body returning per-tile features (B, base, W, H)."""

    def __init__(self, in_ch: int, base: int = 48):
        super().__init__()
        self.enc1 = _DoubleConv(in_ch, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.enc4 = _DoubleConv(base * 4, base * 8)
        self.bot = _DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)
        self.base = base

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bot(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


class TinyUNet(nn.Module):
    """Single-head U-Net for tile prediction."""

    def __init__(self, in_ch: int, out_classes: int, base: int = 48):
        super().__init__()
        self.trunk = _UNetTrunk(in_ch, base=base)
        self.head = nn.Conv2d(base, out_classes, 1)

    def forward(self, x):
        return self.head(self.trunk(x))


# -------------------------- End-to-end model --------------------------

class EndToEndTextureNet(nn.Module):
    """Combines ObjectStamper + style embedding + TinyUNet."""

    def __init__(self, n_types: int, n_owners: int, n_classes: int,
                 *, n_styles: int = 8, obj_embed_dim: int = 32,
                 style_embed_dim: int = 8, base: int = 48):
        super().__init__()
        self.stamper = ObjectStamper(n_types, n_owners, embed_dim=obj_embed_dim)
        self.style_emb = nn.Embedding(n_styles, style_embed_dim)
        in_ch = 1 + 1 + 2 + obj_embed_dim + style_embed_dim   # elev + water + coord + obj + style
        self.unet = TinyUNet(in_ch, n_classes, base=base)
        self.in_ch = in_ch
        self.obj_embed_dim = obj_embed_dim
        self.style_embed_dim = style_embed_dim

    def forward(self, elev: torch.Tensor, water: torch.Tensor, coord: torch.Tensor,
                style_id: torch.Tensor, batch_objects: List[List[Dict]]) -> torch.Tensor:
        B, _, W, H = elev.shape
        obj_grid = self.stamper(batch_objects, W, H)                       # (B, D, W, H)
        style_vec = self.style_emb(style_id)                                # (B, S)
        style_grid = style_vec[:, :, None, None].expand(-1, -1, W, H)       # (B, S, W, H)
        x = torch.cat([elev, water, coord, obj_grid, style_grid], dim=1)
        return self.unet(x)


class MultiHeadTextureNet(nn.Module):
    """Shared U-Net trunk with arbitrary per-target classifier heads.

    Heads are 1x1 convs from the trunk's (B, base, W, H) output. Used to
    jointly predict tiles + blends + single_edge_blends + cliff_blends from
    a single forward pass.
    """

    def __init__(self, n_types: int, n_owners: int, head_classes: Dict[str, int],
                 *, n_styles: int = 8, obj_embed_dim: int = 32,
                 style_embed_dim: int = 8, base: int = 48):
        super().__init__()
        self.stamper = ObjectStamper(n_types, n_owners, embed_dim=obj_embed_dim)
        self.style_emb = nn.Embedding(n_styles, style_embed_dim)
        in_ch = 1 + 1 + 2 + obj_embed_dim + style_embed_dim
        self.trunk = _UNetTrunk(in_ch, base=base)
        self.heads = nn.ModuleDict({k: nn.Conv2d(base, c, 1) for k, c in head_classes.items()})
        self.head_classes = dict(head_classes)
        self.in_ch = in_ch
        self.obj_embed_dim = obj_embed_dim
        self.style_embed_dim = style_embed_dim

    def forward(self, elev: torch.Tensor, water: torch.Tensor, coord: torch.Tensor,
                style_id: torch.Tensor, batch_objects: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        B, _, W, H = elev.shape
        obj_grid = self.stamper(batch_objects, W, H)
        style_vec = self.style_emb(style_id)
        style_grid = style_vec[:, :, None, None].expand(-1, -1, W, H)
        x = torch.cat([elev, water, coord, obj_grid, style_grid], dim=1)
        feats = self.trunk(x)
        return {name: head(feats) for name, head in self.heads.items()}


class _BlendDecompHead(nn.Module):
    """Cascaded blend-prediction head.

    Consumes the trunk feature map AND the upstream tile head's logits so
    the model can condition blend predictions on "what texture is here, what
    texture is at my neighbours". A 4-conv refinement net combines the two
    before splitting into present/secondary/direction sub-heads.

    Tile logits are DETACHED before being concatenated so blend gradients do
    not corrupt the tile-head representations.
    """

    def __init__(self, trunk_ch: int, palette_size: int,
                 n_directions: int = 17, hidden: int = 96):
        super().__init__()
        in_ch = trunk_ch + palette_size  # trunk feats + tile softmax (detached)
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
        )
        self.present = nn.Conv2d(hidden, 1, 1)
        self.secondary = nn.Conv2d(hidden, palette_size, 1)
        self.direction = nn.Conv2d(hidden, n_directions, 1)

    def forward(self, trunk_feats: torch.Tensor, tile_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        tile_probs = F.softmax(tile_logits, dim=1).detach()  # detach: blend grads don't flow into tile head
        x = torch.cat([trunk_feats, tile_probs], dim=1)
        feats = self.refine(x)
        return {
            "present": self.present(feats),
            "secondary": self.secondary(feats),
            "direction": self.direction(feats),
        }


class CascadeTextureNet(nn.Module):
    """Two-stage texture predictor: tile head first, then blend heads conditioned on tile output.

    Forward returns a dict whose top level is:
      tiles                : (B, palette_size, W, H)
      blend                : dict from _BlendDecompHead
      single_edge          : dict from _BlendDecompHead (no neighbor head)
    """

    def __init__(self, n_types: int, n_owners: int, palette_size: int,
                 *, n_styles: int = 8, obj_embed_dim: int = 32,
                 style_embed_dim: int = 8, base: int = 48,
                 n_directions: int = 17, blend_hidden: int = 96):
        super().__init__()
        self.stamper = ObjectStamper(n_types, n_owners, embed_dim=obj_embed_dim)
        self.style_emb = nn.Embedding(n_styles, style_embed_dim)
        in_ch = 1 + 1 + 2 + obj_embed_dim + style_embed_dim
        self.trunk = _UNetTrunk(in_ch, base=base)
        self.tile_head = nn.Conv2d(base, palette_size, 1)
        self.blend_head = _BlendDecompHead(base, palette_size, n_directions, blend_hidden)
        self.single_edge_head = _BlendDecompHead(base, palette_size, n_directions, blend_hidden)
        self.in_ch = in_ch
        self.obj_embed_dim = obj_embed_dim
        self.style_embed_dim = style_embed_dim
        self.palette_size = palette_size
        self.n_directions = n_directions

    def forward(self, elev, water, coord, style_id, batch_objects):
        B, _, W, H = elev.shape
        obj_grid = self.stamper(batch_objects, W, H)
        style_vec = self.style_emb(style_id)
        style_grid = style_vec[:, :, None, None].expand(-1, -1, W, H)
        x = torch.cat([elev, water, coord, obj_grid, style_grid], dim=1)
        feats = self.trunk(x)
        tile_logits = self.tile_head(feats)
        return {
            "tiles": tile_logits,
            "blend": self.blend_head(feats, tile_logits),
            "single_edge": self.single_edge_head(feats, tile_logits),
        }
