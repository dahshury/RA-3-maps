"""
Predict blends for a map using a trained model checkpoint.

Given an original map with blends:
1) Creates a blendless version
2) Uses the trained model to predict blend info for each cell
3) Reconstructs blend_info entries from predictions
4) Saves both the ground truth (original) and predicted maps for comparison

Usage:
    python scripts/predict_blends.py \
        --checkpoint "../blendinfo dataset/_generated/hf_deit_v4_realrun/checkpoint-20000" \
        --data-dir "../blendinfo dataset/_generated/prepared_w5_elev_full" \
        --input-map "../RA3 Official maps/2 II/map_mp_2_rao1.map" \
        --out-dir "../RA3 Official maps/2 II/test"

Phase 0 inference improvements:
    --tta                  Enable flipX test-time augmentation (average original + flipped logits)
    --logit_adjustment_tau TAU  Post-hoc logit adjustment for direction (0=disabled, 1.0=full adjustment)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map  # noqa: E402
from map_processor.assets.terrain.blend_tile_data import BlendTileData  # noqa: E402
from map_processor.assets.terrain.blend_info import BlendInfo  # noqa: E402
from map_processor.assets.terrain.blend_direction import BlendDirection  # noqa: E402
from map_processor.assets.terrain.height_map_data import HeightMapData  # noqa: E402


# =============================================================================
# Texture/tile encoding (from generate_blendinfo_dataset.py)
# =============================================================================

def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """C# BlendTileData.GetTexture inverse"""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _get_tile_from_texture(x: int, y: int, texture_id: int) -> int:
    """Encode texture_id and position into secondary_texture_tile format."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return texture_id * 64 + current


def _decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    """Convert tile grid to texture id grid."""
    tiles = np.asarray(blend.tiles, dtype=np.int32)
    w, h = tiles.shape
    tex = np.zeros((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


# =============================================================================
# Feature extraction (matching generate_blendinfo_dataset.py)
# =============================================================================

# Neighbor indices: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
# CRITICAL: These are (row_offset, col_offset) to match numpy indexing!
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_NEIGHBOR_NAMES = ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]

# Texture type vocabulary
_TEX_TYPES = [
    "Grass", "Snow", "Rock", "Sand", "Pavement", "Cliff", "Dirt", "Reef",
    "Transition", "Mud", "Pave", "Gravel", "Asphalt", "SteelDeck", "RA3",
    "BB", "FortressBlackEdge", "Other"
]
_TEX_TYPE_TO_IDX = {t: i for i, t in enumerate(_TEX_TYPES)}

# Texture biome vocabulary
_TEX_BIOMES = [
    "Yucatan", "Solvang", "Iceland", "Hawaii", "Cannes", "Gibraltar", "Heidelberg",
    "CapeCod", "MtRush", "Easter", "Mykonos", "Geneva", "SantaMonica", "SaintPetersburg",
    "Amsterdam", "TokyoHarbor", "Kremlin", "Vlad", "NewYork", "Havana", "Golf", "Odessa",
    "Fortress", "Grid", "Elevation", "Ocean", "Gypsy", "Heidel", "Other"
]
_TEX_BIOME_TO_IDX = {b: i for i, b in enumerate(_TEX_BIOMES)}


def _parse_tex_type(name: str) -> int:
    """Extract texture type index from texture name like 'Grass_Yucatan02'."""
    if not name:
        return _TEX_TYPE_TO_IDX["Other"]
    if name.startswith("RA3"):
        return _TEX_TYPE_TO_IDX["RA3"]
    if name.startswith("BB_"):
        return _TEX_TYPE_TO_IDX["BB"]
    if name.startswith("FortressBlackEdge"):
        return _TEX_TYPE_TO_IDX["FortressBlackEdge"]
    if name.startswith("SteelDeck"):
        return _TEX_TYPE_TO_IDX["SteelDeck"]
    if name.startswith("Asphalt"):
        return _TEX_TYPE_TO_IDX["Asphalt"]
    parts = name.split("_")
    if parts:
        prefix = parts[0]
        if prefix in _TEX_TYPE_TO_IDX:
            return _TEX_TYPE_TO_IDX[prefix]
    return _TEX_TYPE_TO_IDX["Other"]


def _parse_tex_biome(name: str) -> int:
    """Extract texture biome/location index from texture name."""
    if not name:
        return _TEX_BIOME_TO_IDX["Other"]
    if "Grid" in name:
        return _TEX_BIOME_TO_IDX["Grid"]
    if "Elevation" in name:
        return _TEX_BIOME_TO_IDX["Elevation"]
    if "Ocean" in name or "SeaFloor" in name:
        return _TEX_BIOME_TO_IDX["Ocean"]
    if "Fortress" in name:
        return _TEX_BIOME_TO_IDX["Fortress"]
    for biome in _TEX_BIOMES:
        if biome in name:
            return _TEX_BIOME_TO_IDX[biome]
    return _TEX_BIOME_TO_IDX["Other"]


def _build_tex_type_biome_grids(
    tex_grid: np.ndarray,
    textures: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build grids of texture type and biome indices from texture grid."""
    w, h = tex_grid.shape
    n_tex = len(textures)
    type_lut = np.array([_parse_tex_type(t.name) for t in textures], dtype=np.int8)
    biome_lut = np.array([_parse_tex_biome(t.name) for t in textures], dtype=np.int8)
    tex_clipped = np.clip(tex_grid, 0, max(n_tex - 1, 0))
    type_grid = type_lut[tex_clipped] if n_tex > 0 else np.zeros_like(tex_grid, dtype=np.int8)
    biome_grid = biome_lut[tex_clipped] if n_tex > 0 else np.zeros_like(tex_grid, dtype=np.int8)
    return type_grid, biome_grid


def _compute_extra_features(
    tex_grid: np.ndarray,
    elev_grid: Optional[np.ndarray],
    type_grid: np.ndarray,
    biome_grid: np.ndarray,
    x: int,
    y: int,
    elev_mean: float = 168.5,
    elev_std: float = 113.85,
) -> np.ndarray:
    """Compute extra features for a single cell (matching training)."""
    w, h = tex_grid.shape
    
    # Pad grids
    tex_pad = np.pad(tex_grid, pad_width=((1, 1), (1, 1)), mode="edge")
    typ_pad = np.pad(type_grid, pad_width=((1, 1), (1, 1)), mode="edge")
    bio_pad = np.pad(biome_grid, pad_width=((1, 1), (1, 1)), mode="edge")
    
    px, py = x + 1, y + 1
    c = tex_pad[px, py]
    c_type = typ_pad[px, py]
    c_biome = bio_pad[px, py]
    
    # Get neighbor values
    neigh8 = [tex_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS]
    neigh_type = [typ_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS]
    neigh_biome = [bio_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS]
    
    feats = []
    
    # diff_{TL,T,TR,L,R,BL,B,BR} - 8 features
    diff_mask = [(n != c) for n in neigh8]
    feats.extend([float(d) for d in diff_mask])
    
    # tex_diff8, tex_diff4
    diff8 = sum(diff_mask)
    offs4_indices = [3, 4, 1, 6]  # L, R, T, B
    diff4 = sum(diff_mask[i] for i in offs4_indices)
    feats.extend([float(diff8), float(diff4)])
    
    # center_type_norm, center_biome_norm
    n_types = len(_TEX_TYPES)
    n_biomes = len(_TEX_BIOMES)
    feats.append(float(c_type) / n_types)
    feats.append(float(c_biome) / n_biomes)
    
    # same_type_diff_{TL..BR} - 8 features
    same_type_diff_tex = [(neigh_type[i] == c_type) and (neigh8[i] != c) for i in range(8)]
    feats.extend([float(s) for s in same_type_diff_tex])
    
    # diff_type_{TL..BR} - 8 features
    diff_type_mask = [neigh_type[i] != c_type for i in range(8)]
    feats.extend([float(d) for d in diff_type_mask])
    
    # diff_biome_{TL..BR} - 8 features
    diff_biome_mask = [neigh_biome[i] != c_biome for i in range(8)]
    feats.extend([float(d) for d in diff_biome_mask])
    
    # center_tex_idx_norm
    max_tex = max(tex_grid.max(), 1)
    feats.append(float(c) / max_tex)
    
    # can_blend_{TL..BR} - 8 features (center <= neighbor for different textures)
    can_blend = []
    for i in range(8):
        if neigh8[i] != c:
            can_blend.append(float(c <= neigh8[i]))
        else:
            can_blend.append(0.0)
    feats.extend(can_blend)
    
    # can_blend_fraction
    feats.append(sum(can_blend) / 8.0)
    
    # is_lower_idx_same_type
    is_lower_sum = 0.0
    same_type_count = sum(same_type_diff_tex)
    for i in range(8):
        if same_type_diff_tex[i] and c <= neigh8[i]:
            is_lower_sum += 1.0
    if same_type_count > 0:
        feats.append(is_lower_sum / same_type_count)
    else:
        feats.append(0.0)
    
    # Elevation features if available
    if elev_grid is not None:
        z_pad = np.pad(elev_grid, pad_width=((1, 1), (1, 1)), mode="edge")
        cz = z_pad[px, py]
        left = z_pad[px - 1, py]
        right = z_pad[px + 1, py]
        up = z_pad[px, py - 1]
        down = z_pad[px, py + 1]
        dx = (right - left) * 0.5
        dy = (down - up) * 0.5
        slope = np.sqrt(dx * dx + dy * dy)
        lap = (up + down + left + right) - (4.0 * cz)
        feats.extend([float(dx), float(dy), float(slope), float(lap)])
    
    return np.array(feats, dtype=np.float32)


# =============================================================================
# Model loading and inference
# =============================================================================

def _load_model_from_checkpoint(checkpoint_dir: Path, meta: dict, arch: str = "deit", hidden: int = 384, n_layers: int = 6):
    """Load trained model from checkpoint."""
    import torch
    import torch.nn as nn

    num_neighbor_classes = meta["neighbor"]["num_classes"]
    dir_num_classes = meta["direction"]["num_classes"]
    extra_dim = meta.get("extra_dim", 0)
    map_style_dim = meta.get("map_style_dim", 0)
    num_textures = meta["vocab"]["num_textures"]

    if arch == "token":
        # Token transformer model (matches train_blend_model_hf.py TokenBlendModel)
        seq_len = 25
        center_idx = 12
        # hidden and n_layers are passed as parameters
        n_heads = 8  # Match training script default (hidden must be divisible by n_heads)
        dropout = 0.1
        
        class TokenBlendModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_textures = int(num_textures)
                self.hidden = int(hidden)
                
                self.tex_emb = nn.Embedding(self.num_textures, self.hidden)
                self.elev_proj = nn.Linear(1, self.hidden, bias=True)
                self.pos_emb = nn.Embedding(seq_len, self.hidden)
                self.local_proj = nn.Linear(1, self.hidden, bias=True)
                
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.hidden,
                    nhead=int(n_heads),
                    dim_feedforward=int(self.hidden * 4),
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
                
                self.extra_proj = nn.Linear(int(extra_dim), self.hidden) if int(extra_dim) > 0 else None
                self.map_style_proj = nn.Linear(int(map_style_dim), self.hidden) if int(map_style_dim) > 0 else None
                
                # Heads - mask-based (8 bits for neighbor mask)
                self.blend_present = nn.Linear(self.hidden, 1)
                self.blend_mask = nn.Linear(self.hidden, 8)
                self.blend_dir = nn.Linear(self.hidden, int(dir_num_classes))
                
                # Hierarchical direction heads (for loading v9+ checkpoints)
                self.neighbor_idxs = [6, 7, 8, 11, 13, 16, 17, 18]
                self.dir_query = nn.Parameter(torch.randn(1, 1, self.hidden) * 0.02)
                self.dir_attn = nn.MultiheadAttention(self.hidden, num_heads=4, batch_first=True)
                self.blend_dir_row = nn.Linear(self.hidden, 3)
                self.blend_dir_col = nn.Linear(self.hidden, 3)
                self.blend_dir_type = nn.Linear(self.hidden, 3)
                
                self.se_present = nn.Linear(self.hidden, 1)
                self.se_mask = nn.Linear(self.hidden, 8)
                self.se_dir = nn.Linear(self.hidden, int(dir_num_classes))
                self.se_dir_row = nn.Linear(self.hidden, 3)
                self.se_dir_col = nn.Linear(self.hidden, 3)
                self.se_dir_type = nn.Linear(self.hidden, 3)
                
                # Direction class weights (buffer for loading)
                self.register_buffer('dir_class_weights', torch.ones(17, dtype=torch.float32))
            
            def _get_dir_pooled(self, x):
                B = x.shape[0]
                neighbor_emb = x[:, self.neighbor_idxs, :]
                query = self.dir_query.expand(B, -1, -1)
                attn_out, _ = self.dir_attn(query, neighbor_emb, neighbor_emb)
                center = x[:, center_idx, :]
                return center + attn_out.squeeze(1)
            
            def forward(self, tex, elev_z, tex_local_norm=None, extra_features=None, map_style=None, labels=None):
                B = tex.shape[0]
                pos = torch.arange(seq_len, device=tex.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
                
                x = self.tex_emb(torch.clamp(tex, 0, self.num_textures - 1))
                x = x + self.elev_proj(elev_z.unsqueeze(-1)) + self.pos_emb(pos)
                if tex_local_norm is not None:
                    x = x + self.local_proj(tex_local_norm.unsqueeze(-1))
                x = self.encoder(x)
                
                pooled = x[:, center_idx, :]
                if self.extra_proj is not None and extra_features is not None:
                    pooled = pooled + self.extra_proj(extra_features)
                if self.map_style_proj is not None and map_style is not None:
                    pooled = pooled + self.map_style_proj(map_style)
                
                # Use position-aware pooling for direction
                dir_pooled = self._get_dir_pooled(x)
                if self.extra_proj is not None and extra_features is not None:
                    dir_pooled = dir_pooled + self.extra_proj(extra_features)
                if self.map_style_proj is not None and map_style is not None:
                    dir_pooled = dir_pooled + self.map_style_proj(map_style)
                
                b_present = self.blend_present(pooled).squeeze(-1)
                b_mask = self.blend_mask(pooled)
                b_dir = self.blend_dir(dir_pooled)
                se_present = self.se_present(pooled).squeeze(-1)
                se_mask = self.se_mask(pooled)
                se_dir = self.se_dir(dir_pooled)
                
                return {
                    "logits_blend_present": b_present,
                    "logits_blend_mask": b_mask,
                    "logits_blend_dir": b_dir,
                    "logits_se_present": se_present,
                    "logits_se_mask": se_mask,
                    "logits_se_dir": se_dir,
                }
        
        model = TokenBlendModel()
    else:
        # DeiT model (legacy)
        from transformers import AutoConfig, AutoModel
        model_name = "facebook/deit-tiny-patch16-224"
        cfg = AutoConfig.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, config=cfg)
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "dim", 768)))
        
        class MultiHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.extra_dim = int(extra_dim)
                self.extra_proj = None
                if self.extra_dim > 0:
                    self.extra_proj = nn.Linear(self.extra_dim, hidden)
                self.blend_present = nn.Linear(hidden, 1)
                self.blend_sec = nn.Linear(hidden, int(num_neighbor_classes))
                self.blend_dir = nn.Linear(hidden, int(dir_num_classes))
                self.se_present = nn.Linear(hidden, 1)
                self.se_sec = nn.Linear(hidden, int(num_neighbor_classes))
                self.se_dir = nn.Linear(hidden, int(dir_num_classes))

            def forward(self, pixel_values, extra_features=None, labels=None):
                out = self.backbone(pixel_values=pixel_values)
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    pooled = out.last_hidden_state[:, 0]
                if self.extra_proj is not None and extra_features is not None:
                    pooled = pooled + self.extra_proj(extra_features)
                b_present = self.blend_present(pooled).squeeze(-1)
                b_sec = self.blend_sec(pooled)
                b_dir = self.blend_dir(pooled)
                se_present = self.se_present(pooled).squeeze(-1)
                se_sec = self.se_sec(pooled)
                se_dir = self.se_dir(pooled)
                return {
                    "logits_blend_present": b_present,
                    "logits_blend_sec": b_sec,
                    "logits_blend_dir": b_dir,
                    "logits_se_present": se_present,
                    "logits_se_sec": se_sec,
                    "logits_se_dir": se_dir,
                }
        
        model = MultiHead()
    
    # Load checkpoint weights
    from safetensors.torch import load_file
    weights_path = checkpoint_dir / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict, strict=True)
        print(f"  -> Loaded weights from {weights_path}", flush=True)
    else:
        raise FileNotFoundError(f"No model.safetensors found in {checkpoint_dir}")
    
    model.eval()
    return model


def _build_palette(num_textures: int, seed: int = 42) -> np.ndarray:
    """Build RGB palette for textures (matching training)."""
    rng = np.random.default_rng(seed)
    pal = rng.random((max(1, num_textures), 3)).astype(np.float32)
    pal[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return pal


def _prepare_batch_deit(
    tex_grid: np.ndarray,
    elev_grid: Optional[np.ndarray],
    type_grid: np.ndarray,
    biome_grid: np.ndarray,
    coords: List[Tuple[int, int]],
    local_to_global: Dict[int, int],
    palette: np.ndarray,
    elev_mean: float,
    elev_std: float,
    device,
):
    """Prepare a batch of samples for DeiT model input."""
    import torch
    import torch.nn.functional as F
    
    win = 5
    pad = win // 2
    w, h = tex_grid.shape
    
    # Pad grids
    tex_pad = np.pad(tex_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
    elev_pad = None
    if elev_grid is not None:
        elev_pad = np.pad(elev_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
    
    tex_windows = []
    elev_windows = []
    extra_list = []
    
    for x, y in coords:
        px, py = x + pad, y + pad
        
        # Extract 5x5 window
        tex_win = tex_pad[px - pad:px + pad + 1, py - pad:py + pad + 1].flatten()
        # Map local to global texture IDs
        tex_win_global = np.array([local_to_global.get(int(t), 0) for t in tex_win], dtype=np.int64)
        tex_windows.append(tex_win_global)
        
        if elev_pad is not None:
            elev_win = elev_pad[px - pad:px + pad + 1, py - pad:py + pad + 1].flatten()
            elev_windows.append(elev_win.astype(np.float32))
        
        # Extra features
        extra = _compute_extra_features(
            tex_grid, elev_grid, type_grid, biome_grid, x, y, elev_mean, elev_std
        )
        extra_list.append(extra)
    
    tex = torch.tensor(np.stack(tex_windows), dtype=torch.long, device=device)
    elev = None
    if elev_windows:
        elev = torch.tensor(np.stack(elev_windows), dtype=torch.float32, device=device)
    extra = torch.tensor(np.stack(extra_list), dtype=torch.float32, device=device)
    
    # Build RGB from palette
    pal_t = torch.tensor(palette, dtype=torch.float32, device=device)
    tex_clamped = torch.clamp(tex, 0, pal_t.shape[0] - 1)
    rgb = pal_t[tex_clamped]  # [B, 25, 3]
    rgb = rgb.view(-1, 5, 5, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, 5, 5]
    
    # Elevation modulation
    if elev is not None:
        elev_img = elev.view(-1, 1, 5, 5)
        z = (elev_img - elev_mean) / elev_std
        bright = torch.sigmoid(z)
        rgb = rgb * (0.65 + 0.35 * bright)
    
    # Upsample to 224x224
    rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
    
    # Normalize with ImageNet stats (DeiT uses these)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std
    
    return {"pixel_values": rgb, "extra_features": extra}


def _prepare_batch_token(
    tex_grid: np.ndarray,
    elev_grid: Optional[np.ndarray],
    coords: List[Tuple[int, int]],
    local_to_global: Dict[int, int],
    elev_mean: float,
    elev_std: float,
    n_local_tex: int,
    device,
):
    """Prepare a batch of samples for token transformer model input."""
    import torch
    
    win = 5
    pad = win // 2
    
    # Pad grids
    tex_pad = np.pad(tex_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
    elev_pad = None
    if elev_grid is not None:
        elev_pad = np.pad(elev_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
    
    tex_windows = []
    elev_windows = []
    local_norm_windows = []
    
    for x, y in coords:
        px, py = x + pad, y + pad
        
        # Extract 5x5 window
        tex_win_local = tex_pad[px - pad:px + pad + 1, py - pad:py + pad + 1].flatten()
        # Map local to global texture IDs for tex
        tex_win_global = np.array([local_to_global.get(int(t), 0) for t in tex_win_local], dtype=np.int64)
        tex_windows.append(tex_win_global)
        
        # Compute local norm for each position in window
        if n_local_tex > 1:
            local_norm = tex_win_local.astype(np.float32) / (n_local_tex - 1)
        else:
            local_norm = np.zeros_like(tex_win_local, dtype=np.float32)
        local_norm_windows.append(local_norm)
        
        if elev_pad is not None:
            elev_win = elev_pad[px - pad:px + pad + 1, py - pad:py + pad + 1].flatten()
            elev_windows.append(elev_win.astype(np.float32))
        else:
            elev_windows.append(np.zeros(25, dtype=np.float32))
    
    tex = torch.tensor(np.stack(tex_windows), dtype=torch.long, device=device)
    elev = torch.tensor(np.stack(elev_windows), dtype=torch.float32, device=device)
    local_norm = torch.tensor(np.stack(local_norm_windows), dtype=torch.float32, device=device)
    
    # Normalize elevation
    elev_z = (elev - elev_mean) / max(elev_std, 1e-6)
    
    return {"tex": tex, "elev_z": elev_z, "tex_local_norm": local_norm}


# =============================================================================
# FlipX Test-Time Augmentation (TTA) helpers
# =============================================================================

# Direction class index remapping under horizontal flip.
# dir_values: [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
# Index:        0  1  2  3  4   5   6   7   8   9  10  11  12  13  14  15  16
# Names:      N/A  L  B ExTR ExTL R  T  ExBR ExBL ?   ?   BL  BR   ?   ?  TL  TR
#
# Horizontal flip swaps Left<->Right in direction semantics:
#   Left (1) <-> Right (5)
#   Bottom (2) stays (2)
#   ExceptTopRight (3) <-> ExceptTopLeft (4)
#   Top (6) stays (6)
#   ExceptBottomRight (7) <-> ExceptBottomLeft (8)
#   BottomLeft (11) <-> BottomRight (12)
#   TopLeft (15) <-> TopRight (16)
#   Invalid/unknown (0, 9, 10, 13, 14) stay the same
_DIR_FLIPX_MAP = {
    0: 0, 1: 5, 2: 2, 3: 4, 4: 3, 5: 1, 6: 6, 7: 8, 8: 7,
    9: 9, 10: 10, 11: 12, 12: 11, 13: 13, 14: 14, 15: 16, 16: 15,
}

# Neighbor mask bit remapping under horizontal flip.
# Bits: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
# Swap: TL(0)<->TR(2), L(3)<->R(4), BL(5)<->BR(7); T(1) and B(6) stay
_MASK_FLIPX_MAP = [2, 1, 0, 4, 3, 7, 6, 5]  # new_bit[i] = old_bit[_MASK_FLIPX_MAP[i]]

# 5x5 grid column flip permutation (flattened row-major).
# For each position i in the flattened 25-element array, compute the
# position it maps to when columns are flipped: col' = 4 - col
_GRID_FLIPX_PERM = []
for _r in range(5):
    for _c in range(5):
        _GRID_FLIPX_PERM.append(_r * 5 + (4 - _c))


def _build_dir_logit_adjustment(dir_values: list, tau: float) -> np.ndarray:
    """
    Build post-hoc logit adjustment offsets for direction classes.

    Uses hardcoded class frequency priors from verified training data analysis:
    - Cardinal directions (Left=1, Right=5, Top=6, Bottom=2): ~9.75% each => ~39% total
    - Except-corner (ExTR=3, ExTL=4, ExBR=7, ExBL=8): ~12.25% each => ~49% total
    - Full-corner (TL=15, TR=16, BL=11, BR=12): ~3% each => ~12% total
    - Invalid/unknown (0, 9, 10, 13, 14): negligible (~0.01% each placeholder)

    Returns log-prior offsets of shape [num_dir_classes].
    """
    num_classes = len(dir_values)
    # Approximate class prior (normalized counts)
    prior = np.zeros(num_classes, dtype=np.float64)

    # Class indices and approximate frequencies (from training data distributions):
    # Cardinal (~39% total, 4 classes)
    for idx in [1, 2, 5, 6]:
        if idx < num_classes:
            prior[idx] = 0.0975
    # Except-corner (~49% total, 4 classes)
    for idx in [3, 4, 7, 8]:
        if idx < num_classes:
            prior[idx] = 0.1225
    # Full-corner (~12% total, 4 classes)
    for idx in [11, 12, 15, 16]:
        if idx < num_classes:
            prior[idx] = 0.03
    # Invalid/unknown classes get small epsilon
    for idx in [0, 9, 10, 13, 14]:
        if idx < num_classes:
            prior[idx] = 0.0001

    # Normalize to ensure it sums to 1
    prior = prior / prior.sum()

    # log-prior adjustment: tau * log(prior)
    log_prior = tau * np.log(prior + 1e-12)
    return log_prior.astype(np.float32)


def _flipx_tex_batch(tex: "torch.Tensor") -> "torch.Tensor":
    """
    Horizontally flip a batch of flattened 5x5 texture ID grids.

    tex: [B, 25] long tensor of global texture IDs
    Returns: [B, 25] with columns flipped (0<->4, 1<->3, 2 stays)
    """
    return tex[:, _GRID_FLIPX_PERM]


def _flipx_elev_batch(elev_z: "torch.Tensor") -> "torch.Tensor":
    """
    Horizontally flip a batch of flattened 5x5 elevation grids.

    elev_z: [B, 25] float tensor of normalized elevations
    Returns: [B, 25] with columns flipped
    """
    return elev_z[:, _GRID_FLIPX_PERM]


def _flipx_local_norm_batch(local_norm: "torch.Tensor") -> "torch.Tensor":
    """
    Horizontally flip a batch of flattened 5x5 local norm grids.

    local_norm: [B, 25] float tensor
    Returns: [B, 25] with columns flipped
    """
    return local_norm[:, _GRID_FLIPX_PERM]


def _unflip_dir_logits(dir_logits: "torch.Tensor") -> "torch.Tensor":
    """
    Remap direction logits from flipped coordinate frame back to original.

    dir_logits: [B, num_dir_classes] float tensor
    Returns: [B, num_dir_classes] with columns permuted so that
             unflipped_logits[:, orig_class] = dir_logits[:, flipped_class]
    """
    import torch
    num_classes = dir_logits.shape[1]
    # Build permutation: for each original class i, the flipped class is _DIR_FLIPX_MAP[i]
    perm = [_DIR_FLIPX_MAP.get(i, i) for i in range(num_classes)]
    perm_tensor = torch.tensor(perm, dtype=torch.long, device=dir_logits.device)
    # Gather: for original class i, take logit from flipped position _DIR_FLIPX_MAP[i]
    return torch.gather(dir_logits, 1, perm_tensor.unsqueeze(0).expand(dir_logits.shape[0], -1))


def _unflip_mask_logits(mask_logits: "torch.Tensor") -> "torch.Tensor":
    """
    Remap 8-bit neighbor mask logits from flipped coordinate frame back to original.

    mask_logits: [B, 8] float tensor
    Returns: [B, 8] with bits remapped: TL<->TR, L<->R, BL<->BR
    """
    import torch
    perm_tensor = torch.tensor(_MASK_FLIPX_MAP, dtype=torch.long, device=mask_logits.device)
    return torch.gather(mask_logits, 1, perm_tensor.unsqueeze(0).expand(mask_logits.shape[0], -1))


# =============================================================================
# Main inference
# =============================================================================

def _print(msg: str):
    """Print with immediate flush for real-time output."""
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Predict blends for a map using trained model.")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    ap.add_argument("--data-dir", required=True, help="Path to prepared dataset dir (for metadata)")
    ap.add_argument("--input-map", required=True, help="Path to input .map file (with original blends)")
    ap.add_argument("--out-dir", required=True, help="Output directory for comparison files")
    ap.add_argument("--arch", default="token", choices=["deit", "token"],
                    help="Model architecture (default: token)")
    ap.add_argument("--hidden", type=int, default=384, help="Hidden size for token model (384, 768, etc.)")
    ap.add_argument("--layers", type=int, default=6, help="Number of transformer layers (default: 6)")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--blend-present-threshold", type=float, default=0.3,
                    help="Threshold for blend_present prediction (sigmoid output, 0.3 optimal for F1)")
    ap.add_argument("--use-rule-for-present", action="store_true",
                    help="Use deterministic rule for blend_present (center_local < some_diff_neighbor_local)")
    ap.add_argument("--tta", action="store_true",
                    help="Enable flipX test-time augmentation (average original + flipped logits)")
    ap.add_argument("--logit_adjustment_tau", type=float, default=0.0,
                    help="Post-hoc logit adjustment tau for direction (0=disabled, 1.0=full). "
                         "Adds tau*log(class_prior) to direction logits before argmax.")
    args = ap.parse_args()
    
    _print("=== Blend Prediction Script ===")
    _print(f"Checkpoint: {args.checkpoint}")
    _print(f"Data dir: {args.data_dir}")
    _print(f"Input map: {args.input_map}")
    _print(f"TTA (flipX): {args.tta}")
    _print(f"Logit adjustment tau: {args.logit_adjustment_tau}")
    
    import torch
    
    checkpoint_dir = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    input_map = Path(args.input_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _print(f"Using device: {device}")
    
    # Load metadata
    _print("Loading metadata...")
    meta_path = data_dir / "prepared_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _print(f"  -> {meta['n_samples']} samples, {meta['vocab']['num_textures']} textures")
    
    # Build global texture name -> id mapping
    global_tex_name_to_id = meta["vocab"]["texture_name_to_id"]
    global_tex_id_to_name = {v: k for k, v in global_tex_name_to_id.items()}
    
    # Direction values mapping
    dir_values = meta["direction"]["values"]

    # Build direction logit adjustment (post-hoc correction for class imbalance)
    dir_log_prior = None
    if args.logit_adjustment_tau != 0.0:
        dir_log_prior = _build_dir_logit_adjustment(dir_values, tau=args.logit_adjustment_tau)
        _print(f"  -> Direction logit adjustment enabled (tau={args.logit_adjustment_tau})")
        _print(f"     log_prior range: [{dir_log_prior.min():.3f}, {dir_log_prior.max():.3f}]")

    # Load model
    _print(f"Loading model from checkpoint (arch={args.arch}, hidden={args.hidden}, layers={args.layers})...")
    model = _load_model_from_checkpoint(checkpoint_dir, meta, arch=args.arch, hidden=args.hidden, n_layers=args.layers)
    model = model.to(device)
    _print("  -> Model loaded and moved to device")
    
    # Build palette
    palette = _build_palette(meta["vocab"]["num_textures"], seed=42)
    
    # Load input map
    _print(f"Loading map: {input_map}")
    m = Ra3Map(str(input_map))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)
    height = ctx.get_asset_by_type(HeightMapData)
    
    if blend is None:
        raise ValueError("BlendTileData not found in map")
    
    w, h = blend.tiles.shape
    _print(f"  -> Map size: {w}x{h} = {w*h} cells")
    
    # Build local -> global texture ID mapping for this map
    local_tex_names = [t.name for t in (blend.textures or [])]
    local_to_global = {}
    for i, name in enumerate(local_tex_names):
        if name in global_tex_name_to_id:
            local_to_global[i] = global_tex_name_to_id[name]
        else:
            _print(f"  Warning: texture '{name}' not in global vocab, using 0")
            local_to_global[i] = 0
    
    # Decode texture grid
    tex_grid = _decode_texture_grid(blend)
    elev_grid = None
    if height is not None and height.elevations is not None:
        elev_grid = np.asarray(height.elevations, dtype=np.float32)
    
    # Build type/biome grids
    type_grid, biome_grid = _build_tex_type_biome_grids(tex_grid, blend.textures or [])
    
    elev_mean = meta["elevation_norm"]["mean"]
    elev_std = meta["elevation_norm"]["std"]
    
    # Store original blends for comparison
    orig_blends = np.asarray(blend.blends).copy()
    orig_se = np.asarray(blend.single_edge_blends).copy()
    orig_blend_info = list(blend.blend_info) if blend.blend_info else []
    
    _print(f"Original stats: {(orig_blends > 0).sum()} blend cells, {(orig_se > 0).sum()} SE cells, {len(orig_blend_info)} blend_info entries")
    
    # Prepare predicted storage
    # IMPORTANT: We must CREATE NEW blend_info entries, NOT reuse original ones!
    # Attempting to reuse original blend_info via lookup causes WorldBuilder crashes
    # because the lookup collapses to few unique indices and breaks SE predictions.
    # Each predicted blend needs a properly encoded BlendInfo with:
    # - secondary_texture_tile: position-encoded texture ID via _get_tile_from_texture()
    # - blend_direction: from model prediction
    # - _blend_direction_raw: computed via _from_blend_direction()
    pred_blends = np.zeros_like(orig_blends)
    pred_se = np.zeros_like(orig_se)
    pred_blend_info: List[BlendInfo] = []
    blend_info_lookup: Dict[Tuple[int, int, int], int] = {}  # (sec_tex_tile, dir, layer) -> index
    
    # Collect all coordinates
    all_coords = [(x, y) for x in range(w) for y in range(h)]
    n_local_tex = len(local_tex_names)
    
    # Convert logit adjustment to torch tensor if enabled
    dir_log_prior_t = None
    if dir_log_prior is not None:
        dir_log_prior_t = torch.tensor(dir_log_prior, dtype=torch.float32, device=device)

    _print(f"Running inference on {len(all_coords)} cells (arch={args.arch})...")

    def _mask_to_neighbor_idx(mask_logits):
        """Convert 8-bit mask logits to neighbor index (first active bit)."""
        mask_probs = torch.sigmoid(mask_logits)  # [B, 8]
        # Pick the highest probability neighbor
        return mask_probs.argmax(dim=-1).cpu().numpy()

    def _run_model_token(batch_dict):
        """Run token model and return raw logits dict (on device)."""
        return model(
            tex=batch_dict["tex"],
            elev_z=batch_dict["elev_z"],
            tex_local_norm=batch_dict["tex_local_norm"]
        )

    def _run_model_deit(batch_dict):
        """Run DeiT model and return raw logits dict (on device)."""
        return model(batch_dict["pixel_values"], extra_features=batch_dict["extra_features"])

    def _average_logits(out_orig, out_flip, arch):
        """Average original and un-flipped logits for TTA."""
        result = {}
        # Present logits: scalar per sample, symmetric under flip -> just average
        result["logits_blend_present"] = (out_orig["logits_blend_present"] + out_flip["logits_blend_present"]) / 2.0
        result["logits_se_present"] = (out_orig["logits_se_present"] + out_flip["logits_se_present"]) / 2.0

        # Direction logits: un-flip the flipped output, then average
        result["logits_blend_dir"] = (out_orig["logits_blend_dir"] + _unflip_dir_logits(out_flip["logits_blend_dir"])) / 2.0
        result["logits_se_dir"] = (out_orig["logits_se_dir"] + _unflip_dir_logits(out_flip["logits_se_dir"])) / 2.0

        if arch == "token":
            # Mask logits (8-bit neighbor mask): un-flip then average
            result["logits_blend_mask"] = (out_orig["logits_blend_mask"] + _unflip_mask_logits(out_flip["logits_blend_mask"])) / 2.0
            result["logits_se_mask"] = (out_orig["logits_se_mask"] + _unflip_mask_logits(out_flip["logits_se_mask"])) / 2.0
        else:
            # DeiT uses sec logits (8 neighbor classes, same remapping as mask)
            result["logits_blend_sec"] = (out_orig["logits_blend_sec"] + _unflip_mask_logits(out_flip["logits_blend_sec"])) / 2.0
            result["logits_se_sec"] = (out_orig["logits_se_sec"] + _unflip_mask_logits(out_flip["logits_se_sec"])) / 2.0

        return result

    with torch.no_grad():
        for batch_start in range(0, len(all_coords), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(all_coords))
            batch_coords = all_coords[batch_start:batch_end]

            if args.arch == "token":
                # Token model batch - original
                batch = _prepare_batch_token(
                    tex_grid, elev_grid, batch_coords,
                    local_to_global, elev_mean, elev_std, n_local_tex, device
                )
                outputs = _run_model_token(batch)

                # TTA: run flipped version and average logits
                if args.tta:
                    batch_flip = {
                        "tex": _flipx_tex_batch(batch["tex"]),
                        "elev_z": _flipx_elev_batch(batch["elev_z"]),
                        "tex_local_norm": _flipx_local_norm_batch(batch["tex_local_norm"]),
                    }
                    outputs_flip = _run_model_token(batch_flip)
                    outputs = _average_logits(outputs, outputs_flip, arch="token")

                # Apply direction logit adjustment
                b_dir_logits = outputs["logits_blend_dir"]
                se_dir_logits = outputs["logits_se_dir"]
                if dir_log_prior_t is not None:
                    b_dir_logits = b_dir_logits + dir_log_prior_t
                    se_dir_logits = se_dir_logits + dir_log_prior_t

                # Token model outputs mask logits
                b_present_probs = torch.sigmoid(outputs["logits_blend_present"]).cpu().numpy()
                b_sec_preds = _mask_to_neighbor_idx(outputs["logits_blend_mask"])
                b_dir_preds = b_dir_logits.argmax(dim=-1).cpu().numpy()

                se_present_probs = torch.sigmoid(outputs["logits_se_present"]).cpu().numpy()
                se_sec_preds = _mask_to_neighbor_idx(outputs["logits_se_mask"])
                se_dir_preds = se_dir_logits.argmax(dim=-1).cpu().numpy()
            else:
                # DeiT model batch - original
                batch = _prepare_batch_deit(
                    tex_grid, elev_grid, type_grid, biome_grid, batch_coords,
                    local_to_global, palette, elev_mean, elev_std, device
                )
                outputs = _run_model_deit(batch)

                # TTA for DeiT: would need to flip the 224x224 image and remap
                # For now, TTA is only supported for token arch since DeiT uses
                # pixel-level RGB images (5x5 upsampled to 224x224) which would
                # require image-level flipping. We skip TTA for DeiT gracefully.
                if args.tta and batch_start == 0:
                    _print("  Warning: TTA (--tta) is not supported for DeiT arch, proceeding without TTA")

                # Apply direction logit adjustment
                b_dir_logits = outputs["logits_blend_dir"]
                se_dir_logits = outputs["logits_se_dir"]
                if dir_log_prior_t is not None:
                    b_dir_logits = b_dir_logits + dir_log_prior_t
                    se_dir_logits = se_dir_logits + dir_log_prior_t

                # DeiT model outputs neighbor class
                b_present_probs = torch.sigmoid(outputs["logits_blend_present"]).cpu().numpy()
                b_sec_preds = outputs["logits_blend_sec"].argmax(dim=-1).cpu().numpy()
                b_dir_preds = b_dir_logits.argmax(dim=-1).cpu().numpy()

                se_present_probs = torch.sigmoid(outputs["logits_se_present"]).cpu().numpy()
                se_sec_preds = outputs["logits_se_sec"].argmax(dim=-1).cpu().numpy()
                se_dir_preds = se_dir_logits.argmax(dim=-1).cpu().numpy()
            
            # Convert predictions to blend_info entries
            for i, (x, y) in enumerate(batch_coords):
                center_tex = tex_grid[x, y]

                # Pre-compute valid neighbors (those with different texture than center)
                valid_neighbors = []
                for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and tex_grid[nx, ny] != center_tex:
                        valid_neighbors.append(ni)

                # Optionally use rule for blend_present
                if args.use_rule_for_present:
                    # Deterministic rule: blend if center_local < any different neighbor's local
                    should_blend = False
                    for dx, dy in _NEIGHBOR_OFFSETS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            n_tex = tex_grid[nx, ny]
                            if n_tex != center_tex and center_tex < n_tex:
                                should_blend = True
                                break
                    b_present = should_blend
                else:
                    b_present = b_present_probs[i] > args.blend_present_threshold
                
                # Blend layer
                if b_present and valid_neighbors:
                    # Constrain secondary texture prediction to valid neighbors only
                    if args.arch == "token":
                        # Mask invalid neighbors in logits before argmax
                        sec_logits = outputs["logits_blend_mask"][i].clone()
                        for ni in range(8):
                            if ni not in valid_neighbors:
                                sec_logits[ni] = float('-inf')
                        neighbor_idx = int(sec_logits.argmax())
                    else:
                        # DeiT model: mask invalid neighbors in sec logits
                        sec_logits = outputs["logits_blend_sec"][i].clone()
                        for ni in range(8):
                            if ni not in valid_neighbors:
                                sec_logits[ni] = float('-inf')
                        neighbor_idx = int(sec_logits.argmax())

                    dir_class = int(b_dir_preds[i])

                    # Get neighbor texture (guaranteed valid due to masking)
                    dx, dy = _NEIGHBOR_OFFSETS[neighbor_idx]
                    nx, ny = x + dx, y + dy
                    sec_tex = tex_grid[nx, ny]

                    # Encode secondary_texture_tile
                    sec_tex_tile = _get_tile_from_texture(x, y, int(sec_tex))

                    # Get direction value from class
                    dir_value = dir_values[dir_class] if dir_class < len(dir_values) else 0

                    # Find or create blend_info entry
                    key = (sec_tex_tile, dir_value, 0)  # simplified key
                    if key not in blend_info_lookup:
                        bi = BlendInfo()
                        bi.secondary_texture_tile = sec_tex_tile
                        bi.blend_direction = BlendDirection(dir_value)
                        bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
                        bi.i3 = 0xFFFFFFFF  # -1 as uint32
                        bi.i4 = 2061107200  # Magic value from original maps
                        pred_blend_info.append(bi)
                        blend_info_lookup[key] = len(pred_blend_info)

                    pred_blends[x, y] = blend_info_lookup[key]
                
                # Single edge layer (only model-based, rule doesn't apply well)
                if se_present_probs[i] > args.blend_present_threshold and valid_neighbors:
                    # Constrain secondary texture prediction to valid neighbors only
                    if args.arch == "token":
                        # Mask invalid neighbors in logits before argmax
                        sec_logits = outputs["logits_se_mask"][i].clone()
                        for ni in range(8):
                            if ni not in valid_neighbors:
                                sec_logits[ni] = float('-inf')
                        neighbor_idx = int(sec_logits.argmax())
                    else:
                        # DeiT model: mask invalid neighbors in sec logits
                        sec_logits = outputs["logits_se_sec"][i].clone()
                        for ni in range(8):
                            if ni not in valid_neighbors:
                                sec_logits[ni] = float('-inf')
                        neighbor_idx = int(sec_logits.argmax())

                    dir_class = int(se_dir_preds[i])

                    # Get neighbor texture (guaranteed valid due to masking)
                    dx, dy = _NEIGHBOR_OFFSETS[neighbor_idx]
                    nx, ny = x + dx, y + dy
                    sec_tex = tex_grid[nx, ny]

                    sec_tex_tile = _get_tile_from_texture(x, y, int(sec_tex))
                    dir_value = dir_values[dir_class] if dir_class < len(dir_values) else 0

                    key = (sec_tex_tile, dir_value, 1)  # different key for SE
                    if key not in blend_info_lookup:
                        bi = BlendInfo()
                        bi.secondary_texture_tile = sec_tex_tile
                        bi.blend_direction = BlendDirection(dir_value)
                        bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
                        bi.i3 = 0xFFFFFFFF  # -1 as uint32
                        bi.i4 = 2061107200  # Magic value from original maps
                        pred_blend_info.append(bi)
                        blend_info_lookup[key] = len(pred_blend_info)

                    pred_se[x, y] = blend_info_lookup[key]
            
            if (batch_end % 5000) == 0 or batch_end == len(all_coords):
                pct = 100.0 * batch_end / len(all_coords)
                _print(f"  Processed {batch_end}/{len(all_coords)} cells ({pct:.1f}%)...")
    
    _print(f"Prediction stats: {(pred_blends > 0).sum()} blend cells, {(pred_se > 0).sum()} SE cells, {len(pred_blend_info)} blend_info entries")
    
    # Compute accuracy metrics
    blend_match = ((orig_blends > 0) == (pred_blends > 0)).mean()
    se_match = ((orig_se > 0) == (pred_se > 0)).mean()
    _print(f"Blend presence match: {blend_match:.4f}")
    _print(f"SE presence match: {se_match:.4f}")
    
    # Save blendless version
    # NOTE: Keep blend_info and blends_count unchanged - only zero the blend arrays
    # This matches WorldBuilder's "Remove all texture blends" behavior
    blendless_path = out_dir / f"{input_map.stem}_blendless.map"
    m_blendless = Ra3Map(str(input_map))
    m_blendless.parse()
    ctx_bl = m_blendless.get_context()
    blend_bl = ctx_bl.get_asset_by_type(BlendTileData)
    blend_bl.blends = np.zeros_like(blend_bl.blends, dtype=np.uint16)
    blend_bl.single_edge_blends = np.zeros_like(blend_bl.single_edge_blends, dtype=np.uint16)
    # Keep blend_info and blends_count unchanged (matches observed WB output behavior)
    # FIX: Recompute raw bytes to avoid values > 1 that crash WorldBuilder
    for bi in blend_bl.blend_info:
        bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
    m_blendless.save(str(blendless_path), compress=True)
    _print(f"Saved blendless map: {blendless_path}")
    
    # Save original (copy)
    orig_path = out_dir / f"{input_map.stem}_original.map"
    import shutil
    shutil.copy(str(input_map), str(orig_path))
    _print(f"Copied original map: {orig_path}")
    
    # Save predicted version
    pred_path = out_dir / f"{input_map.stem}_predicted.map"
    m_pred = Ra3Map(str(input_map))
    m_pred.parse()
    ctx_pred = m_pred.get_context()
    blend_pred = ctx_pred.get_asset_by_type(BlendTileData)
    blend_pred.blends = pred_blends.astype(np.uint16)
    blend_pred.single_edge_blends = pred_se.astype(np.uint16)
    # CRITICAL: Use newly created blend_info list, NOT original!
    # Reusing original blend_info causes WorldBuilder crashes.
    blend_pred.blend_info = pred_blend_info
    blend_pred.blends_count = len(pred_blend_info)
    m_pred.save(str(pred_path), compress=True)
    _print(f"Saved predicted map: {pred_path}")
    
    # Save comparison stats
    stats = {
        "input_map": str(input_map),
        "settings": {
            "tta": args.tta,
            "logit_adjustment_tau": args.logit_adjustment_tau,
            "arch": args.arch,
            "blend_present_threshold": args.blend_present_threshold,
            "use_rule_for_present": args.use_rule_for_present,
        },
        "original": {
            "blend_cells": int((orig_blends > 0).sum()),
            "se_cells": int((orig_se > 0).sum()),
            "blend_info_count": len(orig_blend_info),
        },
        "predicted": {
            "blend_cells": int((pred_blends > 0).sum()),
            "se_cells": int((pred_se > 0).sum()),
            "blend_info_count": len(pred_blend_info),
        },
        "metrics": {
            "blend_presence_match": float(blend_match),
            "se_presence_match": float(se_match),
        }
    }
    stats_path = out_dir / f"{input_map.stem}_comparison_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    _print(f"Saved comparison stats: {stats_path}")
    
    _print("\nDone! Compare the maps visually in-game or using a diff tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


