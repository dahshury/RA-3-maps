"""
End-to-end dataset generator for RA3 blend synthesis.

Given a root directory containing many .map files, this script will:
1) Dedupe identical maps (by hashing core BlendTileData content) and remove duplicates.
2) Remove maps with low blend coverage (default < 0.70).
3) (Optional) Create rotation augmentations (default: disabled).
4) Generate blendless maps for all kept originals/augmentations.
5) Extract supervised samples and write a combined dataset .npz + metadata .json.

Notes
-----
- "Remove" is implemented as moving files into a quarantine folder under the root
  (default: <root>/_pruned). This satisfies "delete from dataset" while keeping
  an undo path.
- Blend ratio metric (default "any"):
    blendinfo_usage = (#unique blend_info indices referenced by blends+single_edge) / (len(blend_info))
    any_cells = (#cells where blends OR single_edge_blends is nonzero) / (#total cells)
    combined_cells = (#nonzero blends + #nonzero single_edge) / (2*#total cells)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_cdt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map  # noqa: E402
from map_processor.assets.terrain.blend_tile_data import BlendTileData  # noqa: E402
from map_processor.assets.terrain.height_map_data import HeightMapData  # noqa: E402
from map_processor.utils.map_rotation import rotate_context_right_angles  # noqa: E402


def _tqdm(iterable, total: Optional[int], desc: str):
    """
    tqdm wrapper that degrades gracefully if tqdm isn't installed.
    (tqdm is in requirements.txt, but this keeps the script robust.)
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        return iterable


def _is_blendless_name(p: Path) -> bool:
    return "blendless" in p.stem.lower()


def _iter_map_files(root: Path) -> List[Path]:
    if root.is_file() and root.suffix.lower() == ".map":
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.map"))
    return []


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    # C# BlendTileData.GetTexture inverse
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    tiles = np.asarray(blend.tiles)
    w, h = tiles.shape
    tex = np.empty((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


def _labels_for_layer(blend: BlendTileData, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy: returns (present, secondary_texture_id, direction)."""
    w, h = grid.shape
    present = (grid > 0).astype(np.uint8)
    sec = np.full((w, h), -1, dtype=np.int16)
    direction = np.zeros((w, h), dtype=np.int16)

    info = blend.blend_info or []
    for x in range(w):
        for y in range(h):
            idx = int(grid[x, y])
            if idx <= 0:
                continue
            if idx > len(info):
                continue
            bi = info[idx - 1]
            sec[x, y] = int(_get_texture_from_tile(x, y, int(bi.secondary_texture_tile)))
            direction[x, y] = int(bi.blend_direction)
    return present, sec, direction


# Neighbor indices: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
# CRITICAL: These are (row_offset, col_offset) to match numpy indexing!
# The 5x5 window is indexed row-major where center is at (2,2).
# TL = row-1, col-1; T = row-1, col; TR = row-1, col+1; etc.
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_NEIGHBOR_NAMES = ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]

# Texture type vocabulary (extracted from texture name prefix)
_TEX_TYPES = [
    "Grass", "Snow", "Rock", "Sand", "Pavement", "Cliff", "Dirt", "Reef",
    "Transition", "Mud", "Pave", "Gravel", "Asphalt", "SteelDeck", "RA3",
    "BB", "FortressBlackEdge", "Other"
]
_TEX_TYPE_TO_IDX = {t: i for i, t in enumerate(_TEX_TYPES)}

# Texture biome vocabulary (extracted from texture name suffix)
# Comprehensive list from scanning MapCreatorCore/Resources.cs and actual map data
_TEX_BIOMES = [
    "Yucatan", "Solvang", "Iceland", "Hawaii", "Cannes", "Gibraltar", "Heidelberg",
    "CapeCod", "MtRush", "Easter", "Mykonos", "Geneva", "SantaMonica", "SaintPetersburg",
    "Amsterdam", "TokyoHarbor", "Kremlin", "Vlad", "NewYork", "Havana", "Golf", "Odessa",
    "Fortress", "Grid", "Elevation", "Ocean", "Gypsy", "Heidel", "Other"  # Added Gypsy, Heidel
]
_TEX_BIOME_TO_IDX = {b: i for i, b in enumerate(_TEX_BIOMES)}


def _parse_tex_type(name: str) -> int:
    """Extract texture type index from texture name like 'Grass_Yucatan02'."""
    if not name:
        return _TEX_TYPE_TO_IDX["Other"]
    # Handle special cases
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
    # Standard format: Type_Location##
    parts = name.split("_")
    if parts:
        prefix = parts[0]
        if prefix in _TEX_TYPE_TO_IDX:
            return _TEX_TYPE_TO_IDX[prefix]
    return _TEX_TYPE_TO_IDX["Other"]


def _parse_tex_biome(name: str) -> int:
    """Extract texture biome/location index from texture name like 'Grass_Yucatan02'."""
    if not name:
        return _TEX_BIOME_TO_IDX["Other"]
    # Handle special cases
    if "Grid" in name:
        return _TEX_BIOME_TO_IDX["Grid"]
    if "Elevation" in name:
        return _TEX_BIOME_TO_IDX["Elevation"]
    if "Ocean" in name or "SeaFloor" in name:
        return _TEX_BIOME_TO_IDX["Ocean"]
    if "Fortress" in name:
        return _TEX_BIOME_TO_IDX["Fortress"]
    # Check each biome
    for biome in _TEX_BIOMES:
        if biome in name:
            return _TEX_BIOME_TO_IDX[biome]
    return _TEX_BIOME_TO_IDX["Other"]


def _build_tex_type_biome_grids(
    tex_grid: np.ndarray,
    textures: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build grids of texture type and biome indices from texture grid.
    Returns (type_grid, biome_grid) both shape [W, H] int8.
    """
    w, h = tex_grid.shape
    type_grid = np.zeros((w, h), dtype=np.int8)
    biome_grid = np.zeros((w, h), dtype=np.int8)
    
    # Build lookup from local texture index to type/biome
    n_tex = len(textures)
    type_lut = np.array([_parse_tex_type(t.name) for t in textures], dtype=np.int8)
    biome_lut = np.array([_parse_tex_biome(t.name) for t in textures], dtype=np.int8)
    
    # Vectorized lookup
    tex_clipped = np.clip(tex_grid, 0, n_tex - 1)
    type_grid = type_lut[tex_clipped]
    biome_grid = biome_lut[tex_clipped]
    
    return type_grid, biome_grid


def _compute_distance_to_boundary(tex_grid: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev distance to the nearest texture boundary for every cell.

    A cell is on a boundary (distance=0) if any of its 4-connected neighbors has
    a different base texture.  For interior cells the distance increases by 1 per
    Chebyshev ring.

    Uses ``scipy.ndimage.distance_transform_cdt`` with ``metric='chessboard'``
    (Chebyshev) on a binary mask where boundary cells are 0 and interior cells are 1.

    Parameters
    ----------
    tex_grid : ndarray, shape (W, H), int
        Decoded base-texture index grid.

    Returns
    -------
    dist : ndarray, shape (W, H), float32
        Chebyshev distance to the nearest texture boundary.  Boundary cells have
        distance 0.
    """
    tex = np.asarray(tex_grid, dtype=np.int32)
    # Pad with edge values so that map-edge cells never look like boundaries
    # just because they are at the array boundary.
    tex_pad = np.pad(tex, pad_width=1, mode="edge")

    # A cell is a boundary cell if ANY 4-connected neighbor differs.
    is_boundary = np.zeros(tex_pad.shape, dtype=np.bool_)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(tex_pad, dx, axis=0), dy, axis=1)
        is_boundary |= (shifted != tex_pad)

    # Remove the padding artefacts on the outer ring of is_boundary.
    # (roll introduces wrap-around; just mask the outer 1-pixel ring.)
    is_boundary[0, :] = False
    is_boundary[-1, :] = False
    is_boundary[:, 0] = False
    is_boundary[:, -1] = False

    # Crop back to original size.
    is_boundary = is_boundary[1:-1, 1:-1]

    # distance_transform_cdt expects 0=background (boundary), positive=interior.
    # Boundary cells should have distance 0; interior cells > 0.
    interior_mask = (~is_boundary).astype(np.uint8)
    dist = distance_transform_cdt(interior_mask, metric="chessboard").astype(np.float32)
    return dist


def _compute_soft_blend_present(
    blend_present: np.ndarray,
    dist_to_boundary: np.ndarray,
    flat_idx: np.ndarray,
    w: int,
    h: int,
    label_smooth_pos: float = 0.95,
    rate_boundary: float = 0.45,
    rate_near: float = 0.10,
    rate_interior: float = 0.02,
) -> np.ndarray:
    """
    Compute GeoLS soft targets for blend_present based on distance to texture boundary.

    Parameters
    ----------
    blend_present : ndarray, shape (W*H,) or (W, H), uint8
        Hard binary labels (0 or 1).
    dist_to_boundary : ndarray, shape (W, H), float32
        Chebyshev distance to nearest texture boundary (from ``_compute_distance_to_boundary``).
    flat_idx : ndarray
        Flat indices into (W*H) for the sampled cells.
    w, h : int
        Map dimensions.
    label_smooth_pos : float
        Soft target for positive samples (default 0.95).
    rate_boundary : float
        Soft target for negative samples ON the boundary (dist=0).  Empirically ~55%
        of boundary cells have blends, so 45% of negatives are boundary cells
        without blends.
    rate_near : float
        Soft target for negative samples at distance 1 (anticipatory blend zone).
    rate_interior : float
        Soft target for negative samples at distance >= 2 (deep interior).

    Returns
    -------
    soft : ndarray, shape (len(flat_idx),), float32
    """
    bp = np.asarray(blend_present).reshape(-1)[flat_idx].astype(np.float32)
    dist_flat = dist_to_boundary.reshape(-1)[flat_idx]

    soft = np.where(bp > 0.5, label_smooth_pos, 0.0).astype(np.float32)

    neg_mask = bp < 0.5
    d0 = neg_mask & (dist_flat == 0)
    d1 = neg_mask & (dist_flat == 1)
    d2 = neg_mask & (dist_flat >= 2)

    soft[d0] = rate_boundary
    soft[d1] = rate_near
    soft[d2] = rate_interior

    return soft


def _labels_for_layer_v3(
    blend: BlendTileData,
    grid: np.ndarray,
    tex_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (present, sec_mask8, direction, neighbor_idx_first).

    - sec_mask8: uint8 bitmask of which of the 8 neighbors match the secondary texture.
      Bit i corresponds to _NEIGHBOR_OFFSETS[i]. This is a MULTI-LABEL target and
      is the key missing signal for direction and boundary-side selection.
        - 0 means: no neighbor matches the secondary (rare / noisy).
        - 255 means: IGNORE (we set this when present==1 but mask would be 0, i.e. secondary not found).

    - neighbor_idx_first: int8 single neighbor index (0-7) of the first matching neighbor, or -1.
      Kept for backward compatibility / debugging, but training should prefer sec_mask8.
    """
    w, h = grid.shape
    present = (grid > 0).astype(np.uint8)
    direction = np.full((w, h), -1, dtype=np.int16)
    sec_tex_grid = np.full((w, h), -1, dtype=np.int16)

    info = blend.blend_info or []
    for x in range(w):
        for y in range(h):
            idx = int(grid[x, y])
            if idx <= 0:
                continue
            if idx > len(info):
                continue
            bi = info[idx - 1]
            sec_tex = int(_get_texture_from_tile(x, y, int(bi.secondary_texture_tile)))
            sec_tex_grid[x, y] = np.int16(sec_tex)
            direction[x, y] = np.int16(int(bi.blend_direction))

    # Build neighbor match mask8 (vectorized over the whole grid)
    tex = np.asarray(tex_grid, dtype=np.int32, order="C")
    tex_pad = np.pad(tex, pad_width=((1, 1), (1, 1)), mode="edge")
    sec_pad = np.pad(sec_tex_grid.astype(np.int32, copy=False), pad_width=((1, 1), (1, 1)), mode="edge")

    mask8 = np.zeros((w, h), dtype=np.uint8)
    neighbor_idx_first = np.full((w, h), -1, dtype=np.int8)
    for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
        neigh = tex_pad[1 + dx : 1 + dx + w, 1 + dy : 1 + dy + h]
        sec = sec_pad[1:1 + w, 1:1 + h]
        hit = (present != 0) & (sec >= 0) & (neigh == sec)
        bit = (hit.astype(np.uint8) << np.uint8(ni))
        mask8 = np.bitwise_or(mask8, bit)
        # store first matching neighbor index (deterministic)
        set_first = hit & (neighbor_idx_first < 0)
        neighbor_idx_first[set_first] = np.int8(ni)

    # If present==1 but no neighbor matches secondary, mark as IGNORE for mask training
    mask8 = np.where((present != 0) & (mask8 == 0), np.uint8(255), mask8).astype(np.uint8, copy=False)

    return present, mask8, direction, neighbor_idx_first


def _sample_indices(w: int, h: int, stride: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    xs = np.arange(0, w, stride, dtype=np.int32)
    ys = np.arange(0, h, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    flat = (grid_x * h + grid_y).reshape(-1)
    if max_samples > 0 and flat.size > max_samples:
        sel = rng.choice(flat.size, size=max_samples, replace=False)
        flat = flat[sel]
    return flat


def _blend_ratio(blend: BlendTileData, metric: str) -> float:
    blends = np.asarray(blend.blends)
    se = np.asarray(blend.single_edge_blends)
    area = blends.size
    if area == 0:
        return 0.0
    if metric == "combined_cells":
        return float((blends > 0).sum() + (se > 0).sum()) / float(2 * area)
    if metric == "any_cells":
        return float(((blends > 0) | (se > 0)).sum()) / float(area)
    # "blendinfo_usage"
    info_len = len(blend.blend_info or [])
    if info_len <= 0:
        return 0.0
    used = set(np.unique(blends[blends > 0]).tolist()) | set(np.unique(se[se > 0]).tolist())
    return float(len(used)) / float(info_len)


def _hash_core(blend: BlendTileData, height: Optional[HeightMapData], include_elevation: bool) -> str:
    """
    Stable content hash for dedupe: textures list + tiles + blends + single_edge,
    and optionally elevation.
    """
    h = hashlib.sha1()
    tex_names = [t.name for t in (blend.textures or [])]
    h.update(("\n".join(tex_names)).encode("utf-8", errors="ignore"))
    h.update(np.asarray(blend.tiles).tobytes())
    h.update(np.asarray(blend.blends).tobytes())
    h.update(np.asarray(blend.single_edge_blends).tobytes())
    if include_elevation and height is not None and height.elevations is not None:
        h.update(np.asarray(height.elevations, dtype=np.float32).tobytes())
    return h.hexdigest()


def _move_to_pruned(src: Path, pruned_root: Path, reason: str) -> None:
    pruned_root.mkdir(parents=True, exist_ok=True)
    dst_dir = pruned_root / reason
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # ensure uniqueness
    if dst.exists():
        dst = dst_dir / f"{src.stem}__dup{src.suffix}"
    shutil.move(str(src), str(dst))


def _write_blendless_map(orig_map_path: Path, out_path: Path) -> None:
    m = Ra3Map(str(orig_map_path))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)
    if blend is None:
        raise ValueError(f"BlendTileData not found in {orig_map_path}")
    blend.blends = np.zeros_like(blend.blends, dtype=np.uint16)
    blend.single_edge_blends = np.zeros_like(blend.single_edge_blends, dtype=np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=True)


def _write_rotated_map(orig_map_path: Path, out_path: Path, degrees: int) -> None:
    m = Ra3Map(str(orig_map_path))
    m.parse()
    ctx = m.get_context()
    rotate_context_right_angles(ctx, degrees=degrees, clockwise=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=True)


_ROT_90_DIR = {
    1: 18,
    18: 17,
    17: 2,
    2: 1,
    4: 8,
    8: 20,
    20: 24,
    24: 4,
    36: 52,
    52: 56,
    56: 40,
    40: 36,
    33: 50,
    50: 49,
    49: 34,
    34: 33,
}


def _rotate_dir_values_90(vals: np.ndarray, turns_cw: int) -> np.ndarray:
    """
    Rotate BlendDirection bitfield values for sample augmentation.
    For turns_cw==0 returns input; for turns_cw>0 applies 90° mapping turns_cw times.
    Unknown values remain unchanged (identity).
    """
    turns_cw = int(turns_cw) % 4
    if turns_cw == 0:
        return vals

    # Build a LUT only once per call; small cost vs per-element dict lookups.
    lut = np.arange(65536, dtype=np.int32)
    for k, v in _ROT_90_DIR.items():
        lut[k] = v

    out = vals.astype(np.int32, copy=True)
    for _ in range(turns_cw):
        mask = (out >= 0) & (out < 65536)
        out[mask] = lut[out[mask]]
    return out.astype(vals.dtype, copy=False)


def _rotate_neighbor_idx_90(vals: np.ndarray, turns_cw: int) -> np.ndarray:
    """
    Rotate neighbor indices (0-7) clockwise.
    Neighbor layout: 0=TL, 1=T, 2=TR, 3=L, 4=R, 5=BL, 6=B, 7=BR
    
    After 90° CW rotation:
      TL->TR, T->R, TR->BR, L->T, R->B, BL->TL, B->L, BR->BL
    i.e. 0->2, 1->4, 2->7, 3->1, 4->6, 5->0, 6->3, 7->5
    """
    turns_cw = int(turns_cw) % 4
    if turns_cw == 0:
        return vals
    
    # LUT for 90° CW rotation
    rot90_lut = np.array([-1, 2, 4, 7, 1, 6, 0, 3, 5], dtype=np.int8)  # -1 stays -1, then 0->2, 1->4, ...
    
    out = vals.copy()
    for _ in range(turns_cw):
        mask = (out >= 0) & (out < 8)
        out[mask] = rot90_lut[out[mask] + 1]  # +1 to handle -1 case
    return out


def _augment_samples_rotations(
    X: np.ndarray,
    win: int,
    y_dir: np.ndarray,
    y_neighbor: np.ndarray,
    rotations: List[int],
    include_elevation: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create augmented samples by rotating the window features and rotating labels.
    rotations: list of degrees CW (e.g., [90], [90,180,270]).
    Returns (X_aug, y_dir_aug, y_neighbor_aug) including the original samples first.
    """
    if not rotations:
        return X, y_dir, y_neighbor

    feat_per_field = win * win
    tex = X[:, :feat_per_field].reshape(-1, win, win)
    elev = None
    if include_elevation:
        elev = X[:, feat_per_field : 2 * feat_per_field].reshape(-1, win, win)

    X_list = [X]
    ydir_list = [y_dir]
    yneigh_list = [y_neighbor]
    for deg in rotations:
        turns = (deg // 90) % 4
        if turns == 0:
            continue
        # clockwise: np.rot90 with k=-turns
        tex_r = np.rot90(tex, k=-turns, axes=(1, 2)).reshape(-1, feat_per_field)
        parts = [tex_r]
        if include_elevation and elev is not None:
            elev_r = np.rot90(elev, k=-turns, axes=(1, 2)).reshape(-1, feat_per_field)
            parts.append(elev_r)
        Xr = np.concatenate(parts, axis=1)
        yr_dir = _rotate_dir_values_90(y_dir, turns_cw=turns)
        yr_neigh = _rotate_neighbor_idx_90(y_neighbor, turns_cw=turns)
        X_list.append(Xr)
        ydir_list.append(yr_dir)
        yneigh_list.append(yr_neigh)

    return np.concatenate(X_list, axis=0), np.concatenate(ydir_list, axis=0), np.concatenate(yneigh_list, axis=0)


def _extra_signal_features(
    tex_grid: np.ndarray,
    elev_grid: Optional[np.ndarray],
    type_grid: Optional[np.ndarray],
    biome_grid: Optional[np.ndarray],
    impassable_grid: Optional[np.ndarray],
    dist_to_boundary: Optional[np.ndarray],
    flat_idx: np.ndarray,
    w: int,
    h: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute additional per-cell signals that help blend prediction:
      - 8 binary features: is each neighbor different from center texture? (TL,T,TR,L,R,BL,B,BR)
      - texture boundary counts (diff8, diff4 against center texture)
      - Texture type and biome features for center and 8 neighbors (for priority rules)
      - 8 binary features: is each neighbor same-type but different texture?
      - 8 binary features: is each neighbor different-biome?
      - Center texture palette index (lower index = higher priority within same type)
      - Relative priority: is center index < min neighbor index with same type?
      - Distance to texture boundary (center + 5x5 neighborhood grid)
      - elevation slope/curvature (dx, dy, slope_mag, laplacian) if elevation is provided

    Returns (extra_feats [N,K] float32, feature_names).
    """
    flat_idx = flat_idx.astype(np.int64, copy=False)
    x = (flat_idx // int(h)).astype(np.int32, copy=False)
    y = (flat_idx % int(h)).astype(np.int32, copy=False)

    # boundary features from textures (use edge padding)
    tex = np.asarray(tex_grid, dtype=np.int32, order="C")
    tex_pad = np.pad(tex, pad_width=((1, 1), (1, 1)), mode="edge")
    px = x + 1
    py = y + 1
    c = tex_pad[px, py]  # center texture id

    # 8-neighborhood binary mask: is each neighbor different from center?
    # Order: TL=0, T=1, TR=2, L=3, R=4, BL=5, B=6, BR=7 (matches _NEIGHBOR_OFFSETS)
    neigh8 = np.stack([tex_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS], axis=1)  # [N,8]
    diff_mask = (neigh8 != c[:, None]).astype(np.float32)  # [N,8] binary
    
    # Also keep aggregate counts for additional signal
    diff8 = diff_mask.sum(axis=1).astype(np.float32)  # total different neighbors
    
    offs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # L, R, T, B (edge neighbors only)
    neigh4 = np.stack([tex_pad[px + dx, py + dy] for dx, dy in offs4], axis=1)  # [N,4]
    diff4 = np.sum(neigh4 != c[:, None], axis=1).astype(np.float32)

    # Build feature list: 8 binary neighbor-diff features + 2 aggregate counts
    feats: List[np.ndarray] = [diff_mask[:, i] for i in range(8)]
    names = [f"diff_{n}" for n in _NEIGHBOR_NAMES]  # diff_TL, diff_T, diff_TR, diff_L, diff_R, diff_BL, diff_B, diff_BR
    
    feats.extend([diff8, diff4])
    names.extend(["tex_diff8", "tex_diff4"])

    # ==================== BOUNDARY SHAPE FEATURES ====================
    # Critical for direction prediction: encode the SHAPE of the texture boundary.
    #
    # FINDING: Direction error scales with boundary complexity:
    #   - 1 different neighbor: ~1% error (trivial)
    #   - 4 different neighbors: ~30% error
    #   - 7 different neighbors: ~50% error
    #
    # The model needs to understand boundary geometry, not just counts.

    # Convert 8-bit diff pattern to integer (0-255) and normalized
    # Each bit represents a neighbor: TL=bit0, T=bit1, ..., BR=bit7
    diff_pattern_bits = diff_mask.astype(np.uint8)  # [N, 8]
    diff_pattern_int = np.zeros(len(flat_idx), dtype=np.int32)
    for bit_idx in range(8):
        diff_pattern_int |= (diff_pattern_bits[:, bit_idx].astype(np.int32) << bit_idx)
    feats.append(diff_pattern_int.astype(np.float32) / 255.0)
    names.append("boundary_pattern_norm")

    # Count "runs" of consecutive different neighbors (boundary shape complexity)
    # Walking around the ring: TL->T->TR->R->BR->B->BL->L->back to TL
    # A single edge has 1-3 runs, a corner has 1 run, scattered has many runs
    ring_order = [0, 1, 2, 4, 7, 6, 5, 3]  # TL, T, TR, R, BR, B, BL, L (clockwise)
    diff_ring = diff_pattern_bits[:, ring_order]  # [N, 8] in ring order

    # Count transitions from 0->1 (entering a "different" region)
    transitions = np.zeros(len(flat_idx), dtype=np.int32)
    for i in range(8):
        prev_i = (i - 1) % 8
        # Transition: prev=0 (same), curr=1 (different)
        is_transition = (diff_ring[:, prev_i] == 0) & (diff_ring[:, i] == 1)
        transitions += is_transition.astype(np.int32)
    # Edge case: if first and last are both 1 but we counted a false transition, adjust
    # Actually, this counts "starts" of runs, which is correct
    num_runs = transitions.astype(np.float32)
    # Handle all-same (0 runs) and all-different (1 continuous run)
    num_runs = np.where(diff8 == 0, 0.0, num_runs)
    num_runs = np.where(diff8 == 8, 1.0, num_runs)  # All different = 1 continuous boundary
    feats.append(num_runs / 4.0)  # Normalize (max 4 runs for alternating pattern)
    names.append("boundary_num_runs")

    # Specific pattern indicators (computed from 8-bit pattern)
    # These are the most common boundary shapes:

    # Single neighbor different (isolated contact point)
    is_single_contact = (diff8 == 1).astype(np.float32)
    feats.append(is_single_contact)
    names.append("is_single_contact")

    # Corner pattern: exactly 3 consecutive different neighbors (L-shape)
    # Corners: {TL,T,L}, {T,TR,R}, {R,BR,B}, {B,BL,L}
    corner_patterns = [
        [0, 1, 3],  # TL, T, L
        [1, 2, 4],  # T, TR, R
        [4, 7, 6],  # R, BR, B
        [6, 5, 3],  # B, BL, L
    ]
    is_corner = np.zeros(len(flat_idx), dtype=np.float32)
    for corner in corner_patterns:
        mask = np.ones(len(flat_idx), dtype=bool)
        for idx in corner:
            mask &= (diff_pattern_bits[:, idx] == 1)
        # And all others must be 0
        for idx in range(8):
            if idx not in corner:
                mask &= (diff_pattern_bits[:, idx] == 0)
        is_corner = np.maximum(is_corner, mask.astype(np.float32))
    feats.append(is_corner)
    names.append("is_corner_boundary")

    # Straight edge: 2-4 consecutive same-side neighbors different
    # Horizontal edges: top row (TL,T,TR) or bottom row (BL,B,BR)
    # Vertical edges: left col (TL,L,BL) or right col (TR,R,BR)
    edge_patterns = [
        [0, 1, 2],     # Top edge: TL, T, TR
        [5, 6, 7],     # Bottom edge: BL, B, BR
        [0, 3, 5],     # Left edge: TL, L, BL
        [2, 4, 7],     # Right edge: TR, R, BR
    ]
    is_straight_edge = np.zeros(len(flat_idx), dtype=np.float32)
    for edge in edge_patterns:
        mask = np.ones(len(flat_idx), dtype=bool)
        for idx in edge:
            mask &= (diff_pattern_bits[:, idx] == 1)
        # And opposite side must be 0
        opposite = [i for i in range(8) if i not in edge]
        for idx in opposite:
            mask &= (diff_pattern_bits[:, idx] == 0)
        is_straight_edge = np.maximum(is_straight_edge, mask.astype(np.float32))
    feats.append(is_straight_edge)
    names.append("is_straight_edge")

    # Island: center is completely surrounded (all 8 different)
    is_island = (diff8 == 8).astype(np.float32)
    feats.append(is_island)
    names.append("is_island_center")

    # Peninsula: only 1-2 same neighbors (mostly surrounded)
    is_peninsula = ((diff8 >= 6) & (diff8 <= 7)).astype(np.float32)
    feats.append(is_peninsula)
    names.append("is_peninsula")

    # ==================== NEW: Texture Type & Biome Features ====================
    # These help the model learn the priority hierarchy for blending:
    # - Which texture TYPE should "receive" the blend
    # - Same-type textures: lower palette index gets the blend
    # - Different-biome textures may not blend at all
    #
    # NOTE: We always emit these features (with zero fallback if grids are missing)
    # to guarantee a consistent feature count across all maps.

    # Fallback: if type/biome grids are missing, create zero grids so that
    # the feature count is always the same.  In practice type_grid and biome_grid
    # are always computed by the caller, but this guards against edge cases.
    if type_grid is None:
        type_grid = np.zeros_like(tex_grid, dtype=np.int8)
    if biome_grid is None:
        biome_grid = np.zeros_like(tex_grid, dtype=np.int8)

    typ = np.asarray(type_grid, dtype=np.int32, order="C")
    bio = np.asarray(biome_grid, dtype=np.int32, order="C")
    typ_pad = np.pad(typ, pad_width=((1, 1), (1, 1)), mode="edge")
    bio_pad = np.pad(bio, pad_width=((1, 1), (1, 1)), mode="edge")

    c_type = typ_pad[px, py]  # center type
    c_biome = bio_pad[px, py]  # center biome

    # Neighbor types and biomes
    neigh_type = np.stack([typ_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS], axis=1)  # [N,8]
    neigh_biome = np.stack([bio_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS], axis=1)  # [N,8]

    # Center type (one-hot would be too many dims, use normalized index instead)
    n_types = len(_TEX_TYPES)
    n_biomes = len(_TEX_BIOMES)
    center_type_norm = c_type.astype(np.float32) / n_types
    center_biome_norm = c_biome.astype(np.float32) / n_biomes
    feats.extend([center_type_norm, center_biome_norm])
    names.extend(["center_type_norm", "center_biome_norm"])

    # Is each neighbor SAME type but DIFFERENT texture? (same-type blending case)
    same_type_diff_tex = ((neigh_type == c_type[:, None]) & (neigh8 != c[:, None])).astype(np.float32)  # [N,8]
    for i in range(8):
        feats.append(same_type_diff_tex[:, i])
        names.append(f"same_type_diff_{_NEIGHBOR_NAMES[i]}")

    # Is each neighbor DIFFERENT type? (cross-type blending case)
    diff_type_mask = (neigh_type != c_type[:, None]).astype(np.float32)  # [N,8]
    for i in range(8):
        feats.append(diff_type_mask[:, i])
        names.append(f"diff_type_{_NEIGHBOR_NAMES[i]}")

    # Is each neighbor DIFFERENT biome? (may indicate incompatible pair)
    diff_biome_mask = (neigh_biome != c_biome[:, None]).astype(np.float32)  # [N,8]
    for i in range(8):
        feats.append(diff_biome_mask[:, i])
        names.append(f"diff_biome_{_NEIGHBOR_NAMES[i]}")

    # Center texture palette index (normalized) - for priority rule
    # DISCOVERED: MapGenerator.BlendTextures() uses: if (centerTexture <= tex)
    # Only add blend if center has lower or equal palette index!
    center_tex_norm = c.astype(np.float32) / max(tex.max(), 1)
    feats.append(center_tex_norm)
    names.append("center_tex_idx_norm")

    # For each neighbor: is center index <= neighbor index? (the actual blend rule!)
    # This directly encodes the priority condition from the game's blend algorithm
    for i in range(8):
        # Only relevant when textures are different
        is_diff = neigh8[:, i] != c
        # The condition for receiving blend: center <= neighbor
        can_receive = (c <= neigh8[:, i]).astype(np.float32)
        # Set to 0 when textures are same (no blend possible)
        can_receive = np.where(is_diff, can_receive, 0.0)
        feats.append(can_receive)
        names.append(f"can_blend_{_NEIGHBOR_NAMES[i]}")

    # Global: count of neighbors where center can receive blend (center <= neighbor)
    can_blend_count = np.zeros(len(flat_idx), dtype=np.float32)
    for i in range(8):
        is_diff = neigh8[:, i] != c
        can_blend_count += (is_diff & (c <= neigh8[:, i])).astype(np.float32)
    feats.append(can_blend_count / 8.0)  # Normalized
    names.append("can_blend_fraction")

    # Is center the LOWER index among same-type neighbors? (same-type priority signal)
    # For same-type textures, the palette order determines who blends into whom
    is_lower_priority = np.zeros(len(flat_idx), dtype=np.float32)
    for i in range(8):
        same_type_cond = (neigh_type[:, i] == c_type) & (neigh8[:, i] != c)
        is_lower = c <= neigh8[:, i]  # Using <= to match the actual rule
        is_lower_priority += (same_type_cond & is_lower).astype(np.float32)
    # Normalize by count of same-type different neighbors
    same_type_count = same_type_diff_tex.sum(axis=1)
    is_lower_priority = np.where(same_type_count > 0, is_lower_priority / same_type_count, 0.0)
    feats.append(is_lower_priority)
    names.append("is_lower_idx_same_type")

    # ==================== SAME-TYPE VARIANT FEATURES ====================
    # FINDING: 57-65% of false negatives are same-type variants!
    # The model misses blends like Dirt_Yucatan01 -> Dirt_Yucatan02 because
    # it sees "same type" and thinks no blend is needed.
    #
    # Key insight: Same-type variants SHOULD blend when:
    #   - center has lower palette index than neighbor
    #   - This is identical to cross-type rule, but model doesn't learn it
    #
    # Solution: Add explicit features that highlight same-type blending cases.

    # Count of same-type different-variant neighbors
    same_type_count_feat = same_type_diff_tex.sum(axis=1)
    feats.append(same_type_count_feat / 8.0)
    names.append("same_type_variant_count")

    # Is this a PURE same-type boundary? (no cross-type neighbors)
    # This is harder to predict because model trained mostly on cross-type
    any_diff_type = diff_type_mask.max(axis=1)  # 1 if ANY neighbor is different type
    pure_same_type_boundary = ((same_type_count_feat > 0) & (any_diff_type == 0)).astype(np.float32)
    feats.append(pure_same_type_boundary)
    names.append("pure_same_type_boundary")

    # For same-type neighbors: should this cell receive blend? (center < neighbor index)
    # This is the key signal the model is missing!
    same_type_should_blend = np.zeros(len(flat_idx), dtype=np.float32)
    for i in range(8):
        # Same type, different texture, AND center has strictly lower index
        should_blend_i = (same_type_diff_tex[:, i] == 1) & (c < neigh8[:, i])
        same_type_should_blend = np.maximum(same_type_should_blend, should_blend_i.astype(np.float32))
    feats.append(same_type_should_blend)
    names.append("same_type_should_blend")

    # Which same-type neighbor has the HIGHEST index? (likely secondary texture)
    # Encode as one-hot-ish: for each direction, is it the max same-type neighbor?
    same_type_neighbor_idx = np.where(same_type_diff_tex == 1, neigh8, -1)  # [N, 8]
    max_same_type_idx = same_type_neighbor_idx.max(axis=1)  # [N]
    for i in range(8):
        is_max_same_type = (same_type_neighbor_idx[:, i] == max_same_type_idx) & (same_type_neighbor_idx[:, i] >= 0)
        feats.append(is_max_same_type.astype(np.float32))
        names.append(f"max_same_type_{_NEIGHBOR_NAMES[i]}")

    # ==================== CRITICAL: Rule-based blend prediction ====================
    # The deterministic rule is: blend_present=1 if center_local < some_different_neighbor_local
    # This achieves 100% precision on "consistent" maps (those created with standard blend tool).
    # We encode this directly so the model can learn when to trust vs override it.
    # NOTE: 'c' is the LOCAL texture index (from tex_grid which comes from decoded tiles).
    rule_predicts_blend = np.zeros(len(flat_idx), dtype=np.float32)
    for i in range(8):
        is_diff = neigh8[:, i] != c
        is_higher = neigh8[:, i] > c  # Strict: secondary must be HIGHER index
        rule_predicts_blend = np.maximum(rule_predicts_blend, (is_diff & is_higher).astype(np.float32))
    feats.append(rule_predicts_blend)
    names.append("rule_predicts_blend")

    # Also count how many different neighbors have higher index (strength of rule signal)
    rule_neighbor_count = np.zeros(len(flat_idx), dtype=np.float32)
    for i in range(8):
        is_diff = neigh8[:, i] != c
        is_higher = neigh8[:, i] > c
        rule_neighbor_count += (is_diff & is_higher).astype(np.float32)
    feats.append(rule_neighbor_count / 8.0)
    names.append("rule_neighbor_fraction")

    # ==================== PATTERN MATCHING FEATURES ====================
    # The blend algorithm checks patterns in a specific priority order.
    # We encode which pattern matches and whether the blend is valid.
    # This directly tells the model the result of the game's pattern matching.

    # Get the 4 cardinal + 4 diagonal neighbors
    # Offsets: TL=0, T=1, TR=2, L=3, R=4, BL=5, B=6, BR=7
    left = neigh8[:, 3]  # L
    right = neigh8[:, 4]  # R
    top = neigh8[:, 1]  # T
    bottom = neigh8[:, 6]  # B
    topLeft = neigh8[:, 0]  # TL
    topRight = neigh8[:, 2]  # TR
    bottomLeft = neigh8[:, 5]  # BL
    bottomRight = neigh8[:, 7]  # BR

    # Compute pattern match result (which tex wins priority, or -1 if none)
    # Priority order from MapGenerator.BlendTextures
    pattern_tex = np.full(len(flat_idx), -1, dtype=np.int32)
    pattern_dir = np.zeros(len(flat_idx), dtype=np.int32)  # encoded direction

    # Pattern 1: left==top && top!=center -> BottomRight, tex=top
    mask = (left == top) & (top != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, top, pattern_tex)
    pattern_dir = np.where(mask, 1, pattern_dir)

    # Pattern 2: right==top && top!=center -> BottomLeft, tex=top
    mask = (right == top) & (top != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, top, pattern_tex)
    pattern_dir = np.where(mask, 2, pattern_dir)

    # Pattern 3: right==bottom && bottom!=center -> TopLeft, tex=bottom
    mask = (right == bottom) & (bottom != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, bottom, pattern_tex)
    pattern_dir = np.where(mask, 3, pattern_dir)

    # Pattern 4: left==bottom && bottom!=center -> TopRight, tex=bottom
    mask = (left == bottom) & (bottom != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, bottom, pattern_tex)
    pattern_dir = np.where(mask, 4, pattern_dir)

    # Pattern 5-8: Single edges
    mask = (left != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, left, pattern_tex)
    pattern_dir = np.where(mask, 5, pattern_dir)

    mask = (right != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, right, pattern_tex)
    pattern_dir = np.where(mask, 6, pattern_dir)

    mask = (top != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, top, pattern_tex)
    pattern_dir = np.where(mask, 7, pattern_dir)

    mask = (bottom != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, bottom, pattern_tex)
    pattern_dir = np.where(mask, 8, pattern_dir)

    # Pattern 9-12: Diagonals only
    mask = (topLeft != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, topLeft, pattern_tex)
    pattern_dir = np.where(mask, 9, pattern_dir)

    mask = (topRight != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, topRight, pattern_tex)
    pattern_dir = np.where(mask, 10, pattern_dir)

    mask = (bottomRight != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, bottomRight, pattern_tex)
    pattern_dir = np.where(mask, 11, pattern_dir)

    mask = (bottomLeft != c) & (pattern_tex < 0)
    pattern_tex = np.where(mask, bottomLeft, pattern_tex)
    pattern_dir = np.where(mask, 12, pattern_dir)

    # Key feature: does the pattern-matched tex satisfy center <= tex?
    # This is the EXACT rule the blend algorithm uses
    has_pattern = pattern_tex >= 0
    pattern_valid = has_pattern & (c <= pattern_tex)
    feats.append(pattern_valid.astype(np.float32))
    names.append("blend_rule_valid")

    # Also add the pattern direction as a normalized feature
    pattern_dir_norm = pattern_dir.astype(np.float32) / 12.0
    feats.append(pattern_dir_norm)
    names.append("pattern_dir_norm")

    # ==================== PASSABILITY FEATURES ====================
    # HIGH IMPACT: Blends don't cross impassable terrain (cliffs/walls).
    # This is a hard constraint we encode as features.
    # NOTE: Always emit passability features (with zero fallback if grid is missing)
    # to guarantee a consistent feature count across all maps.
    if impassable_grid is None:
        # Fallback: treat everything as passable (all zeros).
        impassable_grid = np.zeros((w, h), dtype=np.bool_)

    imp = np.asarray(impassable_grid, dtype=np.bool_, order="C")
    imp_pad = np.pad(imp, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=False)

    # Center impassability
    center_imp = imp_pad[px, py].astype(np.float32)
    feats.append(center_imp)
    names.append("center_impassable")

    # 8-neighbor impassability
    neigh_imp = np.stack([imp_pad[px + dx, py + dy] for dx, dy in _NEIGHBOR_OFFSETS], axis=1)  # [N,8]
    for i in range(8):
        feats.append(neigh_imp[:, i].astype(np.float32))
        names.append(f"impassable_{_NEIGHBOR_NAMES[i]}")

    # Aggregate: any neighbor impassable (hard boundary signal)
    any_imp = neigh_imp.any(axis=1).astype(np.float32)
    feats.append(any_imp)
    names.append("any_neighbor_impassable")

    # Count of impassable neighbors (normalized)
    imp_count = neigh_imp.sum(axis=1).astype(np.float32) / 8.0
    feats.append(imp_count)
    names.append("impassable_neighbor_fraction")

    # Blend rule should not apply across impassable boundaries
    # For each neighbor: if it's impassable, blend rule is invalid
    for i in range(8):
        is_diff = neigh8[:, i] != c
        can_blend_if_passable = (c <= neigh8[:, i]) & is_diff
        # Mask out if neighbor is impassable
        blend_valid_passable = can_blend_if_passable & ~neigh_imp[:, i]
        feats.append(blend_valid_passable.astype(np.float32))
        names.append(f"can_blend_passable_{_NEIGHBOR_NAMES[i]}")

    # ==================== DISTANCE-TO-TEXTURE-BOUNDARY FEATURES ====================
    # Chebyshev distance to the nearest texture boundary for each cell.
    # Key insight: blend probability drops sharply with distance from boundary.
    # - Distance 0 (ON boundary): ~55% of cells have blends
    # - Distance 1 (near boundary): ~10% (anticipatory blends)
    # - Distance >= 2 (interior): ~2% (very rare)
    #
    # We include both the center cell distance and the full 5x5 neighborhood
    # distance grid for richer spatial context.
    if dist_to_boundary is not None:
        dtb = np.asarray(dist_to_boundary, dtype=np.float32)
    else:
        # Fallback: compute it on the fly (should not happen in normal usage)
        dtb = _compute_distance_to_boundary(tex_grid)

    # Center cell distance (single scalar, capped at 10 and normalized)
    center_dist = dtb.reshape(-1)[flat_idx]
    center_dist_capped = np.minimum(center_dist, 10.0) / 10.0
    feats.append(center_dist_capped)
    names.append("dist_to_boundary")

    # 5x5 neighborhood distance grid (25 values, capped and normalized)
    # Use same padding strategy as for texture windows
    win = 5  # neighborhood window size (fixed for this feature)
    dtb_pad = np.pad(dtb, pad_width=((2, 2), (2, 2)), mode="edge")
    dtb_windows = np.lib.stride_tricks.sliding_window_view(dtb_pad, (win, win))
    dtb_flat = dtb_windows.reshape(w * h, win * win)
    dtb_sampled = dtb_flat[flat_idx]
    dtb_sampled = np.minimum(dtb_sampled, 10.0) / 10.0
    for idx_5x5 in range(win * win):
        r, cc = divmod(idx_5x5, win)
        feats.append(dtb_sampled[:, idx_5x5])
        names.append(f"dist_to_boundary_{r}_{cc}")

    # ==================== ELEVATION FEATURES ====================
    # Always emit elevation features (with zero fallback if grid is missing)
    # to guarantee a consistent feature count across all maps.
    if elev_grid is not None:
        z = np.asarray(elev_grid, dtype=np.float32, order="C")
        zpad = np.pad(z, pad_width=((1, 1), (1, 1)), mode="edge")
        cz = zpad[px, py]
        zleft = zpad[px - 1, py]
        zright = zpad[px + 1, py]
        zup = zpad[px, py - 1]
        zdown = zpad[px, py + 1]
        dx = (zright - zleft) * 0.5
        dy = (zdown - zup) * 0.5
        slope = np.sqrt(dx * dx + dy * dy)
        lap = (zup + zdown + zleft + zright) - (4.0 * cz)

        feats.extend([dx, dy, slope, lap])
        names.extend(["elev_dx", "elev_dy", "elev_slope", "elev_laplacian"])
    else:
        # Fallback: zero elevation features so feature count is always consistent
        n = len(flat_idx)
        feats.extend([np.zeros(n, dtype=np.float32)] * 4)
        names.extend(["elev_dx", "elev_dy", "elev_slope", "elev_laplacian"])

    extra = np.stack(feats, axis=1).astype(np.float32, copy=False)
    return extra, names


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate RA3 blendinfo supervised dataset from a directory of maps.")
    ap.add_argument("--root", required=True, help="Root dir containing .map files (recursively scanned)")
    ap.add_argument("--out", required=True, help="Output dataset .npz path")
    ap.add_argument("--meta-out", default="", help="Optional metadata JSON path (default: <out>.json)")
    ap.add_argument("--limit-maps", type=int, default=0, help="Only process the first N original maps (for quick tests)")

    ap.add_argument("--window", type=int, default=5, help="Odd neighborhood window size (3/5/7...)")
    ap.add_argument("--include-elevation", action="store_true", help="Append elevation neighborhood to X")
    ap.add_argument(
        "--no-extra-signal",
        action="store_true",
        help="Disable extra per-cell signals (phase/boundary/slope). Not recommended.",
    )
    ap.add_argument("--stride", type=int, default=1, help="Sample every N cells in x/y")
    ap.add_argument("--max-samples-per-map", type=int, default=50000, help="Cap samples per map (default 50k)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")

    ap.add_argument("--dedupe", action="store_true", help="Dedupe identical maps by content hash")
    ap.add_argument("--min-blend-ratio", type=float, default=0.70, help="Remove maps below this blend ratio")
    ap.add_argument(
        "--blend-ratio-metric",
        choices=["blendinfo_usage", "any_cells", "combined_cells"],
        default="blendinfo_usage",
        help="Which blend 'ratio' to use for filtering (default: blendinfo_usage).",
    )
    ap.add_argument(
        "--augment-sample-rotations",
        default="",
        help="Comma-separated degrees (CW) for *sample-level* augmentation, e.g. '90' or '90,180,270'. Empty disables (default).",
    )
    ap.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable global shuffling of (X,y, map_id) before writing dataset (not recommended).",
    )
    ap.add_argument("--write-blendless-maps", action="store_true", help="Write blendless .map files to workdir")
    ap.add_argument(
        "--generated-no-compress",
        action="store_true",
        help="Write generated maps uncompressed (faster; fine for training data).",
    )
    ap.add_argument("--workdir", default="", help="Working directory (default: <root>/_generated)")
    ap.add_argument("--pruned-dir", default="", help="Where to move pruned maps (default: <root>/_pruned)")
    ap.add_argument("--force", action="store_true", help="Recreate generated maps even if they exist")

    args = ap.parse_args()

    win = int(args.window)
    if win <= 0 or win % 2 != 1:
        raise SystemExit("--window must be a positive odd integer")
    pad = win // 2

    root = Path(args.root)
    workdir = Path(args.workdir) if args.workdir else (root / "_generated")
    pruned = Path(args.pruned_dir) if args.pruned_dir else (root / "_pruned")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "blendless").mkdir(parents=True, exist_ok=True)

    all_maps = _iter_map_files(root)
    # Only treat non-blendless maps as originals; blendless are generated artifacts.
    originals = [
        p
        for p in all_maps
        if not _is_blendless_name(p)
        and "_generated" not in p.parts
        and "_pruned" not in p.parts
        and not any(part.startswith("_") for part in p.relative_to(root).parts)  # ignore internal folders under root
    ]
    if int(args.limit_maps) > 0:
        originals = originals[: int(args.limit_maps)]
    if not originals:
        raise SystemExit("No original .map files found under --root")

    rng = np.random.default_rng(int(args.seed))
    rng_shuffle = np.random.default_rng(int(args.seed) + 99991)

    # ---- pass 1: filter by blend ratio and compute hash for dedupe ----
    kept: List[Path] = []
    hash_to_path: Dict[str, Path] = {}
    deduped = 0
    removed_low = 0
    meta_prune: List[Dict[str, object]] = []

    for p in _tqdm(originals, total=len(originals), desc="Filter/dedupe"):
        try:
            m = Ra3Map(str(p))
            m.parse()
            ctx = m.get_context()
            blend = ctx.get_asset_by_type(BlendTileData)
            height = ctx.get_asset_by_type(HeightMapData)
            if blend is None:
                _move_to_pruned(p, pruned, "missing_blendtiledata")
                meta_prune.append({"path": str(p), "reason": "missing_blendtiledata"})
                continue

            ratio = _blend_ratio(blend, metric=str(args.blend_ratio_metric))
            if ratio < float(args.min_blend_ratio):
                _move_to_pruned(p, pruned, "low_blend_ratio")
                removed_low += 1
                meta_prune.append({"path": str(p), "reason": "low_blend_ratio", "ratio": ratio})
                continue

            if args.dedupe:
                hsh = _hash_core(blend, height, include_elevation=bool(args.include_elevation))
                if hsh in hash_to_path:
                    _move_to_pruned(p, pruned, "duplicate")
                    deduped += 1
                    meta_prune.append({"path": str(p), "reason": "duplicate", "same_as": str(hash_to_path[hsh])})
                    continue
                hash_to_path[hsh] = p

            kept.append(p)
        except Exception as e:
            _move_to_pruned(p, pruned, "parse_error")
            meta_prune.append({"path": str(p), "reason": "parse_error", "error": str(e)})

    print(f"Found originals: {len(originals)}")
    print(f"Kept after filtering: {len(kept)} (deduped={deduped}, low_blend_removed={removed_low})")

    if not kept:
        raise SystemExit("No maps left after filtering/deduping.")

    # Shuffle map order (keeps map-level IDs consistent with meta["pairs"], but avoids deterministic ordering).
    kept = list(kept)
    rng_shuffle.shuffle(kept)

    # sample-level augmentation rotations
    rotations = [int(x.strip()) for x in str(args.augment_sample_rotations).split(",") if x.strip()]
    rotations = [r for r in rotations if r % 90 == 0 and (r % 360) != 0]
    if rotations:
        print(f"Sample-level augment rotations: {rotations}")
    if rotations and (not bool(args.no_extra_signal)):
        raise SystemExit(
            "Rotation augmentation with extra signal features is not supported (it would require rotating phase/slope). "
            "Re-run with --augment-sample-rotations '' (recommended) or --no-extra-signal."
        )

    # ---- extract dataset ----
    X_parts: List[np.ndarray] = []
    ybp_parts: List[np.ndarray] = []
    # New v3 mask labels: uint8 bitmask (0-255, where 255=ignore)
    ybm_parts: List[np.ndarray] = []
    ybs_parts: List[np.ndarray] = []
    ybd_parts: List[np.ndarray] = []
    ysp_parts: List[np.ndarray] = []
    ysm_parts: List[np.ndarray] = []
    yss_parts: List[np.ndarray] = []
    ysd_parts: List[np.ndarray] = []
    # GeoLS soft targets for blend_present (float32, distance-weighted)
    ybp_soft_parts: List[np.ndarray] = []
    ysp_soft_parts: List[np.ndarray] = []
    map_id_parts: List[np.ndarray] = []

    meta: Dict[str, object] = {
        "root": str(root),
        "window": win,
        "include_elevation": bool(args.include_elevation),
        "stride": int(args.stride),
        "max_samples_per_map": int(args.max_samples_per_map),
        "seed": int(args.seed),
        "min_blend_ratio": float(args.min_blend_ratio),
        "blend_ratio_metric": str(args.blend_ratio_metric),
        "dedupe": bool(args.dedupe),
        "augment_sample_rotations": rotations,
        "extra_signal": (not bool(args.no_extra_signal)),
        "write_blendless_maps": bool(args.write_blendless_maps),
        "generated_no_compress": bool(args.generated_no_compress),
        "pairs": [],
        "pruned": meta_prune,
    }

    for i, p in enumerate(_tqdm(kept, total=len(kept), desc="Extract samples")):
        m = Ra3Map(str(p))
        m.parse()
        ctx = m.get_context()
        b = ctx.get_asset_by_type(BlendTileData)
        if b is None:
            continue
        w, h = b.tiles.shape

        # Optional: write a blendless .map for this original (fast path: no extra parse)
        if args.write_blendless_maps:
            blendless_path = workdir / "blendless" / f"{p.stem}_blendless.map"
            if args.force or (not blendless_path.exists()):
                # store labels first (we need blends to build y), then zero and save
                orig_blends = np.asarray(b.blends).copy()
                orig_se = np.asarray(b.single_edge_blends).copy()
                b.blends = np.zeros_like(b.blends, dtype=np.uint16)
                b.single_edge_blends = np.zeros_like(b.single_edge_blends, dtype=np.uint16)
                m.save(str(blendless_path), compress=(not args.generated_no_compress))
                # restore so labels match original
                b.blends = orig_blends
                b.single_edge_blends = orig_se
            else:
                blendless_path = None

        # Features from tiles (blendless == same tiles)
        tex_grid = _decode_texture_grid(b).astype(np.int16)
        tex_pad = np.pad(tex_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
        tex_windows = np.lib.stride_tricks.sliding_window_view(tex_pad, (win, win))
        tex_feat = tex_windows.reshape(w * h, win * win)

        # Build texture type and biome grids for priority features
        type_grid, biome_grid = _build_tex_type_biome_grids(tex_grid, b.textures or [])

        feat_list = [tex_feat]
        elev_grid = None
        if args.include_elevation:
            bh = ctx.get_asset_by_type(HeightMapData)
            if bh is None or bh.elevations is None:
                continue
            elev_grid = np.asarray(bh.elevations, dtype=np.float32)
            if elev_grid.shape != (w, h):
                continue
            elev_pad = np.pad(elev_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
            elev_windows = np.lib.stride_tricks.sliding_window_view(elev_pad, (win, win))
            elev_feat = elev_windows.reshape(w * h, win * win)
            feat_list.append(elev_feat)

        X_full = np.concatenate(feat_list, axis=1)

        # Use v3 labels: 8-bit neighbor mask (multi-label) + direction.
        # This captures the full secondary-neighbor pattern and is critical for direction.
        ybp, ybm, ybd, ybs_first = _labels_for_layer_v3(b, np.asarray(b.blends), tex_grid)
        ysp, ysm, ysd, yss_first = _labels_for_layer_v3(b, np.asarray(b.single_edge_blends), tex_grid)

        flat_idx = _sample_indices(
            w,
            h,
            stride=max(1, int(args.stride)),
            max_samples=int(args.max_samples_per_map),
            rng=rng,
        )

        Xs = X_full[flat_idx]
        ybp_s = ybp.reshape(-1)[flat_idx]
        ybm_s = ybm.reshape(-1)[flat_idx]
        ybd_s = ybd.reshape(-1)[flat_idx]
        ysp_s = ysp.reshape(-1)[flat_idx]
        ysm_s = ysm.reshape(-1)[flat_idx]
        ysd_s = ysd.reshape(-1)[flat_idx]

        # Keep legacy single-neighbor labels for debugging/back-compat (not used by new model)
        ybs_s = ybs_first.reshape(-1)[flat_idx]
        yss_s = yss_first.reshape(-1)[flat_idx]

        # Extra per-cell signals (computed on sampled indices)
        extra_s = None
        extra_names: List[str] = []
        # Get impassable grid from BlendTileData (HIGH IMPACT feature)
        impassable_grid = None
        if hasattr(b, 'impassable') and b.impassable is not None:
            impassable_grid = np.asarray(b.impassable, dtype=np.bool_)

        # Pre-compute distance-to-texture-boundary for the full map
        dist_to_boundary_grid = _compute_distance_to_boundary(tex_grid)

        if not bool(args.no_extra_signal):
            extra_s, extra_names = _extra_signal_features(
                tex_grid=tex_grid,
                elev_grid=elev_grid,
                type_grid=type_grid,
                biome_grid=biome_grid,
                impassable_grid=impassable_grid,
                dist_to_boundary=dist_to_boundary_grid,
                flat_idx=flat_idx,
                w=w,
                h=h,
            )
            Xs = np.concatenate([Xs.astype(np.float32, copy=False), extra_s], axis=1)

        # Compute GeoLS soft targets for blend_present based on boundary distance.
        # These are always computed (independent of --no-extra-signal) so training
        # can optionally use them for the ASL/GeoLS loss.
        ybp_soft_s = _compute_soft_blend_present(
            ybp, dist_to_boundary_grid, flat_idx, w, h,
        )
        ysp_soft_s = _compute_soft_blend_present(
            ysp, dist_to_boundary_grid, flat_idx, w, h,
        )

        # sample-level rotation augmentation (fast; avoids full-map rotation)
        if rotations:
            Xs, ybd_s, ybs_s = _augment_samples_rotations(
                Xs, win=win, y_dir=ybd_s, y_neighbor=ybs_s,
                rotations=rotations, include_elevation=bool(args.include_elevation)
            )
            # apply same augment to single-edge
            _, ysd_s, yss_s = _augment_samples_rotations(
                X_full[flat_idx],
                win=win,
                y_dir=ysd_s,
                y_neighbor=yss_s,
                rotations=rotations,
                include_elevation=bool(args.include_elevation),
            )
            # NOTE: mask labels are rotation-dependent too. We don't rotate masks currently because
            # rotations are disabled by default (and recommended off). If you enable rotations later,
            # implement a mask rotation LUT similar to _rotate_neighbor_idx_90.
            # presence is rotation-invariant for sample augmentation
            rep = 1 + len(rotations)
            ybp_s = np.tile(ybp_s, rep)
            ysp_s = np.tile(ysp_s, rep)
            # Soft targets are also rotation-invariant (scalar per cell)
            ybp_soft_s = np.tile(ybp_soft_s, rep)
            ysp_soft_s = np.tile(ysp_soft_s, rep)

        X_parts.append(Xs)
        ybp_parts.append(ybp_s)
        ybd_parts.append(ybd_s)
        ysp_parts.append(ysp_s)
        ysd_parts.append(ysd_s)

        # GeoLS soft targets
        ybp_soft_parts.append(ybp_soft_s)
        ysp_soft_parts.append(ysp_soft_s)

        # New mask labels (uint8 bitmask, 255=ignore)
        ybm_parts.append(ybm_s)
        ysm_parts.append(ysm_s)

        # Legacy neighbor-index labels (int8, -1=unknown)
        ybs_parts.append(ybs_s)
        yss_parts.append(yss_s)
        map_id_parts.append(np.full((Xs.shape[0],), i, dtype=np.int32))

        # Per-map regime stats: used later as map-style features so one model can handle mixed sources.
        # These are computed on the full map (not just samples).
        bpos_n = float(np.sum(ybp != 0))
        spos_n = float(np.sum(ysp != 0))
        area = float(w * h)
        blend_mask_valid = float(np.sum((ybp != 0) & (ybm != np.uint8(255))))
        se_mask_valid = float(np.sum((ysp != 0) & (ysm != np.uint8(255))))

        # Compute rule adherence: for blend_present=1 cells with valid mask,
        # how often is center_local < secondary_local?
        # This identifies "consistent" maps where the standard blend rule applies.
        rule_adherence = 0.0
        blend_with_mask = (ybp != 0) & (ybm != np.uint8(0)) & (ybm != np.uint8(255))
        blend_with_mask_n = int(np.sum(blend_with_mask))
        if blend_with_mask_n > 0:
            tex_full_flat = tex_feat.reshape(w * h, -1)  # Already computed above
            rule_holds = 0
            rule_total = 0
            # Sample up to 1000 cells for efficiency
            blend_indices = np.where(blend_with_mask.flatten())[0]
            check_indices = blend_indices[:1000]
            for idx in check_indices:
                mask_val = int(ybm.flatten()[idx])
                tex_win = tex_full_flat[idx].reshape(5, 5)
                center = int(tex_win[2, 2])
                # Check if center < any masked neighbor
                NEIGH_POS = [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)]
                for bit, (r, c) in enumerate(NEIGH_POS):
                    if (mask_val >> bit) & 1:
                        sec = int(tex_win[r, c])
                        if center < sec:
                            rule_holds += 1
                        rule_total += 1
                        break  # Just check first secondary
            if rule_total > 0:
                rule_adherence = float(rule_holds) / float(rule_total)
        meta["pairs"].append(
            {
                "id": p.stem,
                "orig": str(p),
                "blendless_written": bool(args.write_blendless_maps),
                "textures": [t.name for t in (b.textures or [])],
                "w": int(w),
                "h": int(h),
                "samples": int(Xs.shape[0]),
                "stats": {
                    "blend_pos_rate": float(bpos_n / max(1.0, area)),
                    "se_pos_rate": float(spos_n / max(1.0, area)),
                    "blend_mask_valid_rate": float(blend_mask_valid / max(1.0, bpos_n)),
                    "se_mask_valid_rate": float(se_mask_valid / max(1.0, spos_n)),
                    "rule_adherence": float(rule_adherence),  # 1.0 = consistent map, <0.7 = inconsistent
                },
            }
        )

        if (not bool(args.no_extra_signal)) and ("extra_feature_names" not in meta):
            meta["extra_feature_names"] = extra_names

    if not X_parts:
        raise SystemExit("No samples extracted (check filters and paths).")

    X = np.concatenate(X_parts, axis=0)
    # Strong global shuffle across all maps/samples (helps training stability and prevents map-order artifacts).
    if not bool(args.no_shuffle):
        perm = rng_shuffle.permutation(int(X.shape[0]))
        X = X[perm]
        ybp_all = np.concatenate(ybp_parts, axis=0)[perm]
        ybm_all = np.concatenate(ybm_parts, axis=0)[perm]
        ybs_all = np.concatenate(ybs_parts, axis=0)[perm]
        ybd_all = np.concatenate(ybd_parts, axis=0)[perm]
        ysp_all = np.concatenate(ysp_parts, axis=0)[perm]
        ysm_all = np.concatenate(ysm_parts, axis=0)[perm]
        yss_all = np.concatenate(yss_parts, axis=0)[perm]
        ysd_all = np.concatenate(ysd_parts, axis=0)[perm]
        ybp_soft_all = np.concatenate(ybp_soft_parts, axis=0)[perm]
        ysp_soft_all = np.concatenate(ysp_soft_parts, axis=0)[perm]
        map_id_all = np.concatenate(map_id_parts, axis=0)[perm]
    else:
        ybp_all = np.concatenate(ybp_parts, axis=0)
        ybm_all = np.concatenate(ybm_parts, axis=0)
        ybs_all = np.concatenate(ybs_parts, axis=0)
        ybd_all = np.concatenate(ybd_parts, axis=0)
        ysp_all = np.concatenate(ysp_parts, axis=0)
        ysm_all = np.concatenate(ysm_parts, axis=0)
        yss_all = np.concatenate(yss_parts, axis=0)
        ysd_all = np.concatenate(ysd_parts, axis=0)
        ybp_soft_all = np.concatenate(ybp_soft_parts, axis=0)
        ysp_soft_all = np.concatenate(ysp_soft_parts, axis=0)
        map_id_all = np.concatenate(map_id_parts, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_blend_present=ybp_all,
        y_blend_mask=ybm_all,
        y_blend_sec=ybs_all,
        y_blend_dir=ybd_all,
        y_se_present=ysp_all,
        y_se_mask=ysm_all,
        y_se_sec=yss_all,
        y_se_dir=ysd_all,
        y_blend_present_soft=ybp_soft_all,
        y_se_present_soft=ysp_soft_all,
        map_id=map_id_all,
    )
    print(f"Wrote dataset: {out_path} (samples={X.shape[0]}, feat_dim={X.shape[1]})")

    meta_out = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".json")
    meta["shuffled"] = (not bool(args.no_shuffle))
    meta["shuffle_seed"] = int(args.seed) + 99991
    # Document the new label encodings
    meta["label_encoding"] = {
        # NEW: 8-bit neighbor mask (bit i => neighbor i equals secondary texture), 255=ignore/unknown.
        # This is the primary label for secondary texture structure (multi-label).
        "y_blend_mask": "neighbor_mask8_u8",
        "y_se_mask": "neighbor_mask8_u8",
        # Legacy: first matching neighbor index (kept for debugging/back-compat)
        "y_blend_sec": "neighbor_index_first",
        "y_se_sec": "neighbor_index_first",
        "neighbor_names": _NEIGHBOR_NAMES,  # ["TL", "T", "TR", "L", "R", "BL", "B", "BR"]
        "num_neighbor_classes": 8,
        # GeoLS soft targets for blend_present (float32).
        # Training can use these instead of hard binary labels for
        # better calibration near texture boundaries.
        # Values: 0.95 (positive), 0.45 (neg on boundary), 0.10 (neg near boundary),
        #         0.02 (neg interior).
        "y_blend_present_soft": "geols_soft_f32",
        "y_se_present_soft": "geols_soft_f32",
    }
    # Document texture type and biome vocabularies (used for priority features)
    meta["texture_type_vocab"] = _TEX_TYPES
    meta["texture_biome_vocab"] = _TEX_BIOMES
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


