"""Raw, unengineered inputs for end-to-end texture prediction.

We pull only what's natively in a textureless map file:
  - Elevation grid (z per tile)
  - Water mask (rasterised once from StandingWater/River/Wave area polygons)
  - Object set (each: tile_x, tile_y, z, angle_deg, type_name, owner)
  - MP spawn positions

The model is responsible for learning every spatial relationship from these
raw inputs; this module deliberately does NO distance-fields, slope, curvature,
density-at-radius, bucket categorisation, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RawObject:
    tile_x: float
    tile_y: float
    z: float
    angle_deg: float
    type_name: str
    owner: str


@dataclass
class BlendVocab:
    """Compact-id encoding of a uint16 blend array.

    For inference writeback: original_value = values[predicted_compact_id].
    """
    array: np.ndarray         # (W, H) int32 compact ids in [0, len(values))
    values: List[int]         # values[i] = original uint16 for compact id i


# Direction class vocabulary. Matches existing scripts/train_blend_unet.py so
# that downstream tools (eval, inference) stay compatible. Class 0 = "no blend"
# (raw value -1); classes 1..16 are the 16 real BlendDirection enum values.
DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
DIR_VAL_TO_CLASS = {v: i for i, v in enumerate(DIRECTION_VALUES)}
NUM_DIR_CLASSES = len(DIRECTION_VALUES)


@dataclass
class BlendDecomp:
    """Decomposed targets for a uint16 blend-index array.

    Each entry of the source array indexes a blend_info record. We decompose
    that record into per-tile properties so the model can predict tractable
    quantities (~17 directions, palette_size secondary textures) instead of
    a 17000-class softmax over raw blend indices.
    """
    present: np.ndarray         # (W, H) int32 in {0, 1}
    secondary_tex: np.ndarray   # (W, H) int32 palette idx; -1 where present=0
    direction: np.ndarray       # (W, H) int32 in [0, NUM_DIR_CLASSES); -1 where present=0
    neighbor_mask: Optional[np.ndarray] = None  # (W, H) uint8 8-bit mask; 255 = ignore


@dataclass
class RawInputs:
    elev: np.ndarray          # (W, H) float32
    water: np.ndarray         # (W, H) float32 in {0, 1}
    width: int
    height: int
    objects: List[RawObject] = field(default_factory=list)
    mp_spawns: List[Tuple[float, float]] = field(default_factory=list)
    style_id: Optional[int] = None
    target_tiles: Optional[np.ndarray] = None     # (W, H) int32
    palette: Optional[List[str]] = None
    # Decomposed blend-array targets (optional, present when extract_target=True)
    blends: Optional[BlendDecomp] = None
    single_edge_blends: Optional[BlendDecomp] = None
    cliff_blends: Optional[BlendDecomp] = None


def _rasterise_water_areas(ctx, W: int, H: int) -> np.ndarray:
    out = np.zeros((W, H), dtype=np.float32)
    standing = ctx.get_asset("StandingWaterAreas")
    rivers = ctx.get_asset("RiverAreas")
    waves = ctx.get_asset("StandingWaveAreas")

    def _stamp(areas, attr_points: str = "polygon"):
        for a in areas:
            pts = getattr(a, attr_points, None) or getattr(a, "points", None)
            if not pts:
                continue
            xs = [p[0] / 10.0 for p in pts]
            ys = [p[1] / 10.0 for p in pts]
            x0, x1 = max(0, int(min(xs))), min(W - 1, int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(H - 1, int(max(ys)))
            if x1 > x0 and y1 > y0:
                out[x0:x1 + 1, y0:y1 + 1] = 1.0

    if standing is not None:
        _stamp(getattr(standing, "water_areas", []))
    if rivers is not None:
        _stamp(getattr(rivers, "areas", []))
    if waves is not None:
        _stamp(getattr(waves, "areas", []))
    return out


def extract_raw_inputs(ra3_map, *, extract_target: bool = False,
                       style_id: Optional[int] = None) -> RawInputs:
    ctx = ra3_map.get_context()
    blend = ctx.get_asset("BlendTileData")
    h_asset = ctx.get_asset("HeightMapData")
    if blend is None or h_asset is None:
        raise ValueError("Map missing BlendTileData or HeightMapData")

    W = int(blend.map_width)
    H = int(blend.map_height)

    elev = np.asarray(h_asset.elevations, dtype=np.float32)
    eW, eH = elev.shape
    out_elev = np.zeros((W, H), dtype=np.float32)
    cw, ch = min(W, eW), min(H, eH)
    out_elev[:cw, :ch] = elev[:cw, :ch]
    # Normalise to [-1, 1] using percentile clipping for stability across maps.
    lo, hi = float(np.percentile(out_elev, 1)), float(np.percentile(out_elev, 99))
    if hi - lo > 1e-6:
        out_elev = np.clip((out_elev - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0)
    else:
        out_elev = np.zeros_like(out_elev)

    water = _rasterise_water_areas(ctx, W, H)

    objects: List[RawObject] = []
    objs_asset = ctx.get_asset("ObjectsList")
    if objs_asset is not None:
        for obj in objs_asset.map_objects:
            name = obj.type_name or ""
            if not name:
                continue
            # Filter out ambient sound emitters — they are invisible runtime
            # entities (cricket loops, wave loops, etc.) that carry no texture
            # signal and are not the kind of object a user would place at the
            # "just made the terrain" stage.
            if name.startswith("Amb_") or name.startswith("amb_"):
                continue
            objects.append(RawObject(
                tile_x=obj.position[0] / 10.0,
                tile_y=obj.position[1] / 10.0,
                z=obj.position[2],
                angle_deg=float(obj.angle or 0.0),
                type_name=name,
                owner=obj.original_owner or "",
            ))

    mp_spawns: List[Tuple[float, float]] = []
    if objs_asset is not None:
        for obj in objs_asset.map_objects:
            n = (obj.type_name or "").lower()
            if "player" in n and "start" in n:
                mp_spawns.append((obj.position[0] / 10.0, obj.position[1] / 10.0))

    target_tiles = None
    palette = None
    blends_d = None
    single_edge_d = None
    cliff_d = None
    if extract_target:
        target_tiles = np.zeros((W, H), dtype=np.int32)
        for x in range(W):
            for y in range(H):
                target_tiles[x, y] = blend.get_texture(x, y)
        palette = [t.name for t in blend.textures]

        def _crop(arr):
            a = np.asarray(arr)
            out = np.zeros((W, H), dtype=a.dtype)
            cw, ch = min(W, a.shape[0]), min(H, a.shape[1])
            out[:cw, :ch] = a[:cw, :ch]
            return out

        blend_info = list(blend.blend_info or [])
        blends_d = _decompose_blend_array(_crop(blend.blends), blend_info, target_tiles, with_neighbor_mask=True)
        single_edge_d = _decompose_blend_array(_crop(blend.single_edge_blends), blend_info, target_tiles)
        if blend.cliff_blends is not None:
            cliff_d = _decompose_blend_array(_crop(blend.cliff_blends), blend_info, target_tiles)

    return RawInputs(
        elev=out_elev,
        water=water,
        width=W,
        height=H,
        objects=objects,
        mp_spawns=mp_spawns,
        style_id=style_id,
        target_tiles=target_tiles,
        palette=palette,
        blends=blends_d,
        single_edge_blends=single_edge_d,
        cliff_blends=cliff_d,
    )


# Neighbor offsets for the 8-bit neighbor mask. Order matches the existing
# blend U-Net (TL, T, TR, L, R, BL, B, BR) so checkpoints stay compatible.
_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _position_pattern_grid(W: int, H: int) -> np.ndarray:
    xs = np.arange(W, dtype=np.int32)[:, None]
    ys = np.arange(H, dtype=np.int32)[None, :]
    row_first = (ys % 8) // 2 * 16 + (ys % 2) * 2
    return (xs % 8) // 2 * 4 + (xs % 2) + row_first


def _decompose_blend_array(arr: np.ndarray, blend_info: list, tex_grid: np.ndarray,
                           *, with_neighbor_mask: bool = False) -> BlendDecomp:
    """Convert a (W, H) blend-index array into (present, sec_tex, direction[, neighbor]).

    Each value `idx` in the source array indexes blend_info[idx-1]; a value of
    0 means "no blend". The blend_info entry carries:
      - secondary_texture_tile = sec_tex_id * 64 + position_pattern(x, y)
      - blend_direction         = raw int (decoded into 17-class space)
    """
    a = np.asarray(arr, dtype=np.int32)
    W, H = a.shape
    pos = _position_pattern_grid(W, H)

    present = (a > 0).astype(np.int32)
    sec_tex = np.full((W, H), -1, dtype=np.int32)
    direction = np.full((W, H), -1, dtype=np.int32)

    # Decode (sec_tex, direction) per nonzero cell. Fast path: vectorised by
    # stable indexing into a precomputed (info_idx -> (sec_tex_tile, dir_raw))
    # array, then position-decoding sec_tex_tile per cell.
    n_info = len(blend_info)
    if n_info > 0:
        info_sec_tile = np.array([int(bi.secondary_texture_tile) for bi in blend_info], dtype=np.int32)
        info_dir_raw = np.array([int(bi.blend_direction) for bi in blend_info], dtype=np.int32)
        nz = a > 0
        if nz.any():
            idxs = a[nz] - 1
            valid = (idxs >= 0) & (idxs < n_info)
            xs, ys = np.where(nz)
            xs = xs[valid]; ys = ys[valid]; idxs = idxs[valid]
            sec_tile = info_sec_tile[idxs]
            sec_decoded = (sec_tile - pos[xs, ys]) // 64
            sec_tex[xs, ys] = sec_decoded
            dir_raw = info_dir_raw[idxs]
            # Map raw direction values to class indices (17-way). Unknown raw
            # values fall through as -1 (will be ignored by loss masks).
            for raw_val, cls_idx in DIR_VAL_TO_CLASS.items():
                direction[xs[dir_raw == raw_val], ys[dir_raw == raw_val]] = cls_idx

    neighbor_mask = None
    if with_neighbor_mask:
        # 8-bit per-cell mask of "this neighbor's primary texture matches my
        # secondary texture". 255 = ignore (we have a blend but no neighbour
        # matched, so the mask is undefined for the existing model).
        tex_pad = np.pad(tex_grid, ((1, 1), (1, 1)), mode="edge")
        sec_pad = np.pad(sec_tex,  ((1, 1), (1, 1)), mode="edge")
        mask8 = np.zeros((W, H), dtype=np.uint8)
        for ni, (dx, dy) in enumerate(_NEIGHBOR_OFFSETS):
            neigh = tex_pad[1 + dx: 1 + dx + W, 1 + dy: 1 + dy + H]
            sec = sec_pad[1:1 + W, 1:1 + H]
            hit = (present != 0) & (sec >= 0) & (neigh == sec)
            mask8 = np.bitwise_or(mask8, (hit.astype(np.uint8) << np.uint8(ni)))
        mask8 = np.where((present != 0) & (mask8 == 0), np.uint8(255), mask8).astype(np.uint8)
        neighbor_mask = mask8

    return BlendDecomp(present=present, secondary_tex=sec_tex,
                       direction=direction, neighbor_mask=neighbor_mask)
