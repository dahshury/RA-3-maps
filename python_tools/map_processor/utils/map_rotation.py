"""
Map rotation utilities (right-angle rotations) for RA3 maps.

Key gotcha (important!):
- `BlendTileData.tiles[x,y]` is NOT a pure texture index grid.
  It stores the result of C# `BlendTileData.GetTile(x,y,texture)` which encodes
  BOTH texture index and (x%8,y%8) pattern.
- `BlendInfo.secondary_texture_tile` is also stored as that same tile-value form.
- Therefore, rotating terrain is not just `np.rot90` on the `tiles` grid: you must
  extract texture indices at the old position, then recompute tile values for the
  new position.

This module rotates:
- HeightMapData (elevations)
- BlendTileData (tiles + blends + single_edge_blends + all related grids)
- ObjectsList (positions + yaw)
- Water areas / rivers / waves / trigger polygons
- Script argument positions (if present)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..assets.terrain.blend_direction import BlendDirection

WORLD_UNITS_PER_TILE = 10.0


@dataclass(frozen=True)
class RotationSpec:
    degrees: int
    clockwise: bool = True

    @property
    def turns_cw(self) -> int:
        deg = int(self.degrees) % 360
        if deg % 90 != 0:
            raise ValueError(f"Rotation must be a multiple of 90 degrees, got {self.degrees}")
        turns = (deg // 90) % 4
        return turns if self.clockwise else (-turns) % 4


def _rotate_grid_xy(arr: Optional[np.ndarray], turns_cw: int) -> Optional[np.ndarray]:
    """Rotate a (width, height) [x,y] array by CW turns (0..3)."""
    if arr is None:
        return None
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return arr
    mat_yx = np.asarray(arr).T  # (h,w)
    rotated_yx = np.rot90(mat_yx, k=-turns_cw)  # negative => clockwise
    return rotated_yx.T.copy()


def _encode_bool_grid_raw_xy(arr: np.ndarray) -> bytes:
    """Encode a (width, height) bool grid as the RA3 bit-packed format."""
    bits_yx = np.asarray(arr, dtype=np.bool_).T  # (height, width)
    packed = np.packbits(bits_yx, axis=1, bitorder="little")
    return packed.tobytes()


def _rotate_world_xy(
    x: float,
    y: float,
    map_width_tiles: int,
    map_height_tiles: int,
    turns_cw: int,
) -> Tuple[float, float]:
    """Rotate a world-space point within playable-area bounds (tile-space * 10)."""
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return (x, y)

    max_x = float(map_width_tiles) * WORLD_UNITS_PER_TILE
    max_y = float(map_height_tiles) * WORLD_UNITS_PER_TILE

    if turns_cw == 1:
        return (y, max_x - x)
    if turns_cw == 2:
        return (max_x - x, max_y - y)
    return (max_y - y, x)


def _rotate_angle_degrees(angle_deg: float, turns_cw: int) -> float:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return angle_deg
    return (angle_deg - 90.0 * turns_cw) % 360.0


# --- BlendDirection rotation (handles non-enum combos like 33,49 that appear in real maps) ---
_ROT_180 = {
    1: 17,
    17: 1,
    2: 18,
    18: 2,
    4: 24,
    24: 4,
    8: 20,
    20: 8,
    36: 56,
    56: 36,
    40: 52,
    52: 40,
    33: 49,  # observed: Left|0x20 -> Right|0x20
    49: 33,
}

_ROT_90 = {
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
    33: 50,  # 0x21 -> 0x32 (best-effort based on observed usage)
    50: 49,
    49: 34,
    34: 33,
}


def _rotate_blend_direction_value(val: int, turns_cw: int) -> int:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return val
    if turns_cw == 2:
        return _ROT_180.get(val, val)
    out = val
    for _ in range(turns_cw):
        out = _ROT_90.get(out, out)
    return out


def _rotate_blend_direction(dir_in: BlendDirection, turns_cw: int) -> BlendDirection:
    return BlendDirection(_rotate_blend_direction_value(int(dir_in), turns_cw))


# --- BlendTileData helpers (C# semantics) ---
def _get_tile_value(x: int, y: int, texture: int) -> int:
    """C# BlendTileData.GetTile(x,y,texture)."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return current + 64 * texture


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """Inverse of GetTile at a given (x,y): returns texture index."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _rotate_xy_index(old_x: int, old_y: int, old_w: int, old_h: int, turns_cw: int) -> Tuple[int, int]:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return old_x, old_y
    if turns_cw == 1:
        return old_y, old_w - 1 - old_x
    if turns_cw == 2:
        return old_w - 1 - old_x, old_h - 1 - old_y
    return old_h - 1 - old_y, old_x


def rotate_context_right_angles(context, degrees: int, clockwise: bool = True) -> None:
    spec = RotationSpec(degrees=degrees, clockwise=clockwise)
    turns_cw = spec.turns_cw
    if turns_cw == 0:
        return

    old_w_full = int(context.map_width)
    old_h_full = int(context.map_height)
    if old_w_full <= 0 or old_h_full <= 0:
        raise ValueError(f"Invalid context dimensions: map_width={old_w_full}, map_height={old_h_full}")

    new_w_full, new_h_full = (old_h_full, old_w_full) if turns_cw % 2 == 1 else (old_w_full, old_h_full)

    # --- HeightMapData ---
    from ..assets.terrain.height_map_data import HeightMapData

    height: Optional[HeightMapData] = context.get_asset_by_type(HeightMapData)
    if height is not None:
        height._elevations_raw = _rotate_grid_xy(height._elevations_raw, turns_cw)
        height.elevations = _rotate_grid_xy(height.elevations, turns_cw)
        height.map_width = new_w_full
        height.map_height = new_h_full
        height.playable_width = height.map_width - 2 * height.border_width
        height.playable_height = height.map_height - 2 * height.border_width
        height.area = height.map_width * height.map_height

    # Update context dims for assets that read from context
    context.map_width = new_w_full
    context.map_height = new_h_full

    # --- BlendTileData ---
    from ..assets.terrain.blend_tile_data import BlendTileData
    from ..assets.terrain.passability import Passability
    from ..assets.terrain.blend_info import BlendInfo

    blend: Optional[BlendTileData] = context.get_asset_by_type(BlendTileData)
    if blend is not None:
        old_tiles = np.asarray(blend.tiles)
        old_blends = np.asarray(blend.blends)
        old_se = np.asarray(blend.single_edge_blends)
        old_info = list(blend.blend_info or [])

        # new grids
        new_tiles = np.zeros((new_w_full, new_h_full), dtype=old_tiles.dtype)
        new_blends = np.zeros((new_w_full, new_h_full), dtype=old_blends.dtype)
        new_se = np.zeros((new_w_full, new_h_full), dtype=old_se.dtype)

        # rebuild blend_info using exact per-cell remap (prevents “10% wrong blends”)
        new_info: List[BlendInfo] = []
        key_to_idx: dict[Tuple[int, int, int, int], int] = {}  # (sec_tile, dir, i3, i4) -> 1-based idx

        def get_or_add(sec_tile: int, dir_val: int, i3: int, i4: int) -> int:
            k = (int(sec_tile), int(dir_val), int(i3), int(i4))
            idx = key_to_idx.get(k)
            if idx is not None:
                return idx
            bi = BlendInfo()
            bi.secondary_texture_tile = int(sec_tile)
            bi.blend_direction = BlendDirection(int(dir_val))
            bi._blend_direction_raw = bi._from_blend_direction(bi.blend_direction)
            bi.i3 = int(i3)
            bi.i4 = int(i4)
            new_info.append(bi)
            idx = len(new_info)  # 1-based
            key_to_idx[k] = idx
            return idx

        # rotate tile + blend layers
        for ox in range(old_w_full):
            for oy in range(old_h_full):
                nx, ny = _rotate_xy_index(ox, oy, old_w_full, old_h_full, turns_cw)

                # tiles: recompute by texture index at old pos
                tex = _get_texture_from_tile(ox, oy, int(old_tiles[ox, oy]))
                new_tiles[nx, ny] = np.uint16(_get_tile_value(nx, ny, tex))

                # blends layer
                old_idx = int(old_blends[ox, oy])
                if old_idx > 0 and 1 <= old_idx <= len(old_info):
                    oi = old_info[old_idx - 1]
                    sec_tex = _get_texture_from_tile(ox, oy, int(oi.secondary_texture_tile))
                    sec_tile_new = _get_tile_value(nx, ny, sec_tex)
                    dir_new = _rotate_blend_direction(oi.blend_direction, turns_cw)
                    new_idx = get_or_add(sec_tile_new, int(dir_new), oi.i3, oi.i4)
                    new_blends[nx, ny] = np.uint16(new_idx)

                # single edge blends layer (same indexing scheme)
                old_idx2 = int(old_se[ox, oy])
                if old_idx2 > 0 and 1 <= old_idx2 <= len(old_info):
                    oi2 = old_info[old_idx2 - 1]
                    sec_tex2 = _get_texture_from_tile(ox, oy, int(oi2.secondary_texture_tile))
                    sec_tile_new2 = _get_tile_value(nx, ny, sec_tex2)
                    dir_new2 = _rotate_blend_direction(oi2.blend_direction, turns_cw)
                    new_idx2 = get_or_add(sec_tile_new2, int(dir_new2), oi2.i3, oi2.i4)
                    new_se[nx, ny] = np.uint16(new_idx2)

        blend.map_width = new_w_full
        blend.map_height = new_h_full
        blend.area = new_w_full * new_h_full
        blend.tiles = new_tiles
        blend.blends = new_blends
        blend.single_edge_blends = new_se

        # rotate the rest (pure grids)
        blend.cliff_blends = _rotate_grid_xy(blend.cliff_blends, turns_cw)
        blend.dynamic_shrubbery = _rotate_grid_xy(blend.dynamic_shrubbery, turns_cw)
        blend.passability = _rotate_grid_xy(blend.passability, turns_cw)

        if blend.passability is not None:
            p = np.asarray(blend.passability, dtype=np.int32)
            impassable = p == int(Passability.Impassable)
            impassable_to_players = p == int(Passability.ImpassableToPlayers)
            impassable_to_air_units = p == int(Passability.ImpassableToAirUnits)
            extra_passable = p == int(Passability.ExtraPassable)

            blend.impassable = impassable
            blend.passage_width = _rotate_grid_xy(blend.passage_width, turns_cw)
            blend.visibility = _rotate_grid_xy(blend.visibility, turns_cw)
            blend.buildability = _rotate_grid_xy(blend.buildability, turns_cw)
            blend.tiberium_growability = _rotate_grid_xy(blend.tiberium_growability, turns_cw)

            blend._impassable_raw = _encode_bool_grid_raw_xy(impassable)
            blend._impassable_to_players_raw = _encode_bool_grid_raw_xy(impassable_to_players)
            blend._extra_passable_raw = _encode_bool_grid_raw_xy(extra_passable)
            blend._impassable_to_air_units_raw = _encode_bool_grid_raw_xy(impassable_to_air_units)
            if blend.passage_width is not None:
                blend._passage_width_raw = _encode_bool_grid_raw_xy(blend.passage_width)
            if blend.visibility is not None:
                blend._visibility_raw = _encode_bool_grid_raw_xy(blend.visibility)
            if blend.buildability is not None:
                blend._buildability_raw = _encode_bool_grid_raw_xy(blend.buildability)
            if blend.tiberium_growability is not None:
                blend._tiberium_growability_raw = _encode_bool_grid_raw_xy(blend.tiberium_growability)

        blend.blend_info = new_info
        blend.blends_count = len(new_info)

    # --- ObjectsList (positions + yaw) ---
    from ..assets.objects.objects_list import ObjectsList

    border_tiles = int(getattr(context, "border", 0) or 0)
    old_play_w = int(getattr(height, "playable_width", old_w_full - 2 * border_tiles)) if height is not None else int(old_w_full - 2 * border_tiles)
    old_play_h = int(getattr(height, "playable_height", old_h_full - 2 * border_tiles)) if height is not None else int(old_h_full - 2 * border_tiles)
    if old_play_w <= 0 or old_play_h <= 0:
        old_play_w, old_play_h = old_w_full, old_h_full

    objs: Optional[ObjectsList] = context.get_asset_by_type(ObjectsList)
    if objs is not None:
        for obj in objs.map_objects:
            x, y, z = obj.position
            nx, ny = _rotate_world_xy(x, y, old_play_w, old_play_h, turns_cw)
            obj.position = (float(nx), float(ny), float(z))
            obj.angle = float(_rotate_angle_degrees(obj.angle, turns_cw))

    # --- Water areas / rivers / waves / triggers ---
    def _rotate_points_in_place(points: List[Tuple[float, float]]) -> None:
        for i, (px, py) in enumerate(points):
            rx, ry = _rotate_world_xy(px, py, old_play_w, old_play_h, turns_cw)
            points[i] = (float(rx), float(ry))

    from ..assets.water.standing_water_areas import StandingWaterAreas
    from ..assets.water.standing_wave_areas import StandingWaveAreas
    from ..assets.water.river_areas import RiverAreas
    from ..assets.triggers.trigger_areas import TriggerAreas

    swa = context.get_asset_by_type(StandingWaterAreas)
    if swa is not None:
        for area in swa.water_areas:
            _rotate_points_in_place(area.points)

    swv = context.get_asset_by_type(StandingWaveAreas)
    if swv is not None:
        for area in swv.areas:
            _rotate_points_in_place(area.points)

    rivers = context.get_asset_by_type(RiverAreas)
    if rivers is not None:
        for area in rivers.areas:
            _rotate_points_in_place(area.points)

    trig = context.get_asset_by_type(TriggerAreas)
    if trig is not None:
        for area in trig.areas:
            _rotate_points_in_place(area.points)

    # --- Script argument positions (if any) ---
    from ..assets.scripts.player_scripts_list import PlayerScriptsList
    from ..assets.scripts.script_argument import ScriptArgument

    ps = context.get_asset_by_type(PlayerScriptsList)
    if ps is not None:
        for sl in ps.script_lists:
            for script in getattr(sl, "scripts", []):
                for action in getattr(script, "script_action_on_true", []):
                    for arg in getattr(action, "arguments", []):
                        if isinstance(arg, ScriptArgument) and arg.argument_type == 16:
                            ax, ay, az = arg.position
                            rx, ry = _rotate_world_xy(ax, ay, old_play_w, old_play_h, turns_cw)
                            arg.position = (float(rx), float(ry), float(az))
                for action in getattr(script, "script_action_on_false", []):
                    for arg in getattr(action, "arguments", []):
                        if isinstance(arg, ScriptArgument) and arg.argument_type == 16:
                            ax, ay, az = arg.position
                            rx, ry = _rotate_world_xy(ax, ay, old_play_w, old_play_h, turns_cw)
                            arg.position = (float(rx), float(ry), float(az))
