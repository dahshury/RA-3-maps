"""
Map composition utilities for RA3 maps.

Composes one or more parsed maps into a single canvas by placing each map at a
chosen layout position. Composition is a strict superset of duplication:

  - "Duplication" = every layout cell points at the same source map. Use
    `duplicate_context(ctx, nx, ny, ...)` for this case.
  - "Composition" = layout cells point at potentially different source maps.
    Use `compose_context(target_ctx, spec)` for this case. Presets such as
    `row`, `col`, and the four `triangle_*` shapes lower into a free-form
    spec (see `presets`).

Why playable-only stitching: simply tiling the full grid puts each source's
border inside the new map, which breaks pathing (mid-map impassable strips
between players). Playable-only stitching concatenates source playable areas
seamlessly; the new canvas keeps a single outer border drawn from each
top/bottom/left/right source's own border range.

Mirrors the bookkeeping `map_rotation.py` does for a single source, just per-
source for multi-source composition:
- HeightMapData (elevations + dimensions)
- BlendTileData (tiles + blends + single_edge_blends + boolean grids +
  blend_info + texture table). Texture tables are merged across sources by
  name and `cell_start` is reassigned to `i*64` so per-cell tile encoding
  stays consistent.
- ObjectsList (positions; player-start unique IDs renumbered globally; other
  unique IDs reissued with a fresh global sequence so WB accepts them)
- Water areas / rivers / waves / trigger polygons (replicated per source with
  per-source offsets)
- SidesList / Teams / PlayerScriptsList / BuildLists (each new player's
  templates come from THAT player's source map, not the canvas, so faction
  defaults from the canvas don't leak into other sources' players)

Note: tile-value encoding depends on (x%8, y%8). Tile values must be
recomputed at the new (gx, gy) coordinates -- not just copied -- exactly
like rotation.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..assets.terrain.blend_direction import BlendDirection

WORLD_UNITS_PER_TILE = 10.0

PLAYER_START_IDS = {
    "Player_1_Start",
    "Player_2_Start",
    "Player_3_Start",
    "Player_4_Start",
    "Player_5_Start",
    "Player_6_Start",
}

MAX_PLAYERS = 6


# ---------------------------------------------------------------------------
# Public composition spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompositionCell:
    """One layout slot.

    Coordinates are in *layout-grid units* (not tiles). `col` and `row` are the
    cell's anchor, `span_cols` / `span_rows` extend the cell across multiple
    grid positions (used by the triangle presets so the lone source spans the
    full row/column width). `align_x` / `align_y` control where the source's
    playable region sits inside its allocation when the source is smaller than
    the column-width / row-height max ("center" | "left" | "right" /
    "center" | "top" | "bottom").

    The `source` field is a parsed `MapDataContext` (e.g. from
    `Ra3Map.parse(); m.get_context()`). Multiple cells may share the same
    `source` -- the duplicate case is exactly that.
    """
    source: object  # MapDataContext (avoid hard import to skip cycles)
    col: int = 0
    row: int = 0
    span_cols: int = 1
    span_rows: int = 1
    align_x: str = "center"
    align_y: str = "center"

    def __post_init__(self) -> None:
        if self.col < 0 or self.row < 0:
            raise ValueError(f"col/row must be >= 0, got col={self.col}, row={self.row}")
        if self.span_cols < 1 or self.span_rows < 1:
            raise ValueError(
                f"span_cols/span_rows must be >= 1, got "
                f"span_cols={self.span_cols}, span_rows={self.span_rows}"
            )
        if self.align_x not in ("center", "left", "right"):
            raise ValueError(f"align_x must be center|left|right, got {self.align_x}")
        if self.align_y not in ("center", "top", "bottom"):
            raise ValueError(f"align_y must be center|top|bottom, got {self.align_y}")


@dataclass(frozen=True)
class CompositionSpec:
    """A list of cells plus inter-cell padding (water-filled gaps in tiles)."""
    cells: Tuple[CompositionCell, ...]
    pad_x: int = 0
    pad_y: int = 0

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("CompositionSpec.cells must not be empty")
        if self.pad_x < 0 or self.pad_y < 0:
            raise ValueError(f"padding must be >= 0, got pad_x={self.pad_x}, pad_y={self.pad_y}")


# ---------------------------------------------------------------------------
# Preset builders -- each returns a CompositionSpec from a list of source
# contexts. The slot ordering mirrors what `compose_map.py --preset NAME
# --maps A B C ...` accepts.
# ---------------------------------------------------------------------------

class presets:  # namespace
    @staticmethod
    def duplicate(source, nx: int, ny: int, pad_x: int = 0, pad_y: int = 0) -> CompositionSpec:
        """N x M grid of the same source (the original duplication case)."""
        if nx < 1 or ny < 1:
            raise ValueError(f"nx/ny must be >= 1, got nx={nx}, ny={ny}")
        cells = tuple(
            CompositionCell(source=source, col=i, row=j)
            for j in range(ny) for i in range(nx)
        )
        return CompositionSpec(cells=cells, pad_x=pad_x, pad_y=pad_y)

    @staticmethod
    def row(sources, pad_x: int = 0) -> CompositionSpec:
        """1xN row -- left to right."""
        cells = tuple(CompositionCell(source=s, col=i, row=0) for i, s in enumerate(sources))
        return CompositionSpec(cells=cells, pad_x=pad_x, pad_y=0)

    @staticmethod
    def col(sources, pad_y: int = 0) -> CompositionSpec:
        """Nx1 column -- top to bottom."""
        cells = tuple(CompositionCell(source=s, col=0, row=j) for j, s in enumerate(sources))
        return CompositionSpec(cells=cells, pad_x=0, pad_y=pad_y)

    @staticmethod
    def triangle_top(a, b, c, pad_x: int = 0, pad_y: int = 0) -> CompositionSpec:
        """2 maps top + 1 bottom-centered (spans both columns)."""
        return CompositionSpec(cells=(
            CompositionCell(source=a, col=0, row=0),
            CompositionCell(source=b, col=1, row=0),
            CompositionCell(source=c, col=0, row=1, span_cols=2),
        ), pad_x=pad_x, pad_y=pad_y)

    @staticmethod
    def triangle_bottom(a, b, c, pad_x: int = 0, pad_y: int = 0) -> CompositionSpec:
        """1 map top-centered (spans both columns) + 2 maps bottom."""
        return CompositionSpec(cells=(
            CompositionCell(source=a, col=0, row=0, span_cols=2),
            CompositionCell(source=b, col=0, row=1),
            CompositionCell(source=c, col=1, row=1),
        ), pad_x=pad_x, pad_y=pad_y)

    @staticmethod
    def triangle_left(a, b, c, pad_x: int = 0, pad_y: int = 0) -> CompositionSpec:
        """2 maps left (top + bottom) + 1 map right-centered (spans both rows)."""
        return CompositionSpec(cells=(
            CompositionCell(source=a, col=0, row=0),
            CompositionCell(source=b, col=0, row=1),
            CompositionCell(source=c, col=1, row=0, span_rows=2),
        ), pad_x=pad_x, pad_y=pad_y)

    @staticmethod
    def triangle_right(a, b, c, pad_x: int = 0, pad_y: int = 0) -> CompositionSpec:
        """1 map left-centered (spans both rows) + 2 maps right (top + bottom)."""
        return CompositionSpec(cells=(
            CompositionCell(source=a, col=0, row=0, span_rows=2),
            CompositionCell(source=b, col=1, row=0),
            CompositionCell(source=c, col=1, row=1),
        ), pad_x=pad_x, pad_y=pad_y)


# ---------------------------------------------------------------------------
# Tile-encoding helpers -- pure functions, identical to the rotation module's
# semantics. Inlined here so this file is self-contained.
# ---------------------------------------------------------------------------

def _get_tile_value(x: int, y: int, texture: int) -> int:
    """C# BlendTileData.GetTile(x, y, texture). Encodes a tile value at (x, y)
    for the given (global) texture index."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return current + 64 * texture


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """Inverse of GetTile at (x, y): returns the (local) texture index."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _encode_sage_float16(h_world: float) -> int:
    """Encode a world-space elevation into RA3's custom SageFloat16 uint16."""
    h = max(0.0, float(h_world))
    upper = min(255, int(h // 10))
    residual = h - upper * 10.0
    lower = int(round(residual * 256.0 / 9.96))
    lower = min(255, max(0, lower))
    return (upper << 8) | lower


def _encode_bool_grid_raw_xy(arr: np.ndarray) -> bytes:
    """Encode a (W, H) bool grid as the RA3 bit-packed raw format."""
    bits_yx = np.asarray(arr, dtype=np.bool_).T  # (H, W)
    packed = np.packbits(bits_yx, axis=1, bitorder="little")
    return packed.tobytes()


def _clone_property_collection(src, target_ctx=None):
    """Shallow-clone an AssetPropertyCollection so we can mutate per-copy fields.

    When target_ctx is given, every property's `id` is re-registered in
    target_ctx's string pool. This is required when cloning from a different
    source context (whose string-pool indices don't match the canvas's).
    For the duplicate case (target_ctx is the source) this is idempotent.
    """
    from ..assets.assets.asset_property import AssetProperty, AssetPropertyCollection

    out = AssetPropertyCollection()
    for name, prop in src.property_map.items():
        np_ = AssetProperty()
        np_.property_type = prop.property_type
        np_.name = prop.name
        np_.data = copy.copy(prop.data)
        if target_ctx is not None and prop.name:
            np_.id = target_ctx.map_struct.register_string(prop.name)
        else:
            np_.id = prop.id
        out.property_map[name] = np_
    return out


def _set_property_data(prop_collection, name: str, data) -> None:
    """Set a property's data if it exists (mutates in place)."""
    prop = prop_collection.property_map.get(name)
    if prop is not None:
        prop.data = data


# ---------------------------------------------------------------------------
# Source snapshot -- read all needed data from each unique source ctx BEFORE
# we mutate the canvas. This way duplication (canvas == source) and
# composition (canvas is one of the sources) both work uniformly.
# ---------------------------------------------------------------------------

@dataclass
class _SourceSnapshot:
    """Frozen pre-mutation view of one source map's relevant data."""
    ctx: object
    map_w: int
    map_h: int
    border: int
    pw: int
    ph: int

    # HeightMap
    heights_raw: Optional[np.ndarray] = None
    heights_float: Optional[np.ndarray] = None
    min_elev_world: float = 0.0

    # BlendTileData
    tiles: Optional[np.ndarray] = None
    blends: Optional[np.ndarray] = None
    single_edge_blends: Optional[np.ndarray] = None
    cliff_blends: Optional[np.ndarray] = None
    dynamic_shrubbery: Optional[np.ndarray] = None
    passability: Optional[np.ndarray] = None
    passage_width: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None
    buildability: Optional[np.ndarray] = None
    tiberium_growability: Optional[np.ndarray] = None
    blend_info: List[object] = field(default_factory=list)
    textures: List[object] = field(default_factory=list)

    # Per-player metadata templates -- snapshotted as live references because
    # we always *clone* before appending to the canvas. We MUST capture these
    # before we mutate target_ctx (else the duplicate case wipes them out).
    sides_players: List[object] = field(default_factory=list)  # SidesList.players
    teams_list: List[object] = field(default_factory=list)     # Teams.teams
    script_lists: List[object] = field(default_factory=list)   # PlayerScriptsList.script_lists
    build_lists_list: List[object] = field(default_factory=list)  # BuildLists.build_list

    # Polygon-area lists. Same reason as the player templates: in the
    # duplicate case the canvas IS every source, and the area-replicate pass
    # clears canvas.water_areas before reading -- so without a snapshot we'd
    # iterate an empty list and lose every original water area.
    swa_areas: List[object] = field(default_factory=list)
    swv_areas: List[object] = field(default_factory=list)
    river_areas: List[object] = field(default_factory=list)
    trigger_areas: List[object] = field(default_factory=list)


def _snapshot_source(ctx) -> _SourceSnapshot:
    from ..assets.terrain.height_map_data import HeightMapData
    from ..assets.terrain.blend_tile_data import BlendTileData
    from ..assets.sides.sides_list import SidesList
    from ..assets.teams.teams import Teams
    from ..assets.scripts.player_scripts_list import PlayerScriptsList
    from ..assets.build.build_lists import BuildLists
    from ..assets.water.standing_water_areas import StandingWaterAreas
    from ..assets.water.standing_wave_areas import StandingWaveAreas
    from ..assets.water.river_areas import RiverAreas
    from ..assets.triggers.trigger_areas import TriggerAreas

    border = int(getattr(ctx, "border", 0) or 0)
    map_w = int(ctx.map_width)
    map_h = int(ctx.map_height)
    pw = map_w - 2 * border
    ph = map_h - 2 * border

    snap = _SourceSnapshot(ctx=ctx, map_w=map_w, map_h=map_h, border=border, pw=pw, ph=ph)

    height: Optional[HeightMapData] = ctx.get_asset_by_type(HeightMapData)
    if height is not None:
        if height._elevations_raw is not None:
            snap.heights_raw = np.array(height._elevations_raw, copy=True)
        if height.elevations is not None:
            snap.heights_float = np.array(height.elevations, copy=True)
            snap.min_elev_world = float(np.min(height.elevations))

    blend: Optional[BlendTileData] = ctx.get_asset_by_type(BlendTileData)
    if blend is not None:
        snap.tiles = np.array(blend.tiles, copy=True) if blend.tiles is not None else None
        snap.blends = np.array(blend.blends, copy=True) if blend.blends is not None else None
        snap.single_edge_blends = (
            np.array(blend.single_edge_blends, copy=True)
            if blend.single_edge_blends is not None else None
        )
        snap.cliff_blends = (
            np.array(blend.cliff_blends, copy=True)
            if blend.cliff_blends is not None else None
        )
        snap.dynamic_shrubbery = (
            np.array(blend.dynamic_shrubbery, copy=True)
            if blend.dynamic_shrubbery is not None else None
        )
        snap.passability = (
            np.array(blend.passability, copy=True)
            if blend.passability is not None else None
        )
        snap.passage_width = (
            np.array(blend.passage_width, copy=True)
            if blend.passage_width is not None else None
        )
        snap.visibility = (
            np.array(blend.visibility, copy=True) if blend.visibility is not None else None
        )
        snap.buildability = (
            np.array(blend.buildability, copy=True)
            if blend.buildability is not None else None
        )
        snap.tiberium_growability = (
            np.array(blend.tiberium_growability, copy=True)
            if blend.tiberium_growability is not None else None
        )
        snap.blend_info = list(blend.blend_info or [])
        snap.textures = list(blend.textures or [])

    sides = ctx.get_asset_by_type(SidesList)
    if sides is not None:
        snap.sides_players = list(sides.players or [])
    teams = ctx.get_asset_by_type(Teams)
    if teams is not None:
        snap.teams_list = list(teams.teams or [])
    psl = ctx.get_asset_by_type(PlayerScriptsList)
    if psl is not None:
        snap.script_lists = list(psl.script_lists or [])
    blists = ctx.get_asset_by_type(BuildLists)
    if blists is not None:
        snap.build_lists_list = list(blists.build_list or [])

    swa = ctx.get_asset_by_type(StandingWaterAreas)
    if swa is not None:
        snap.swa_areas = list(swa.water_areas or [])
    swv = ctx.get_asset_by_type(StandingWaveAreas)
    if swv is not None:
        snap.swv_areas = list(swv.areas or [])
    rivers = ctx.get_asset_by_type(RiverAreas)
    if rivers is not None:
        snap.river_areas = list(rivers.areas or [])
    trig = ctx.get_asset_by_type(TriggerAreas)
    if trig is not None:
        snap.trigger_areas = list(trig.areas or [])

    return snap


# ---------------------------------------------------------------------------
# Layout solver
# ---------------------------------------------------------------------------

@dataclass
class _CellLayout:
    """Resolved per-cell placement on the canvas, in tile units."""
    cell: CompositionCell
    snap: _SourceSnapshot
    col: int
    row: int
    span_cols: int
    span_rows: int
    # Allocated playable region on canvas (in tiles, 0-based, excluding outer
    # canvas border):
    dst_x_alloc: int
    dst_y_alloc: int
    dst_w_alloc: int
    dst_h_alloc: int
    # Where the source's playable region sits inside its allocation (alignment).
    src_align_dx: int
    src_align_dy: int


@dataclass
class _Layout:
    border: int
    n_cols: int
    n_rows: int
    col_widths: List[int]
    row_heights: List[int]
    pad_x: int
    pad_y: int
    new_pw: int
    new_ph: int
    new_w: int
    new_h: int
    cells: List[_CellLayout]
    grid: np.ndarray
    top_cell_for_col: List[int]
    bot_cell_for_col: List[int]
    left_cell_for_row: List[int]
    right_cell_for_row: List[int]
    col_dst_x: List[int]
    row_dst_y: List[int]


def _solve_layout(spec: CompositionSpec, snaps: Dict[int, _SourceSnapshot]) -> _Layout:
    """Compute the canvas layout from a CompositionSpec."""
    cells = list(spec.cells)
    if not cells:
        raise ValueError("CompositionSpec must have at least one cell")

    n_cols = max(c.col + c.span_cols for c in cells)
    n_rows = max(c.row + c.span_rows for c in cells)

    # Pass 1: per-column max width = max source playable width among
    # *non-spanning* cells anchored at that column. Same for rows.
    col_widths = [0] * n_cols
    row_heights = [0] * n_rows
    for c in cells:
        snap = snaps[id(c.source)]
        if c.span_cols == 1:
            col_widths[c.col] = max(col_widths[c.col], snap.pw)
        if c.span_rows == 1:
            row_heights[c.row] = max(row_heights[c.row], snap.ph)

    # Pass 2: redistribute deficits from spanning cells whose width / height
    # exceeds the sum of the columns / rows they span.
    def _redistribute(values: List[int], start: int, span: int, required: int) -> None:
        current = sum(values[start:start + span])
        if required <= current:
            return
        deficit = required - current
        zeros = [i for i in range(start, start + span) if values[i] == 0]
        if zeros:
            per = deficit // len(zeros)
            rem = deficit - per * len(zeros)
            for i, idx in enumerate(zeros):
                values[idx] += per + (1 if i < rem else 0)
        else:
            per = deficit // span
            rem = deficit - per * span
            for i in range(span):
                values[start + i] += per + (1 if i < rem else 0)

    for c in cells:
        snap = snaps[id(c.source)]
        if c.span_cols > 1:
            _redistribute(col_widths, c.col, c.span_cols, snap.pw)
        if c.span_rows > 1:
            _redistribute(row_heights, c.row, c.span_rows, snap.ph)

    pad_x = spec.pad_x if n_cols > 1 else 0
    pad_y = spec.pad_y if n_rows > 1 else 0

    new_pw = sum(col_widths) + (n_cols - 1) * pad_x
    new_ph = sum(row_heights) + (n_rows - 1) * pad_y

    # Pick the canvas border from the cell at (0, 0). All sources should have
    # the same border thickness in practice; warn-but-continue if they differ.
    primary = next(c for c in cells if c.col == 0 and c.row == 0)
    primary_snap = snaps[id(primary.source)]
    border = primary_snap.border
    for c in cells:
        s = snaps[id(c.source)]
        if s.border != border:
            print(
                f"[map_composition] WARNING: source border={s.border} differs "
                f"from canvas border={border}; using canvas border"
            )

    new_w = new_pw + 2 * border
    new_h = new_ph + 2 * border

    col_dst_x = [0] * n_cols
    acc = 0
    for i, w in enumerate(col_widths):
        col_dst_x[i] = acc
        acc += w + (pad_x if i < n_cols - 1 else 0)

    row_dst_y = [0] * n_rows
    acc = 0
    for j, h in enumerate(row_heights):
        row_dst_y[j] = acc
        acc += h + (pad_y if j < n_rows - 1 else 0)

    cell_layouts: List[_CellLayout] = []
    grid = np.full((n_cols, n_rows), -1, dtype=np.int64)
    for idx, c in enumerate(cells):
        snap = snaps[id(c.source)]
        dst_w_alloc = (
            sum(col_widths[c.col:c.col + c.span_cols])
            + (c.span_cols - 1) * pad_x
        )
        dst_h_alloc = (
            sum(row_heights[c.row:c.row + c.span_rows])
            + (c.span_rows - 1) * pad_y
        )
        slack_x = max(0, dst_w_alloc - snap.pw)
        slack_y = max(0, dst_h_alloc - snap.ph)
        if c.align_x == "left":
            sx = 0
        elif c.align_x == "right":
            sx = slack_x
        else:
            sx = slack_x // 2
        if c.align_y == "top":
            sy = 0
        elif c.align_y == "bottom":
            sy = slack_y
        else:
            sy = slack_y // 2
        layout = _CellLayout(
            cell=c, snap=snap,
            col=c.col, row=c.row,
            span_cols=c.span_cols, span_rows=c.span_rows,
            dst_x_alloc=col_dst_x[c.col],
            dst_y_alloc=row_dst_y[c.row],
            dst_w_alloc=dst_w_alloc,
            dst_h_alloc=dst_h_alloc,
            src_align_dx=sx,
            src_align_dy=sy,
        )
        cell_layouts.append(layout)
        for ci in range(c.col, c.col + c.span_cols):
            for ri in range(c.row, c.row + c.span_rows):
                if grid[ci, ri] != -1:
                    raise ValueError(
                        f"layout cells overlap at (col={ci}, row={ri}); "
                        f"cell {idx} conflicts with cell {grid[ci, ri]}"
                    )
                grid[ci, ri] = idx

    def _first_nonneg(seq):
        for v in seq:
            if v != -1:
                return int(v)
        return -1

    def _last_nonneg(seq):
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != -1:
                return int(seq[i])
        return -1

    top_cell_for_col = [_first_nonneg(grid[c, :]) for c in range(n_cols)]
    bot_cell_for_col = [_last_nonneg(grid[c, :]) for c in range(n_cols)]
    left_cell_for_row = [_first_nonneg(grid[:, r]) for r in range(n_rows)]
    right_cell_for_row = [_last_nonneg(grid[:, r]) for r in range(n_rows)]

    return _Layout(
        border=border,
        n_cols=n_cols, n_rows=n_rows,
        col_widths=col_widths, row_heights=row_heights,
        pad_x=pad_x, pad_y=pad_y,
        new_pw=new_pw, new_ph=new_ph,
        new_w=new_w, new_h=new_h,
        cells=cell_layouts,
        grid=grid,
        top_cell_for_col=top_cell_for_col,
        bot_cell_for_col=bot_cell_for_col,
        left_cell_for_row=left_cell_for_row,
        right_cell_for_row=right_cell_for_row,
        col_dst_x=col_dst_x,
        row_dst_y=row_dst_y,
    )


def _col_at_play_x(layout: _Layout, gx_play: int) -> int:
    """Return column idx for canvas-playable x; -1 if it falls in a pad gap
    or off the right edge."""
    for i in range(layout.n_cols):
        start = layout.col_dst_x[i]
        end = start + layout.col_widths[i]
        if start <= gx_play < end:
            return i
        if i < layout.n_cols - 1 and end <= gx_play < end + layout.pad_x:
            return -1
    return -1


def _row_at_play_y(layout: _Layout, gy_play: int) -> int:
    for i in range(layout.n_rows):
        start = layout.row_dst_y[i]
        end = start + layout.row_heights[i]
        if start <= gy_play < end:
            return i
        if i < layout.n_rows - 1 and end <= gy_play < end + layout.pad_y:
            return -1
    return -1


# ---------------------------------------------------------------------------
# Per-cell ownership lookup
# ---------------------------------------------------------------------------

def _lookup(layout: _Layout, gx: int, gy: int) -> Optional[Tuple[_CellLayout, int, int]]:
    """Map a canvas tile (gx, gy) to its owning source + local source coords.

    Returns (cell_layout, sx_full, sy_full) where sx_full / sy_full are tile
    indices within the source's full grid (i.e. include the source's own
    border). Returns None if the canvas tile falls in a pad-gap or alignment
    padding within a cell (water-fill).
    """
    border = layout.border
    new_w = layout.new_w
    new_h = layout.new_h

    in_top = gy < border
    in_bot = gy >= new_h - border
    in_left = gx < border
    in_right = gx >= new_w - border

    if in_top and in_left:
        cell_idx = layout.top_cell_for_col[0]
        if cell_idx == -1:
            cell_idx = layout.left_cell_for_row[0]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        return (cell, gx, gy)

    if in_top and in_right:
        cell_idx = layout.top_cell_for_col[layout.n_cols - 1]
        if cell_idx == -1:
            cell_idx = layout.right_cell_for_row[0]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        sx = cell.snap.map_w - (new_w - gx)
        return (cell, sx, gy)

    if in_bot and in_left:
        cell_idx = layout.bot_cell_for_col[0]
        if cell_idx == -1:
            cell_idx = layout.left_cell_for_row[layout.n_rows - 1]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        sy = cell.snap.map_h - (new_h - gy)
        return (cell, gx, sy)

    if in_bot and in_right:
        cell_idx = layout.bot_cell_for_col[layout.n_cols - 1]
        if cell_idx == -1:
            cell_idx = layout.right_cell_for_row[layout.n_rows - 1]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        sx = cell.snap.map_w - (new_w - gx)
        sy = cell.snap.map_h - (new_h - gy)
        return (cell, sx, sy)

    if in_top or in_bot:
        gx_play = gx - border
        col = _col_at_play_x(layout, gx_play)
        if col == -1:
            return None
        cell_idx = layout.top_cell_for_col[col] if in_top else layout.bot_cell_for_col[col]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        sx_play = (gx_play - cell.dst_x_alloc) - cell.src_align_dx
        if not (0 <= sx_play < cell.snap.pw):
            return None
        sx = sx_play + cell.snap.border
        if in_top:
            return (cell, sx, gy)
        sy = cell.snap.map_h - (new_h - gy)
        return (cell, sx, sy)

    if in_left or in_right:
        gy_play = gy - border
        row = _row_at_play_y(layout, gy_play)
        if row == -1:
            return None
        cell_idx = layout.left_cell_for_row[row] if in_left else layout.right_cell_for_row[row]
        if cell_idx == -1:
            return None
        cell = layout.cells[cell_idx]
        sy_play = (gy_play - cell.dst_y_alloc) - cell.src_align_dy
        if not (0 <= sy_play < cell.snap.ph):
            return None
        sy = sy_play + cell.snap.border
        if in_left:
            return (cell, gx, sy)
        sx = cell.snap.map_w - (new_w - gx)
        return (cell, sx, sy)

    # Interior playable
    gx_play = gx - border
    gy_play = gy - border
    col = _col_at_play_x(layout, gx_play)
    row = _row_at_play_y(layout, gy_play)
    if col == -1 or row == -1:
        return None
    cell_idx = int(layout.grid[col, row])
    if cell_idx == -1:
        return None
    cell = layout.cells[cell_idx]
    sx_play = (gx_play - cell.dst_x_alloc) - cell.src_align_dx
    sy_play = (gy_play - cell.dst_y_alloc) - cell.src_align_dy
    if not (0 <= sx_play < cell.snap.pw and 0 <= sy_play < cell.snap.ph):
        return None
    sx = sx_play + cell.snap.border
    sy = sy_play + cell.snap.border
    return (cell, sx, sy)


# ---------------------------------------------------------------------------
# Texture-table merge
# ---------------------------------------------------------------------------

def _merge_textures(layout: _Layout) -> Tuple[List[object], Dict[int, np.ndarray]]:
    """Merge texture tables from all unique source contexts.

    Returns:
      merged: a fresh list of Texture objects with `cell_start = i * 64`.
      per_source_remap: id(snap) -> int64 array of length len(snap.textures);
        remap[local_tex_idx] == global_tex_idx.
    """
    from ..assets.terrain.texture import Texture

    seen_by_name: Dict[str, int] = {}
    merged: List[object] = []
    per_source_remap: Dict[int, np.ndarray] = {}

    for cell in layout.cells:
        snap = cell.snap
        key = id(snap)
        if key in per_source_remap:
            continue
        if not snap.textures:
            per_source_remap[key] = np.zeros(0, dtype=np.int64)
            continue
        remap = np.zeros(len(snap.textures), dtype=np.int64)
        for local_idx, tex in enumerate(snap.textures):
            name = tex.name
            if name in seen_by_name:
                remap[local_idx] = seen_by_name[name]
                continue
            new_idx = len(merged)
            new_tex = Texture()
            new_tex.cell_start = new_idx * 64
            new_tex.cell_count = tex.cell_count
            new_tex.cell_size = tex.cell_size
            new_tex.magic_value = tex.magic_value
            new_tex.name = name
            merged.append(new_tex)
            seen_by_name[name] = new_idx
            remap[local_idx] = new_idx
        per_source_remap[key] = remap

    return merged, per_source_remap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compose_context(target_ctx, spec: CompositionSpec) -> None:
    """Compose `spec` into `target_ctx`, mutating it in place.

    `target_ctx` MUST be the source for the (col=0, row=0) cell -- the
    composer treats it as the canvas and rewrites its assets in-place.

    Raises:
        ValueError: if total players > MAX_PLAYERS, if a cell has < 2 players,
        or if cells overlap.
    """
    cells = list(spec.cells)

    primary_cells = [c for c in cells if c.col == 0 and c.row == 0]
    if not primary_cells:
        raise ValueError("CompositionSpec needs a cell anchored at (0, 0)")
    if not any(c.source is target_ctx for c in primary_cells):
        raise ValueError(
            "target_ctx must match the source of the (0, 0) cell so we can "
            "mutate it as the composition canvas"
        )

    snaps: Dict[int, _SourceSnapshot] = {}
    for c in cells:
        if id(c.source) not in snaps:
            snaps[id(c.source)] = _snapshot_source(c.source)

    for snap in snaps.values():
        if snap.map_w <= 0 or snap.map_h <= 0:
            raise ValueError(f"source has invalid dims: {snap.map_w}x{snap.map_h}")
        if snap.pw <= 0 or snap.ph <= 0:
            raise ValueError(
                f"source has invalid playable dims after border={snap.border}: "
                f"{snap.pw}x{snap.ph}"
            )

    layout = _solve_layout(spec, snaps)

    from ..assets.objects.objects_list import ObjectsList
    from ..assets.objects.map_object import MapObject

    per_cell_player_starts: List[List[Tuple[int, "MapObject"]]] = []
    total_players = 0
    for c in cells:
        cobjs: Optional[ObjectsList] = c.source.get_asset_by_type(ObjectsList)
        ps: List[Tuple[int, MapObject]] = []
        if cobjs is not None:
            for obj in cobjs.map_objects:
                uid = obj.unique_id
                if uid in PLAYER_START_IDS:
                    try:
                        n = int(uid.split("_")[1])
                    except Exception:
                        continue
                    ps.append((n, obj))
        if len(ps) < 2:
            raise ValueError(
                f"cell at (col={c.col}, row={c.row}) source has only {len(ps)} "
                f"player(s); all sources must have >= 2 players"
            )
        per_cell_player_starts.append(ps)
        total_players += len(ps)

    if total_players > MAX_PLAYERS:
        raise ValueError(
            f"refusing composition: total players = {total_players} > "
            f"MAX_PLAYERS ({MAX_PLAYERS})"
        )

    # Trivial: 1x1 layout with the single cell == target_ctx -- no-op.
    if (
        layout.n_cols == 1 and layout.n_rows == 1
        and len(cells) == 1 and cells[0].source is target_ctx
    ):
        return

    border = layout.border
    new_w = layout.new_w
    new_h = layout.new_h

    merged_textures, tex_remap = _merge_textures(layout)

    # ---- HeightMapData ----
    from ..assets.terrain.height_map_data import HeightMapData
    height: Optional[HeightMapData] = target_ctx.get_asset_by_type(HeightMapData)
    if height is not None:
        pad_elev_world = min(
            (s.min_elev_world for s in snaps.values() if s.heights_float is not None),
            default=0.0,
        )
        pad_raw = np.uint16(_encode_sage_float16(pad_elev_world))
        new_h_raw = np.full((new_w, new_h), pad_raw, dtype=np.uint16)
        new_h_float = np.full((new_w, new_h), float(pad_elev_world), dtype=np.float32)

        for gy in range(new_h):
            for gx in range(new_w):
                hit = _lookup(layout, gx, gy)
                if hit is None:
                    continue
                cell, sx, sy = hit
                snap = cell.snap
                if snap.heights_raw is not None:
                    new_h_raw[gx, gy] = snap.heights_raw[sx, sy]
                if snap.heights_float is not None:
                    new_h_float[gx, gy] = snap.heights_float[sx, sy]

        height._elevations_raw = new_h_raw
        height.elevations = new_h_float
        height.map_width = new_w
        height.map_height = new_h
        height.border_width = border
        height.playable_width = layout.new_pw
        height.playable_height = layout.new_ph
        height.area = new_w * new_h

        for hb in height.borders or []:
            hb.width = layout.new_pw
            hb.height = layout.new_ph

    target_ctx.map_width = new_w
    target_ctx.map_height = new_h

    # ---- BlendTileData ----
    from ..assets.terrain.blend_tile_data import BlendTileData
    from ..assets.terrain.passability import Passability
    from ..assets.terrain.blend_info import BlendInfo

    blend: Optional[BlendTileData] = target_ctx.get_asset_by_type(BlendTileData)
    if blend is not None:
        new_tiles = np.zeros((new_w, new_h), dtype=np.uint16)
        new_blends = np.zeros((new_w, new_h), dtype=np.uint16)
        new_se = np.zeros((new_w, new_h), dtype=np.uint16)
        new_cliff = np.zeros((new_w, new_h), dtype=np.uint16)
        new_shrub = np.zeros((new_w, new_h), dtype=np.uint8)
        new_pass = np.full((new_w, new_h), int(Passability.Passable), dtype=np.int32)
        new_pwidth = np.zeros((new_w, new_h), dtype=np.bool_)
        new_vis = np.ones((new_w, new_h), dtype=np.bool_)
        new_build = np.zeros((new_w, new_h), dtype=np.bool_)
        new_tib = np.zeros((new_w, new_h), dtype=np.bool_)

        new_info: List["BlendInfo"] = []
        key_to_idx: Dict[Tuple[int, int, int, int], int] = {}

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
            idx = len(new_info)
            key_to_idx[k] = idx
            return idx

        for gy in range(new_h):
            for gx in range(new_w):
                hit = _lookup(layout, gx, gy)
                if hit is None:
                    new_tiles[gx, gy] = np.uint16(_get_tile_value(gx, gy, 0))
                    continue
                cell, sx, sy = hit
                snap = cell.snap
                source_remap = tex_remap[id(snap)]

                if snap.tiles is not None:
                    local_tex = _get_texture_from_tile(sx, sy, int(snap.tiles[sx, sy]))
                    if 0 <= local_tex < len(source_remap):
                        global_tex = int(source_remap[local_tex])
                    else:
                        global_tex = 0
                    new_tiles[gx, gy] = np.uint16(_get_tile_value(gx, gy, global_tex))

                if snap.blends is not None and snap.blend_info:
                    old_idx = int(snap.blends[sx, sy])
                    if 1 <= old_idx <= len(snap.blend_info):
                        oi = snap.blend_info[old_idx - 1]
                        local_sec_tex = _get_texture_from_tile(
                            sx, sy, int(oi.secondary_texture_tile)
                        )
                        global_sec_tex = (
                            int(source_remap[local_sec_tex])
                            if 0 <= local_sec_tex < len(source_remap) else 0
                        )
                        sec_tile_new = _get_tile_value(gx, gy, global_sec_tex)
                        nidx = get_or_add(sec_tile_new, int(oi.blend_direction), oi.i3, oi.i4)
                        new_blends[gx, gy] = np.uint16(nidx)

                if snap.single_edge_blends is not None and snap.blend_info:
                    old_idx2 = int(snap.single_edge_blends[sx, sy])
                    if 1 <= old_idx2 <= len(snap.blend_info):
                        oi2 = snap.blend_info[old_idx2 - 1]
                        local_sec_tex2 = _get_texture_from_tile(
                            sx, sy, int(oi2.secondary_texture_tile)
                        )
                        global_sec_tex2 = (
                            int(source_remap[local_sec_tex2])
                            if 0 <= local_sec_tex2 < len(source_remap) else 0
                        )
                        sec_tile_new2 = _get_tile_value(gx, gy, global_sec_tex2)
                        nidx2 = get_or_add(sec_tile_new2, int(oi2.blend_direction), oi2.i3, oi2.i4)
                        new_se[gx, gy] = np.uint16(nidx2)

                if snap.cliff_blends is not None:
                    new_cliff[gx, gy] = snap.cliff_blends[sx, sy]
                if snap.dynamic_shrubbery is not None:
                    new_shrub[gx, gy] = snap.dynamic_shrubbery[sx, sy]
                if snap.passability is not None:
                    new_pass[gx, gy] = snap.passability[sx, sy]
                if snap.passage_width is not None:
                    new_pwidth[gx, gy] = snap.passage_width[sx, sy]
                if snap.visibility is not None:
                    new_vis[gx, gy] = snap.visibility[sx, sy]
                if snap.buildability is not None:
                    new_build[gx, gy] = snap.buildability[sx, sy]
                if snap.tiberium_growability is not None:
                    new_tib[gx, gy] = snap.tiberium_growability[sx, sy]

        if int(new_tiles.max()) > np.iinfo(np.uint16).max:
            raise ValueError("tile values overflow uint16; texture count too high?")

        blend.map_width = new_w
        blend.map_height = new_h
        blend.area = new_w * new_h
        blend.tiles = new_tiles
        blend.blends = new_blends
        blend.single_edge_blends = new_se
        blend.cliff_blends = new_cliff
        blend.dynamic_shrubbery = new_shrub
        blend.passability = new_pass
        blend.passage_width = new_pwidth
        blend.visibility = new_vis
        blend.buildability = new_build
        blend.tiberium_growability = new_tib
        blend.blend_info = new_info
        blend.blends_count = len(new_info)
        blend.textures = merged_textures

        passability = np.asarray(new_pass, dtype=np.int32)
        impassable = passability == int(Passability.Impassable)
        impassable_to_players = passability == int(Passability.ImpassableToPlayers)
        impassable_to_air_units = passability == int(Passability.ImpassableToAirUnits)
        extra_passable = passability == int(Passability.ExtraPassable)
        blend.impassable = impassable
        blend._impassable_raw = _encode_bool_grid_raw_xy(impassable)
        blend._impassable_to_players_raw = _encode_bool_grid_raw_xy(impassable_to_players)
        blend._extra_passable_raw = _encode_bool_grid_raw_xy(extra_passable)
        blend._impassable_to_air_units_raw = _encode_bool_grid_raw_xy(impassable_to_air_units)
        blend._passage_width_raw = _encode_bool_grid_raw_xy(new_pwidth)
        blend._visibility_raw = _encode_bool_grid_raw_xy(new_vis)
        blend._buildability_raw = _encode_bool_grid_raw_xy(new_build)
        blend._tiberium_growability_raw = _encode_bool_grid_raw_xy(new_tib)

    # ---- ObjectsList: replicate per cell with per-source offsets ----
    def _world_offset_for_cell(cell: _CellLayout) -> Tuple[float, float]:
        """World-space (dx, dy) to add to a source object's position for this
        cell. Source full-grid origin maps to canvas tile coords:
            (canvas_border + cell.dst_x_alloc + cell.src_align_dx - source_border)
        """
        snap = cell.snap
        dx_tile = layout.border + cell.dst_x_alloc + cell.src_align_dx - snap.border
        dy_tile = layout.border + cell.dst_y_alloc + cell.src_align_dy - snap.border
        return (dx_tile * WORLD_UNITS_PER_TILE, dy_tile * WORLD_UNITS_PER_TILE)

    target_objs: Optional[ObjectsList] = target_ctx.get_asset_by_type(ObjectsList)
    if target_objs is not None:
        new_objects: List[MapObject] = []
        uid_re = re.compile(r"^(?P<base>.*?)\s+(?P<num>\d+)$")

        max_seq = 0
        max_wp_id = 0
        for snap in snaps.values():
            objs_asset: Optional[ObjectsList] = snap.ctx.get_asset_by_type(ObjectsList)
            if objs_asset is None:
                continue
            for obj in objs_asset.map_objects:
                m = uid_re.match(obj.unique_id or "")
                if m:
                    try:
                        max_seq = max(max_seq, int(m.group("num")))
                    except ValueError:
                        pass
                wid_prop = obj.asset_property_collection.property_map.get("waypointID")
                if wid_prop and wid_prop.data is not None:
                    try:
                        max_wp_id = max(max_wp_id, int(wid_prop.data))
                    except (TypeError, ValueError):
                        pass
        next_seq = [max_seq + 1]
        wp_id_stride = max(max_wp_id, 1)

        global_player_n = 0
        for cell_idx, (cell_layout, cell_player_starts) in enumerate(
            zip(layout.cells, per_cell_player_starts)
        ):
            objs_asset = cell_layout.snap.ctx.get_asset_by_type(ObjectsList)
            if objs_asset is None:
                global_player_n += len(cell_player_starts)
                continue
            dx_world, dy_world = _world_offset_for_cell(cell_layout)

            local_to_global: Dict[int, int] = {}
            sorted_starts = sorted(cell_player_starts, key=lambda kv: kv[0])
            for offset, (local_n, _obj) in enumerate(sorted_starts, start=1):
                local_to_global[local_n] = global_player_n + offset
            global_player_n += len(sorted_starts)

            for obj in objs_asset.map_objects:
                new_obj = MapObject()
                # MajorAsset.id is a string-pool index for the asset name;
                # re-register in target so cross-source clones don't carry
                # stale indices from the source's own string pool.
                new_obj.name = obj.name
                new_obj.id = (
                    target_ctx.map_struct.register_string(obj.name)
                    if obj.name else int(obj.id)
                )
                new_obj.version = obj.version
                new_obj.position = (
                    float(obj.position[0]) + dx_world,
                    float(obj.position[1]) + dy_world,
                    float(obj.position[2]),
                )
                new_obj.angle = float(obj.angle)
                new_obj.road_option = int(obj.road_option)
                new_obj.type_name = obj.type_name
                new_obj.asset_property_collection = _clone_property_collection(
                    obj.asset_property_collection, target_ctx
                )

                old_uid = obj.unique_id
                new_uid: Optional[str] = None
                if old_uid in PLAYER_START_IDS:
                    try:
                        local_n = int(old_uid.split("_")[1])
                    except Exception:
                        local_n = 0
                    if local_n in local_to_global:
                        new_uid = f"Player_{local_to_global[local_n]}_Start"
                elif old_uid:
                    m = uid_re.match(old_uid)
                    if m:
                        new_uid = f"{m.group('base')} {next_seq[0]}"
                        next_seq[0] += 1
                    else:
                        new_uid = f"{old_uid} {next_seq[0]}"
                        next_seq[0] += 1

                if new_uid is not None:
                    _set_property_data(
                        new_obj.asset_property_collection, "uniqueID", new_uid
                    )

                wid_prop = new_obj.asset_property_collection.property_map.get("waypointID")
                if wid_prop and wid_prop.data is not None:
                    try:
                        old_wid = int(wid_prop.data)
                    except (TypeError, ValueError):
                        old_wid = 0
                    if old_wid > 0:
                        wid_prop.data = old_wid + cell_idx * wp_id_stride

                if old_uid and new_uid and new_uid != old_uid:
                    for pname, prop in new_obj.asset_property_collection.property_map.items():
                        if pname == "uniqueID":
                            continue
                        if isinstance(prop.data, str) and prop.data == old_uid:
                            prop.data = new_uid

                new_objects.append(new_obj)

        target_objs.map_objects = new_objects

    # ---- Per-player metadata: SidesList / Teams / PlayerScriptsList / BuildLists ----
    from ..assets.sides.sides_list import SidesList
    from ..assets.sides.player import Player
    from ..assets.teams.teams import Teams
    from ..assets.teams.team import Team
    from ..assets.scripts.player_scripts_list import PlayerScriptsList
    from ..assets.scripts.script_list import ScriptList
    from ..assets.build.build_lists import BuildLists
    from ..assets.build.build_list import BuildList

    target_sides = target_ctx.get_asset_by_type(SidesList)
    target_teams = target_ctx.get_asset_by_type(Teams)
    target_psl = target_ctx.get_asset_by_type(PlayerScriptsList)
    target_blists = target_ctx.get_asset_by_type(BuildLists)

    def _index_of_player_in(players_list, name: str) -> int:
        for idx, pl in enumerate(players_list):
            pn = pl.asset_property_collection.property_map.get("playerName")
            if pn and pn.data == name:
                return idx
        return -1

    pre_canvas_players = list(target_sides.players) if target_sides else []
    pre_canvas_teams = list(target_teams.teams) if target_teams else []

    def _is_player_n_entry(prop_collection, key: str = "playerName") -> bool:
        pn = prop_collection.property_map.get(key)
        if pn is None or pn.data is None:
            return False
        return bool(re.match(r"^Player_\d+$", str(pn.data)))

    if target_sides is not None:
        target_sides.players = [
            p for p in pre_canvas_players
            if not _is_player_n_entry(p.asset_property_collection, "playerName")
        ]
    if target_teams is not None:
        def _team_belongs_to_player_n(t) -> bool:
            owner = t.property_collection.property_map.get("teamOwner")
            if owner is None or owner.data is None:
                return False
            return bool(re.match(r"^Player_\d+$", str(owner.data)))
        target_teams.teams = [
            t for t in pre_canvas_teams if not _team_belongs_to_player_n(t)
        ]
    # PlayerScriptsList and BuildLists are positional and parallel to
    # SidesList's Player_N entries; rebuild them entirely from cell sources.
    if target_psl is not None:
        target_psl.script_lists = []
    if target_blists is not None:
        target_blists.build_list = []

    global_player_n = 0
    for cell_layout, cell_player_starts in zip(layout.cells, per_cell_player_starts):
        snap = cell_layout.snap

        sorted_starts = sorted(cell_player_starts, key=lambda kv: kv[0])
        for offset, (local_n, _obj) in enumerate(sorted_starts, start=1):
            new_n = global_player_n + offset
            template_idx = _index_of_player_in(snap.sides_players, f"Player_{local_n}")
            if template_idx < 0:
                continue

            if target_sides is not None and snap.sides_players:
                src_pl = snap.sides_players[template_idx]
                new_pl = Player()
                new_pl.asset_property_collection = _clone_property_collection(
                    src_pl.asset_property_collection, target_ctx
                )
                new_pl.build_list_items = list(src_pl.build_list_items)
                _set_property_data(new_pl.asset_property_collection, "playerName", f"Player_{new_n}")
                _set_property_data(new_pl.asset_property_collection, "playerDisplayName", f"Player_{new_n}")
                target_sides.players.append(new_pl)

            if target_teams is not None and template_idx < len(snap.teams_list):
                src_team = snap.teams_list[template_idx]
                new_team = Team()
                new_team.property_collection = _clone_property_collection(
                    src_team.property_collection, target_ctx
                )
                _set_property_data(new_team.property_collection, "teamName", f"teamPlayer_{new_n}")
                _set_property_data(new_team.property_collection, "teamOwner", f"Player_{new_n}")
                target_teams.teams.append(new_team)

            if target_psl is not None and template_idx < len(snap.script_lists):
                src_sl = snap.script_lists[template_idx]
                new_sl = ScriptList()
                new_sl.scripts = []
                new_sl.script_groups = []
                new_sl._child_order = []
                new_sl.name = src_sl.name
                new_sl.id = (
                    target_ctx.map_struct.register_string(src_sl.name)
                    if src_sl.name else int(src_sl.id)
                )
                new_sl.version = src_sl.version
                target_psl.script_lists.append(new_sl)

            if target_blists is not None and template_idx < len(snap.build_lists_list):
                src_bl = snap.build_lists_list[template_idx]
                new_bl = BuildList()
                new_bl.faction = src_bl.faction
                new_bl.count = 0
                target_blists.build_list.append(new_bl)

        global_player_n += len(sorted_starts)

    # ---- Water areas / rivers / waves / triggers ----
    # Sources are read from snapshots (taken before this mutation pass) so
    # the duplicate case -- where canvas IS every cell's source -- doesn't
    # clear the very list it tries to read from.
    from ..assets.water.standing_water_areas import StandingWaterAreas
    from ..assets.water.standing_wave_areas import StandingWaveAreas
    from ..assets.water.river_areas import RiverAreas
    from ..assets.triggers.trigger_areas import TriggerAreas

    def _replicate_areas_into(target_asset_obj, snap_attr: str) -> None:
        if target_asset_obj is None:
            return
        if hasattr(target_asset_obj, "water_areas"):
            list_name = "water_areas"
        elif hasattr(target_asset_obj, "areas"):
            list_name = "areas"
        else:
            return
        target_list = getattr(target_asset_obj, list_name)
        target_list.clear()

        for cell_layout in layout.cells:
            cell_list = getattr(cell_layout.snap, snap_attr, None) or []
            if not cell_list:
                continue
            dx_world, dy_world = _world_offset_for_cell(cell_layout)
            for area in cell_list:
                new_area = copy.deepcopy(area)
                pts = new_area.points
                for k, (px, py) in enumerate(pts):
                    pts[k] = (float(px) + dx_world, float(py) + dy_world)
                target_list.append(new_area)

    target_swa = target_ctx.get_asset_by_type(StandingWaterAreas)
    _replicate_areas_into(target_swa, "swa_areas")
    target_swv = target_ctx.get_asset_by_type(StandingWaveAreas)
    _replicate_areas_into(target_swv, "swv_areas")
    target_rivers = target_ctx.get_asset_by_type(RiverAreas)
    _replicate_areas_into(target_rivers, "river_areas")
    target_trig = target_ctx.get_asset_by_type(TriggerAreas)
    _replicate_areas_into(target_trig, "trigger_areas")

    # ---- Pad-water cover rectangles for gaps and empty layout cells ----
    if target_swa is not None and target_swa.water_areas:
        template = target_swa.water_areas[0]
        play_origin_x = layout.border * WORLD_UNITS_PER_TILE
        play_origin_y = layout.border * WORLD_UNITS_PER_TILE

        def _add_water_rect(x0_play: float, y0_play: float, x1_play: float, y1_play: float) -> None:
            x0 = play_origin_x + x0_play
            y0 = play_origin_y + y0_play
            x1 = play_origin_x + x1_play
            y1 = play_origin_y + y1_play
            new_area = copy.deepcopy(template)
            new_area.points = [
                (float(x0), float(y1)),
                (float(x1), float(y1)),
                (float(x1), float(y0)),
                (float(x0), float(y0)),
            ]
            target_swa.water_areas.append(new_area)

        if layout.pad_x > 0:
            for i in range(layout.n_cols - 1):
                gap_x0 = (layout.col_dst_x[i] + layout.col_widths[i]) * WORLD_UNITS_PER_TILE
                gap_x1 = gap_x0 + layout.pad_x * WORLD_UNITS_PER_TILE
                _add_water_rect(gap_x0, 0.0, gap_x1, layout.new_ph * WORLD_UNITS_PER_TILE)
        if layout.pad_y > 0:
            for j in range(layout.n_rows - 1):
                gap_y0 = (layout.row_dst_y[j] + layout.row_heights[j]) * WORLD_UNITS_PER_TILE
                gap_y1 = gap_y0 + layout.pad_y * WORLD_UNITS_PER_TILE
                _add_water_rect(0.0, gap_y0, layout.new_pw * WORLD_UNITS_PER_TILE, gap_y1)

        for col in range(layout.n_cols):
            for row in range(layout.n_rows):
                if layout.grid[col, row] != -1:
                    continue
                x0 = layout.col_dst_x[col] * WORLD_UNITS_PER_TILE
                y0 = layout.row_dst_y[row] * WORLD_UNITS_PER_TILE
                x1 = (layout.col_dst_x[col] + layout.col_widths[col]) * WORLD_UNITS_PER_TILE
                y1 = (layout.row_dst_y[row] + layout.row_heights[row]) * WORLD_UNITS_PER_TILE
                _add_water_rect(x0, y0, x1, y1)


def duplicate_context(
    target_ctx,
    nx: int = 1,
    ny: int = 1,
    pad_x: int = 0,
    pad_y: int = 0,
) -> None:
    """Backwards-compatible duplication: tile target_ctx into an Nx*Ny grid.

    Equivalent to:
        compose_context(target_ctx,
                        presets.duplicate(target_ctx, nx, ny, pad_x, pad_y))
    """
    if int(nx) == 1 and int(ny) == 1:
        return
    spec = presets.duplicate(target_ctx, int(nx), int(ny), int(pad_x), int(pad_y))
    compose_context(target_ctx, spec)
