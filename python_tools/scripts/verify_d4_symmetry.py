"""
Verify whether D4 symmetry augmentation (rotations + reflections) preserves
blend rules in real RA3 maps.

The 12 deterministic blend rules from MapGenerator.BlendTextures():
  Priority order (first match wins, using the C# coordinate system where
  left=(x-1,y), right=(x+1,y), top=(x,y+1), bottom=(x,y-1)):

  1. left==top && top!=center       -> BottomRight (0x28)  tex=top
  2. right==top && top!=center      -> BottomLeft  (0x24)  tex=top
  3. right==bottom && bottom!=center-> TopLeft     (0x34)  tex=bottom
  4. left==bottom && bottom!=center -> TopRight    (0x38)  tex=bottom
  5. left!=center                   -> Right       (0x11)  tex=left
  6. right!=center                  -> Left        (0x01)  tex=right
  7. top!=center                    -> Bottom      (0x02)  tex=top
  8. bottom!=center                 -> Top         (0x12)  tex=bottom
  9. topLeft!=center                -> ExceptTopLeft     (0x04) tex=topLeft
  10. topRight!=center              -> ExceptTopRight    (0x08) tex=topRight
  11. bottomRight!=center           -> ExceptBottomRight (0x14) tex=bottomRight
  12. bottomLeft!=center            -> ExceptBottomLeft  (0x18) tex=bottomLeft

  Final gate: only blend if centerTexture <= tex.

Investigation:
  1. Load real maps and extract cells with known blends.
  2. For each cell, extract the 5x5 texture neighborhood.
  3. Apply each of 8 D4 transforms to the neighborhood.
  4. Re-apply the 12 rules to the transformed neighborhood.
  5. Check if the resulting direction matches the expected remapped direction.
  6. Also check: does position-encoding break under rotation?
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np

# Setup path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.terrain.blend_info import BlendInfo
from map_processor.assets.terrain.blend_direction import BlendDirection

# =============================================================================
# BlendDirection values (from blend_direction.py)
# =============================================================================
DIR_BottomRight     = 0x28  # 40
DIR_Bottom          = 0x02  # 2
DIR_BottomLeft      = 0x24  # 36
DIR_Right           = 0x11  # 17
DIR_Left            = 0x01  # 1
DIR_TopRight        = 0x38  # 56
DIR_Top             = 0x12  # 18
DIR_TopLeft         = 0x34  # 52
DIR_ExceptBottomRight = 0x14  # 20
DIR_ExceptBottomLeft  = 0x18  # 24
DIR_ExceptTopRight    = 0x04  # 4
DIR_ExceptTopLeft     = 0x08  # 8

ALL_DIRS = [
    DIR_BottomRight, DIR_Bottom, DIR_BottomLeft, DIR_Right, DIR_Left,
    DIR_TopRight, DIR_Top, DIR_TopLeft,
    DIR_ExceptBottomRight, DIR_ExceptBottomLeft, DIR_ExceptTopRight, DIR_ExceptTopLeft,
]

DIR_NAMES = {
    DIR_BottomRight: "BottomRight",
    DIR_Bottom: "Bottom",
    DIR_BottomLeft: "BottomLeft",
    DIR_Right: "Right",
    DIR_Left: "Left",
    DIR_TopRight: "TopRight",
    DIR_Top: "Top",
    DIR_TopLeft: "TopLeft",
    DIR_ExceptBottomRight: "ExceptBottomRight",
    DIR_ExceptBottomLeft: "ExceptBottomLeft",
    DIR_ExceptTopRight: "ExceptTopRight",
    DIR_ExceptTopLeft: "ExceptTopLeft",
}

# =============================================================================
# Coordinate system note
# =============================================================================
# In the C# code, BlendTextures iterates with (x,y) where:
#   left  = (x-1, y)    right = (x+1, y)
#   top   = (x, y+1)    bottom = (x, y-1)
#   topLeft = (x-1, y+1)  etc.
#
# In our numpy tex_grid[x, y], the same convention holds:
#   axis 0 = x (left/right), axis 1 = y (top/bottom)
#
# 5x5 neighborhood around center (cx, cy):
#   grid[cx-2..cx+2, cy-2..cy+2]
#
# The 8 immediate neighbors in C# terms (offsets from center in x,y space):
#   TL=(-1,+1), T=(0,+1), TR=(+1,+1)
#   L =(-1, 0),           R =(+1, 0)
#   BL=(-1,-1), B=(0,-1), BR=(+1,-1)

# =============================================================================
# Apply the 12 rules (pure reimplementation of C# BlendTextures)
# =============================================================================
def apply_blend_rules(neighborhood_5x5: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    Given a 5x5 texture grid (x-indexed, y-indexed), apply the 12 blend rules
    to the center cell [2,2].

    Returns (direction, secondary_texture) or (None, None) if no blend.

    Neighborhood indexing:  grid[x_offset+2, y_offset+2]
      center = grid[2,2]
      left   = grid[1,2]  (x-1)
      right  = grid[3,2]  (x+1)
      top    = grid[2,3]  (y+1)
      bottom = grid[2,1]  (y-1)
      topLeft = grid[1,3]
      topRight = grid[3,3]
      bottomLeft = grid[1,1]
      bottomRight = grid[3,1]
    """
    g = neighborhood_5x5
    center = int(g[2, 2])
    left   = int(g[1, 2])
    right  = int(g[3, 2])
    top    = int(g[2, 3])
    bottom = int(g[2, 1])
    topLeft     = int(g[1, 3])
    topRight    = int(g[3, 3])
    bottomLeft  = int(g[1, 1])
    bottomRight = int(g[3, 1])

    direction = None
    tex = center  # will be overwritten

    # Rule 1: left==top && top!=center -> BottomRight
    if left == top and top != center:
        direction = DIR_BottomRight
        tex = top
    # Rule 2: right==top && top!=center -> BottomLeft
    elif right == top and top != center:
        direction = DIR_BottomLeft
        tex = top
    # Rule 3: right==bottom && bottom!=center -> TopLeft
    elif right == bottom and bottom != center:
        direction = DIR_TopLeft
        tex = bottom
    # Rule 4: left==bottom && bottom!=center -> TopRight
    elif left == bottom and bottom != center:
        direction = DIR_TopRight
        tex = bottom
    # Rule 5: left!=center -> Right
    elif left != center:
        direction = DIR_Right
        tex = left
    # Rule 6: right!=center -> Left
    elif right != center:
        direction = DIR_Left
        tex = right
    # Rule 7: top!=center -> Bottom
    elif top != center:
        direction = DIR_Bottom
        tex = top
    # Rule 8: bottom!=center -> Top
    elif bottom != center:
        direction = DIR_Top
        tex = bottom
    # Rule 9: topLeft!=center -> ExceptTopLeft
    elif topLeft != center:
        direction = DIR_ExceptTopLeft
        tex = topLeft
    # Rule 10: topRight!=center -> ExceptTopRight
    elif topRight != center:
        direction = DIR_ExceptTopRight
        tex = topRight
    # Rule 11: bottomRight!=center -> ExceptBottomRight
    elif bottomRight != center:
        direction = DIR_ExceptBottomRight
        tex = bottomRight
    # Rule 12: bottomLeft!=center -> ExceptBottomLeft
    elif bottomLeft != center:
        direction = DIR_ExceptBottomLeft
        tex = bottomLeft
    else:
        return (None, None)

    # Final gate: only blend if center <= tex
    if center <= tex:
        return (direction, tex)
    else:
        return (None, None)


# =============================================================================
# D4 group transforms on 5x5 grid
# =============================================================================
# We work in (x, y) space. The grid is indexed as grid[x, y].
# All transforms keep the center at (2,2).
#
# Identity:        new[x, y] = old[x, y]
# Rot90CW:         new[y, 4-x] = old[x, y]   =>  new[x,y] = old[4-y, x]
# Rot180:          new[4-x, 4-y] = old[x, y]  =>  new[x,y] = old[4-x, 4-y]
# Rot270CW:        new[4-y, x] = old[x, y]    =>  new[x,y] = old[y, 4-x]
# FlipX:           new[4-x, y] = old[x, y]    =>  new[x,y] = old[4-x, y]
# FlipY:           new[x, 4-y] = old[x, y]    =>  new[x,y] = old[x, 4-y]
# FlipDiag(x=y):   new[y, x] = old[x, y]      =>  new[x,y] = old[y, x]
# FlipAntiDiag:    new[4-y, 4-x] = old[x, y]  =>  new[x,y] = old[4-y, 4-x]

def transform_grid(g: np.ndarray, transform_name: str) -> np.ndarray:
    """Apply a D4 transform to a 5x5 grid in (x,y) indexing."""
    n = 4  # max index
    result = np.zeros_like(g)
    for x in range(5):
        for y in range(5):
            if transform_name == "identity":
                ox, oy = x, y
            elif transform_name == "rot90cw":
                ox, oy = n - y, x
            elif transform_name == "rot180":
                ox, oy = n - x, n - y
            elif transform_name == "rot270cw":
                ox, oy = y, n - x
            elif transform_name == "flipX":
                # Flip the x-axis (left-right flip)
                ox, oy = n - x, y
            elif transform_name == "flipY":
                # Flip the y-axis (top-bottom flip)
                ox, oy = x, n - y
            elif transform_name == "flipDiag":
                # Flip along x=y diagonal
                ox, oy = y, x
            elif transform_name == "flipAntiDiag":
                # Flip along x+y=4 anti-diagonal
                ox, oy = n - y, n - x
            else:
                raise ValueError(f"Unknown transform: {transform_name}")
            result[x, y] = g[ox, oy]
    return result


# Direction remapping under D4 transforms.
# When we rotate/flip the grid, the blend direction must be remapped accordingly.
#
# Under rotation, the neighbor positions rotate:
#   Rot90CW: (dx, dy) -> (dy, -dx)
#     Left(-1,0) -> (0,1) = Top
#     Right(1,0) -> (0,-1) = Bottom
#     Top(0,1) -> (-1,0) = Left  (wait, that flips...)
#
# Actually, let's think more carefully.
#
# The blend direction describes WHERE the secondary texture encroaches FROM.
# E.g., "Right" means the secondary texture is to the right of center.
# After a 90 CW rotation of the grid, what was Right is now Bottom.
#
# Under Rot90CW of the coordinate system:
#   new_x = old_y, new_y = -(old_x) + n
#   A neighbor that was at (1,0) relative to center (Right)
#   is now at (0, -1+n) relative. But since we also rotated the grid,
#   the direction label should follow.
#
# Let's derive by thinking about what happens to each direction name:
#
# Rot90CW (grid rotates clockwise => directions rotate clockwise):
#   Left -> Bottom,  Right -> Top,  Top -> Left,  Bottom -> Right
#   TopLeft -> BottomLeft,  TopRight -> TopLeft,
#   BottomRight -> TopRight,  BottomLeft -> BottomRight
#   ExceptTopLeft -> ExceptBottomLeft,  ExceptTopRight -> ExceptTopLeft,
#   ExceptBottomRight -> ExceptTopRight,  ExceptBottomLeft -> ExceptBottomRight

# Wait. Let me think again. The C# convention is:
# top = (x, y+1), i.e. increasing y.
#
# When we apply rot90cw to the grid: new[x,y] = old[4-y, x]
# The neighbor that was at old[cx+1, cy] (=Right in old) is now:
# In new coords, we need old[4-y, x] = old[cx+1, cy].
# So 4-y = cx+1, x = cy  => y = 4-cx-1 = 3-cx = n-1-cx, x = cy.
# If center is at (2,2): y=1, x=2. So the new position is (2,1) relative to center = (0,-1) = Bottom.
#
# So Right -> Bottom under rot90cw. Let me verify all:
#
# old_center = (2,2)
# After rot90cw: new[x,y] = old[4-y, x]
# new center at (2,2): old[4-2, 2] = old[2,2] = center. Good.
#
# old Left = (1,2): In new, we need: new[xn,yn] = old[1,2]
#   4-yn = 1, xn = 2 => yn = 3, xn = 2 => new pos (2,3) = (0,+1) from center = Top
# old Right = (3,2): 4-yn=3, xn=2 => yn=1, xn=2 => new pos (2,1) = (0,-1) = Bottom
# old Top = (2,3): 4-yn=2, xn=3 => yn=2, xn=3 => new pos (3,2) = (+1,0) = Right
# old Bottom = (2,1): 4-yn=2, xn=1 => yn=2, xn=1 => new pos (1,2) = (-1,0) = Left
#
# old TopLeft = (1,3): 4-yn=1, xn=3 => yn=3, xn=3 => new (3,3) = (+1,+1) = TopRight
# old TopRight = (3,3): 4-yn=3, xn=3 => yn=1, xn=3 => new (3,1) = (+1,-1) = BottomRight
# old BottomLeft = (1,1): 4-yn=1, xn=1 => yn=3, xn=1 => new (1,3) = (-1,+1) = TopLeft
# old BottomRight = (3,1): 4-yn=3, xn=1 => yn=1, xn=1 => new (1,1) = (-1,-1) = BottomLeft
#
# So under rot90cw:
#   Left -> Top,  Right -> Bottom,  Top -> Right,  Bottom -> Left
#   TopLeft -> TopRight,  TopRight -> BottomRight,
#   BottomRight -> BottomLeft,  BottomLeft -> TopLeft
#   ExceptTopLeft -> ExceptTopRight, ExceptTopRight -> ExceptBottomRight,
#   ExceptBottomRight -> ExceptBottomLeft, ExceptBottomLeft -> ExceptTopLeft

_REMAP_ROT90CW = {
    DIR_Left: DIR_Top,
    DIR_Right: DIR_Bottom,
    DIR_Top: DIR_Right,
    DIR_Bottom: DIR_Left,
    DIR_TopLeft: DIR_TopRight,
    DIR_TopRight: DIR_BottomRight,
    DIR_BottomRight: DIR_BottomLeft,
    DIR_BottomLeft: DIR_TopLeft,
    DIR_ExceptTopLeft: DIR_ExceptTopRight,
    DIR_ExceptTopRight: DIR_ExceptBottomRight,
    DIR_ExceptBottomRight: DIR_ExceptBottomLeft,
    DIR_ExceptBottomLeft: DIR_ExceptTopLeft,
}

# FlipX: new[x,y] = old[4-x, y]
# old Left = (1,2): 4-1=3 => new (3,2) = Right
# old Right = (3,2): 4-3=1 => new (1,2) = Left
# old Top = (2,3): same y => new (2,3) = Top
# old Bottom = (2,1): same y => new (2,1) = Bottom
# old TopLeft = (1,3): => new (3,3) = TopRight
# old TopRight = (3,3): => new (1,3) = TopLeft
# old BottomLeft = (1,1): => new (3,1) = BottomRight
# old BottomRight = (3,1): => new (1,1) = BottomLeft

_REMAP_FLIPX = {
    DIR_Left: DIR_Right,
    DIR_Right: DIR_Left,
    DIR_Top: DIR_Top,
    DIR_Bottom: DIR_Bottom,
    DIR_TopLeft: DIR_TopRight,
    DIR_TopRight: DIR_TopLeft,
    DIR_BottomRight: DIR_BottomLeft,
    DIR_BottomLeft: DIR_BottomRight,
    DIR_ExceptTopLeft: DIR_ExceptTopRight,
    DIR_ExceptTopRight: DIR_ExceptTopLeft,
    DIR_ExceptBottomRight: DIR_ExceptBottomLeft,
    DIR_ExceptBottomLeft: DIR_ExceptBottomRight,
}

# FlipY: new[x,y] = old[x, 4-y]
# old Left = (1,2): same x => new (1,2) = Left
# old Right = (3,2): => new (3,2) = Right
# old Top = (2,3): 4-3=1 => new (2,1) = Bottom
# old Bottom = (2,1): 4-1=3 => new (2,3) = Top
# old TopLeft = (1,3): => new (1,1) = BottomLeft
# old TopRight = (3,3): => new (3,1) = BottomRight
# old BottomLeft = (1,1): => new (1,3) = TopLeft
# old BottomRight = (3,1): => new (3,3) = TopRight

_REMAP_FLIPY = {
    DIR_Left: DIR_Left,
    DIR_Right: DIR_Right,
    DIR_Top: DIR_Bottom,
    DIR_Bottom: DIR_Top,
    DIR_TopLeft: DIR_BottomLeft,
    DIR_TopRight: DIR_BottomRight,
    DIR_BottomRight: DIR_TopRight,
    DIR_BottomLeft: DIR_TopLeft,
    DIR_ExceptTopLeft: DIR_ExceptBottomLeft,
    DIR_ExceptTopRight: DIR_ExceptBottomRight,
    DIR_ExceptBottomRight: DIR_ExceptTopRight,
    DIR_ExceptBottomLeft: DIR_ExceptTopLeft,
}


def compose_remap(remap_a: Dict[int, int], remap_b: Dict[int, int]) -> Dict[int, int]:
    """Compose two remappings: result[d] = remap_b[remap_a[d]]."""
    result = {}
    for d in ALL_DIRS:
        result[d] = remap_b[remap_a[d]]
    return result


def build_all_remaps() -> Dict[str, Dict[int, int]]:
    """Build direction remaps for all 8 D4 transforms."""
    identity = {d: d for d in ALL_DIRS}
    rot90 = _REMAP_ROT90CW
    rot180 = compose_remap(rot90, rot90)
    rot270 = compose_remap(rot180, rot90)
    flipX = _REMAP_FLIPX
    flipY = _REMAP_FLIPY
    # flipDiag = rot90cw then flipX  (transpose x,y)
    # Actually let's derive it directly:
    # FlipDiag: new[x,y] = old[y, x]
    # old Left=(1,2) => new(2,1) = Bottom
    # old Right=(3,2) => new(2,3) = Top
    # old Top=(2,3) => new(3,2) = Right
    # old Bottom=(2,1) => new(1,2) = Left
    # old TL=(1,3) => new(3,1) = BottomRight
    # old TR=(3,3) => new(3,3) = TopRight
    # old BL=(1,1) => new(1,1) = BottomLeft
    # old BR=(3,1) => new(1,3) = TopLeft
    flipDiag = {
        DIR_Left: DIR_Bottom,
        DIR_Right: DIR_Top,
        DIR_Top: DIR_Right,
        DIR_Bottom: DIR_Left,
        DIR_TopLeft: DIR_BottomRight,  # Wait, let me recheck
        DIR_TopRight: DIR_TopRight,
        DIR_BottomRight: DIR_TopLeft,  # Wait...
        DIR_BottomLeft: DIR_BottomLeft,
        DIR_ExceptTopLeft: DIR_ExceptBottomRight,
        DIR_ExceptTopRight: DIR_ExceptTopRight,
        DIR_ExceptBottomRight: DIR_ExceptTopLeft,
        DIR_ExceptBottomLeft: DIR_ExceptBottomLeft,
    }
    # Hmm, let me re-derive more carefully.
    # FlipDiag: new[x,y] = old[y, x]
    # If old has a blend from TopLeft direction (meaning secondary is at TopLeft = (cx-1, cy+1)),
    # after flip, old[cx-1, cy+1] maps to new[cy+1, cx-1].
    # Relative to new center (2,2): (cy+1-2, cx-1-2) = (cy-1, cx-3)
    # For center at (2,2): (2-1, 2-3) = (1, -1) => (1,-1) from center = BottomRight? No...
    # (dx, dy) = (cy+1 - 2, cx-1 - 2) = (cy-1, cx-3).
    # With cx=2, cy=2: (1, -1). So new neighbor at (+1, -1) from center.
    # In C# terms: +1 in x = Right, -1 in y = bottom. So BottomRight.
    # OK so TopLeft -> BottomRight? Hmm, that seems wrong. Let me re-derive.
    #
    # Actually wait, the direction name describes the blend direction, not the neighbor position.
    # Direction "TopLeft" (0x34) means the blend covers TopLeft corner.
    # In the C# code:
    #   left==bottom && bottom!=center -> TopRight
    # This means: if bottom and left have the same secondary, blend direction = TopRight.
    # TopRight blend means the secondary encroaches into the TopRight of the center cell.
    #
    # Actually the blend direction names are confusing. Let me look at the rules:
    # Rule 1: left==top -> BottomRight means "fill BottomRight" with secondary (secondary is at top-left)
    # Rule 9: topLeft!=center -> ExceptTopLeft means "fill everywhere EXCEPT TopLeft corner"
    #
    # For remapping purposes, the direction label describes a geometric region of the cell.
    # Under flipDiag (swapping x and y axes):
    #   "Top" (y+) becomes "Right" (x+)  [wait, no: flip along x=y means x->y, y->x]
    #   So what was "Top" (positive y direction) becomes... in the flipped coords, the same
    #   physical position is now in the positive x direction = "Right".
    #
    # Let me just re-derive from the neighbor positions.
    # FlipDiag: new[x,y] = old[y,x]. Center stays at (2,2).
    # The neighbor old[1,2] (Left) is at new[2,1] (Bottom).
    # The neighbor old[3,2] (Right) is at new[2,3] (Top).
    # old[2,3] (Top) -> new[3,2] (Right).
    # old[2,1] (Bottom) -> new[1,2] (Left).
    # old[1,3] (TopLeft) -> new[3,1] (BottomRight).
    # old[3,3] (TopRight) -> new[3,3] (TopRight).  -- STAYS SAME!
    # old[1,1] (BottomLeft) -> new[1,1] (BottomLeft). -- STAYS SAME!
    # old[3,1] (BottomRight) -> new[1,3] (TopLeft).
    #
    # So the remapping for directions (where the direction name refers to which
    # corner/edge of the center cell the secondary texture fills):
    #
    # The direction "BottomRight" means secondary fills bottom-right of center.
    # "Bottom-right" position = (cx+1, cy-1) in old = BottomRight neighbor.
    # After flipDiag, this neighbor is at (cy-1, cx+1) in new = new(1, 3) = TopLeft position.
    # So BottomRight -> TopLeft.
    #
    # Actually, I think I should think of it differently. The direction describes
    # the SHAPE of the blend painted on the cell, not a neighbor position.
    # When we flip the grid, the shape also flips.
    #
    # Let me use a simpler approach: just verify computationally by comparing
    # apply_blend_rules on transformed grids.

    # flipAntiDiag: new[x,y] = old[4-y, 4-x]
    # = flipDiag composed with rot180

    # Instead of manually deriving all 8, let me compute them by running
    # apply_blend_rules on systematically constructed test grids.
    # But actually, for the verification script, I can compute the expected direction
    # differently: just apply the rules to the transformed grid and see if it matches
    # the remapped direction. So I only need the remaps for reporting purposes.
    # Let me compute them automatically.

    remaps = {}
    remaps["identity"] = identity

    # For each transform, we compute the remap by testing what happens when we
    # apply the rules to a grid where ONLY one specific neighbor differs.
    # This gives us the direction remap for single-neighbor cases.
    # For corner cases (rules 1-4), we need to test those patterns specifically.

    for tname in ["rot90cw", "rot180", "rot270cw", "flipX", "flipY", "flipDiag", "flipAntiDiag"]:
        remap = {}

        # Test each single-neighbor rule (rules 5-12)
        # Create a grid where only one neighbor differs from center
        single_neighbor_positions = {
            DIR_Right: (1, 2),    # left neighbor position -> triggers Right
            DIR_Left: (3, 2),     # right neighbor -> triggers Left
            DIR_Bottom: (2, 3),   # top neighbor -> triggers Bottom
            DIR_Top: (2, 1),      # bottom neighbor -> triggers Top
            DIR_ExceptTopLeft: (1, 3),   # topLeft -> ExceptTopLeft
            DIR_ExceptTopRight: (3, 3),  # topRight -> ExceptTopRight
            DIR_ExceptBottomRight: (3, 1),  # bottomRight -> ExceptBottomRight
            DIR_ExceptBottomLeft: (1, 1),   # bottomLeft -> ExceptBottomLeft
        }

        for orig_dir, (nx, ny) in single_neighbor_positions.items():
            g = np.full((5, 5), 0, dtype=np.int32)
            g[nx, ny] = 1  # only this neighbor differs, and 0 <= 1 so blend happens
            tg = transform_grid(g, tname)
            result_dir, result_tex = apply_blend_rules(tg)
            if result_dir is not None:
                remap[orig_dir] = result_dir
            else:
                # This shouldn't happen for the single-neighbor cases with center=0, tex=1
                print(f"WARNING: {tname} single-neighbor test for {DIR_NAMES[orig_dir]} returned no blend!")
                remap[orig_dir] = orig_dir

        # Test corner rules (rules 1-4)
        # Rule 1: left==top -> BottomRight. left=(1,2), top=(2,3)
        corner_tests = {
            DIR_BottomRight: [(1, 2), (2, 3)],  # left==top
            DIR_BottomLeft: [(3, 2), (2, 3)],   # right==top
            DIR_TopLeft: [(3, 2), (2, 1)],       # right==bottom
            DIR_TopRight: [(1, 2), (2, 1)],      # left==bottom
        }

        for orig_dir, positions in corner_tests.items():
            g = np.full((5, 5), 0, dtype=np.int32)
            for (px, py) in positions:
                g[px, py] = 1
            tg = transform_grid(g, tname)
            result_dir, result_tex = apply_blend_rules(tg)
            if result_dir is not None:
                remap[orig_dir] = result_dir
            else:
                print(f"WARNING: {tname} corner test for {DIR_NAMES[orig_dir]} returned no blend!")
                remap[orig_dir] = orig_dir

        remaps[tname] = remap

    return remaps


# =============================================================================
# Position encoding analysis
# =============================================================================
def _get_tile_from_texture(x: int, y: int, texture_id: int) -> int:
    """C# BlendTileData.GetTile(x,y,texture)."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return current + 64 * texture_id


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    """Inverse of GetTile."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def analyze_position_encoding():
    """Check if position encoding breaks under rotation."""
    print("\n" + "=" * 80)
    print("POSITION ENCODING ANALYSIS")
    print("=" * 80)

    print("\nThe tile encoding formula: tile_value = texture_id * 64 + offset(x, y)")
    print("where offset(x,y) = ((y%8)//2)*16 + (y%2)*2 + ((x%8)//2)*4 + (x%2)")
    print()

    # Show the offset pattern for an 8x8 block
    print("Offset pattern for (x,y) in 8x8 block:")
    print("       y=0  y=1  y=2  y=3  y=4  y=5  y=6  y=7")
    for x in range(8):
        offsets = []
        for y in range(8):
            off = _get_tile_from_texture(x, y, 0)
            offsets.append(f"{off:3d}")
        print(f"  x={x}  {'  '.join(offsets)}")

    # Check: if we rotate position (x,y) to (y, W-1-x), does offset change?
    print("\n--- Does the offset change under rotation? ---")
    print("Under rot90cw in tile space: (x,y) -> (y, W-1-x)")
    print("For W=100 (typical map width):")
    W = 100
    mismatches = 0
    total = 0
    for x in range(8):
        for y in range(8):
            nx, ny = y, W - 1 - x  # rot90cw
            old_off = _get_tile_from_texture(x, y, 0)
            new_off = _get_tile_from_texture(nx, ny, 0)
            if old_off != new_off:
                mismatches += 1
            total += 1
    print(f"  Offset mismatches: {mismatches}/{total}")
    print(f"  This means secondary_texture_tile MUST be recomputed after rotation!")

    # Show that texture ID extraction is position-dependent
    print("\n--- Texture ID extraction depends on position ---")
    print("  For the SAME texture_id=5:")
    for x in range(4):
        for y in range(4):
            tile_val = _get_tile_from_texture(x, y, 5)
            recovered = _get_texture_from_tile(x, y, tile_val)
            wrong = _get_texture_from_tile(x + 1, y, tile_val)
            print(f"    ({x},{y}): tile_val={tile_val}, recover at ({x},{y})={recovered}, "
                  f"wrong recovery at ({x+1},{y})={wrong}")

    print("\n==> CONCLUSION: Position encoding makes tile_value position-dependent.")
    print("    When augmenting, you CANNOT simply rotate tile values.")
    print("    You MUST: 1) decode texture at old pos, 2) re-encode at new pos.")
    print("    This is already handled by map_rotation.py but would be a problem")
    print("    for naive D4 augmentation of raw tile arrays.")


# =============================================================================
# Main analysis: load maps, verify D4 equivariance
# =============================================================================
def decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    """Return (w,h) int32 grid of texture indices decoded from tile-values."""
    tiles = np.asarray(blend.tiles)
    w, h = tiles.shape
    tex = np.empty((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


def extract_ground_truth(blend: BlendTileData, tex_grid: np.ndarray) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Extract ground truth blend direction and secondary texture for each blended cell.
    Returns {(x,y): (direction, secondary_tex_id)}.
    """
    w, h = tex_grid.shape
    blends = np.asarray(blend.blends)
    info = blend.blend_info or []
    gt = {}
    for x in range(w):
        for y in range(h):
            idx = int(blends[x, y])
            if idx <= 0 or idx > len(info):
                continue
            bi = info[idx - 1]
            sec_tex = _get_texture_from_tile(x, y, int(bi.secondary_texture_tile))
            direction = int(bi.blend_direction)
            gt[(x, y)] = (direction, sec_tex)
    return gt


def verify_d4_on_map(map_path: Path, remaps: Dict[str, Dict[int, int]], max_samples: int = 5000) -> Dict[str, dict]:
    """
    For a single map, verify D4 equivariance of blend rules.

    Returns per-transform stats.
    """
    print(f"\n--- Loading: {map_path.name} ---")
    m = Ra3Map(str(map_path))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)
    if blend is None:
        print("  No BlendTileData found, skipping.")
        return {}

    w, h = blend.tiles.shape
    tex_grid = decode_texture_grid(blend)
    gt = extract_ground_truth(blend, tex_grid)
    print(f"  Map size: {w}x{h}, blended cells: {len(gt)}")

    if len(gt) == 0:
        print("  No blended cells, skipping.")
        return {}

    # Filter to cells with enough border (need 5x5 neighborhood)
    valid_cells = [(x, y, d, s) for (x, y), (d, s) in gt.items()
                   if 2 <= x < w - 2 and 2 <= y < h - 2]

    if len(valid_cells) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(valid_cells), max_samples, replace=False)
        valid_cells = [valid_cells[i] for i in sorted(indices)]

    print(f"  Testing {len(valid_cells)} cells with 5x5 neighborhood")

    # First: verify that the rules reproduce ground truth on the original grid
    rules_match_gt = 0
    rules_no_blend = 0
    rules_wrong_dir = 0
    rules_wrong_tex = 0
    gt_dir_not_in_12 = 0

    for x, y, gt_dir, gt_sec in valid_cells:
        neighborhood = tex_grid[x - 2:x + 3, y - 2:y + 3].copy()
        result_dir, result_tex = apply_blend_rules(neighborhood)

        if result_dir is None:
            rules_no_blend += 1
        elif result_dir == gt_dir and result_tex == gt_sec:
            rules_match_gt += 1
        elif result_dir != gt_dir:
            rules_wrong_dir += 1
        else:
            rules_wrong_tex += 1

    total = len(valid_cells)
    print(f"\n  --- Rules vs Ground Truth (original grid) ---")
    print(f"  Exact match:     {rules_match_gt:6d} / {total} = {100*rules_match_gt/total:.2f}%")
    print(f"  No blend (rule says skip): {rules_no_blend:6d} / {total} = {100*rules_no_blend/total:.2f}%")
    print(f"  Wrong direction: {rules_wrong_dir:6d} / {total} = {100*rules_wrong_dir/total:.2f}%")
    print(f"  Wrong texture:   {rules_wrong_tex:6d} / {total} = {100*rules_wrong_tex/total:.2f}%")

    # Now test D4 transforms
    transform_names = ["identity", "rot90cw", "rot180", "rot270cw", "flipX", "flipY", "flipDiag", "flipAntiDiag"]
    results = {}

    for tname in transform_names:
        remap = remaps[tname]

        exact_match = 0
        no_blend_orig = 0
        no_blend_transformed = 0
        both_no_blend = 0
        dir_match_tex_match = 0
        dir_match_tex_mismatch = 0
        dir_mismatch = 0
        both_blend_total = 0

        for x, y, gt_dir, gt_sec in valid_cells:
            neighborhood = tex_grid[x - 2:x + 3, y - 2:y + 3].copy()

            # Apply rules to original
            orig_dir, orig_tex = apply_blend_rules(neighborhood)

            # Transform the grid
            transformed = transform_grid(neighborhood, tname)

            # Apply rules to transformed
            trans_dir, trans_tex = apply_blend_rules(transformed)

            # Expected: if orig said direction D, transformed should say remap[D]
            if orig_dir is None and trans_dir is None:
                both_no_blend += 1
                exact_match += 1
            elif orig_dir is None:
                no_blend_orig += 1
            elif trans_dir is None:
                no_blend_transformed += 1
            else:
                both_blend_total += 1
                expected_dir = remap.get(orig_dir, orig_dir)
                if trans_dir == expected_dir:
                    dir_match_tex_match += 1
                    exact_match += 1
                else:
                    dir_mismatch += 1

        results[tname] = {
            "total": total,
            "exact_match": exact_match,
            "both_no_blend": both_no_blend,
            "no_blend_orig_only": no_blend_orig,
            "no_blend_trans_only": no_blend_transformed,
            "both_blend": both_blend_total,
            "dir_match": dir_match_tex_match,
            "dir_mismatch": dir_mismatch,
        }

        match_rate = 100.0 * exact_match / total if total > 0 else 0
        print(f"\n  --- {tname} ---")
        print(f"  Exact equivariance: {exact_match}/{total} = {match_rate:.2f}%")
        if both_blend_total > 0:
            dir_rate = 100.0 * dir_match_tex_match / both_blend_total
            print(f"  Among both-blend cells: dir match = {dir_match_tex_match}/{both_blend_total} = {dir_rate:.2f}%")
        if dir_mismatch > 0:
            print(f"  Direction mismatches: {dir_mismatch}")
        if no_blend_orig + no_blend_transformed > 0:
            print(f"  Asymmetric no-blend: orig_only={no_blend_orig}, trans_only={no_blend_transformed}")

    return results


def verify_rule_equivariance_synthetic():
    """
    Verify D4 equivariance of the 12 rules using exhaustive synthetic neighborhoods.

    This tests whether, for ANY 5x5 texture configuration, transforming the grid
    and re-applying rules gives a direction consistent with the remap.

    We test a large set of random 5x5 grids with controlled texture distributions.
    """
    print("\n" + "=" * 80)
    print("SYNTHETIC EQUIVARIANCE TEST")
    print("=" * 80)

    remaps = build_all_remaps()
    transform_names = ["rot90cw", "rot180", "rot270cw", "flipX", "flipY", "flipDiag", "flipAntiDiag"]

    rng = np.random.default_rng(12345)
    n_tests = 100_000

    # Generate random 5x5 grids with 2-4 texture types
    # This covers all possible rule trigger patterns
    results_per_transform = defaultdict(lambda: {"total": 0, "match": 0, "mismatch": 0,
                                                  "both_none": 0, "orig_none": 0, "trans_none": 0,
                                                  "mismatch_examples": []})

    for i in range(n_tests):
        n_tex = rng.integers(2, 6)  # 2-5 texture types
        g = rng.integers(0, n_tex, size=(5, 5), dtype=np.int32)

        orig_dir, orig_tex = apply_blend_rules(g)

        for tname in transform_names:
            remap = remaps[tname]
            tg = transform_grid(g, tname)
            trans_dir, trans_tex = apply_blend_rules(tg)

            stats = results_per_transform[tname]
            stats["total"] += 1

            if orig_dir is None and trans_dir is None:
                stats["both_none"] += 1
                stats["match"] += 1
            elif orig_dir is None and trans_dir is not None:
                stats["orig_none"] += 1
                stats["mismatch"] += 1
                if len(stats["mismatch_examples"]) < 3:
                    stats["mismatch_examples"].append({
                        "type": "orig_none_trans_blend",
                        "grid": g.tolist(),
                        "trans_dir": DIR_NAMES.get(trans_dir, str(trans_dir)),
                    })
            elif orig_dir is not None and trans_dir is None:
                stats["trans_none"] += 1
                stats["mismatch"] += 1
                if len(stats["mismatch_examples"]) < 3:
                    stats["mismatch_examples"].append({
                        "type": "orig_blend_trans_none",
                        "grid": g.tolist(),
                        "orig_dir": DIR_NAMES.get(orig_dir, str(orig_dir)),
                    })
            else:
                expected = remap.get(orig_dir, orig_dir)
                if trans_dir == expected:
                    stats["match"] += 1
                else:
                    stats["mismatch"] += 1
                    if len(stats["mismatch_examples"]) < 5:
                        stats["mismatch_examples"].append({
                            "type": "dir_mismatch",
                            "grid": g.tolist(),
                            "orig_dir": DIR_NAMES.get(orig_dir, str(orig_dir)),
                            "expected_dir": DIR_NAMES.get(expected, str(expected)),
                            "actual_dir": DIR_NAMES.get(trans_dir, str(trans_dir)),
                        })

    print(f"\nTested {n_tests} random 5x5 grids per transform:")
    for tname in transform_names:
        s = results_per_transform[tname]
        match_rate = 100.0 * s["match"] / s["total"]
        print(f"\n  {tname}:")
        print(f"    Match: {s['match']}/{s['total']} = {match_rate:.4f}%")
        print(f"    Both none: {s['both_none']}, Orig none: {s['orig_none']}, Trans none: {s['trans_none']}")
        print(f"    Direction mismatch: {s['mismatch'] - s['orig_none'] - s['trans_none']}")

        if s["mismatch_examples"]:
            print(f"    Sample mismatches:")
            for ex in s["mismatch_examples"][:3]:
                if ex["type"] == "dir_mismatch":
                    print(f"      {ex['type']}: orig={ex['orig_dir']}, expected={ex['expected_dir']}, actual={ex['actual_dir']}")
                    grid = np.array(ex["grid"])
                    print(f"        Center={grid[2,2]}, L={grid[1,2]}, R={grid[3,2]}, T={grid[2,3]}, B={grid[2,1]}")
                    print(f"        TL={grid[1,3]}, TR={grid[3,3]}, BL={grid[1,1]}, BR={grid[3,1]}")
                elif ex["type"] == "orig_none_trans_blend":
                    print(f"      {ex['type']}: trans_dir={ex['trans_dir']}")
                elif ex["type"] == "orig_blend_trans_none":
                    print(f"      {ex['type']}: orig_dir={ex['orig_dir']}")

    return results_per_transform


def analyze_rule_symmetry_breaking():
    """
    Deep dive: identify exactly WHY and WHEN D4 equivariance breaks.

    The 12 rules have a priority ordering. Rotations/reflections can cause
    a different rule to fire first because the priority order is not symmetric.

    Specifically:
    - Rules 1-4 (corner rules) check left+top before right+bottom.
      left==top is rule 1, but after rot90cw, this becomes top==right
      which is rule 2 (different priority).
    - Rules 5-8 (edge rules) check left before right before top before bottom.
      After rotation, the check order changes.

    This function quantifies how often the priority breaks equivariance.
    """
    print("\n" + "=" * 80)
    print("SYMMETRY-BREAKING ANALYSIS")
    print("=" * 80)

    print("\nThe 12 rules have this priority order:")
    print("  1. L==T (corner)  2. R==T (corner)  3. R==B (corner)  4. L==B (corner)")
    print("  5. L (edge)  6. R (edge)  7. T (edge)  8. B (edge)")
    print("  9. TL (diag) 10. TR (diag) 11. BR (diag) 12. BL (diag)")
    print()
    print("This order is NOT symmetric under D4 transformations!")
    print("Example: Under rot90cw, 'left' becomes 'top' and 'top' becomes 'right'.")
    print("  Rule 5 (L!=center) checks before Rule 7 (T!=center).")
    print("  After rot90cw, old-L is now at Top, old-T is now at Right.")
    print("  So old-Rule-5 should map to Rule 7, but old-Rule-7 maps to Rule 6.")
    print("  If both fire, priority order gives a different winner!")
    print()

    # Concrete example: center=0, left=1, top=2 (both differ from center)
    # Original: Rule 1 checks left==top: 1!=2, fails. Rule 5: left!=center, fires. Dir=Right.
    # After rot90cw: old-left(=1) at Top, old-top(=2) at Right.
    # Grid: center=0, right=2, top=1 (and others=0).
    # Rule 1: left(=0)==top(=1)? No. Rule 2: right(=2)==top(=1)? No.
    # Rule 5: left(=0)!=center(=0)? No.
    # Rule 6: right(=2)!=center(=0)? Yes. Dir=Left. tex=2.
    #
    # Expected: remap[Right] = Bottom.
    # Actual: Left.
    # MISMATCH! Because in original, Rule 5 fired (left first), but after rotation,
    # Rule 6 fires (right before top), giving Left instead of Bottom.

    print("Concrete example:")
    print("  Original: center=0, left=1, top=2")
    g = np.zeros((5, 5), dtype=np.int32)
    g[1, 2] = 1  # left
    g[2, 3] = 2  # top
    orig_dir, orig_tex = apply_blend_rules(g)
    print(f"  Rule fires: dir={DIR_NAMES.get(orig_dir, 'None')}, tex={orig_tex}")

    tg = transform_grid(g, "rot90cw")
    trans_dir, trans_tex = apply_blend_rules(tg)
    print(f"  After rot90cw: dir={DIR_NAMES.get(trans_dir, 'None')}, tex={trans_tex}")

    remaps = build_all_remaps()
    expected = remaps["rot90cw"].get(orig_dir, orig_dir) if orig_dir else None
    print(f"  Expected direction: {DIR_NAMES.get(expected, 'None')}")
    print(f"  MATCH: {trans_dir == expected}")

    print()
    print("This happens because the rule priority order is asymmetric:")
    print("  - Edge rules check L,R,T,B in that fixed order")
    print("  - Corner rules check L+T, R+T, R+B, L+B in that fixed order")
    print("  - Diagonal rules check TL,TR,BR,BL in that fixed order")
    print()
    print("When multiple neighbors differ from center, the FIRST matching rule wins.")
    print("Rotation changes which neighbor is checked first, breaking equivariance.")

    # Quantify: what fraction of blended cells have multiple different neighbors?
    # (These are the ones at risk of equivariance breaking.)
    print("\n--- Quantifying multi-neighbor cases ---")
    rng = np.random.default_rng(42)
    n_tests = 200_000
    n_single_diff = 0
    n_multi_diff = 0
    n_blended = 0

    for _ in range(n_tests):
        n_tex = rng.integers(2, 6)
        g = rng.integers(0, n_tex, size=(5, 5), dtype=np.int32)
        d, t = apply_blend_rules(g)
        if d is None:
            continue
        n_blended += 1

        # Count how many of the 8 immediate neighbors differ
        center = g[2, 2]
        neighbors = [g[1,2], g[3,2], g[2,3], g[2,1], g[1,3], g[3,3], g[1,1], g[3,1]]
        n_diff = sum(1 for n in neighbors if n != center)
        if n_diff <= 1:
            n_single_diff += 1
        else:
            n_multi_diff += 1

    print(f"  Random grids that produce a blend: {n_blended}/{n_tests}")
    if n_blended > 0:
        print(f"  Single different neighbor: {n_single_diff} ({100*n_single_diff/n_blended:.1f}%)")
        print(f"  Multiple different neighbors: {n_multi_diff} ({100*n_multi_diff/n_blended:.1f}%)")
        print(f"  -> Multi-neighbor cases are at risk of equivariance breaking!")


def main():
    print("=" * 80)
    print("D4 SYMMETRY VERIFICATION FOR RA3 BLEND RULES")
    print("=" * 80)

    # Step 1: Build direction remaps
    print("\n[Step 1] Building direction remaps for all 8 D4 transforms...")
    remaps = build_all_remaps()

    for tname, remap in remaps.items():
        if tname == "identity":
            continue
        print(f"\n  {tname}:")
        for d in ALL_DIRS:
            src = DIR_NAMES[d]
            dst = DIR_NAMES.get(remap[d], str(remap[d]))
            print(f"    {src:20s} -> {dst}")

    # Step 2: Position encoding analysis
    analyze_position_encoding()

    # Step 3: Synthetic equivariance test
    synthetic_results = verify_rule_equivariance_synthetic()

    # Step 4: Analyze WHY symmetry breaks
    analyze_rule_symmetry_breaking()

    # Step 5: Test on real maps
    print("\n" + "=" * 80)
    print("REAL MAP VERIFICATION")
    print("=" * 80)

    maps_dir = Path(r"E:\DL\Projects\Ra3 texture gen\RA 3 maps\RA3 Official maps")

    # Collect map files from multiple directories
    map_files = []
    for subdir in sorted(maps_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("_") or subdir.name.startswith("ARCHON") or subdir.name.startswith("Ban"):
            continue
        for f in sorted(subdir.glob("*.map")):
            if "blendless" not in f.stem.lower() and "predicted" not in f.stem.lower():
                map_files.append(f)

    print(f"Found {len(map_files)} map files")

    # Test on a selection of maps (different styles)
    test_maps = map_files[:10]  # First 10 maps

    all_results = {}
    for mp in test_maps:
        try:
            result = verify_d4_on_map(mp, remaps, max_samples=3000)
            if result:
                all_results[mp.name] = result
        except Exception as e:
            print(f"  ERROR: {e}")

    # Step 6: Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL MAPS")
    print("=" * 80)

    transform_names = ["identity", "rot90cw", "rot180", "rot270cw", "flipX", "flipY", "flipDiag", "flipAntiDiag"]

    for tname in transform_names:
        total_cells = 0
        total_match = 0
        total_both_blend = 0
        total_dir_match = 0
        total_dir_mismatch = 0

        for map_name, map_results in all_results.items():
            if tname not in map_results:
                continue
            r = map_results[tname]
            total_cells += r["total"]
            total_match += r["exact_match"]
            total_both_blend += r["both_blend"]
            total_dir_match += r["dir_match"]
            total_dir_mismatch += r["dir_mismatch"]

        if total_cells > 0:
            match_rate = 100.0 * total_match / total_cells
            print(f"\n  {tname}:")
            print(f"    Overall equivariance: {total_match}/{total_cells} = {match_rate:.2f}%")
            if total_both_blend > 0:
                dir_rate = 100.0 * total_dir_match / total_both_blend
                print(f"    Direction match (when both blend): {total_dir_match}/{total_both_blend} = {dir_rate:.2f}%")
                print(f"    Direction mismatches: {total_dir_mismatch}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    # Check synthetic results for the answer
    any_break = False
    for tname in ["rot90cw", "rot180", "rot270cw", "flipX", "flipY", "flipDiag", "flipAntiDiag"]:
        if tname in synthetic_results:
            s = synthetic_results[tname]
            if s["mismatch"] > 0:
                any_break = True
                break

    if any_break:
        print("""
FINDING: D4 symmetry DOES NOT perfectly preserve the 12 blend rules.

The blend rules use a PRIORITY ORDER (left-before-right, top-before-bottom)
that is inherently asymmetric. When a cell has multiple different neighbors,
the first matching rule wins. Rotation/reflection changes which rule fires
first, producing a different direction.

This affects cells where:
  - Multiple cardinal neighbors differ from center (edge priority breaks)
  - Different corner pairs match (corner priority breaks)
  - Mix of diagonal-only neighbors (diagonal priority breaks)

IMPACT ON AUGMENTATION:
  1. For SINGLE different neighbor cells: D4 augmentation is EXACT (100%).
  2. For MULTI different neighbor cells: augmentation is APPROXIMATE.
  3. Position encoding (tile_value = tex_id*64 + offset(x,y)) REQUIRES
     re-encoding at the new position. Raw tile array rotation is WRONG.

RECOMMENDATIONS:
  - D4 augmentation is safe for the 5x5 TEXTURE grid (after decoding from tiles).
  - The augmented blend DIRECTION must be recomputed by re-applying the rules,
    NOT by remapping the original direction (because priority order breaks).
  - Alternative: only augment the texture grid, then predict blends from scratch
    using the rules. This is equivariant by construction.
  - For training data: augmentation introduces label noise proportional to the
    multi-neighbor cell rate. This may be acceptable if the noise rate is low.
""")
    else:
        print("\nFINDING: D4 symmetry perfectly preserves the 12 blend rules.")
        print("This should not happen given the asymmetric priority order...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
