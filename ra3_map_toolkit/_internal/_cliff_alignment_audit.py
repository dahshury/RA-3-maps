"""
Per-cliff-wall alignment audit.

For every cliff-wall object in the source map, measure:
  1. Distance to its nearest impassable tile center (world units, playable-rel).
  2. Direction (degrees CCW from +X) from wall position to that tile center.
  3. Wall's own angle.
  4. Relative angle = (wall_angle - direction_to_cliff)  [in [-180, 180]]
        — this is the "where the cliff sits relative to the wall's facing".

For each transform (rot90cw / rot90ccw / rot180 / flipx / flipy):
  Repeat the measurement on the transformed map.
  Compare the relative angle to the original. For rotations it should be
  preserved exactly; for flips it should negate (or be 180-shifted with the
  +180° mesh fixup applied).

Reports:
  - per-wall distance drift (should be 0)
  - per-wall relative-angle drift (should match the per-transform expectation)
  - any walls that don't behave as predicted

Run:
  python _cliff_alignment_audit.py [--map PATH]
"""
from __future__ import annotations
import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from map_processor.assets.terrain.height_map_data import HeightMapData
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.utils.map_rotation import (
    rotate_context_right_angles,
    flip_context_axis,
    DEFAULT_FLIP_MESH_FIXUP,
)

DEFAULT_MAP = r"e:/DL/Projects/Ra3 texture gen/RA 3 maps/RA3 Official maps/2 II/map_mp_2_rao1.map"
WORLD_UNITS_PER_TILE = 10.0


def _is_cliff_wall_name(name: str) -> bool:
    if not name:
        return False
    n = name.upper()
    return ("CLIFFWALL" in n) or ("SEACLIFFWALL" in n)


def _impassable_full_tile_indices(ctx) -> np.ndarray:
    """Returns (N, 2) array of FULL-MAP (x_tile, y_tile) for every impassable cell."""
    b = ctx.get_asset_by_type(BlendTileData)
    imp = np.asarray(b.impassable, dtype=np.bool_)
    ys, xs = np.where(imp.T)
    return np.stack([xs.astype(np.int64), ys.astype(np.int64)], axis=1)


def _tile_to_playable_world(tx: int, ty: int, border: int) -> Tuple[float, float]:
    return ((tx - border + 0.5) * WORLD_UNITS_PER_TILE,
            (ty - border + 0.5) * WORLD_UNITS_PER_TILE)


def _impassable_centers_playable_world(ctx) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (centers_world_NX2, tile_indices_full_NX2)."""
    h = ctx.get_asset_by_type(HeightMapData)
    border = int(h.border_width)
    tiles = _impassable_full_tile_indices(ctx)
    pxs = (tiles[:, 0] - border + 0.5) * WORLD_UNITS_PER_TILE
    pys = (tiles[:, 1] - border + 0.5) * WORLD_UNITS_PER_TILE
    return np.stack([pxs, pys], axis=1), tiles


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference (a - b) in [-180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def measure_walls(ctx) -> List[Dict]:
    """
    One dict per cliff wall.
    `nearest_tile` is the FULL-MAP (tx, ty) tile coordinate of the nearest
    impassable cell; we keep it so we can re-find that *same* tile after a
    transform, sidestepping argmin ties between equidistant cells.
    """
    objs = ctx.get_asset_by_type(ObjectsList).map_objects
    centers, tile_idx = _impassable_centers_playable_world(ctx)
    out = []
    for i, o in enumerate(objs):
        if not _is_cliff_wall_name(o.type_name):
            continue
        ox, oy = float(o.position[0]), float(o.position[1])
        d2 = (centers[:, 0] - ox) ** 2 + (centers[:, 1] - oy) ** 2
        k = int(np.argmin(d2))
        dist = float(math.sqrt(d2[k]))
        dx = float(centers[k, 0] - ox)
        dy = float(centers[k, 1] - oy)
        dir_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        out.append({
            "idx": i,
            "type": o.type_name,
            "pos": (ox, oy),
            "angle": float(o.angle) % 360.0,
            "dist": dist,
            "dir_to_cliff": dir_deg,
            "rel_angle": _angle_diff_deg(float(o.angle) % 360.0, dir_deg),
            "nearest_tile": (int(tile_idx[k, 0]), int(tile_idx[k, 1])),
        })
    return out


def _predict_tile_after(tx_full: int, ty_full: int, full_w: int, full_h: int,
                        transform: str) -> Tuple[int, int]:
    """Predict where full-map tile (tx, ty) lands under a transform. Index space."""
    if transform == "rot90cw":
        return (ty_full, full_w - 1 - tx_full)
    if transform == "rot90ccw":
        return (full_h - 1 - ty_full, tx_full)
    if transform == "rot180":
        return (full_w - 1 - tx_full, full_h - 1 - ty_full)
    if transform == "flipy":  # mirror x
        return (full_w - 1 - tx_full, ty_full)
    if transform == "flipx":  # mirror y
        return (tx_full, full_h - 1 - ty_full)
    raise ValueError(transform)


def measure_wall_against_known_tile(ctx, wall_idx: int, target_tile: Tuple[int, int]) -> Dict:
    """Measure dist/dir/rel for a specific wall index against a specific full-map tile."""
    h = ctx.get_asset_by_type(HeightMapData)
    border = int(h.border_width)
    o = ctx.get_asset_by_type(ObjectsList).map_objects[wall_idx]
    ox, oy = float(o.position[0]), float(o.position[1])
    tx, ty = _tile_to_playable_world(target_tile[0], target_tile[1], border)
    dx = tx - ox; dy = ty - oy
    dist = math.hypot(dx, dy)
    dir_deg = math.degrees(math.atan2(dy, dx)) % 360.0
    return {
        "type": o.type_name,
        "angle": float(o.angle) % 360.0,
        "dist": dist,
        "dir_to_cliff": dir_deg,
        "rel_angle": _angle_diff_deg(float(o.angle) % 360.0, dir_deg),
    }


def expected_rel_angle_after(transform_name: str, orig_rel: float, fixup_applies: bool) -> float:
    """
    Returns the expected relative-angle after a transform.
      Rotations preserve rel_angle exactly.
      Flips negate it (mesh chirality lost).
      With +180 fixup applied to cliff walls, flip_x  rel = -orig - 180 (mod 360 -> [-180,180]).
                                          flip_y  rel = -orig - 180.
      [The +180 added to the wall's angle shifts (wall_angle - dir_to_cliff) by +180.]
    """
    if transform_name in ("rot90cw", "rot90ccw", "rot180"):
        return orig_rel
    if transform_name in ("flipx", "flipy"):
        rel = -orig_rel
        if fixup_applies:
            rel = rel + 180.0
        # normalize to [-180, 180]
        rel = (rel + 180.0) % 360.0 - 180.0
        return rel
    raise ValueError(transform_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=DEFAULT_MAP)
    args = ap.parse_args()

    src = Ra3Map(args.map)
    src.parse()
    base_ctx = src.get_context()

    base_meas = measure_walls(base_ctx)
    base_by_idx = {m["idx"]: m for m in base_meas}
    print(f"Source: {args.map}")
    print(f"Cliff-wall objects measured: {len(base_meas)}")

    # Print original alignment summary
    rels = [m["rel_angle"] for m in base_meas]
    dists = [m["dist"] for m in base_meas]
    print(f"\nORIGINAL stats:")
    print(f"  rel_angle (wall_angle - dir_to_cliff):  mean={np.mean(rels):+.1f}  "
          f"min={min(rels):+.1f}  max={max(rels):+.1f}")
    print(f"  dist to nearest cliff tile (world u):   mean={np.mean(dists):.1f}  "
          f"min={min(dists):.1f}  max={max(dists):.1f}")

    transforms = [
        ("rot90cw",  lambda c: rotate_context_right_angles(c, 90, clockwise=True)),
        ("rot90ccw", lambda c: rotate_context_right_angles(c, 90, clockwise=False)),
        ("rot180",   lambda c: rotate_context_right_angles(c, 180, clockwise=True)),
        ("flipx",    lambda c: flip_context_axis(c, axis="x")),
        ("flipy",    lambda c: flip_context_axis(c, axis="y")),
    ]

    base_h = base_ctx.get_asset_by_type(HeightMapData)
    full_w_orig = int(base_h.map_width)
    full_h_orig = int(base_h.map_height)

    for tname, fn in transforms:
        ctx = copy.deepcopy(base_ctx)
        fn(ctx)

        bad_dist = 0
        bad_rel = 0
        worst_dist_drift = 0.0
        worst_rel_drift = 0.0
        worst_examples = []

        for idx, om in base_by_idx.items():
            # Predict where the wall's original-nearest cliff tile lands after the transform.
            new_tile = _predict_tile_after(
                om["nearest_tile"][0], om["nearest_tile"][1],
                full_w_orig, full_h_orig, tname,
            )
            nm = measure_wall_against_known_tile(ctx, idx, new_tile)
            d_drift = abs(nm["dist"] - om["dist"])
            fixup_applies = (tname in ("flipx", "flipy")) and any(
                s.upper() in om["type"].upper() for s in DEFAULT_FLIP_MESH_FIXUP
            )
            exp_rel = expected_rel_angle_after(tname, om["rel_angle"], fixup_applies)
            r_drift = abs(_angle_diff_deg(nm["rel_angle"], exp_rel))

            if d_drift > 1.0:
                bad_dist += 1
                worst_dist_drift = max(worst_dist_drift, d_drift)
            if r_drift > 1.0:
                bad_rel += 1
                worst_rel_drift = max(worst_rel_drift, r_drift)
                if len(worst_examples) < 5:
                    worst_examples.append((om["type"], om["nearest_tile"], om["rel_angle"], nm["rel_angle"], exp_rel, r_drift))

        flag = "OK " if (bad_dist == 0 and bad_rel == 0) else "BAD"
        print(f"\n{flag} [{tname}]  (tracking same tile across transform)")
        print(f"     dist drifts > 1u:        {bad_dist}/{len(base_meas)}   worst: {worst_dist_drift:.2f}")
        print(f"     rel-angle drifts > 1deg: {bad_rel}/{len(base_meas)}   worst: {worst_rel_drift:.2f}")
        for tp, ntile, orel, nrel, erel, drift in worst_examples:
            print(f"       {tp} (orig_tile={ntile}): orig_rel={orel:+.1f}  got={nrel:+.1f}  expected={erel:+.1f}  drift={drift:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
