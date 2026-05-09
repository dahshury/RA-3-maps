"""
File-level round-trip verification: parse -> rotate -> save -> reparse -> rotate -> save ...
At the end, compare to the original. Catches save/load bugs that pure in-memory
tests miss.

Generates round-trip outputs you can also load in-game / WorldBuilder to confirm
visually that they match the original.
"""
from __future__ import annotations
import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

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
)


def _arr_eq(a, b):
    if a is None and b is None: return True
    if a is None or b is None: return False
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _resolve_blend_layer(b, layer):
    info = b.blend_info or []
    out = np.zeros(layer.shape, dtype=np.int64)
    for x in range(layer.shape[0]):
        for y in range(layer.shape[1]):
            idx = int(layer[x, y])
            if idx > 0 and idx <= len(info):
                bi = info[idx - 1]
                out[x, y] = (int(bi.secondary_texture_tile) << 32) | int(bi.blend_direction)
    return out


def compare_contexts(c1, c2) -> List[str]:
    fails = []
    h1 = c1.get_asset_by_type(HeightMapData); h2 = c2.get_asset_by_type(HeightMapData)
    if (h1.map_width, h1.map_height) != (h2.map_width, h2.map_height):
        fails.append(f"dims {h1.map_width}x{h1.map_height} vs {h2.map_width}x{h2.map_height}")
    if not _arr_eq(h1.elevations, h2.elevations):
        diff = int(np.sum(np.asarray(h1.elevations) != np.asarray(h2.elevations)))
        fails.append(f"elevations differ ({diff} cells)")
    b1 = c1.get_asset_by_type(BlendTileData); b2 = c2.get_asset_by_type(BlendTileData)
    if not _arr_eq(b1.tiles, b2.tiles):
        diff = int(np.sum(np.asarray(b1.tiles) != np.asarray(b2.tiles)))
        fails.append(f"tiles differ ({diff} cells)")
    if not _arr_eq(_resolve_blend_layer(b1, b1.blends), _resolve_blend_layer(b2, b2.blends)):
        fails.append("blends layer differs (semantic)")
    if not _arr_eq(_resolve_blend_layer(b1, b1.single_edge_blends),
                   _resolve_blend_layer(b2, b2.single_edge_blends)):
        fails.append("single_edge_blends layer differs (semantic)")
    if not _arr_eq(b1.passability, b2.passability):
        fails.append("passability differs")
    if not _arr_eq(b1.cliff_blends, b2.cliff_blends):
        fails.append("cliff_blends differs")
    o1 = c1.get_asset_by_type(ObjectsList); o2 = c2.get_asset_by_type(ObjectsList)
    if len(o1.map_objects) != len(o2.map_objects):
        fails.append(f"object count {len(o1.map_objects)} vs {len(o2.map_objects)}")
    else:
        bad_pos = bad_ang = 0
        for a, b in zip(o1.map_objects, o2.map_objects):
            for i in range(3):
                if abs(a.position[i] - b.position[i]) > 1e-2:
                    bad_pos += 1; break
            d = (a.angle - b.angle) % 360.0; d = min(d, 360.0 - d)
            if d > 1e-2: bad_ang += 1
        if bad_pos: fails.append(f"{bad_pos} objects with position drift")
        if bad_ang: fails.append(f"{bad_ang} objects with angle drift")
    return fails


def apply_step(ctx, step):
    if step[0] == "rot":
        rotate_context_right_angles(ctx, degrees=step[1], clockwise=step[2])
    elif step[0] == "flip":
        # Disable mesh fixup for round-trip tests so flip x flip = identity
        flip_context_axis(ctx, axis=step[1], mesh_fixup={})


def file_roundtrip(orig_path: Path, ops: List[Tuple], out_path: Path) -> List[str]:
    """
    Apply each op in `ops` sequentially. Between ops, write to a fresh .map file
    and reparse. The composition of `ops` should be the identity transform.
    The final saved file is `out_path`; we compare its parsed content to the
    original parsed content.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="rot_rt_"))
    try:
        # Reference: original parsed
        orig_m = Ra3Map(str(orig_path)); orig_m.parse()
        orig_ctx = orig_m.get_context()

        cur_path = orig_path
        for i, op in enumerate(ops):
            m = Ra3Map(str(cur_path)); m.parse()
            apply_step(m.get_context(), op)
            step_out = (tmp_dir / f"step_{i}.map") if i < len(ops) - 1 else out_path
            step_out.parent.mkdir(parents=True, exist_ok=True)
            m.save(str(step_out), compress=True)
            cur_path = step_out

        final_m = Ra3Map(str(out_path)); final_m.parse()
        return compare_contexts(orig_ctx, final_m.get_context())
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    orig = Path(args.map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("rt_rot90cwx4",   [("rot", 90, True)] * 4),
        ("rt_rot90ccwx4",  [("rot", 90, False)] * 4),
        ("rt_rot180x2",    [("rot", 180, True)] * 2),
        ("rt_cw_then_ccw", [("rot", 90, True), ("rot", 90, False)]),
        ("rt_flipx2",      [("flip", "x")] * 2),
        ("rt_flipy2",      [("flip", "y")] * 2),
        ("rt_flipx_flipy_rot180", [("flip", "x"), ("flip", "y"), ("rot", 180, True)]),
    ]

    failures = 0
    print(f"Source: {orig}\nOutput dir: {out_dir}\n")
    for name, ops in cases:
        out_path = out_dir / f"{orig.stem}_{name}.map"
        fails = file_roundtrip(orig, ops, out_path)
        if fails:
            failures += 1
            print(f"FAIL  {name}")
            for f in fails:
                print(f"      - {f}")
            print(f"      output: {out_path}")
        else:
            print(f"PASS  {name}  ({len(ops)} steps)  -> {out_path.name}")

    print(f"\n=== TOTAL FAILURES: {failures} ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
