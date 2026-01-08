"""
Check whether blends from an original map are rotated correctly in a rotated map.

This uses the "blendless" concept implicitly: we treat EVERY nonzero blend cell in the original
as "a blend that exists" and verify it appears at the rotated location in the rotated map.

It checks BOTH layers:
- BlendTileData.blends
- BlendTileData.single_edge_blends

For each original blended cell (x,y):
1) Compute rotated coordinate (rx,ry)
2) Verify the rotated map has a blend at (rx,ry)
3) Decode:
   - primary texture index at (x,y) in orig and at (rx,ry) in rotated
   - secondary texture index from blend_info using GetTexture inverse at that same coordinate
   - blend direction rotated by turns

Reports:
- counts of OK / missing / wrong primary / wrong secondary / wrong direction
- top mismatch types: (layer, primary_orig, secondary_orig, dir_orig) -> count
- sample rows with coordinates and what differed

Usage:
  python scripts/check_rotated_blends.py --orig "<orig.map>" --rot "<rot.map>" --degrees 180 --out report.json

Optionally, pass --blendless "<blendless.map>" to define the blend-set as:
  cells where (orig layer > 0) AND (blendless layer == 0)
This matches the common "Edit -> Remove all texture blends" output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData


def _rotate_xy_index(old_x: int, old_y: int, w: int, h: int, turns_cw: int) -> Tuple[int, int]:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return old_x, old_y
    if turns_cw == 1:
        return old_y, w - 1 - old_x
    if turns_cw == 2:
        return w - 1 - old_x, h - 1 - old_y
    return h - 1 - old_y, old_x


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


_ROT_180 = {1: 17, 17: 1, 2: 18, 18: 2, 4: 24, 24: 4, 8: 20, 20: 8, 36: 56, 56: 36, 40: 52, 52: 40, 33: 49, 49: 33}
_ROT_90 = {1: 18, 18: 17, 17: 2, 2: 1, 4: 8, 8: 20, 20: 24, 24: 4, 36: 52, 52: 56, 56: 40, 40: 36, 33: 50, 50: 49, 49: 34, 34: 33}


def _rotate_dir(v: int, turns_cw: int) -> int:
    turns_cw = turns_cw % 4
    if turns_cw == 0:
        return v
    if turns_cw == 2:
        return _ROT_180.get(v, v)
    out = v
    for _ in range(turns_cw):
        out = _ROT_90.get(out, out)
    return out


def _tex_name(blend: BlendTileData, tex_idx: int) -> str:
    if tex_idx < 0:
        return "<neg>"
    if tex_idx >= len(blend.textures):
        return f"<oor:{tex_idx}>"
    return blend.textures[tex_idx].name


@dataclass
class Sample:
    layer: str
    x: int
    y: int
    rx: int
    ry: int
    primary_orig: str
    primary_rot: str
    secondary_orig: str
    secondary_rot: str
    dir_orig: int
    dir_expected: int
    dir_rot: int
    status: str


def main() -> int:
    p = argparse.ArgumentParser(description="Check if original blends are rotated correctly in a rotated map.")
    p.add_argument("--orig", required=True)
    p.add_argument("--blendless", default="", help="Optional blendless .map (removed blends define the check-set)")
    p.add_argument("--rot", required=True)
    p.add_argument("--degrees", type=int, required=True)
    p.add_argument("--ccw", action="store_true")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--out", default="")
    args = p.parse_args()

    deg = int(args.degrees) % 360
    if deg % 90 != 0:
        raise SystemExit("degrees must be multiple of 90")
    turns = (deg // 90) % 4
    if args.ccw:
        turns = (-turns) % 4

    om = Ra3Map(args.orig); om.parse()
    rm = Ra3Map(args.rot); rm.parse()
    ob = om.get_context().get_asset_by_type(BlendTileData)
    rb = rm.get_context().get_asset_by_type(BlendTileData)
    if ob is None or rb is None:
        raise SystemExit("Missing BlendTileData")

    blendless_blend: BlendTileData | None = None
    if args.blendless:
        bm = Ra3Map(args.blendless); bm.parse()
        blendless_blend = bm.get_context().get_asset_by_type(BlendTileData)
        if blendless_blend is None:
            raise SystemExit("Missing BlendTileData in blendless map")

    w, h = ob.map_width, ob.map_height
    if (w, h) != (rb.map_width, rb.map_height):
        raise SystemExit(f"Map dims differ: orig=({w},{h}) rot=({rb.map_width},{rb.map_height})")
    if blendless_blend is not None and (w, h) != (blendless_blend.map_width, blendless_blend.map_height):
        raise SystemExit(f"Blendless dims differ: orig=({w},{h}) blendless=({blendless_blend.map_width},{blendless_blend.map_height})")

    def check_layer(layer_name: str, ogrid: np.ndarray, rgrid: np.ndarray, bgrid: np.ndarray | None) -> Dict[str, object]:
        ok = missing = wrong_primary = wrong_secondary = wrong_dir = 0
        mismatch_types: Dict[str, int] = {}
        samples: List[Sample] = []

        for x in range(w):
            for y in range(h):
                oidx = int(ogrid[x, y])
                if oidx <= 0:
                    continue
                if bgrid is not None:
                    # Only check blends that were removed in blendless
                    if int(bgrid[x, y]) != 0:
                        continue
                rx, ry = _rotate_xy_index(x, y, w, h, turns)
                ridx = int(rgrid[rx, ry])

                # decode orig textures
                p_o = _get_texture_from_tile(x, y, int(ob.tiles[x, y]))
                p_r = _get_texture_from_tile(rx, ry, int(rb.tiles[rx, ry]))
                prim_o = _tex_name(ob, p_o)
                prim_r = _tex_name(rb, p_r)

                if ridx <= 0:
                    missing += 1
                    key = f"{layer_name}|{prim_o}|<missing>|{oidx}"
                    mismatch_types[key] = mismatch_types.get(key, 0) + 1
                    if len(samples) < args.max_samples:
                        samples.append(Sample(layer_name, x, y, rx, ry, prim_o, prim_r, "<n/a>", "<n/a>", -1, -1, -1, "missing"))
                    continue

                if not (1 <= oidx <= len(ob.blend_info)) or not (1 <= ridx <= len(rb.blend_info)):
                    missing += 1
                    if len(samples) < args.max_samples:
                        samples.append(Sample(layer_name, x, y, rx, ry, prim_o, prim_r, "<idx_oob>", "<idx_oob>", -1, -1, -1, "index_oob"))
                    continue

                obi = ob.blend_info[oidx - 1]
                rbi = rb.blend_info[ridx - 1]

                s_o_idx = _get_texture_from_tile(x, y, int(obi.secondary_texture_tile))
                s_r_idx = _get_texture_from_tile(rx, ry, int(rbi.secondary_texture_tile))
                sec_o = _tex_name(ob, s_o_idx)
                sec_r = _tex_name(rb, s_r_idx)

                d_o = int(obi.blend_direction)
                d_exp = _rotate_dir(d_o, turns)
                d_r = int(rbi.blend_direction)

                status = "ok"
                if p_o != p_r:
                    wrong_primary += 1
                    status = "wrong_primary"
                if s_o_idx != s_r_idx:
                    wrong_secondary += 1
                    status = "wrong_secondary" if status == "ok" else status + "+wrong_secondary"
                if d_r != d_exp:
                    wrong_dir += 1
                    status = "wrong_dir" if status == "ok" else status + "+wrong_dir"

                if status == "ok":
                    ok += 1
                else:
                    k = f"{layer_name}|{prim_o}|{sec_o}|{d_o}|{status}"
                    mismatch_types[k] = mismatch_types.get(k, 0) + 1
                    if len(samples) < args.max_samples:
                        samples.append(Sample(layer_name, x, y, rx, ry, prim_o, prim_r, sec_o, sec_r, d_o, d_exp, d_r, status))

        top = sorted(mismatch_types.items(), key=lambda kv: kv[1], reverse=True)[:200]
        return {
            "ok": ok,
            "missing": missing,
            "wrong_primary": wrong_primary,
            "wrong_secondary": wrong_secondary,
            "wrong_dir": wrong_dir,
            "mismatch_types_top": top,
            "samples": [asdict(s) for s in samples],
        }

    report = {
        "orig": str(Path(args.orig)),
        "blendless": str(Path(args.blendless)) if args.blendless else "",
        "rot": str(Path(args.rot)),
        "degrees": args.degrees,
        "turns_cw": turns,
        "dims": {"w": w, "h": h},
        "layers": {
            "blends": check_layer(
                "blends",
                np.asarray(ob.blends),
                np.asarray(rb.blends),
                np.asarray(blendless_blend.blends) if blendless_blend is not None else None,
            ),
            "single_edge_blends": check_layer(
                "single_edge_blends",
                np.asarray(ob.single_edge_blends),
                np.asarray(rb.single_edge_blends),
                np.asarray(blendless_blend.single_edge_blends) if blendless_blend is not None else None,
            ),
        },
    }

    text = json.dumps(report, indent=2)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
        print(f"Wrote: {outp}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


