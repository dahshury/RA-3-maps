"""
Remove all texture blends from an RA3 map (WorldBuilder-style "blendless" map).

Empirically, WB's "Remove all texture blends" output corresponds to:
- BlendTileData.blends -> all zeros
- BlendTileData.single_edge_blends -> all zeros
while leaving:
- BlendTileData.tiles (base textures)
- BlendTileData.blend_info table
- BlendTileData.blends_count
unchanged.

This script applies that transformation and writes a new .map.

Usage:
  python scripts/remove_all_texture_blends.py --in "<map.map>" --out "<map_blendless.map>"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData


def main() -> int:
    p = argparse.ArgumentParser(description="Remove all texture blends from an RA3 map.")
    p.add_argument("--in", dest="in_path", required=True, help="Input .map")
    p.add_argument("--out", dest="out_path", required=True, help="Output .map")
    p.add_argument("--no-compress", action="store_true", help="Write uncompressed output")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    m = Ra3Map(str(in_path))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset_by_type(BlendTileData)
    if blend is None:
        raise SystemExit("BlendTileData not found")

    # Zero both blend layers, preserving dtype/shape
    if blend.blends is not None:
        blend.blends = np.zeros_like(blend.blends, dtype=np.uint16)
    if blend.single_edge_blends is not None:
        blend.single_edge_blends = np.zeros_like(blend.single_edge_blends, dtype=np.uint16)

    # Keep blend_info and blends_count unchanged (matches observed WB output behavior)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=(not args.no_compress))
    print(f"Wrote blendless map: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










