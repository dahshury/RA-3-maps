"""
Diff BlendTileData blends between an original map and a "blendless" map.

This tool answers: "What blends exist in the original that are missing in blendless?"
It examines BOTH layers:
- BlendTileData.blends
- BlendTileData.single_edge_blends

It reports:
- Counts of nonzero cells per layer in each map
- Counts of removed cells (orig nonzero -> blendless zero)
- A breakdown of removed blend *types*:
    (layer, primary_texture_name, secondary_texture_name, blend_direction_value) -> count
- A small sample of removed cells with coordinates and decoded textures/direction.

Usage:
  python scripts/diff_blends.py --orig "<orig.map>" --blendless "<blendless.map>" --out "<report.json>"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# allow direct run
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    # C# BlendTileData.GetTexture inverse
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _tex_name(blend: BlendTileData, tex_index: int) -> str:
    if tex_index < 0:
        return "<neg>"
    if tex_index >= len(blend.textures):
        return f"<out_of_range:{tex_index}>"
    t = blend.textures[tex_index]
    return getattr(t, "name", str(t))


@dataclass
class RemovedSample:
    layer: str
    x: int
    y: int
    primary_texture: str
    secondary_texture: str
    direction: int
    orig_index: int


def _analyze_layer(
    layer_name: str,
    orig_blend: BlendTileData,
    blendless_blend: BlendTileData,
    orig_grid: np.ndarray,
    blendless_grid: np.ndarray,
    max_samples: int,
) -> Tuple[Dict[Tuple[str, str, str, int], int], List[RemovedSample]]:
    """
    Returns:
    - counts: (layer, primary_name, secondary_name, dir_val) -> count
    - samples: list of RemovedSample
    """
    counts: Dict[Tuple[str, str, str, int], int] = {}
    samples: List[RemovedSample] = []

    w, h = orig_grid.shape
    for x in range(w):
        for y in range(h):
            oidx = int(orig_grid[x, y])
            if oidx <= 0:
                continue
            bidx = int(blendless_grid[x, y])
            if bidx != 0:
                continue  # not removed

            if not (1 <= oidx <= len(orig_blend.blend_info)):
                continue

            bi = orig_blend.blend_info[oidx - 1]
            primary_tex_i = _get_texture_from_tile(x, y, int(orig_blend.tiles[x, y]))
            secondary_tex_i = _get_texture_from_tile(x, y, int(bi.secondary_texture_tile))
            primary_name = _tex_name(orig_blend, primary_tex_i)
            secondary_name = _tex_name(orig_blend, secondary_tex_i)
            dir_val = int(bi.blend_direction)

            key = (layer_name, primary_name, secondary_name, dir_val)
            counts[key] = counts.get(key, 0) + 1

            if len(samples) < max_samples:
                samples.append(
                    RemovedSample(
                        layer=layer_name,
                        x=x,
                        y=y,
                        primary_texture=primary_name,
                        secondary_texture=secondary_name,
                        direction=dir_val,
                        orig_index=oidx,
                    )
                )

    return counts, samples


def main() -> int:
    p = argparse.ArgumentParser(description="Diff original vs blendless blends.")
    p.add_argument("--orig", required=True, help="Original .map")
    p.add_argument("--blendless", required=True, help="Blendless .map")
    p.add_argument("--out", default="", help="Optional JSON output path")
    p.add_argument("--max-samples", type=int, default=50, help="Max sample cells to include")
    args = p.parse_args()

    om = Ra3Map(args.orig)
    om.parse()
    bm = Ra3Map(args.blendless)
    bm.parse()

    ob = om.get_context().get_asset_by_type(BlendTileData)
    bb = bm.get_context().get_asset_by_type(BlendTileData)
    if ob is None or bb is None:
        raise SystemExit("Missing BlendTileData in one of the maps")

    report: Dict[str, object] = {}
    report["orig"] = str(Path(args.orig))
    report["blendless"] = str(Path(args.blendless))
    report["dims"] = {"w": ob.map_width, "h": ob.map_height}
    report["counts"] = {
        "orig_blends_nonzero": int((ob.blends > 0).sum()),
        "blendless_blends_nonzero": int((bb.blends > 0).sum()),
        "orig_single_edge_nonzero": int((ob.single_edge_blends > 0).sum()),
        "blendless_single_edge_nonzero": int((bb.single_edge_blends > 0).sum()),
        "orig_blend_info_len": int(len(ob.blend_info)),
        "blendless_blend_info_len": int(len(bb.blend_info)),
    }

    # Analyze removed cells for both layers
    counts_blends, samples_blends = _analyze_layer(
        "blends", ob, bb, np.asarray(ob.blends), np.asarray(bb.blends), args.max_samples
    )
    counts_se, samples_se = _analyze_layer(
        "single_edge_blends",
        ob,
        bb,
        np.asarray(ob.single_edge_blends),
        np.asarray(bb.single_edge_blends),
        args.max_samples,
    )

    merged_counts: Dict[str, int] = {}
    for (layer, prim, sec, d), c in {**counts_blends, **{k: v for k, v in counts_se.items()}}.items():
        key = f"{layer}|{prim}|{sec}|{d}"
        merged_counts[key] = merged_counts.get(key, 0) + int(c)

    # Top-N types
    top_types = sorted(merged_counts.items(), key=lambda kv: kv[1], reverse=True)
    report["removed_types_top"] = top_types[:200]
    report["removed_types_total_unique"] = len(merged_counts)
    report["samples"] = [asdict(s) for s in (samples_blends + samples_se)[: args.max_samples]]

    # Also count removed cell totals precisely
    report["removed_cells"] = {
        "blends": int(((ob.blends > 0) & (bb.blends == 0)).sum()),
        "single_edge_blends": int(((ob.single_edge_blends > 0) & (bb.single_edge_blends == 0)).sum()),
    }

    txt = json.dumps(report, indent=2)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(txt, encoding="utf-8")
        print(f"Wrote: {outp}")
    else:
        print(txt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())










