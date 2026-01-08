"""
Extract supervised learning samples for "blend generation" from RA3 maps.

Given pairs of:
- an ORIGINAL map (with blends)
- a BLENDLESS map (same tiles, but blends + single_edge_blends cleared to 0)

we build a dataset:
  X = features derived from BLENDLESS tiles (and optionally elevation)
  y = labels derived from ORIGINAL BlendTileData (blends + single_edge_blends)

Labels per cell for each layer:
- present: 0/1
- secondary_texture_index: int (0..num_textures-1), -1 if absent
- direction: int (BlendDirection bitfield as int), 0 if absent

Features:
- texture neighborhood: window x window of texture indices (decoded from tile-values)
  flattened row-major
- optionally elevation neighborhood (same window), appended after textures

Output:
- .npz with arrays: X, y_blend_present, y_blend_sec, y_blend_dir,
                   y_se_present, y_se_sec, y_se_dir, map_id
- metadata JSON describing window size, include_elevation, and texture names per map

Usage examples:
  python scripts/extract_blend_dataset.py \\
    --pairs-json pairs.json \\
    --out dataset_blends.npz \\
    --window 5 --include-elevation \\
    --max-samples-per-map 50000 --seed 123

Where pairs.json is like:
[
  {"orig": "../RA3 Official maps/2 II/map_mp_2_rao1.map",
   "blendless": "../RA3 Official maps/2 II/map_mp_2_rao1_blendless.map",
   "id": "rao1"},
  ...
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

# allow direct run
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.assets.terrain.height_map_data import HeightMapData


def _get_texture_from_tile(x: int, y: int, tile_value: int) -> int:
    # C# BlendTileData.GetTexture inverse
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return (int(tile_value) - current) // 64


def _decode_texture_grid(blend: BlendTileData) -> np.ndarray:
    """Return (w,h) int32 grid of texture indices decoded from tile-values."""
    tiles = np.asarray(blend.tiles)
    w, h = tiles.shape
    tex = np.empty((w, h), dtype=np.int32)
    for x in range(w):
        for y in range(h):
            tex[x, y] = _get_texture_from_tile(x, y, int(tiles[x, y]))
    return tex


def _labels_for_layer(
    blend: BlendTileData,
    grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a blend layer grid (w,h) of indices into blend_info (1-based), produce:
    - present: uint8 (0/1)
    - secondary texture index: int16 (-1..)
    - direction int16 (0..)
    """
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
                # out-of-range (shouldn't happen in official maps)
                continue
            bi = info[idx - 1]
            sec[x, y] = int(_get_texture_from_tile(x, y, int(bi.secondary_texture_tile)))
            direction[x, y] = int(bi.blend_direction)
    return present, sec, direction


def _sample_indices(
    w: int,
    h: int,
    stride: int,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return flattened indices into (w,h) that respect stride and max_samples.
    """
    xs = np.arange(0, w, stride, dtype=np.int32)
    ys = np.arange(0, h, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    flat = (grid_x * h + grid_y).reshape(-1)
    if max_samples > 0 and flat.size > max_samples:
        sel = rng.choice(flat.size, size=max_samples, replace=False)
        flat = flat[sel]
    return flat


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract supervised samples for blend generation.")
    ap.add_argument("--pairs-json", required=True, help="JSON list of {orig, blendless, id?}")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--meta-out", default="", help="Optional metadata JSON path (default: <out>.json)")

    ap.add_argument("--window", type=int, default=5, help="Odd window size for neighborhood features (e.g., 3,5,7)")
    ap.add_argument("--include-elevation", action="store_true", help="Append elevation neighborhood to features")
    ap.add_argument("--stride", type=int, default=1, help="Sample every N cells in x/y (reduces dataset size)")
    ap.add_argument("--max-samples-per-map", type=int, default=0, help="Cap samples per map (0 = no cap)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for sampling")
    args = ap.parse_args()

    win = int(args.window)
    if win <= 0 or win % 2 != 1:
        raise SystemExit("--window must be a positive odd integer")
    pad = win // 2

    pairs = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise SystemExit("pairs-json must be a non-empty JSON list")

    rng = np.random.default_rng(args.seed)

    X_parts: List[np.ndarray] = []
    ybp_parts: List[np.ndarray] = []
    ybs_parts: List[np.ndarray] = []
    ybd_parts: List[np.ndarray] = []
    ysp_parts: List[np.ndarray] = []
    yss_parts: List[np.ndarray] = []
    ysd_parts: List[np.ndarray] = []
    map_id_parts: List[np.ndarray] = []

    meta: Dict[str, object] = {
        "window": win,
        "include_elevation": bool(args.include_elevation),
        "stride": int(args.stride),
        "max_samples_per_map": int(args.max_samples_per_map),
        "seed": int(args.seed),
        "pairs": [],
    }

    for i, item in enumerate(pairs):
        orig_path = Path(item["orig"])
        blendless_path = Path(item["blendless"])
        map_id = str(item.get("id", orig_path.stem))

        om = Ra3Map(str(orig_path)); om.parse()
        bm = Ra3Map(str(blendless_path)); bm.parse()
        ob = om.get_context().get_asset_by_type(BlendTileData)
        bb = bm.get_context().get_asset_by_type(BlendTileData)
        if ob is None or bb is None:
            raise SystemExit(f"Missing BlendTileData for pair id={map_id}")

        if ob.tiles.shape != bb.tiles.shape:
            raise SystemExit(f"tiles shape mismatch for pair id={map_id}: orig={ob.tiles.shape} blendless={bb.tiles.shape}")

        w, h = ob.tiles.shape

        # Features from blendless textures (decoded)
        tex_grid = _decode_texture_grid(bb).astype(np.int16)
        tex_pad = np.pad(tex_grid, pad_width=((pad, pad), (pad, pad)), mode="edge")
        tex_windows = np.lib.stride_tricks.sliding_window_view(tex_pad, (win, win))
        # tex_windows shape: (w, h, win, win)
        tex_feat = tex_windows.reshape(w * h, win * win)

        feat_list = [tex_feat]

        if args.include_elevation:
            bh = bm.get_context().get_asset_by_type(HeightMapData)
            if bh is None or bh.elevations is None:
                raise SystemExit(f"--include-elevation requested but HeightMapData missing for id={map_id}")
            elev = np.asarray(bh.elevations, dtype=np.float32)
            if elev.shape != (w, h):
                raise SystemExit(f"elevation shape mismatch id={map_id}: elev={elev.shape} tiles={(w,h)}")
            elev_pad = np.pad(elev, pad_width=((pad, pad), (pad, pad)), mode="edge")
            elev_windows = np.lib.stride_tricks.sliding_window_view(elev_pad, (win, win))
            elev_feat = elev_windows.reshape(w * h, win * win)
            feat_list.append(elev_feat)

        X_full = np.concatenate(feat_list, axis=1)

        # Labels from original
        ybp, ybs, ybd = _labels_for_layer(ob, np.asarray(ob.blends))
        ysp, yss, ysd = _labels_for_layer(ob, np.asarray(ob.single_edge_blends))

        # Sample indices
        flat_idx = _sample_indices(w, h, stride=max(1, int(args.stride)), max_samples=int(args.max_samples_per_map), rng=rng)

        X_parts.append(X_full[flat_idx])
        ybp_parts.append(ybp.reshape(-1)[flat_idx])
        ybs_parts.append(ybs.reshape(-1)[flat_idx])
        ybd_parts.append(ybd.reshape(-1)[flat_idx])
        ysp_parts.append(ysp.reshape(-1)[flat_idx])
        yss_parts.append(yss.reshape(-1)[flat_idx])
        ysd_parts.append(ysd.reshape(-1)[flat_idx])
        map_id_parts.append(np.full((flat_idx.shape[0],), i, dtype=np.int16))

        meta["pairs"].append(
            {
                "id": map_id,
                "orig": str(orig_path),
                "blendless": str(blendless_path),
                "textures": [t.name for t in ob.textures],
                "w": int(w),
                "h": int(h),
                "samples": int(flat_idx.shape[0]),
            }
        )

        print(f"[{i+1}/{len(pairs)}] {map_id}: extracted {flat_idx.shape[0]} samples, feat_dim={X_full.shape[1]}")

    X = np.concatenate(X_parts, axis=0)
    y_blend_present = np.concatenate(ybp_parts, axis=0)
    y_blend_sec = np.concatenate(ybs_parts, axis=0)
    y_blend_dir = np.concatenate(ybd_parts, axis=0)
    y_se_present = np.concatenate(ysp_parts, axis=0)
    y_se_sec = np.concatenate(yss_parts, axis=0)
    y_se_dir = np.concatenate(ysd_parts, axis=0)
    map_id_arr = np.concatenate(map_id_parts, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_blend_present=y_blend_present,
        y_blend_sec=y_blend_sec,
        y_blend_dir=y_blend_dir,
        y_se_present=y_se_present,
        y_se_sec=y_se_sec,
        y_se_dir=y_se_dir,
        map_id=map_id_arr,
    )
    print(f"Wrote dataset: {out_path} (samples={X.shape[0]}, feat_dim={X.shape[1]})")

    meta_out = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".json")
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










