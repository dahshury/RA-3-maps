"""
Pre-extract RA3 maps to per-map .npz files for fast U-Net training.

Each .npz contains the full 2D tensors needed to train a spatial model:
    tex_grid       [W, H] int32   - texture IDs per cell (decoded from tile values)
    elev_grid      [W, H] float32 - raw elevation per cell (0 if missing)
    blend_present  [W, H] uint8   - 1 where blend exists, else 0
    blend_mask     [W, H] uint8   - 8-bit neighbor mask (255 = ignore: blend present
                                    but secondary texture not in the 8 neighbors)
    blend_dir      [W, H] int16   - direction class (0..16), -1 where no blend
    se_present, se_mask, se_dir   - same for single_edge_blends layer

A small JSON manifest is written alongside with global vocab and per-map dims.

Run with multiprocessing across all .map files. Vectorizes the per-cell loops
that made the previous extractor slow.

Usage:
    python scripts/prepare_unet_dataset.py \\
        --maps_dir "../RA3 Official maps" \\
        --out_dir  "../blendinfo dataset/_generated/unet_data_v1" \\
        --workers  8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Bit layout for the 8-neighbor mask (matches train_blend_unet.py).
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# All distinct BlendDirection raw values observed in the v8 dataset.
DIRECTION_VALUES = [-1, 1, 2, 4, 8, 17, 18, 20, 24, 33, 34, 36, 40, 49, 50, 52, 56]
DIR_VAL_TO_CLASS = {v: i for i, v in enumerate(DIRECTION_VALUES)}
NUM_DIR_CLASSES = len(DIRECTION_VALUES)


def _decode_texture_grid(tiles: np.ndarray) -> np.ndarray:
    """Vectorized inverse of BlendTileData.GetTexture for a [W, H] tile grid."""
    w, h = tiles.shape
    xs = np.arange(w, dtype=np.int32)[:, None]
    ys = np.arange(h, dtype=np.int32)[None, :]
    row_first = (ys % 8) // 2 * 16 + (ys % 2) * 2
    current = (xs % 8) // 2 * 4 + (xs % 2) + row_first
    return ((tiles.astype(np.int32) - current) // 64).astype(np.int32)


def _decode_secondary_tile(tile_value: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized texture decode for secondary_texture_tile values at given (x, y)."""
    row_first = (y % 8) // 2 * 16 + (y % 2) * 2
    current = (x % 8) // 2 * 4 + (x % 2) + row_first
    return ((tile_value.astype(np.int32) - current) // 64).astype(np.int32)


def _layer_labels(
    blends_arr: np.ndarray,
    blend_info_secondary_tile: np.ndarray,
    blend_info_dir_raw: np.ndarray,
    tex_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (present, mask8, dir_class) labels for one blend layer.

    blends_arr:                  [W, H] uint16 - 1-indexed blend_info index, 0 = none
    blend_info_secondary_tile:   [N+1] int32   - sec tile per blend_info (index 0 unused)
    blend_info_dir_raw:          [N+1] int32   - raw direction value per blend_info
    tex_grid:                    [W, H] int32  - decoded center texture per cell

    Returns:
        present  [W, H] uint8   - 1 where blends_arr > 0
        mask8    [W, H] uint8   - 8-bit neighbor match mask, 255 = present-but-no-match
        dir_cls  [W, H] int16   - mapped direction class (0..16), -1 if no blend
    """
    w, h = blends_arr.shape
    present = (blends_arr > 0).astype(np.uint8)

    # Per-cell secondary texture and raw direction (vectorized indexing).
    blends_idx = blends_arr.astype(np.int32)  # [W, H], 0 = no blend
    # blend_info arrays are 1-indexed; blends_idx == 0 maps to "no blend" (sentinel 0)
    sec_tile_per_cell = blend_info_secondary_tile[blends_idx]   # [W, H]
    dir_raw_per_cell = blend_info_dir_raw[blends_idx]           # [W, H]

    # Decode secondary texture id per cell (vectorized).
    xs = np.arange(w, dtype=np.int32)[:, None]
    ys = np.arange(h, dtype=np.int32)[None, :]
    sec_tex = _decode_secondary_tile(sec_tile_per_cell, xs, ys)
    # Mark "no blend" cells with sec_tex = -1 so they never match a neighbor.
    sec_tex = np.where(present > 0, sec_tex, -1)

    # Build the 8-bit neighbor match mask (which neighbors equal the secondary texture).
    tex_pad = np.pad(tex_grid, ((1, 1), (1, 1)), mode="edge")
    mask8 = np.zeros((w, h), dtype=np.uint16)
    for ni, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        neigh = tex_pad[1 + dx: 1 + dx + w, 1 + dy: 1 + dy + h]
        hit = (present > 0) & (sec_tex >= 0) & (neigh == sec_tex)
        mask8 |= hit.astype(np.uint16) << np.uint16(ni)
    mask8 = mask8.astype(np.uint8)
    # If blend present but no neighbor matched, sentinel = 255.
    mask8 = np.where((present > 0) & (mask8 == 0), np.uint8(255), mask8).astype(np.uint8)

    # Direction class lookup: any value not in DIR_VAL_TO_CLASS becomes -1 (ignore).
    dir_cls = np.full((w, h), -1, dtype=np.int16)
    for raw_val, cls_idx in DIR_VAL_TO_CLASS.items():
        dir_cls[(present > 0) & (dir_raw_per_cell == raw_val)] = cls_idx
    return present, mask8, dir_cls


def _flatten_blend_info(blend_info_list) -> Tuple[np.ndarray, np.ndarray]:
    """Pack a Python list of BlendInfo objects into two parallel ndarrays.

    Returns (sec_tile, dir_raw) arrays of length N+1, with index 0 reserved as
    a sentinel ('no blend') so cells with blends_arr==0 can be vectorized-indexed
    safely.
    """
    n = len(blend_info_list)
    sec_tile = np.zeros(n + 1, dtype=np.int32)
    dir_raw = np.full(n + 1, -1, dtype=np.int32)
    for i, bi in enumerate(blend_info_list, start=1):
        sec_tile[i] = int(bi.secondary_texture_tile)
        dir_raw[i] = int(bi.blend_direction)
    return sec_tile, dir_raw


def _compute_distance_to_boundary(tex_grid: np.ndarray) -> np.ndarray:
    """Chebyshev distance from each cell to the nearest 4-neighbor boundary."""
    from scipy.ndimage import distance_transform_cdt
    w, h = tex_grid.shape
    pad = np.pad(tex_grid, ((1, 1), (1, 1)), mode="edge")
    diff = (
        (pad[1:1 + w, 1:1 + h] != pad[:w, 1:1 + h]) |
        (pad[1:1 + w, 1:1 + h] != pad[2:2 + w, 1:1 + h]) |
        (pad[1:1 + w, 1:1 + h] != pad[1:1 + w, :h]) |
        (pad[1:1 + w, 1:1 + h] != pad[1:1 + w, 2:2 + h])
    )
    return distance_transform_cdt(~diff, metric="chessboard").astype(np.float32)


def _compute_pattern_code(tex_grid: np.ndarray) -> np.ndarray:
    """Per-cell deterministic blend pattern code (0..12).

    Replicates generate_blendinfo_dataset.py L887-944. Empirically, on val
    maps, argmax(P(blend_dir | pattern_code)) gives ~75% accuracy alone --
    this is the feature the token model uses to leapfrog the U-Net's 0.43
    dir_acc to 0.89.
    """
    c = tex_grid
    padded = np.pad(c, 1, mode="edge")
    left   = padded[1:-1,  :-2]
    right  = padded[1:-1, 2:  ]
    top    = padded[ :-2, 1:-1]
    bottom = padded[2:  , 1:-1]
    tl     = padded[ :-2,  :-2]
    tr     = padded[ :-2, 2:  ]
    bl     = padded[2:  ,  :-2]
    br     = padded[2:  , 2:  ]

    pattern_tex = np.full(c.shape, -1, dtype=np.int32)
    pattern_dir = np.zeros(c.shape, dtype=np.int8)

    def _try(mask, tex, code):
        m = mask & (pattern_tex < 0)
        pattern_tex[m] = tex[m]
        pattern_dir[m] = code

    _try((left == top)    & (top != c),    top, 1)
    _try((right == top)   & (top != c),    top, 2)
    _try((right == bottom)& (bottom != c), bottom, 3)
    _try((left == bottom) & (bottom != c), bottom, 4)
    _try(left   != c, left,    5)
    _try(right  != c, right,   6)
    _try(top    != c, top,     7)
    _try(bottom != c, bottom,  8)
    _try(tl != c, tl, 9)
    _try(tr != c, tr, 10)
    _try(br != c, br, 11)
    _try(bl != c, bl, 12)
    return pattern_dir


def extract_one_map(map_path: str, out_path: str) -> Dict[str, object]:
    """
    Worker: parse a single .map file, build per-map tensors, save .npz.
    Returns a small dict with stats (or {'error': ...} on failure).
    """
    try:
        from map_processor.ra3map import Ra3Map
        from map_processor.assets.terrain.blend_tile_data import BlendTileData
        from map_processor.assets.terrain.height_map_data import HeightMapData

        m = Ra3Map(map_path)
        m.parse()
        ctx = m.get_context()
        blend = ctx.get_asset_by_type(BlendTileData)
        height = ctx.get_asset_by_type(HeightMapData)
        if blend is None or blend.tiles is None:
            return {"path": map_path, "error": "no BlendTileData"}

        tiles = np.asarray(blend.tiles, dtype=np.int32)
        w, h = tiles.shape
        tex_grid = _decode_texture_grid(tiles)
        dist_grid = _compute_distance_to_boundary(tex_grid)
        pattern_code = _compute_pattern_code(tex_grid)

        if height is not None and height.elevations is not None:
            eg = np.asarray(height.elevations, dtype=np.float32)
            elev_grid = eg if eg.shape == (w, h) else np.zeros((w, h), dtype=np.float32)
        else:
            elev_grid = np.zeros((w, h), dtype=np.float32)

        # Texture name list (stable per-map vocab; ordered as stored in the file).
        tex_names = [t.name for t in (blend.textures or [])]

        # Blend layer (regular blends).
        blend_info = blend.blend_info or []
        sec_tile, dir_raw = _flatten_blend_info(blend_info)
        b_present, b_mask, b_dir = _layer_labels(
            np.asarray(blend.blends, dtype=np.uint16),
            sec_tile, dir_raw, tex_grid,
        )

        # Single-edge blends (re-use the same blend_info list — they share it).
        if blend.single_edge_blends is not None:
            se_present, se_mask, se_dir = _layer_labels(
                np.asarray(blend.single_edge_blends, dtype=np.uint16),
                sec_tile, dir_raw, tex_grid,
            )
        else:
            se_present = np.zeros((w, h), dtype=np.uint8)
            se_mask = np.zeros((w, h), dtype=np.uint8)
            se_dir = np.full((w, h), -1, dtype=np.int16)

        np.savez_compressed(
            out_path,
            tex_grid=tex_grid,
            elev_grid=elev_grid,
            blend_present=b_present,
            blend_mask=b_mask,
            blend_dir=b_dir,
            se_present=se_present,
            se_mask=se_mask,
            se_dir=se_dir,
            dist_grid=dist_grid,
            pattern_code=pattern_code,
        )

        return {
            "path": map_path,
            "out": out_path,
            "w": int(w),
            "h": int(h),
            "n_blend": int(b_present.sum()),
            "n_se": int(se_present.sum()),
            "n_textures": len(tex_names),
            "tex_names": tex_names,
        }
    except Exception as e:
        import traceback
        return {"path": map_path, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=3)}


def find_map_files(root: Path) -> List[Path]:
    """Recursively find .map files, excluding generated/blendless/predicted artifacts."""
    maps = []
    for p in sorted(root.rglob("*.map")):
        name_lower = p.stem.lower()
        if "blendless" in name_lower:
            continue
        if any(token in name_lower for token in ("_predicted", "_original", "unet_", "archon_test")):
            continue
        if any(part.startswith("_") for part in p.relative_to(root).parts):
            continue
        maps.append(p)
    return maps


def main():
    ap = argparse.ArgumentParser(description="Pre-extract RA3 maps to .npz tensors for U-Net training.")
    ap.add_argument("--maps_dir", required=True, help="Root directory of .map files")
    ap.add_argument("--out_dir", required=True, help="Output directory for .npz files")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="Parallel worker processes (default: cpu_count - 1)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process this many maps (for smoke testing)")
    args = ap.parse_args()

    maps_root = Path(args.maps_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    map_files = find_map_files(maps_root)
    if args.limit > 0:
        map_files = map_files[:args.limit]
    print(f"Extracting {len(map_files)} maps from {maps_root}")
    print(f"Output dir: {out_dir}")
    print(f"Workers:    {args.workers}")
    if not map_files:
        print("No maps found.")
        return

    t0 = time.time()
    results: List[Dict] = []
    failures: List[Dict] = []

    if args.workers <= 1:
        for p in map_files:
            out_path = str(out_dir / f"{p.stem}.npz")
            r = extract_one_map(str(p), out_path)
            results.append(r)
            if "error" in r:
                failures.append(r)
                print(f"  [FAIL] {p.name}: {r['error']}")
            else:
                print(f"  [ OK ] {p.name}: {r['w']}x{r['h']} blend={r['n_blend']} se={r['n_se']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for p in map_files:
                out_path = str(out_dir / f"{p.stem}.npz")
                futures[pool.submit(extract_one_map, str(p), out_path)] = p

            done = 0
            for fut in as_completed(futures):
                p = futures[fut]
                r = fut.result()
                results.append(r)
                done += 1
                if "error" in r:
                    failures.append(r)
                    print(f"  [FAIL {done}/{len(map_files)}] {p.name}: {r['error']}", flush=True)
                else:
                    print(f"  [ OK {done}/{len(map_files)}] {p.name}: "
                          f"{r['w']}x{r['h']} blend={r['n_blend']} se={r['n_se']}", flush=True)

    elapsed = time.time() - t0

    # Build a global texture vocab from all per-map texture name lists.
    global_vocab: List[str] = []
    seen = {}
    for r in results:
        if "error" in r:
            continue
        for name in r["tex_names"]:
            if name not in seen:
                seen[name] = len(global_vocab)
                global_vocab.append(name)

    # Save manifest.
    manifest = {
        "maps_dir": str(maps_root),
        "out_dir": str(out_dir),
        "num_maps_total": len(map_files),
        "num_maps_ok": sum(1 for r in results if "error" not in r),
        "num_failures": len(failures),
        "elapsed_seconds": round(elapsed, 2),
        "global_texture_vocab_size": len(global_vocab),
        "direction_values": DIRECTION_VALUES,
        "neighbor_offsets": NEIGHBOR_OFFSETS,
        "maps": [
            {k: v for k, v in r.items() if k != "tex_names"}
            for r in results
        ],
        "global_texture_vocab": global_vocab,
        "failures": failures,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    print(f"Done in {elapsed:.1f}s. {manifest['num_maps_ok']}/{len(map_files)} OK, "
          f"{len(failures)} failed. Global vocab: {len(global_vocab)} textures.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
