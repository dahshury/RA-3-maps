#!/usr/bin/env python3
"""Smoke-test the comprehensive feature extractor on a single map.

Prints shape, channel names, palette, and saves a grid of channel heatmaps so
features can be eyeballed before any training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor import Ra3Map  # noqa: E402
from map_processor.features import extract_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG grid; default: <src_stem>_features.png next to src")
    ap.add_argument("--no-grid", action="store_true",
                    help="Skip writing the channel-grid PNG (still prints summary)")
    args = ap.parse_args()

    print(f"Source: {args.src}")
    m = Ra3Map(str(args.src)); m.parse()
    fs = extract_features(m, extract_target=True, style_id=0, n_styles=8)

    print(f"Map size: {fs.width} x {fs.height}")
    print(f"Feature stack: {fs.array.shape}, dtype={fs.array.dtype}")
    print(f"Channels ({len(fs.names)}):")
    for i, n in enumerate(fs.names):
        c = fs.array[i]
        print(f"  {i:3d}  {n:32s}  min={c.min():+.3f}  max={c.max():+.3f}  mean={c.mean():+.3f}")
    print(f"Palette ({len(fs.palette or [])}): {fs.palette}")
    print(f"Object tokens: {fs.object_tokens.shape if fs.object_tokens is not None else None}")
    if fs.target_tiles is not None:
        unique, counts = np.unique(fs.target_tiles, return_counts=True)
        print(f"Target tile distribution (palette_idx: count):")
        for u, c in zip(unique, counts):
            name = fs.palette[u] if 0 <= u < len(fs.palette) else "?"
            print(f"  {u:3d}  {name:32s}  {c}")

    if args.no_grid:
        return 0

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available — skipping grid")
        return 0

    out_path = args.out or (args.src.parent / f"{args.src.stem}_features.png")
    C = fs.array.shape[0]
    cols = 6
    rows = (C + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if i < C:
            ax.imshow(fs.array[i].T, origin="lower", cmap="viridis")
            ax.set_title(fs.names[i], fontsize=6)
        else:
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110)
    print(f"Saved channel grid: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
