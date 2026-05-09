#!/usr/bin/env python3
"""Per-biome texture lookup tables (Tier 2.7).

For each (style_id, segmentation_class) pair, compute:
  - The empirical distribution of texture-vocab indices observed at that
    seg-class in maps belonging to that style
  - Top-K allowed textures for hard-constraint inference
  - Class probability prior for soft-constraint inference

Output: training_outputs/texture_transfer/biome_lookup.json
  {
    "<style_id>": {
      "<seg_class>": {
        "top_k": [vocab_idx, ...],
        "probs": [(vocab_idx, prob), ...]   # sorted desc, truncated
      }
    }
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_python_tools_root()))
from map_processor.models.spade_texture_unet import discretize_input, SEG_N_CLASSES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path,
                    default=_python_tools_root() / "training_outputs" / "texture_transfer")
    ap.add_argument("--index_file", type=str, default="curated_index.json")
    ap.add_argument("--top_k", type=int, default=12, help="textures to retain per (style, seg_class)")
    ap.add_argument("--prob_keep", type=int, default=24, help="prob entries kept per cell")
    args = ap.parse_args()

    index = json.loads((args.data_dir / args.index_file).read_text(encoding="utf-8"))
    n_styles = index["n_styles"]
    vocab_size = index["vocab_size"]
    ignore_index = index["ignore_index"]

    # Per-style, per-seg-class texture counter
    counters: dict[tuple[int, int], Counter] = defaultdict(Counter)

    train_recs = [r for r in index["records"] if r["split"] == "train"]
    print(f"Aggregating from {len(train_recs)} training maps...")

    for i, r in enumerate(train_recs, 1):
        d = np.load(args.data_dir / r["npz"])
        X = torch.from_numpy(d["X"]).unsqueeze(0).float()  # (1, C, W, H)
        y = d["y"]                                          # (W, H)
        seg = discretize_input(X)[0].numpy()                # (W, H)
        style = int(d["style_id"])

        # Aggregate (seg_class, texture_idx) pairs
        valid = y != ignore_index
        for cls in range(SEG_N_CLASSES):
            mask = (seg == cls) & valid
            if not mask.any():
                continue
            tex_indices = y[mask]
            for t in np.unique(tex_indices):
                counters[(style, cls)][int(t)] += int((tex_indices == t).sum())
        if i % 25 == 0:
            print(f"  [{i}/{len(train_recs)}]  cells filled: {len(counters)}")

    # Build the lookup output
    out: dict[str, dict] = {}
    for s in range(n_styles):
        per_seg: dict[str, dict] = {}
        for c in range(SEG_N_CLASSES):
            cnt = counters.get((s, c), Counter())
            if not cnt:
                continue
            top = cnt.most_common(args.top_k)
            top_k = [t for t, _ in top]
            tot = sum(cnt.values())
            probs = [(t, n / tot) for t, n in cnt.most_common(args.prob_keep)]
            per_seg[str(c)] = {
                "top_k": top_k,
                "probs": probs,
                "total_pixels": int(tot),
            }
        out[str(s)] = per_seg

    out_path = args.data_dir / "biome_lookup.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print a quick summary
    print(f"\nBuilt lookup for {n_styles} styles x {SEG_N_CLASSES} seg classes")
    for s in range(n_styles):
        n_filled = len(out.get(str(s), {}))
        print(f"  style {s}: {n_filled}/{SEG_N_CLASSES} seg classes filled")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
