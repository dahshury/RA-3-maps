"""
Auto-detect cliff-wall mirror-pair variants by comparing top-down screenshots.

For each variant N: flip its screenshot horizontally and vertically. Then compare
each flipped image to every other variant's unflipped image (using normalized
cross-correlation on luminance, robust to slight render differences). The
best-matching variant for each flip is the candidate mirror partner.

Output: a ranked list per variant of best matches, plus a summary of likely
mirror pairs (where A.flipH ~ B and B.flipH ~ A).
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

DEFAULT_DIR = r"e:/DL/Projects/Ra3 texture gen/RA 3 maps/wb/data/objectScreenShot"
DEFAULT_PREFIX = "YU_CliffWall"


def _load_lum(path: Path, size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    # Subtract sky/snow background so the rock blob dominates the correlation.
    a -= float(np.median(a))
    return a


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation in [-1, 1]."""
    ax, bx = a.flatten(), b.flatten()
    ax = ax - ax.mean(); bx = bx - bx.mean()
    na = float(np.linalg.norm(ax)); nb = float(np.linalg.norm(bx))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(ax, bx) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--top", type=int, default=3, help="Show top-N matches per variant")
    ap.add_argument("--threshold", type=float, default=0.85, help="NCC threshold for confident pair")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.dir) / f"{args.prefix}*.jpg")))
    if not files:
        print(f"no files matching {args.prefix}*.jpg in {args.dir}")
        return 1
    pat = re.compile(rf"{re.escape(args.prefix)}(\d+)\.jpg$", re.IGNORECASE)
    items: List[Tuple[int, Path, np.ndarray]] = []
    for f in files:
        m = pat.search(Path(f).name)
        if not m: continue
        idx = int(m.group(1))
        items.append((idx, Path(f), _load_lum(Path(f))))
    items.sort()
    print(f"Loaded {len(items)} variants of {args.prefix}.\n")

    # Pre-compute originals, h-flipped, v-flipped, 180-rotated.
    variants = {idx: img for idx, _, img in items}
    h_flip   = {idx: img[:, ::-1].copy() for idx, img in variants.items()}
    v_flip   = {idx: img[::-1, :].copy() for idx, img in variants.items()}
    rot180   = {idx: img[::-1, ::-1].copy() for idx, img in variants.items()}

    # For each variant, find best matches under each transform.
    print(f"{'variant':>10}  best_hflip(top{args.top})              best_vflip(top{args.top})              best_rot180(top{args.top})")
    for src_idx in sorted(variants):
        candidates_h = []
        candidates_v = []
        candidates_r = []
        for tgt_idx in sorted(variants):
            tgt = variants[tgt_idx]
            candidates_h.append((tgt_idx, _ncc(h_flip[src_idx], tgt)))
            candidates_v.append((tgt_idx, _ncc(v_flip[src_idx], tgt)))
            candidates_r.append((tgt_idx, _ncc(rot180[src_idx], tgt)))
        candidates_h.sort(key=lambda x: -x[1])
        candidates_v.sort(key=lambda x: -x[1])
        candidates_r.sort(key=lambda x: -x[1])
        def fmt(cs):
            return ", ".join(f"{i}:{s:+.2f}" for i, s in cs[:args.top])
        print(f"  {args.prefix}{src_idx:02d}  {fmt(candidates_h):<38}  {fmt(candidates_v):<38}  {fmt(candidates_r):<38}")

    # Mutual-best pairs under H-flip (A.flipH ~ B AND B.flipH ~ A).
    print("\nLikely mirror-pair (mutual best H-flip match, NCC >= threshold):")
    best_h = {}
    best_h_score = {}
    for src_idx in variants:
        best = max(((tgt, _ncc(h_flip[src_idx], variants[tgt])) for tgt in variants), key=lambda x: x[1])
        best_h[src_idx] = best[0]; best_h_score[src_idx] = best[1]
    seen = set()
    for a in sorted(best_h):
        b = best_h[a]
        if a == b: continue
        if (a, b) in seen or (b, a) in seen: continue
        if best_h.get(b) == a and best_h_score[a] >= args.threshold and best_h_score[b] >= args.threshold:
            print(f"  {args.prefix}{a:02d}  <->  {args.prefix}{b:02d}    (NCC: {best_h_score[a]:.2f} / {best_h_score[b]:.2f})")
            seen.add((a, b))

    # Self-symmetric (A.flipH ~ A) — these don't need a substitution.
    print("\nSelf-symmetric variants (mesh is its own H-mirror; no substitution needed):")
    for a in sorted(variants):
        score = _ncc(h_flip[a], variants[a])
        if score >= args.threshold:
            print(f"  {args.prefix}{a:02d}    self-NCC: {score:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
