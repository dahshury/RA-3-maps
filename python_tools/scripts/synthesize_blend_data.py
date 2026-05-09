"""
Procedurally synthesize .npz training data matching the official RA3
WorldBuilder blend algorithm.

The algorithm is ported from MapCreatorCore/Core/MapGenerator.cs:BlendTextures.
For each cell it inspects the 4 cardinal + 4 diagonal neighbors and emits ONE
blend with the first matching priority rule. Only the 12 documented
BlendDirection values are produced -- the four anomalous values seen in some
hand-edited community maps (raw 33, 34, 49, 50) are NOT in the algorithm and
cannot be synthesized faithfully.

Output .npz files are drop-in compatible with prepare_unet_dataset.py output
and can be combined into a training run via train_blend_unet.py
--prepared_dir "real_dir,synth_dir".

Usage:
    python scripts/synthesize_blend_data.py \\
        --out_dir "../blendinfo dataset/_generated/unet_data_synth" \\
        --n_maps 80 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the same helpers used at training time so the produced .npz files
# carry pattern_code/dist/etc identical to real-map output.
from scripts.prepare_unet_dataset import (  # noqa: E402
    _compute_distance_to_boundary,
    _compute_pattern_code,
    DIR_VAL_TO_CLASS,
    NEIGHBOR_OFFSETS,
)


# Direction raw values matching the C# BlendDirection enum.
DIR_BOTTOM_RIGHT = 0x28
DIR_BOTTOM = 0x2
DIR_BOTTOM_LEFT = 0x24
DIR_RIGHT = 0x11
DIR_LEFT = 0x1
DIR_TOP_RIGHT = 0x38
DIR_TOP = 0x12
DIR_TOP_LEFT = 0x34
DIR_EXCEPT_BOTTOM_RIGHT = 0x14
DIR_EXCEPT_BOTTOM_LEFT = 0x18
DIR_EXCEPT_TOP_RIGHT = 0x4
DIR_EXCEPT_TOP_LEFT = 0x8


def _compute_blend_labels(tex_grid: np.ndarray):
    """
    Vectorized port of MapGenerator.BlendTextures (C#).

    For each interior cell, picks the first matching rule from the C# priority
    order. Sets blend_present, secondary texture, and dir_raw. Sets mask8 from
    which neighbors equal the secondary texture.

    Returns:
        present  [W, H] uint8
        mask8    [W, H] uint8  (255 where present but no neighbor matches)
        dir_cls  [W, H] int16  (class index 0..16, -1 elsewhere)
    """
    # tex_grid is shape (W, H) where axis 0 = x, axis 1 = y (matches C#).
    # C# convention: top = (x, y+1), bottom = (x, y-1), left = (x-1, y), etc.
    w, h = tex_grid.shape
    pad = np.pad(tex_grid, 1, mode="edge")
    L  = pad[ :-2, 1:-1]   # (x-1, y)
    R  = pad[2:  , 1:-1]   # (x+1, y)
    T  = pad[1:-1, 2:  ]   # (x, y+1)
    B  = pad[1:-1,  :-2]   # (x, y-1)
    TL = pad[ :-2, 2:  ]   # (x-1, y+1)
    TR = pad[2:  , 2:  ]   # (x+1, y+1)
    BL = pad[ :-2,  :-2]   # (x-1, y-1)
    BR = pad[2:  ,  :-2]   # (x+1, y-1)
    C  = tex_grid

    sec = np.full((w, h), -1, dtype=np.int32)
    dir_raw = np.full((w, h), 0, dtype=np.int16)

    def _try(cond, sec_src, dir_val):
        # First-rule-wins: only fill cells whose sec is still unset.
        m = cond & (sec < 0)
        sec[m] = sec_src[m]
        dir_raw[m] = dir_val

    # Priority order copied verbatim from the C# code.
    _try((L == T) & (T != C), T, DIR_BOTTOM_RIGHT)
    _try((R == T) & (T != C), T, DIR_BOTTOM_LEFT)
    _try((R == B) & (B != C), B, DIR_TOP_LEFT)
    _try((L == B) & (B != C), B, DIR_TOP_RIGHT)
    _try(L != C, L, DIR_RIGHT)
    _try(R != C, R, DIR_LEFT)
    _try(T != C, T, DIR_BOTTOM)
    _try(B != C, B, DIR_TOP)
    _try(TL != C, TL, DIR_EXCEPT_TOP_LEFT)
    _try(TR != C, TR, DIR_EXCEPT_TOP_RIGHT)
    _try(BR != C, BR, DIR_EXCEPT_BOTTOM_RIGHT)
    _try(BL != C, BL, DIR_EXCEPT_BOTTOM_LEFT)

    present_bool = sec >= 0
    # Note: the C# auto-generator filters with `centerTexture <= tex` to dedupe
    # boundary pairs. Real maps don't follow that filter (both sides keep blends
    # in the saved data), so we emit on every cell that has a winning neighbor.

    # Build 8-bit neighbor mask (which neighbors match the secondary texture).
    neighbors = [L, T, R, B, TL, TR, BR, BL]  # but use NEIGHBOR_OFFSETS order
    # NEIGHBOR_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # Map offset to neighbor patch: pad indexed as (y+1+dy, x+1+dx)
    # For consistency with prepare_unet_dataset _layer_labels.
    mask8 = np.zeros((w, h), dtype=np.uint16)
    for ni, (dx, dy) in enumerate(NEIGHBOR_OFFSETS):
        neigh = pad[1 + dx: 1 + dx + w, 1 + dy: 1 + dy + h]
        hit = present_bool & (sec >= 0) & (neigh == sec)
        mask8 |= hit.astype(np.uint16) << np.uint16(ni)
    mask8 = mask8.astype(np.uint8)
    mask8 = np.where(present_bool & (mask8 == 0), np.uint8(255), mask8).astype(np.uint8)

    # Direction class lookup.
    dir_cls = np.full((w, h), -1, dtype=np.int16)
    for raw_val, cls_idx in DIR_VAL_TO_CLASS.items():
        dir_cls[present_bool & (dir_raw == raw_val)] = cls_idx

    return present_bool.astype(np.uint8), mask8, dir_cls


def _voronoi_tex_grid(w: int, h: int, n_textures: int, n_seeds: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Voronoi-cell texture map: scatter seeds, assign each seed a texture id,
    each cell takes the texture of its nearest seed (squared L2)."""
    seeds_xy = rng.integers(0, [w, h], size=(n_seeds, 2))
    seed_tex = rng.integers(0, n_textures, size=n_seeds)
    # Vectorized nearest-seed assignment (chunked to limit memory).
    out = np.empty((w, h), dtype=np.int32)
    chunk = max(1, 8_000_000 // max(1, n_seeds))  # cells per chunk
    flat_idx = 0
    grid_xy = np.indices((w, h)).transpose(1, 2, 0).reshape(-1, 2)
    for s in range(0, grid_xy.shape[0], chunk):
        block = grid_xy[s:s + chunk]
        d2 = ((block[:, None, :] - seeds_xy[None, :, :]) ** 2).sum(axis=-1)
        nearest = d2.argmin(axis=1)
        out.flat[s:s + chunk] = seed_tex[nearest]
    return out


def _stripes_tex_grid(w: int, h: int, n_textures: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Horizontal/vertical stripes for cleaner cardinal/Except patterns."""
    out = np.zeros((w, h), dtype=np.int32)
    axis = int(rng.integers(0, 2))
    n_bands = int(rng.integers(2, 8))
    boundaries = sorted(rng.integers(1, [w, h][axis] - 1, size=n_bands - 1).tolist())
    bands = [0] + boundaries + [[w, h][axis]]
    band_tex = rng.integers(0, n_textures, size=len(bands) - 1)
    for i in range(len(band_tex)):
        if axis == 0:
            out[bands[i]:bands[i + 1], :] = band_tex[i]
        else:
            out[:, bands[i]:bands[i + 1]] = band_tex[i]
    return out


def _checker_tex_grid(w: int, h: int, n_textures: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Checkerboard of variable cell size — produces lots of corner patterns."""
    cs = int(rng.integers(4, 32))
    grid = (np.arange(w)[:, None] // cs + np.arange(h)[None, :] // cs)
    if n_textures >= 4:
        # Use 4 textures cycled for richer corners
        tex_pool = rng.choice(n_textures, size=4, replace=False)
    else:
        tex_pool = np.arange(n_textures)
    return tex_pool[grid % len(tex_pool)].astype(np.int32)


def _perlin_elev(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap pseudo-Perlin via low-pass filtered noise."""
    from scipy.ndimage import gaussian_filter
    noise = rng.standard_normal((w, h)).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=10.0)
    elev = (smooth - smooth.min()) / max(1e-6, (smooth.max() - smooth.min()))
    return (elev * 200.0 + 50.0).astype(np.float32)


def synthesize_one(name: str, out_path: Path, rng: np.random.Generator):
    """Generate one synthetic map .npz."""
    w = int(rng.integers(192, 513))
    h = int(rng.integers(192, 513))
    n_textures = int(rng.integers(4, 25))
    style = rng.choice(["voronoi", "voronoi", "voronoi", "stripes", "checker"])
    if style == "voronoi":
        n_seeds = int(rng.integers(8, 80))
        tex_grid = _voronoi_tex_grid(w, h, n_textures, n_seeds, rng)
    elif style == "stripes":
        tex_grid = _stripes_tex_grid(w, h, n_textures, rng)
    else:
        tex_grid = _checker_tex_grid(w, h, n_textures, rng)

    elev_grid = _perlin_elev(w, h, rng)
    blend_present, blend_mask, blend_dir = _compute_blend_labels(tex_grid)

    # Single-edge layer: synthetic data has no se layer, set all to "absent".
    se_present = np.zeros_like(blend_present, dtype=np.uint8)
    se_mask = np.full_like(blend_mask, 255, dtype=np.uint8)
    se_dir = np.full_like(blend_dir, -1, dtype=np.int16)

    dist_grid = _compute_distance_to_boundary(tex_grid)
    pattern_code = _compute_pattern_code(tex_grid)

    np.savez_compressed(
        out_path,
        tex_grid=tex_grid.astype(np.int32),
        elev_grid=elev_grid.astype(np.float32),
        blend_present=blend_present.astype(np.uint8),
        blend_mask=blend_mask.astype(np.uint8),
        blend_dir=blend_dir.astype(np.int16),
        se_present=se_present,
        se_mask=se_mask,
        se_dir=se_dir,
        dist_grid=dist_grid.astype(np.float32),
        pattern_code=pattern_code.astype(np.int8),
    )
    return dict(
        name=name,
        H=int(w), W=int(h),
        n_pres=int(blend_present.sum()),
        density=float(blend_present.mean()),
        n_tex=int(np.unique(tex_grid).size),
        style=str(style),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="Output dir for synthetic .npz")
    ap.add_argument("--n_maps", type=int, default=80, help="How many maps to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    summaries = []
    for i in range(args.n_maps):
        sub_seed = int(rng.integers(0, 2**31))
        sub_rng = np.random.default_rng(sub_seed)
        name = f"synth_{i:04d}.npz"
        out_path = out_dir / name
        try:
            s = synthesize_one(name, out_path, sub_rng)
            summaries.append(s)
            if (i + 1) % 10 == 0 or i == args.n_maps - 1:
                avg_d = float(np.mean([x["density"] for x in summaries]))
                print(f"  [{i+1}/{args.n_maps}] avg density={avg_d:.3f} "
                      f"last: {s['style']} {s['W']}x{s['H']} "
                      f"ntex={s['n_tex']} npres={s['n_pres']}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{args.n_maps}] FAIL: {e}", flush=True)

    # Aggregate.
    if summaries:
        d_arr = np.array([x["density"] for x in summaries])
        n_arr = np.array([x["n_pres"] for x in summaries])
        print(f"\nGenerated {len(summaries)} maps in {out_dir}")
        print(f"  density: min={d_arr.min():.3f} med={np.median(d_arr):.3f} "
              f"max={d_arr.max():.3f}")
        print(f"  blend cells total: {int(n_arr.sum()):,}")


if __name__ == "__main__":
    main()
