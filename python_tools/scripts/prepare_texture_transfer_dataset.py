#!/usr/bin/env python3
"""Prepare per-map training tensors for the texture transfer U-Net.

For each official map:
  Inputs (10-channel float32, shape (10, H, W)):
    0  heightmap  (normalized 0-1 globally per map)
    1  slope      (gradient magnitude, normalized 0-1)
    2  water_mask (binary; impassable + low elevation)
    3  buildability
    4  passability
    5  density_resource     (oil derricks, ore nodes)
    6  density_building     (Allied/Soviet/Japan/civilian + garrisons)
    7  density_decoration   (trees, palms, rocks, statues, etc.)
    8  density_road
    9  density_cliff        (CLIFFWALL/SEACLIFFWALL)

  Label (int32, shape (H, W)):
    Texture vocab index per tile (V = len(official_textures), 350-ish).
    Tiles whose source texture is not in the vocab get IGNORE_INDEX = -1.

  Style id: scalar int (cluster id from clusters_k8_full.json).

Output:
  training_outputs/texture_transfer/
    vocab.json       # ordered list of texture names + ignore index
    index.json       # split, per-map records (path, style, w, h, valid_tile_count)
    maps/<id>.npz    # X (10,H,W) float32, y (H,W) int32, style_id, map_file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_ra3map():
    tools_root = _python_tools_root()
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from map_processor import Ra3Map  # noqa: WPS433
    return Ra3Map


sys.path.insert(0, str(_python_tools_root()))
from map_processor.utils.style_features import (  # noqa: E402
    categorize, gaussian_blur, compute_slope, normalize_height, CATEGORY_NAMES,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps_root", type=Path, action="append", default=None,
                    help="Root folder(s) to scan recursively for .map files. "
                         "Repeat the flag to add more roots. Default: official + side/.")
    ap.add_argument("--min_valid_frac", type=float, default=0.6,
                    help="Skip maps whose share of tiles with official textures is below this. "
                         "Filters out community maps using mostly custom textures.")
    ap.add_argument("--browser_dir", type=Path,
                    default=_python_tools_root() / "style_clusters" / "browser")
    ap.add_argument("--out_dir", type=Path,
                    default=_python_tools_root() / "training_outputs" / "texture_transfer")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--object_sigma", type=float, default=2.0,
                    help="Gaussian blur sigma (in tiles) for object-density channels.")
    ap.add_argument("--world_to_tile", type=float, default=10.0,
                    help="World units per tile (RA3 default = 10).")
    args = ap.parse_args()

    Ra3Map = _import_ra3map()

    # Vocab from official_inventory.json
    inv_path = args.browser_dir / "official_inventory.json"
    if not inv_path.exists():
        raise SystemExit(f"Missing {inv_path}; build with the inventory pass first.")
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    vocab = sorted(set(inv["official_textures"]))
    name_to_idx = {n: i for i, n in enumerate(vocab)}
    print(f"Vocabulary: {len(vocab)} official texture names")

    # Cluster assignments
    clusters_path = args.browser_dir / "clusters_k8_full.json"
    if not clusters_path.exists():
        raise SystemExit(f"Missing {clusters_path}")
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    style_by_path = {a["map_file"]: int(a["cluster"]) for a in clusters["assignments"]}

    roots = args.maps_root if args.maps_root else [
        _python_tools_root().parent / "RA3 Official maps",
        _python_tools_root().parent / "side",
    ]
    roots = [r for r in roots if r.exists()]
    files: List[Path] = []
    seen = set()
    for r in roots:
        for f in sorted(r.rglob("*.map")):
            ap_resolved = f.resolve()
            if ap_resolved in seen:
                continue
            seen.add(ap_resolved)
            files.append(f)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .map files under: {roots}")
    print(f"Scanning {len(files)} maps across {len(roots)} root(s)")

    out_maps = args.out_dir / "maps"
    out_maps.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    records: List[dict] = []
    skipped = 0

    for i, f in enumerate(files, 1):
        # Style id: try exact path match, else relative path lookup
        style_id = None
        for k, v in style_by_path.items():
            if f.samefile(k) if Path(k).exists() else False:
                style_id = v; break
        if style_id is None:
            # Fallback: search by filename
            for k, v in style_by_path.items():
                if Path(k).name == f.name and Path(k).parent.name == f.parent.name:
                    style_id = v; break
        if style_id is None:
            print(f"  [skip] {f.name}: no cluster assignment")
            skipped += 1
            continue

        try:
            m = Ra3Map(str(f)); m.parse(); ctx = m.get_context()
            h_asset = ctx.get_asset("HeightMapData")
            blend = ctx.get_asset("BlendTileData")
            objs = ctx.get_asset("ObjectsList")

            # Both arrays are (mapWidth, mapHeight) per BlendTileData convention,
            # but the heightmap can include extra border padding. Crop to the min.
            tW, tH = blend.tiles.shape
            elev_full = h_asset.elevations.astype(np.float32)
            eW, eH = elev_full.shape
            W = min(tW, eW)
            H = min(tH, eH)
            elev = elev_full[:W, :H]
            height_n = normalize_height(elev)
            slope = compute_slope(elev)

            # masks (cropped to W,H)
            buildability = blend.buildability[:W, :H].astype(np.float32)
            impassable_b = blend.impassable[:W, :H]
            impassable_f = impassable_b.astype(np.float32)
            passability = (1.0 - impassable_f).astype(np.float32)
            # Heuristic water mask: low elevation + impassable
            low_thresh = float(np.quantile(elev, 0.05))
            water_mask = ((elev <= low_thresh + 1e-3) & impassable_b).astype(np.float32)

            # object density channels
            channels_obj = {c: np.zeros((W, H), dtype=np.float32) for c in CATEGORY_NAMES}
            for obj in objs.map_objects:
                cat = categorize(obj.type_name)
                if cat is None:
                    continue
                tx = int(obj.position[0] / args.world_to_tile)
                ty = int(obj.position[1] / args.world_to_tile)
                if 0 <= tx < W and 0 <= ty < H:
                    channels_obj[cat][tx, ty] += 1.0
            for cat in CATEGORY_NAMES:
                channels_obj[cat] = gaussian_blur(channels_obj[cat], args.object_sigma)
                mx = channels_obj[cat].max()
                if mx > 0:
                    channels_obj[cat] /= mx

            # Stack channels: (10, W, H)
            X = np.stack([
                height_n,
                slope,
                water_mask,
                buildability,
                passability,
                channels_obj["resource"],
                channels_obj["building"],
                channels_obj["decoration"],
                channels_obj["road"],
                channels_obj["cliff"],
            ], axis=0)

            # Label: texture vocab index per tile (cropped to W,H)
            tex_idx_per_tile = (blend.tiles[:W, :H] // 64).astype(np.int32)  # (W, H)
            # Map texture index in this map -> texture name -> vocab index
            local_to_vocab = np.full(len(blend.textures), -1, dtype=np.int32)
            for li, tex in enumerate(blend.textures):
                local_to_vocab[li] = name_to_idx.get(tex.name, -1)
            y = np.where(
                (tex_idx_per_tile >= 0) & (tex_idx_per_tile < len(blend.textures)),
                local_to_vocab[np.clip(tex_idx_per_tile, 0, len(blend.textures) - 1)],
                np.int32(-1),
            )
            valid = int((y != -1).sum())
            total = int(y.size)
            valid_frac = valid / max(total, 1)
            if valid_frac < args.min_valid_frac:
                print(f"  [skip] {f.parent.name}/{f.stem}: valid_frac={valid_frac:.2f} "
                      f"< min_valid_frac={args.min_valid_frac}")
                skipped += 1
                continue

            # Save
            out_path = out_maps / f"{i:04d}_{f.stem}.npz"
            np.savez_compressed(out_path, X=X, y=y, style_id=np.int32(style_id))

            records.append({
                "id": i,
                "map_file": str(f),
                "npz": str(out_path.relative_to(args.out_dir)),
                "style_id": int(style_id),
                "W": int(W), "H": int(H),
                "valid_frac": round(valid_frac, 4),
            })
            print(f"[{i:3d}/{len(files)}]  {f.parent.name}/{f.stem}  W={W} H={H}  style={style_id}  valid={valid_frac:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {f}: {e}")
            skipped += 1

    # Train/val split by random map
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_ids = {r["id"] for r in records[:n_val]}
    for r in records:
        r["split"] = "val" if r["id"] in val_ids else "train"

    # Class frequency for weighted loss
    class_counts = Counter()
    for r in records:
        if r["split"] != "train":
            continue
        d = np.load(args.out_dir / r["npz"])
        ys = d["y"].reshape(-1)
        ys = ys[ys != -1]
        class_counts.update(ys.tolist())
    class_freq = np.zeros(len(vocab), dtype=np.int64)
    for k, v in class_counts.items():
        class_freq[int(k)] = int(v)

    index = {
        "vocab_size": len(vocab),
        "ignore_index": -1,
        "channels": [
            "heightmap", "slope", "water_mask", "buildability", "passability",
            "density_resource", "density_building", "density_decoration",
            "density_road", "density_cliff",
        ],
        "n_channels": 10,
        "n_styles": 8,
        "object_sigma": args.object_sigma,
        "world_to_tile": args.world_to_tile,
        "n_train": sum(1 for r in records if r["split"] == "train"),
        "n_val": sum(1 for r in records if r["split"] == "val"),
        "n_skipped": skipped,
        "class_freq_train": class_freq.tolist(),
        "records": records,
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nDone. {index['n_train']} train + {index['n_val']} val maps.  Skipped: {skipped}")
    print(f"Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
