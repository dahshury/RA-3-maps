#!/usr/bin/env python3
"""
Cluster RA3 maps into style groups based on terrain texture usage ("tile style").

We parse each .map, extract a feature vector from BlendTileData:
- histogram of texture NAME usage (tile_id // 64 mapped to texture.name)
- aggregated family stats (Dirt/Grass/Sand/Rock/Reef/Transition/etc by name prefix)

Then we run:
  StandardScaler -> PCA -> KMeans
We auto-pick K by silhouette score over a small range.

Outputs:
- output_dir/
  - clusters.json (metadata, features, K, assignments)
  - clusters.csv
  - cluster_<k>/<map_folder>/... (copies of the original map folders)

This is intended for:
- dataset splitting by biome/style
- style-consistent augmentation (swap palettes within a cluster)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_parser():
    # Keep import local so script can run from anywhere.
    import sys

    tools_root = _python_tools_root()
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from map_processor.parsing.map_parser import Ra3MapParser  # noqa: WPS433

    return Ra3MapParser


FAMILY_PREFIXES = [
    "Dirt_",
    "Grass_",
    "Sand_",
    "Rock_",
    "Reef_",
    "Transition_",
    "Pave",
    "Pavement_",
    "Snow_",
    "Ice_",
    "Mud_",
    "Gravel_",
    "Road",
    "Sidewalk",
]


def _texture_family(name: str) -> str:
    for p in FAMILY_PREFIXES:
        if name.startswith(p):
            return p.rstrip("_")
    # fallback: biome token (often second segment)
    if "_" in name:
        return name.split("_", 1)[0]
    return "Other"


@dataclass(frozen=True)
class MapExample:
    map_file: Path
    map_dir: Path
    map_name: str
    width: int
    height: int
    texture_counts_by_name: Dict[str, int]
    family_fracs: Dict[str, float]
    num_textures_used: int
    texture_entropy: float


def _find_map_files(root: Path) -> List[Path]:
    # Official maps are stored as folders containing .map files.
    return sorted(root.rglob("*.map"))


def _infer_map_dir(map_file: Path) -> Path:
    # For official maps, the parent folder is the “map folder”.
    return map_file.parent


def _infer_map_name(map_file: Path) -> str:
    return map_file.stem


def _extract_example(parser, map_file: Path) -> MapExample:
    ctx = parser.parse(str(map_file))
    blend = ctx.get_asset("BlendTileData")

    w = int(ctx.map_width)
    h = int(ctx.map_height)

    tiles = blend.tiles  # (W,H) uint16
    tex_indices = (tiles // 64).astype(np.int32)
    flat = tex_indices.reshape(-1)
    counts = Counter(flat.tolist())

    # Map per-index counts to texture name counts
    by_name: Dict[str, int] = {}
    for idx, c in counts.items():
        if idx < 0 or idx >= len(blend.textures):
            name = f"UNKNOWN_{idx}"
        else:
            name = blend.textures[idx].name
        by_name[name] = by_name.get(name, 0) + int(c)

    total = float(tiles.size)
    # Family fractions
    fam_counts = defaultdict(int)
    for name, c in by_name.items():
        fam_counts[_texture_family(name)] += c
    fam_fracs = {k: v / total for k, v in fam_counts.items()}

    # Entropy over texture names (style complexity proxy)
    probs = np.array([c / total for c in by_name.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    used = sum(1 for c in by_name.values() if c > 0)

    map_dir = _infer_map_dir(map_file)
    return MapExample(
        map_file=map_file,
        map_dir=map_dir,
        map_name=_infer_map_name(map_file),
        width=w,
        height=h,
        texture_counts_by_name=by_name,
        family_fracs=fam_fracs,
        num_textures_used=int(used),
        texture_entropy=entropy,
    )


def _build_feature_matrix(examples: List[MapExample], max_vocab: int = 512) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dense feature matrix from:
    - top-N texture name fractions (global vocab)
    - family fractions
    - scalar stats (num_textures_used, entropy)
    """
    # Global texture vocab by total usage
    global_counts = Counter()
    for ex in examples:
        global_counts.update(ex.texture_counts_by_name)

    vocab = [name for name, _ in global_counts.most_common(max_vocab)]

    # Global family vocab
    fam_vocab = sorted({k for ex in examples for k in ex.family_fracs.keys()})

    feature_names: List[str] = []
    feature_names += [f"tex:{n}" for n in vocab]
    feature_names += [f"fam:{n}" for n in fam_vocab]
    feature_names += ["stat:num_textures_used", "stat:texture_entropy"]

    X = np.zeros((len(examples), len(feature_names)), dtype=np.float32)

    for i, ex in enumerate(examples):
        total = float(ex.width * ex.height)
        # tex fractions
        for j, tex_name in enumerate(vocab):
            c = ex.texture_counts_by_name.get(tex_name, 0)
            X[i, j] = c / total

        # family fractions
        base = len(vocab)
        for j, fam in enumerate(fam_vocab):
            X[i, base + j] = float(ex.family_fracs.get(fam, 0.0))

        # stats
        X[i, base + len(fam_vocab) + 0] = float(ex.num_textures_used)
        X[i, base + len(fam_vocab) + 1] = float(ex.texture_entropy)

    return X, feature_names


def _pick_k(X_emb: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    n = X_emb.shape[0]
    if n < 3:
        return 1
    k_min = max(2, k_min)
    k_max = min(k_max, n - 1)
    if k_min > k_max:
        return max(2, min(3, n - 1))

    best_k = k_min
    best_score = -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
        labels = km.fit_predict(X_emb)
        # silhouette needs >1 label
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X_emb, labels))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maps_root",
        type=str,
        default=str(_repo_root() / "RA 3 maps" / "RA3 Official maps"),
        help="Root folder containing RA3 map folders (each contains .map).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=str(_python_tools_root() / "style_clusters" / "ra3_official_tiles"),
        help="Where to write clusters + exported folders.",
    )
    ap.add_argument("--max_vocab", type=int, default=512, help="Top-N texture names to include as features.")
    ap.add_argument("--pca_dim", type=int, default=32, help="PCA output dimension before KMeans.")
    ap.add_argument("--k", type=int, default=0, help="If >0, force cluster count K. Otherwise auto-pick.")
    ap.add_argument("--k_min", type=int, default=2, help="Auto-pick: minimum K.")
    ap.add_argument("--k_max", type=int, default=10, help="Auto-pick: maximum K.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy_mode", choices=["copy", "copy_map_only"], default="copy", help="How to export maps.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N maps (debug).")
    args = ap.parse_args()

    maps_root = Path(args.maps_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_files = _find_map_files(maps_root)
    if args.limit and args.limit > 0:
        map_files = map_files[: args.limit]

    if not map_files:
        raise SystemExit(f"No .map files found under {maps_root}")

    Ra3MapParser = _import_parser()
    parser = Ra3MapParser()

    examples: List[MapExample] = []
    for mf in map_files:
        try:
            examples.append(_extract_example(parser, mf))
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] failed to parse {mf}: {e}")

    if len(examples) < 2:
        raise SystemExit("Need at least 2 parsable maps to cluster.")

    X, feature_names = _build_feature_matrix(examples, max_vocab=args.max_vocab)

    # Scale + PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca_dim = min(args.pca_dim, Xs.shape[1], len(examples) - 1)
    if pca_dim < 2:
        pca_dim = min(2, Xs.shape[1])
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    Xp = pca.fit_transform(Xs)

    # Choose K
    if args.k and args.k > 0:
        K = args.k
    else:
        K = _pick_k(Xp, args.k_min, args.k_max, args.seed)

    km = KMeans(n_clusters=K, n_init="auto", random_state=args.seed)
    labels = km.fit_predict(Xp)

    # Write report
    rows = []
    for ex, lab in zip(examples, labels):
        rows.append(
            {
                "cluster": int(lab),
                "map_name": ex.map_name,
                "map_file": str(ex.map_file),
                "map_dir": str(ex.map_dir),
                "width": ex.width,
                "height": ex.height,
                "num_textures_used": ex.num_textures_used,
                "texture_entropy": ex.texture_entropy,
            }
        )

    # Stable cluster ids: sort clusters by size desc
    counts = Counter(int(x) for x in labels.tolist())
    ordered = [c for c, _ in counts.most_common()]
    remap = {old: new for new, old in enumerate(ordered)}
    labels2 = np.array([remap[int(x)] for x in labels.tolist()], dtype=np.int32)

    for r, lab in zip(rows, labels2.tolist()):
        r["cluster"] = int(lab)

    clusters_json = {
        "maps_root": str(maps_root),
        "output_dir": str(out_dir),
        "num_maps_total_found": len(map_files),
        "num_maps_parsed": len(examples),
        "max_vocab": args.max_vocab,
        "pca_dim": int(pca_dim),
        "k": int(K),
        "cluster_sizes": {str(remap[k]): int(v) for k, v in counts.items()},
        "features": {
            "feature_names_count": len(feature_names),
            "feature_names_preview": feature_names[:50],
        },
        "assignments": rows,
    }
    (out_dir / "clusters.json").write_text(json.dumps(clusters_json, indent=2), encoding="utf-8")
    _write_csv(rows, out_dir / "clusters.csv")

    # Export folders
    export_root = out_dir / "exported"
    if export_root.exists():
        shutil.rmtree(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    for ex, lab in zip(examples, labels2.tolist()):
        cluster_dir = export_root / f"cluster_{lab:02d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        src_dir = ex.map_dir
        dst_dir = cluster_dir / src_dir.name
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        if args.copy_mode == "copy_map_only":
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ex.map_file, dst_dir / ex.map_file.name)
        else:
            shutil.copytree(src_dir, dst_dir)

    print(f"OK: clustered {len(examples)} maps into K={K}. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())










