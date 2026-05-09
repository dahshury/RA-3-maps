#!/usr/bin/env python3
"""Browse RA3 official maps grouped by texture-style cluster.

Workflow:
  # one-time: parse all maps and cache features (slow)
  python scripts/browse_styles.py --build

  # explore K values (instant, uses cache):
  python scripts/browse_styles.py --k 6
  python scripts/browse_styles.py --k 8

  # list every map in a chosen cluster:
  python scripts/browse_styles.py --k 6 --style 2

Cache + cluster output land in:
  python_tools/style_clusters/browser/

Features default to family-level fractions (Snow/Grass/Dirt/Reef/Sand/...),
which is much cleaner for biome separation than raw 512-dim texture names.
Pass --full to mix in per-name texture fractions if family is too coarse.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_maps_roots() -> List[Path]:
    # _repo_root() resolves to ".../RA 3 maps/"
    root = _repo_root()
    return [root / "RA3 Official maps", root / "side"]


def _default_out_dir() -> Path:
    return _python_tools_root() / "style_clusters" / "browser"


FAMILY_PREFIXES = [
    "Snow_", "Ice_", "Cliff_Iceland",   # snow first so Cliff_Iceland wins over generic Cliff_
    "Cliff_", "Reef_",
    "Sand_", "Mud_", "Gravel_",
    "Dirt_", "Grass_",
    "Rock_", "Pavement_", "Pave_", "Pave",
    "Asphalt", "Sidewalk", "Road", "SteelDeck",
    "Transition_",
]


def _texture_family(name: str) -> str:
    for p in FAMILY_PREFIXES:
        if name.startswith(p):
            return p.rstrip("_")
    if "_" in name:
        return name.split("_", 1)[0]
    return "Other"


def _import_parser():
    tools_root = _python_tools_root()
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from map_processor.parsing.map_parser import Ra3MapParser  # noqa: WPS433
    return Ra3MapParser


def build_cache(maps_roots: List[Path], cache_path: Path, limit: int = 0) -> None:
    Ra3MapParser = _import_parser()
    parser = Ra3MapParser()

    files: List[Path] = []
    for root in maps_roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.map")))
    # de-dup by absolute path
    seen = set()
    unique: List[Path] = []
    for f in files:
        ap = f.resolve()
        if ap not in seen:
            seen.add(ap)
            unique.append(f)
    files = unique
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No .map files under any of: {maps_roots}")

    records = []
    for i, mf in enumerate(files, 1):
        try:
            ctx = parser.parse(str(mf))
            blend = ctx.get_asset("BlendTileData")
            tiles = blend.tiles
            tex_idx = (tiles // 64).astype(np.int32).reshape(-1)
            counts = Counter(tex_idx.tolist())
            by_name: Dict[str, int] = {}
            for idx, c in counts.items():
                if 0 <= idx < len(blend.textures):
                    by_name[blend.textures[idx].name] = by_name.get(blend.textures[idx].name, 0) + int(c)
            total = int(tiles.size)
            records.append({
                "map_file": str(mf),
                "map_dir": str(mf.parent),
                "map_name": mf.stem,
                "folder": mf.parent.name,
                "width": int(ctx.map_width),
                "height": int(ctx.map_height),
                "total_tiles": total,
                "texture_counts": by_name,
            })
            print(f"[{i}/{len(files)}] {mf.parent.name}/{mf.stem}  ({len(by_name)} textures)")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {mf}: {e}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(f"\nCached {len(records)} maps -> {cache_path}")


def load_cache(cache_path: Path) -> List[dict]:
    if not cache_path.exists():
        raise SystemExit(f"Cache not found: {cache_path}\nRun with --build first.")
    return json.loads(cache_path.read_text(encoding="utf-8"))["records"]


def build_features(records: List[dict], use_full: bool, max_vocab: int = 256) -> Tuple[np.ndarray, List[str], List[str]]:
    fam_vocab: List[str] = []
    fam_seen = set()

    per_record_fam: List[Dict[str, float]] = []
    for r in records:
        fam_counts = defaultdict(int)
        for name, c in r["texture_counts"].items():
            fam_counts[_texture_family(name)] += c
        total = float(r["total_tiles"])
        fracs = {k: v / total for k, v in fam_counts.items()}
        per_record_fam.append(fracs)
        for k in fracs:
            if k not in fam_seen:
                fam_seen.add(k)
                fam_vocab.append(k)
    fam_vocab.sort()

    feat_names = [f"fam:{f}" for f in fam_vocab]
    name_vocab: List[str] = []

    if use_full:
        global_counts: Counter = Counter()
        for r in records:
            global_counts.update(r["texture_counts"])
        name_vocab = [n for n, _ in global_counts.most_common(max_vocab)]
        feat_names += [f"tex:{n}" for n in name_vocab]

    X = np.zeros((len(records), len(feat_names)), dtype=np.float32)
    for i, r in enumerate(records):
        for j, fam in enumerate(fam_vocab):
            X[i, j] = per_record_fam[i].get(fam, 0.0)
        if use_full:
            total = float(r["total_tiles"])
            base = len(fam_vocab)
            for j, name in enumerate(name_vocab):
                X[i, base + j] = r["texture_counts"].get(name, 0) / total

    # L2 normalize so KMeans behaves like cosine clustering
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X, feat_names, fam_vocab


def cluster(X: np.ndarray, k: int, seed: int = 1337) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    # Sort cluster IDs by size (desc) so cluster 0 is always the biggest
    counts = Counter(int(x) for x in labels.tolist())
    order = [c for c, _ in counts.most_common()]
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(x)] for x in labels.tolist()], dtype=np.int32)


def top_textures(records: List[dict], idx: List[int], topn: int = 8) -> List[Tuple[str, float]]:
    fam_totals: Counter = Counter()
    grand = 0
    for i in idx:
        r = records[i]
        for name, c in r["texture_counts"].items():
            fam_totals[_texture_family(name)] += c
            grand += c
    if grand == 0:
        return []
    return [(fam, c / grand) for fam, c in fam_totals.most_common(topn)]


def top_texture_names(records: List[dict], idx: List[int], topn: int = 6) -> List[Tuple[str, float]]:
    name_totals: Counter = Counter()
    grand = 0
    for i in idx:
        r = records[i]
        for name, c in r["texture_counts"].items():
            name_totals[name] += c
            grand += c
    if grand == 0:
        return []
    return [(n, c / grand) for n, c in name_totals.most_common(topn)]


def print_summary(records: List[dict], labels: np.ndarray) -> None:
    k = int(labels.max()) + 1 if len(labels) else 0
    print(f"\n=== {k} clusters across {len(records)} maps ===\n")
    for cid in range(k):
        idx = [i for i, lab in enumerate(labels) if int(lab) == cid]
        fams = top_textures(records, idx, topn=6)
        fam_str = ", ".join(f"{f}={p*100:.0f}%" for f, p in fams)
        sample = ", ".join(records[i]["folder"] for i in idx[:5])
        more = f" (+{len(idx)-5} more)" if len(idx) > 5 else ""
        print(f"[cluster {cid}]  n={len(idx):3d}   {fam_str}")
        print(f"    sample: {sample}{more}\n")


def print_cluster(records: List[dict], labels: np.ndarray, cid: int) -> None:
    idx = [i for i, lab in enumerate(labels) if int(lab) == cid]
    if not idx:
        print(f"No maps in cluster {cid}.")
        return
    fams = top_textures(records, idx, topn=6)
    names = top_texture_names(records, idx, topn=6)
    print(f"\n=== cluster {cid}: {len(idx)} maps ===")
    print("families : " + ", ".join(f"{f}={p*100:.0f}%" for f, p in fams))
    print("textures : " + ", ".join(f"{n}={p*100:.0f}%" for n, p in names))
    print()
    for i in idx:
        r = records[i]
        print(f"  {r['folder']:<30s}  {r['map_name']}  ({r['width']}x{r['height']})")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps_root", type=Path, action="append", default=None,
                    help="Root folder(s) to scan recursively for .map files. Repeat to add more.")
    ap.add_argument("--out_dir", type=Path, default=_default_out_dir())
    ap.add_argument("--build", action="store_true", help="Parse maps and write feature cache.")
    ap.add_argument("--k", type=int, default=6, help="Number of clusters (default 6).")
    ap.add_argument("--style", type=int, default=None, help="Print maps in this cluster id.")
    ap.add_argument("--full", action="store_true", help="Include per-name texture features (more granular).")
    ap.add_argument("--limit", type=int, default=0, help="Build mode: only first N maps (debug).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cache_path = args.out_dir / "feature_cache.json"
    roots = args.maps_root if args.maps_root else _default_maps_roots()

    if args.build:
        build_cache(roots, cache_path, limit=args.limit)
        return 0

    records = load_cache(cache_path)
    X, _, _ = build_features(records, use_full=args.full)
    labels = cluster(X, k=args.k, seed=args.seed)

    # Persist current clustering so the user can inspect later (small file).
    out = {
        "k": args.k,
        "use_full_features": args.full,
        "assignments": [
            {"cluster": int(lab), "folder": r["folder"], "map_name": r["map_name"], "map_file": r["map_file"]}
            for r, lab in zip(records, labels)
        ],
    }
    (args.out_dir / f"clusters_k{args.k}{'_full' if args.full else ''}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    if args.style is None:
        print_summary(records, labels)
        print(f"To list maps in a cluster:  python scripts/browse_styles.py --k {args.k} --style <id>")
    else:
        print_cluster(records, labels, args.style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
