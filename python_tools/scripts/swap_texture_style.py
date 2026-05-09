#!/usr/bin/env python3
"""Swap a map's texture palette and cliff-objects to match a target style cluster.

Preserved: terrain heightmap, gameplay objects, water, layout, symmetry.
Remapped: BlendTileData.textures[] (by usage rank) and 3D cliff objects
(CLIFFWALL/SEACLIFFWALL types) to the target cluster's dominant cliff biome.

Texture strategy: rank-aligned palette swap.
  - Source's most-used texture  -> target style's most-used texture
  - Source's 2nd-most-used      -> target's 2nd-most-used
  - ...
Target palette is filtered to OFFICIAL textures only (no community palette
pollution -> no black/missing tiles).

Cliff strategy:
  - Detect target cluster's dominant cliff biome prefix (YU/IL/BB/MY/...).
  - For each cliff object, swap prefix to target's prefix only if the same
    suffix (e.g. CLIFFWALL05) exists in the target. Otherwise leave the
    cliff alone (e.g. snow target uses IL which has only SEACLIFFWALL,
    so source CLIFFWALL stays Yucatan).

Usage:
  python scripts/swap_texture_style.py --src MAP.map --style N
  python scripts/swap_texture_style.py --src MAP.map --style 7 --out custom.map

Defaults: out = <src_dir>/<src_stem>_style<N>.map
Requires: feature cache + clusters_k8_full.json + cluster_inventory.json +
official_inventory.json under style_clusters/browser/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CLIFF_RE = re.compile(r"^(.*?)_?(CLIFFWALL\d+|SEACLIFFWALL\d+)$", re.I)


def _python_tools_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_browser_dir() -> Path:
    return _python_tools_root() / "style_clusters" / "browser"


def _import_ra3map():
    tools_root = _python_tools_root()
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from map_processor import Ra3Map  # noqa: WPS433
    return Ra3Map


def load_official_inventory(browser_dir: Path) -> Tuple[set, dict]:
    """Returns (official_texture_names, official_inventory_dict)."""
    p = browser_dir / "official_inventory.json"
    if not p.exists():
        raise SystemExit(
            f"Missing {p}. Build with the inventory script (one-off scan of RA3 Official maps).")
    inv = json.loads(p.read_text(encoding="utf-8"))
    return set(inv["official_textures"]), inv


def load_cluster_inventory(browser_dir: Path) -> dict:
    p = browser_dir / "cluster_inventory.json"
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run: python scripts/build_cluster_inventory.py")
    return json.loads(p.read_text(encoding="utf-8"))


def load_target_palette(
    browser_dir: Path, style_id: int, official_textures: set
) -> List[Tuple[str, int]]:
    """Aggregate texture usage across cluster members, filtered to known-valid textures."""
    clusters_path = browser_dir / "clusters_k8_full.json"
    cache_path = browser_dir / "feature_cache.json"
    if not clusters_path.exists():
        raise SystemExit(
            f"Missing {clusters_path}. Run:  python scripts/browse_styles.py --k 8 --full")
    if not cache_path.exists():
        raise SystemExit(f"Missing {cache_path}. Run:  python scripts/browse_styles.py --build")

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    member_files = {a["map_file"] for a in clusters["assignments"] if int(a["cluster"]) == style_id}
    if not member_files:
        raise SystemExit(f"Style cluster {style_id} has no members.")

    totals: Counter = Counter()
    for rec in cache["records"]:
        if rec["map_file"] in member_files:
            for name, c in rec["texture_counts"].items():
                if name in official_textures:  # filter out community/custom
                    totals[name] += int(c)

    if not totals:
        raise SystemExit(
            f"No officially-known textures found for cluster {style_id}. "
            f"Cluster likely contains only community maps."
        )
    return totals.most_common()


def source_texture_usage(blend) -> List[Tuple[str, int]]:
    """Count tile usage of each texture entry in the source map."""
    tiles = blend.tiles
    texture_idx = (tiles // 64).astype(np.int64).reshape(-1)
    counts = Counter(texture_idx.tolist())

    out: List[Tuple[str, int]] = []
    for tex_idx, tex in enumerate(blend.textures):
        out.append((tex.name, int(counts.get(tex_idx, 0))))
    # Sort by usage desc; ties broken by original index for determinism
    out_with_pos = list(enumerate(out))
    out_with_pos.sort(key=lambda kv: (-kv[1][1], kv[0]))
    return [item for _, item in out_with_pos]


def build_rank_mapping(
    src_usage: List[Tuple[str, int]],
    tgt_palette: List[Tuple[str, int]],
    rng: random.Random,
) -> Dict[str, str]:
    """Map each source texture name to a target texture name, by usage rank."""
    mapping: Dict[str, str] = {}
    n_tgt = len(tgt_palette)
    for rank, (src_name, _) in enumerate(src_usage):
        if rank < n_tgt:
            tgt_name = tgt_palette[rank][0]
        else:
            # Source has more unique textures than target. Pick from target
            # weighted by usage, so overflow textures still feel on-style.
            tgt_name = rng.choices(
                [name for name, _ in tgt_palette],
                weights=[max(1, c) for _, c in tgt_palette],
                k=1,
            )[0]
        mapping[src_name] = tgt_name
    return mapping


def apply_swap(blend, mapping: Dict[str, str]) -> int:
    """Rewrite each texture.name via mapping. Returns count of textures changed."""
    changed = 0
    for tex in blend.textures:
        new_name = mapping.get(tex.name)
        if new_name and new_name != tex.name:
            tex.name = new_name
            changed += 1
    return changed


def swap_cliff_objects(
    objects_list,
    target_prefix: Optional[str],
    target_suffixes: List[str],
) -> Tuple[int, int, int]:
    """Rewrite cliff object type_names to target biome where a matching suffix exists.

    Returns (n_swapped, n_left_alone_no_match, n_total_cliffs).
    """
    if not target_prefix:
        # Cluster has no recognized cliff biome (e.g. only community maps)
        return (0, 0, 0)
    target_suffix_set = {s.upper() for s in target_suffixes}
    swapped = left = total = 0
    for obj in objects_list.map_objects:
        m = CLIFF_RE.match(obj.type_name)
        if not m:
            continue
        total += 1
        src_prefix = m.group(1).upper()
        suffix = m.group(2).upper()
        if src_prefix == target_prefix:
            continue  # already in target biome
        if suffix not in target_suffix_set:
            left += 1  # target biome lacks this variant - per user rule, leave alone
            continue
        # Preserve original casing on the suffix portion
        new_name = f"{target_prefix}_{suffix}"
        obj.type_name = new_name
        swapped += 1
    return (swapped, left, total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Source .map file")
    ap.add_argument("--style", type=int, required=True, help="Target style cluster id (0-7)")
    ap.add_argument("--out", type=Path, default=None, help="Output .map path (default: <src>_style<N>.map)")
    ap.add_argument("--browser_dir", type=Path, default=_default_browser_dir())
    ap.add_argument("--no_compress", action="store_true", help="Save uncompressed (faster, bigger).")
    ap.add_argument("--swap_cliffs", action="store_true",
                    help="Also rename cliff objects by biome prefix. Default OFF: name-substitution "
                         "doesn't preserve mesh/orientation alignment. Real cliff swap is the cliff "
                         "placement model (separate task).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source map not found: {args.src}")

    out_path = args.out or args.src.with_name(f"{args.src.stem}_style{args.style}.map")

    Ra3Map = _import_ra3map()
    rng = random.Random(args.seed)

    print(f"Loading target style {args.style} palette from {args.browser_dir}/")
    official_textures, _official_inv = load_official_inventory(args.browser_dir)
    cluster_inv = load_cluster_inventory(args.browser_dir)
    tgt_palette = load_target_palette(args.browser_dir, args.style, official_textures)
    print(f"  {len(tgt_palette)} valid (official) textures in cluster {args.style}")

    cluster_record = cluster_inv.get(str(args.style), {})
    target_cliff_prefix: Optional[str] = cluster_record.get("dominant_cliff_biome")
    target_cliff_suffixes: List[str] = cluster_record.get(
        "cliff_suffixes_by_prefix", {}
    ).get(target_cliff_prefix, []) if target_cliff_prefix else []
    print(f"  cliff biome: {target_cliff_prefix or '(none)'}, "
          f"{len(target_cliff_suffixes)} cliff variants available")

    print(f"\nParsing source: {args.src}")
    m = Ra3Map(str(args.src))
    m.parse()
    ctx = m.get_context()
    blend = ctx.get_asset("BlendTileData")
    print(f"  {blend.map_width}x{blend.map_height}, {len(blend.textures)} textures in palette")

    src_usage = source_texture_usage(blend)
    mapping = build_rank_mapping(src_usage, tgt_palette, rng)

    print("\nTexture mapping (source -> target, by usage rank):")
    for rank, (src_name, c) in enumerate(src_usage[:10]):
        tgt = mapping[src_name]
        print(f"  #{rank+1:2d}  {src_name:30s} ({c:6d} tiles)  ->  {tgt}")
    if len(src_usage) > 10:
        print(f"  ... +{len(src_usage)-10} more")

    changed = apply_swap(blend, mapping)
    print(f"\nReplaced {changed}/{len(blend.textures)} texture names")

    if args.swap_cliffs:
        objects_list = ctx.get_asset("ObjectsList")
        if objects_list is not None:
            n_swapped, n_left, n_total = swap_cliff_objects(
                objects_list, target_cliff_prefix, target_cliff_suffixes
            )
            if n_total:
                print(f"Cliffs: {n_swapped} swapped to {target_cliff_prefix}, "
                      f"{n_left} left alone, of {n_total} total cliff objects "
                      f"(WARN: name-swap doesn't preserve mesh alignment)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path), compress=not args.no_compress)
    print(f"Wrote: {out_path}")

    # Round-trip sanity check
    m2 = Ra3Map(str(out_path))
    m2.parse()
    b2 = m2.get_context().get_asset("BlendTileData")
    if not np.array_equal(blend.tiles, b2.tiles):
        print("WARN: tiles array changed during save round-trip.")
    saved_names = {t.name for t in b2.textures}
    expected_targets = set(mapping.values())
    overlap = saved_names & expected_targets
    print(f"Verification: {len(overlap)}/{len(expected_targets)} target textures present in saved file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
