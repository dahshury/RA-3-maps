"""
Audit object categorization coverage for RA3 maps.

This helps answer questions like:
- Which object types are being dropped by training filtering?
- Which types look like buildings/garrisons but aren't categorized yet?

Usage:
  python scripts/audit_object_categorization.py "../RA3 Official maps" --limit 200
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add parent directory to path (so `map_processor` is importable when run as a script)
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map  # noqa: E402
from map_processor.parsing.parser_config import ParserConfig  # noqa: E402
from map_processor.utils.object_categories import ObjectCategoryConfig  # noqa: E402
from map_processor.utils.constants import ASSET_ObjectsList, ASSET_HeightMapData  # noqa: E402


def iter_map_files(root: Path):
    yield from sorted(root.rglob("*.map"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit RA3 map object categorization coverage")
    ap.add_argument("maps_directory", type=str, nargs="?", default="../RA3 Official maps")
    ap.add_argument("--limit", type=int, default=150, help="Max number of unknown types to print")
    ap.add_argument(
        "--include-decorative",
        action="store_true",
        help="Also show types that are skipped by decorative/ambient filtering (usually noisy).",
    )
    args = ap.parse_args()

    maps_dir = Path(args.maps_directory)
    if not maps_dir.exists():
        print(f"Error: maps_directory not found: {maps_dir}")
        return 2

    # Parse only the minimal assets needed to inspect object types quickly.
    cfg = ParserConfig(included_assets={ASSET_ObjectsList, ASSET_HeightMapData})
    training_included = ParserConfig.training_config().included_object_categories or set()
    cat = ObjectCategoryConfig()

    total_maps = 0
    unknown_types = Counter()
    unknown_buildingish = Counter()
    dropped_by_training = Counter()
    dropped_buildingish = Counter()
    by_category = Counter()

    # Heuristic: types that *might* be buildings/garrisons/tech and worth reviewing.
    buildingish_tokens = [
        "house",
        "hotel",
        "apartment",
        "townhouse",
        "cabin",
        "tent",
        "church",
        "restaurant",
        "shop",
        "store",
        "warehouse",
        "villa",
        "mansion",
        "lighthouse",
        # tower-like structures are often garrisonable / strategic in RA3
        "tower",
        "watchtower",
        "clocktower",
        "castle",
        "fort",
        "bunker",
        "defense",
        "building",
        "structure",
        "tech",
        "post",
        "port",
        "airport",
        "garage",
        "hospital",
    ]

    def is_training_included_key(key: str) -> bool:
        if not key:
            return False
        if key in training_included:
            return True
        if key.startswith("garrison_") and "garrison" in training_included:
            return True
        if key.startswith("building_") and "building" in training_included:
            return True
        return False

    for map_path in iter_map_files(maps_dir):
        total_maps += 1
        ra3 = Ra3Map(str(map_path), config=cfg)
        ra3.parse()
        ctx = ra3.get_context()
        ol = ctx.get_asset("ObjectsList")
        if not ol:
            continue

        for obj in ol.map_objects:
            t = obj.type_name or ""
            tl = t.lower()

            category_obj, should_draw = cat.get_category_for_object(t)

            # Reverse-map to a category key (best-effort; avoids changing existing APIs).
            category_key = None
            if category_obj is not None:
                for k, v in cat.get_all_categories().items():
                    if v is category_obj:
                        category_key = k
                        break
                if category_key is None:
                    # Fallback by name (should be rare)
                    for k, v in cat.get_all_categories().items():
                        if v.name == category_obj.name:
                            category_key = k
                            break

            if category_obj is not None and should_draw:
                by_category[category_obj.name] += 1

            # Unknown in the categorizer (and not explicitly filtered as decorative/ambient)
            if category_obj is None and should_draw:
                unknown_types[t] += 1
                if any(tok in tl for tok in buildingish_tokens):
                    unknown_buildingish[t] += 1

            # Would this type be dropped by training object filtering?
            # (If categorized to some key but not included, or unknown-but-should_draw.)
            if category_obj is None and should_draw:
                dropped_by_training[t] += 1
                if any(tok in tl for tok in buildingish_tokens):
                    dropped_buildingish[t] += 1
            elif category_key and not is_training_included_key(category_key):
                dropped_by_training[t] += 1
                if any(tok in tl for tok in buildingish_tokens):
                    dropped_buildingish[t] += 1

            # Optionally show noisy stuff too
            if args.include_decorative and category_obj is None and not should_draw:
                unknown_types[t] += 1

    print(f"Maps scanned: {total_maps}")
    if by_category:
        print("\nTop categorized counts (drawn):")
        for name, c in by_category.most_common(20):
            print(f"- {name:20} {c}")

    print("\nUnknown types (not decorative-filtered):")
    for t, c in unknown_types.most_common(args.limit):
        # If include_decorative is false, unknown_types only contains should_draw=True.
        print(f"- {t:40} x{c}")

    print("\nUnknown but building-ish (high-signal to review):")
    for t, c in unknown_buildingish.most_common(args.limit):
        print(f"- {t:40} x{c}")

    print("\nWould be dropped by training filter (type-level):")
    for t, c in dropped_by_training.most_common(min(args.limit, 50)):
        print(f"- {t:40} x{c}")

    print("\nDropped by training filter but building-ish (highest-signal):")
    for t, c in dropped_buildingish.most_common(args.limit):
        print(f"- {t:40} x{c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


