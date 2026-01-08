"""
Deep scan for RA3 object type coverage.

Goal:
  Prevent "missing garrisons/buildings" by continuously inventorying and reviewing:
  - All `typeName`s present in map files
  - All type-like strings referenced in codebases (MapCreatorCore / Ra3Solution / Ra3NewWb)
  - Categorization coverage + training-filter risk lists

Outputs (written under python_tools/reports/):
  - object_types_deep_scan.json
  - object_types_deep_scan.md

Usage:
  python scripts/deep_scan_object_types.py \\
    --maps \"../RA3 Official maps\" \\
    --code \"../../MapCreatorCore\" --code \"../../Ra3Solution\" --code \"../../Ra3NewWb\"

Tip:
  Run this after adding new maps or changing categorization rules. The report highlights
  any gameplay-relevant types that would be dropped or miscategorized.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Add parent directory to path (so `map_processor` is importable when run as a script)
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map  # noqa: E402
from map_processor.parsing.parser_config import ParserConfig  # noqa: E402
from map_processor.utils.object_categories import ObjectCategoryConfig  # noqa: E402


CODE_EXTS = {".cs", ".cpp", ".c", ".h", ".hpp", ".inl", ".resx", ".txt", ".md"}


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in CODE_EXTS:
            yield p


_STR_RE = re.compile(r"\"([^\"\\\\\\n\\r\\t]{2,120})\"")
_GUID_RE = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
_FILE_EXT_RE = re.compile(r".+\\.(map|png|jpg|jpeg|tga|dds|wav|mp3|ogg|zip|7z|dll|exe|csproj|sln)$", re.IGNORECASE)


def extract_string_literals(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    return _STR_RE.findall(text)


def looks_like_type_name(s: str) -> bool:
    if not s:
        return False
    if any(ch.isspace() for ch in s):
        return False
    if len(s) < 3 or len(s) > 80:
        return False
    if "<" in s or ">" in s:
        return False
    if _GUID_RE.match(s):
        return False
    if _FILE_EXT_RE.match(s):
        return False
    # File/path-like strings are not object types.
    sl = s.lower()
    if sl.startswith(("./", ".\\", "../", "..\\")):
        return False
    if ":/" in sl or ":\\" in sl:
        return False
    if "/" in s and any(seg in sl for seg in ["docs", "src", "bin", "obj", "assets"]):
        return False
    # Exclude very common non-type strings quickly
    if s.lower() in {"true", "false", "null", "none"}:
        return False
    # Heuristic: RA3 type names typically have letters/underscores/slashes and often CamelCase or PREFIX_.
    has_letter = any(ch.isalpha() for ch in s)
    if not has_letter:
        return False
    if "/" in s:
        return True
    if "_" in s:
        return True
    if s[0].isupper() and any(ch.isupper() for ch in s[1:]):
        return True
    if re.search(r"[A-Za-z]{2,}\\d+", s):
        return True
    return False


def scan_codebases(code_dirs: list[Path]) -> Counter[str]:
    out: Counter[str] = Counter()
    for root in code_dirs:
        for f in iter_files(root):
            for lit in extract_string_literals(f):
                if looks_like_type_name(lit):
                    out[lit] += 1
    return out


def scan_maps(maps_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    cfg = ParserConfig.default()
    for map_path in sorted(maps_dir.rglob("*.map")):
        try:
            ra3 = Ra3Map(str(map_path), config=cfg)
            ra3.parse()
            ctx = ra3.get_context()
            ol = ctx.get_asset("ObjectsList")
            if not ol:
                continue
            for o in ol.map_objects:
                if o.type_name:
                    counts[o.type_name] += 1
        except Exception:
            # Keep going; report focuses on coverage, not parse failures.
            continue
    return counts


def is_buildingish(type_name: str) -> bool:
    tl = type_name.lower()
    tokens = [
        "house",
        "hotel",
        "apartment",
        "townhouse",
        "church",
        "restaurant",
        "shop",
        "store",
        "warehouse",
        "villa",
        "mansion",
        "lighthouse",
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
    return any(t in tl for t in tokens)


def training_includes(category_key: str, included: set[str]) -> bool:
    if not category_key:
        return False
    if category_key in included:
        return True
    if category_key.startswith("garrison_") and "garrison" in included:
        return True
    if category_key.startswith("building_") and "building" in included:
        return True
    return False


def resolve_category_key(cat_cfg: ObjectCategoryConfig, category_obj) -> str | None:
    if category_obj is None:
        return None
    # Prefer identity match, fallback to name match
    for k, v in cat_cfg.get_all_categories().items():
        if v is category_obj or v.name == category_obj.name:
            return k
    return None


def write_report(out_dir: Path, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "object_types_deep_scan.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines: list[str] = []
    lines.append("# RA3 Object Type Deep Scan Report")
    lines.append("")
    lines.append(f"- **maps_scanned**: {data['maps_scanned']}")
    lines.append(f"- **unique_types_in_maps**: {data['unique_types_in_maps']}")
    lines.append(f"- **unique_type_like_strings_in_code**: {data['unique_type_like_strings_in_code']}")
    lines.append("")

    lines.append("## Highest-signal risk lists")
    lines.append("")

    def emit_list(title: str, items: list[dict], limit: int = 50):
        lines.append(f"### {title}")
        lines.append("")
        if not items:
            lines.append("- (none)")
            lines.append("")
            return
        for row in items[:limit]:
            t = row["type"]
            c = row.get("count", 0)
            cat = row.get("category", None)
            key = row.get("category_key", None)
            lines.append(f"- `{t}` x{c} (category={cat}, key={key})")
        lines.append("")

    emit_list(
        "Dropped by training filter but building-ish (must fix)",
        data["dropped_buildingish"],
        limit=80,
    )
    emit_list(
        "Unknown but building-ish (review + categorize)",
        data["unknown_buildingish"],
        limit=80,
    )
    emit_list(
        "Map types referenced in code (intersection, high signal)",
        data["code_referenced_map_types"],
        limit=120,
    )
    emit_list(
        "Types present in maps but never referenced in code (FYI)",
        data["map_only_types"],
        limit=40,
    )
    emit_list(
        "Type-like strings referenced in code but never seen in maps (FYI)",
        data["code_only_types"],
        limit=40,
    )

    lines.append("## Category counts (drawn)")
    lines.append("")
    for name, c in data["by_category_drawn"][:50]:
        lines.append(f"- **{name}**: {c}")
    lines.append("")

    (out_dir / "object_types_deep_scan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Deep scan RA3 object type coverage")
    ap.add_argument("--maps", type=str, default="../RA3 Official maps", help="Maps directory to scan")
    ap.add_argument(
        "--code",
        type=str,
        action="append",
        default=[],
        help="Code directory to scan (repeatable). Examples: ../../MapCreatorCore",
    )
    ap.add_argument("--out", type=str, default="reports", help="Output directory (relative to python_tools)")
    args = ap.parse_args()

    maps_dir = Path(args.maps)
    if not maps_dir.exists():
        print(f"Error: maps dir not found: {maps_dir}")
        return 2

    code_dirs = [Path(p) for p in args.code]
    for p in code_dirs:
        if not p.exists():
            print(f"Warning: code dir not found (skipping): {p}")

    # Scan maps for actual type occurrences
    map_counts = scan_maps(maps_dir)

    # Scan codebases for type-like string literals
    code_counts = scan_codebases([p for p in code_dirs if p.exists()])

    cat_cfg = ObjectCategoryConfig()
    training_keys = ParserConfig.training_config().included_object_categories or set()

    by_category_drawn = Counter()
    unknown_buildingish: Counter[str] = Counter()
    dropped_buildingish: Counter[str] = Counter()

    for t, c in map_counts.items():
        category_obj, should_draw = cat_cfg.get_category_for_object(t)
        category_name = category_obj.name if category_obj else None
        if should_draw and category_name:
            by_category_drawn[category_name] += c

        cat_key = resolve_category_key(cat_cfg, category_obj) or ""
        included = training_includes(cat_key, training_keys) if cat_key else False

        if category_obj is None and should_draw and is_buildingish(t):
            unknown_buildingish[t] += c

        # Flag only building-ish types that would be dropped by training filter
        if is_buildingish(t):
            if category_obj is None and should_draw:
                dropped_buildingish[t] += c
            elif category_obj is not None and should_draw and not included:
                dropped_buildingish[t] += c

    # Build some diffs for context
    map_types = set(map_counts.keys())
    code_types = set(code_counts.keys())

    # Intersection is the most useful: code that references real in-map types.
    code_referenced_map_types = sorted(map_types & code_types)

    # For the FYI lists, keep them somewhat useful (avoid listing thousands of decorative items).
    map_only = []
    for t in sorted(map_types - code_types):
        category_obj, should_draw = cat_cfg.get_category_for_object(t)
        if should_draw and (category_obj is not None or is_buildingish(t)):
            map_only.append(t)
    code_only = sorted(code_types - map_types)

    def rows(counter: Counter[str]) -> list[dict]:
        out = []
        for t, c in counter.most_common():
            category_obj, should_draw = cat_cfg.get_category_for_object(t)
            if not should_draw:
                continue
            out.append(
                {
                    "type": t,
                    "count": c,
                    "category": category_obj.name if category_obj else None,
                    "category_key": resolve_category_key(cat_cfg, category_obj),
                }
            )
        return out

    data = {
        "maps_scanned": len(list(maps_dir.rglob("*.map"))),
        "unique_types_in_maps": len(map_types),
        "unique_type_like_strings_in_code": len(code_types),
        "by_category_drawn": by_category_drawn.most_common(),
        "unknown_buildingish": rows(unknown_buildingish),
        "dropped_buildingish": rows(dropped_buildingish),
        "code_referenced_map_types": [
            {"type": t, "count": map_counts.get(t, 0), "code_refs": code_counts.get(t, 0)}
            for t in code_referenced_map_types[:2000]
        ],
        "map_only_types": [{"type": t, "count": map_counts.get(t, 0)} for t in map_only[:500]],
        "code_only_types": [{"type": t, "count": code_counts.get(t, 0)} for t in code_only[:500]],
    }

    out_dir = Path(args.out)
    write_report(out_dir, data)
    print(f"Wrote report to: {out_dir / 'object_types_deep_scan.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


