"""
Report parsing coverage for RA3 .map files.

This parses maps using the Python map_processor and reports:
- Which asset names are parsed into concrete classes vs DefaultMajorAsset (raw)
- Counts across maps
- Total bytes covered by parsed vs raw assets (using per-asset data_size)

Usage:
  python scripts/report_parsing_coverage.py --path "../RA3 Official maps/2 II"
  python scripts/report_parsing_coverage.py --path "../RA3 Official maps" --limit 20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# allow direct run
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.ra3map import Ra3Map
from map_processor.core.default_major_asset import DefaultMajorAsset


@dataclass
class AssetStat:
    parsed_count: int = 0
    raw_count: int = 0
    parsed_bytes: int = 0
    raw_bytes: int = 0


def _iter_map_files(p: Path) -> List[Path]:
    if p.is_file() and p.suffix.lower() == ".map":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.map"))
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Report map_parser asset coverage.")
    ap.add_argument("--path", required=True, help="Map file or directory to scan")
    ap.add_argument("--limit", type=int, default=50, help="Limit how many .map files to parse")
    ap.add_argument("--show-raw", action="store_true", help="Print raw (DefaultMajorAsset) asset names")
    ap.add_argument("--show-parsed", action="store_true", help="Print parsed asset names")
    args = ap.parse_args()

    root = Path(args.path)
    maps = _iter_map_files(root)[: args.limit]
    if not maps:
        print("No .map files found")
        return 1

    stats: Dict[str, AssetStat] = {}
    per_map_summary: List[Tuple[str, int, int]] = []  # (map, raw_count, parsed_count)

    for mp in maps:
        m = Ra3Map(str(mp))
        m.parse()
        ctx = m.get_context()
        raw_ct = 0
        parsed_ct = 0
        for asset in ctx.map_struct.assets:
            name = asset.get_asset_name()
            st = stats.setdefault(name, AssetStat())
            if isinstance(asset, DefaultMajorAsset):
                raw_ct += 1
                st.raw_count += 1
                st.raw_bytes += int(getattr(asset, "data_size", 0) or 0)
            else:
                parsed_ct += 1
                st.parsed_count += 1
                st.parsed_bytes += int(getattr(asset, "data_size", 0) or 0)
        per_map_summary.append((mp.name, raw_ct, parsed_ct))

    # Print per-map
    print(f"Parsed {len(maps)} map(s)\n")
    for name, raw_ct, parsed_ct in per_map_summary:
        print(f"- {name}: parsed_assets={parsed_ct} raw_assets={raw_ct}")

    # Totals
    total_parsed_assets = sum(s.parsed_count for s in stats.values())
    total_raw_assets = sum(s.raw_count for s in stats.values())
    total_parsed_bytes = sum(s.parsed_bytes for s in stats.values())
    total_raw_bytes = sum(s.raw_bytes for s in stats.values())

    print("\nTotals:")
    print(f"- parsed_assets: {total_parsed_assets}")
    print(f"- raw_assets:    {total_raw_assets}")
    print(f"- parsed_bytes:  {total_parsed_bytes}")
    print(f"- raw_bytes:     {total_raw_bytes}")

    # Breakdown: sort by raw_bytes descending (most valuable to implement)
    by_raw_bytes = sorted(stats.items(), key=lambda kv: kv[1].raw_bytes, reverse=True)
    print("\nTop raw assets by bytes (good next targets):")
    for name, st in by_raw_bytes[:25]:
        if st.raw_count == 0:
            continue
        print(f"- {name}: raw_count={st.raw_count}, raw_bytes={st.raw_bytes}")

    if args.show_raw:
        print("\nRaw assets (DefaultMajorAsset):")
        for name, st in by_raw_bytes:
            if st.raw_count:
                print(f"- {name}")

    if args.show_parsed:
        print("\nParsed assets:")
        for name, st in sorted(stats.items(), key=lambda kv: kv[0]):
            if st.parsed_count:
                print(f"- {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())










