#!/usr/bin/env python3
"""
Dump waypoint objects (type *Waypoints/Waypoint) from a map:
- waypointName / waypointID
- uniqueID
- originalOwner
- position
Also reports duplicate waypointName values (which can break Archon scripts).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map


def _get_str_prop(obj, name: str) -> str:
    p = obj.asset_property_collection.get_property(name)
    return p.data if p and isinstance(p.data, str) else ""


def _get_int_prop(obj, name: str) -> int | None:
    p = obj.asset_property_collection.get_property(name)
    return p.data if p and isinstance(p.data, int) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="Path to .map")
    ap.add_argument("--filter", default="", help="Substring filter on waypointName")
    ap.add_argument("--show", type=int, default=200, help="Max rows to show")
    args = ap.parse_args()

    m = Ra3Map(args.map)
    m.parse()
    ctx = m.get_context()

    objs = ctx.get_asset("ObjectsList")
    if not objs:
        print("No ObjectsList")
        return 1

    rows: List[Tuple[str, int | None, str, str, Tuple[float, float, float]]] = []
    for o in objs.map_objects:
        if (o.type_name or "") != "*Waypoints/Waypoint":
            continue
        wpn = _get_str_prop(o, "waypointName")
        if args.filter and args.filter.lower() not in wpn.lower():
            continue
        wpid = _get_int_prop(o, "waypointID")
        uid = _get_str_prop(o, "uniqueID")
        owner = _get_str_prop(o, "originalOwner")
        rows.append((wpn, wpid, uid, owner, o.position))

    rows.sort(key=lambda r: (r[1] if r[1] is not None else 10_000_000, r[0]))

    counts = Counter([r[0] for r in rows])
    dups = [k for k, v in counts.items() if v > 1]

    print(f"Map: {args.map}")
    print(f"Waypoints: {len(rows)}")
    print(f"Duplicate waypointName entries: {len(dups)}")
    if dups:
        print("  " + ", ".join(sorted(dups)[:50]) + (" ..." if len(dups) > 50 else ""))
    print()

    print("| waypointID | waypointName | uniqueID | originalOwner | position |")
    print("|-----------:|--------------|----------|---------------|----------|")
    shown = 0
    for wpn, wpid, uid, owner, pos in rows:
        if shown >= args.show:
            break
        pos_s = f"({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})"
        wpid_s = "" if wpid is None else str(wpid)
        # Escape pipes
        wpn = wpn.replace("|", "\\|")
        uid = uid.replace("|", "\\|")
        owner = owner.replace("|", "\\|")
        print(f"| {wpid_s} | {wpn} | {uid} | {owner} | {pos_s} |")
        shown += 1

    if len(rows) > args.show:
        print(f"\n... truncated, showed {args.show} of {len(rows)} waypoints")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

