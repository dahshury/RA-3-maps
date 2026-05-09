#!/usr/bin/env python3
"""
Compare two maps' ObjectsList by uniqueID and report objects missing in either map.

This helps pinpoint missing Archon-critical objects (keepalives, beacons, waypoints, etc.).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map


def _uid(o) -> str:
    p = o.asset_property_collection.get_property("uniqueID")
    return p.data if p and isinstance(p.data, str) else ""


def _owner(o) -> str:
    p = o.asset_property_collection.get_property("originalOwner")
    return p.data if p and isinstance(p.data, str) else ""


def _wpn(o) -> str:
    p = o.asset_property_collection.get_property("waypointName")
    return p.data if p and isinstance(p.data, str) else ""


def _kind(o) -> str:
    return (o.type_name or "").strip()


def load_objects(map_path: str) -> Dict[str, Tuple[str, str, str]]:
    m = Ra3Map(map_path)
    m.parse()
    ctx = m.get_context()
    objs = ctx.get_asset("ObjectsList")
    if not objs:
        return {}
    out: Dict[str, Tuple[str, str, str]] = {}
    for o in objs.map_objects:
        uid = _uid(o)
        if not uid:
            continue
        out[uid] = (_kind(o), _owner(o), _wpn(o))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="Map A")
    ap.add_argument("b", help="Map B")
    ap.add_argument("--only", choices=["a", "b", "both"], default="both", help="Which side to show")
    args = ap.parse_args()

    a = load_objects(args.a)
    b = load_objects(args.b)

    a_only = sorted(set(a.keys()) - set(b.keys()))
    b_only = sorted(set(b.keys()) - set(a.keys()))

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"A objects with uniqueID: {len(a)}")
    print(f"B objects with uniqueID: {len(b)}")
    print(f"Only in A: {len(a_only)}")
    print(f"Only in B: {len(b_only)}")
    print()

    def show(title: str, uids):
        if not uids:
            return
        print(title)
        print("| uniqueID | type | originalOwner | waypointName |")
        print("|----------|------|--------------|--------------|")
        for uid in uids[:200]:
            kind, owner, wpn = (a if "A" in title else b)[uid]
            print(f"| {uid} | {kind} | {owner} | {wpn} |")
        if len(uids) > 200:
            print(f"\n... truncated, showed 200 of {len(uids)}")
        print()

    if args.only in ("a", "both"):
        show("Only in A", a_only)
    if args.only in ("b", "both"):
        show("Only in B", b_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

