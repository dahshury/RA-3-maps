#!/usr/bin/env python3
"""
Analyze whether Archon controller players have a "keepalive" building and any owned objects.

This is meant to debug the "controllers insta-defeated" issue in paired-3p Archon maps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map


def _owner_prefix(player_num: int) -> str:
    return f"Player_{player_num}/"


def _obj_owner(obj) -> str:
    p = obj.asset_property_collection.get_property("originalOwner")
    return p.data if p and isinstance(p.data, str) else ""


def _obj_uid(obj) -> str:
    return obj.unique_id or ""


def _obj_name(obj) -> str:
    p = obj.asset_property_collection.get_property("objectName")
    return p.data if p and isinstance(p.data, str) else ""


def analyze(map_path: Path, controller_numbers: List[int]) -> None:
    m = Ra3Map(str(map_path))
    m.parse()
    ctx = m.get_context()

    objs = ctx.get_asset("ObjectsList")
    if not objs:
        print("No ObjectsList")
        return

    # Collect per-controller owned objects
    owned: Dict[int, List[Tuple[str, str, str, Tuple[float, float, float]]]] = {n: [] for n in controller_numbers}
    keepalive: Dict[int, List[Tuple[str, str, str, Tuple[float, float, float]]]] = {n: [] for n in controller_numbers}

    for o in objs.map_objects:
        owner = _obj_owner(o)
        for n in controller_numbers:
            if owner.startswith(_owner_prefix(n)):
                uid = _obj_uid(o)
                t = getattr(o, "type_name", "") or ""
                name = _obj_name(o)
                owned[n].append((t, uid, name, o.position))
                if t == "EI_EasterIslandHeadDefense":
                    keepalive[n].append((t, uid, name, o.position))
                break

    print(f"Map: {map_path}")
    print(f"Controllers: {controller_numbers}")
    print()

    for n in controller_numbers:
        print(f"Player_{n}: owned_objects={len(owned[n])}, keepalives={len(keepalive[n])}")
        # Show a small sample (prioritize keepalives)
        for t, uid, name, pos in keepalive[n][:5]:
            print(f"  KEEPALIVE: type={t} uid={uid} objectName={name} pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
        non_keep = [x for x in owned[n] if x[0] != "EI_EasterIslandHeadDefense"]
        for t, uid, name, pos in non_keep[:8]:
            print(f"  owned: type={t} uid={uid} objectName={name} pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
        if len(non_keep) > 8:
            print(f"  ... {len(non_keep) - 8} more owned objects")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="Path to a .map file")
    ap.add_argument("--controllers", default="2,4,6", help="Comma-separated controller player numbers")
    args = ap.parse_args()

    controller_numbers = [int(x.strip()) for x in args.controllers.split(",") if x.strip()]
    analyze(Path(args.map), controller_numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

