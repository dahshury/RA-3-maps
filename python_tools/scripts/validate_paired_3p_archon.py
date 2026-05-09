#!/usr/bin/env python3
"""
Heuristic validation for paired-3p Archon maps (builders 1/3/5, controllers 2/4/6).

This does NOT attempt bit-perfect comparison; it checks the structural/common requirements
that tend to cause "controller insta-defeat" when missing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map


def _get_str_prop(obj, name: str) -> str:
    p = obj.asset_property_collection.get_property(name)
    return p.data if p and isinstance(p.data, str) else ""


def _get_bool_prop(obj, name: str) -> Optional[bool]:
    p = obj.asset_property_collection.get_property(name)
    return p.data if p and isinstance(p.data, bool) else None


def _team_name(team) -> str:
    p = team.property_collection.get_property("teamName")
    return p.data if p and isinstance(p.data, str) else ""


def _player_name(player) -> str:
    p = player.asset_property_collection.get_property("playerName")
    return p.data if p and isinstance(p.data, str) else ""


def validate(map_path: str) -> int:
    m = Ra3Map(map_path)
    m.parse()
    ctx = m.get_context()

    issues: List[str] = []

    sides = ctx.get_asset("SidesList")
    teams = ctx.get_asset("Teams")
    objs = ctx.get_asset("ObjectsList")
    psl = ctx.get_asset("PlayerScriptsList")
    build = ctx.get_asset("BuildLists")
    lib = ctx.get_asset("LibraryMapLists")

    if not sides:
        issues.append("Missing SidesList")
    if not teams:
        issues.append("Missing Teams")
    if not objs:
        issues.append("Missing ObjectsList")
    if not psl:
        issues.append("Missing PlayerScriptsList")
    if not build:
        issues.append("Missing BuildLists")
    if not lib:
        issues.append("Missing LibraryMapLists")

    if issues:
        print(f"Map: {map_path}")
        for it in issues:
            print(f"ERROR: {it}")
        return 1

    # --- Players ---
    players = [_player_name(p) for p in sides.players]
    player_set = set(players)
    want_players = {f"Player_{n}" for n in (1, 2, 3, 4, 5, 6)}
    missing_players = sorted(want_players - player_set)
    if missing_players:
        issues.append(f"SidesList missing players: {missing_players}")

    # --- Teams ---
    team_names = [_team_name(t) for t in teams.teams]
    team_set = set(team_names)
    want_teams = {f"teamPlayer_{n}" for n in (1, 2, 3, 4, 5, 6)}
    missing_teams = sorted(want_teams - team_set)
    if missing_teams:
        issues.append(f"Teams missing teamPlayer_*: {missing_teams}")

    # exportWithScript should be True for most teams (except empty + teamPlyrCivilian)
    bad_export: List[str] = []
    for i, t in enumerate(teams.teams):
        tn = _team_name(t)
        if i == 0 or tn == "teamPlyrCivilian":
            continue
        exp = t.property_collection.get_property("exportWithScript")
        if not exp or exp.data is not True:
            bad_export.append(tn or f"<idx {i}>")
    if bad_export:
        issues.append(f"Teams exportWithScript missing/false on {len(bad_export)} team(s) (sample: {bad_export[:8]})")

    # --- Lists counts coherence ---
    if len(build.build_list) != len(sides.players):
        issues.append(f"BuildLists count {len(build.build_list)} != SidesList player count {len(sides.players)}")
    if len(lib.library_maps) != len(sides.players):
        issues.append(f"LibraryMapLists count {len(lib.library_maps)} != SidesList player count {len(sides.players)}")
    if len(psl.script_lists) != len(sides.players):
        # Not always true across all base maps, but true for official paired-3p archon GTs we saw.
        issues.append(f"PlayerScriptsList list count {len(psl.script_lists)} != SidesList player count {len(sides.players)}")

    # --- Waypoints sanity ---
    # Archon paired-3p uses 12 waypoints total with waypointName:
    want_waypoint_names: Set[str] = set()
    want_waypoint_names |= {f"Player_{n}_Start" for n in (1, 2, 3, 4, 5, 6)}
    want_waypoint_names |= {f"Apron Ocuppier Player_{n}" for n in (1, 3, 5)}
    want_waypoint_names |= {f"Linked Airfield Player_{n}" for n in (2, 4, 6)}

    waypoint_names: List[str] = []
    by_name: Dict[str, int] = Counter()
    for o in objs.map_objects:
        if (o.type_name or "") != "*Waypoints/Waypoint":
            continue
        wpn = _get_str_prop(o, "waypointName")
        if not wpn:
            continue
        waypoint_names.append(wpn)
        by_name[wpn] += 1

    missing_waypoints = sorted(want_waypoint_names - set(waypoint_names))
    dup_waypoints = sorted([n for n, c in by_name.items() if c > 1])
    if missing_waypoints:
        issues.append(f"Missing waypointName(s): {missing_waypoints}")
    if dup_waypoints:
        issues.append(f"Duplicate waypointName(s): {dup_waypoints}")

    # --- Beacons ---
    beacons = [o for o in objs.map_objects if (o.type_name or "") == "MultiplayerBeacon"]
    if len(beacons) != 5:
        issues.append(f"Expected 5 MultiplayerBeacon objects, found {len(beacons)}")

    # --- Controller keepalive buildings ---
    for cn in (2, 4, 6):
        owner_prefix = f"Player_{cn}/"
        keep = [
            o for o in objs.map_objects
            if (o.type_name or "") == "EI_EasterIslandHeadDefense"
            and _get_str_prop(o, "originalOwner").startswith(owner_prefix)
        ]
        if len(keep) != 1:
            issues.append(f"Controller Player_{cn} keepalive count expected 1, found {len(keep)}")

    # --- Output ---
    print(f"Map: {map_path}")
    print(f"SidesList players: {len(sides.players)}")
    print(f"Teams: {len(teams.teams)}")
    print(f"PlayerScriptsList lists: {len(psl.script_lists)}")
    print(f"BuildLists: {len(build.build_list)}")
    print(f"LibraryMapLists: {len(lib.library_maps)}")
    print(f"Waypoints (with waypointName): {len(waypoint_names)}")
    print(f"Beacons: {len(beacons)}")
    print()

    if issues:
        for it in issues:
            print(f"FAIL: {it}")
        return 2

    print("OK: No structural issues detected for paired-3p Archon.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="Path to .map")
    args = ap.parse_args()
    return validate(args.map)


if __name__ == "__main__":
    raise SystemExit(main())

