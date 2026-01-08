#!/usr/bin/env python3
"""
Comprehensive Map Comparison Tool for RA3 Maps

This script performs a DEEP bit-by-bit comparison of two RA3 map files.
It identifies EVERY difference and provides actionable fixes.

Goal: Achieve bit-perfect parity between generated and ground truth archon maps.
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from io import BytesIO
import struct
import copy

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map
from map_processor.utils.refpack import RefPackDecompressor
from map_processor.utils.constants import COMPRESSED_FLAG, UNCOMPRESSED_FLAG


def decompress_map(path: str) -> bytes:
    """Decompress a map file and return raw bytes."""
    with open(path, 'rb') as f:
        flag = struct.unpack('<I', f.read(4))[0]
        if flag == COMPRESSED_FLAG:
            f.seek(8)
            output = BytesIO()
            RefPackDecompressor.decompress(f, output)
            return output.getvalue()
        else:
            f.seek(0)
            return f.read()


def compare_maps(map1_path: str, map2_path: str, verbose: bool = False) -> dict:
    """
    Comprehensively compare two RA3 maps with BIT-PERFECT precision.
    
    Returns a dict with comparison results organized by category.
    """
    results = {
        'summary': {},
        'assets': {},
        'differences': [],
        'critical_differences': [],
        'fixes_needed': [],
    }
    
    # Load maps
    print(f"Loading map 1: {map1_path}")
    m1 = Ra3Map(map1_path)
    m1.parse()
    ctx1 = m1.get_context()
    
    print(f"Loading map 2: {map2_path}")
    m2 = Ra3Map(map2_path)
    m2.parse()
    ctx2 = m2.get_context()
    
    # Get map names for display
    name1 = Path(map1_path).stem
    name2 = Path(map2_path).stem
    
    print(f"\n{'=' * 100}")
    print(f"BIT-PERFECT MAP COMPARISON")
    print(f"{'=' * 100}")
    print(f"Map 1 (TARGET): {name1}")
    print(f"Map 2 (GENERATED): {name2}")
    print(f"{'=' * 100}")
    
    # 0. Binary comparison first
    print(f"\n{'-' * 50}")
    print("0. BINARY COMPARISON")
    print(f"{'-' * 50}")
    compare_binary(map1_path, map2_path, name1, name2, results)
    
    # 1. String Pool Comparison
    print(f"\n{'-' * 50}")
    print("1. STRING POOL")
    print(f"{'-' * 50}")
    compare_string_pools(ctx1, ctx2, name1, name2, results)
    
    # 2. Asset-by-asset byte comparison
    print(f"\n{'-' * 50}")
    print("2. ASSET-BY-ASSET BYTE COMPARISON")
    print(f"{'-' * 50}")
    compare_assets_bytewise(ctx1, ctx2, name1, name2, results)
    
    # 3. SidesList deep comparison
    print(f"\n{'-' * 50}")
    print("3. SIDESLIST DEEP COMPARISON")
    print(f"{'-' * 50}")
    compare_sides_deep(ctx1, ctx2, name1, name2, results)
    
    # 4. Teams deep comparison
    print(f"\n{'-' * 50}")
    print("4. TEAMS DEEP COMPARISON")
    print(f"{'-' * 50}")
    compare_teams_deep(ctx1, ctx2, name1, name2, results)
    
    # 5. PlayerScriptsList deep comparison
    print(f"\n{'-' * 50}")
    print("5. PLAYERSCRIPTSLIST DEEP COMPARISON")
    print(f"{'-' * 50}")
    compare_scripts_deep(ctx1, ctx2, name1, name2, results)
    
    # 6. ObjectsList comparison (archon objects)
    print(f"\n{'-' * 50}")
    print("6. OBJECTSLIST ARCHON OBJECTS")
    print(f"{'-' * 50}")
    compare_objects_archon(ctx1, ctx2, name1, name2, results)
    
    # Summary
    print(f"\n{'=' * 100}")
    print("FIXES NEEDED FOR BIT-PERFECT PARITY")
    print(f"{'=' * 100}")
    
    if results['fixes_needed']:
        for i, fix in enumerate(results['fixes_needed'], 1):
            print(f"  {i}. {fix}")
    else:
        print("  Maps are BIT-PERFECT!")
    
    return results


def compare_binary(path1: str, path2: str, name1: str, name2: str, results: dict):
    """Compare raw decompressed binary data."""
    data1 = decompress_map(path1)
    data2 = decompress_map(path2)
    
    print(f"  {name1}: {len(data1):,} bytes")
    print(f"  {name2}: {len(data2):,} bytes")
    print(f"  Difference: {len(data2) - len(data1):+,} bytes")
    
    if data1 == data2:
        print(f"\n  *** MAPS ARE BIT-PERFECT! ***")
        return
    
    # Count differences
    min_len = min(len(data1), len(data2))
    diff_count = sum(1 for i in range(min_len) if data1[i] != data2[i])
    
    print(f"\n  Byte differences in overlapping region: {diff_count:,}")
    
    # Find first difference
    for i in range(min_len):
        if data1[i] != data2[i]:
            print(f"  First difference at byte {i} (0x{i:X})")
            break
    
    results['binary_match'] = False
    results['fixes_needed'].append(f"Binary mismatch: {len(data2) - len(data1):+} bytes, {diff_count:,} byte differences")


def compare_assets_bytewise(ctx1, ctx2, name1: str, name2: str, results: dict):
    """Compare each asset's serialized bytes."""
    assets1 = {a.get_asset_name(): a for a in ctx1.map_struct.assets}
    assets2 = {a.get_asset_name(): a for a in ctx2.map_struct.assets}
    
    all_assets = sorted(set(assets1.keys()) | set(assets2.keys()))
    
    print(f"\n  {'Asset':<25} {'Size1':>10} {'Size2':>10} {'Diff':>10} {'ByteDiff':>12}")
    print(f"  {'-' * 70}")
    
    for asset_name in all_assets:
        a1 = assets1.get(asset_name)
        a2 = assets2.get(asset_name)
        
        if not a1 or not a2:
            status = "MISSING"
            print(f"  {asset_name:<25} {'N/A' if not a1 else a1.data_size:>10} {'N/A' if not a2 else a2.data_size:>10} {status:>10}")
            results['fixes_needed'].append(f"{asset_name}: missing in one map")
            continue
        
        # Serialize both
        buf1 = BytesIO()
        buf2 = BytesIO()
        
        try:
            a1.save(buf1, ctx1)
            a2.save(buf2, ctx2)
            
            b1 = buf1.getvalue()
            b2 = buf2.getvalue()
            
            size_diff = len(b2) - len(b1)
            
            if b1 == b2:
                status = "OK"
                byte_diff = 0
            else:
                min_len = min(len(b1), len(b2))
                byte_diff = sum(1 for i in range(min_len) if b1[i] != b2[i])
                status = "DIFF"
            
            marker = " ***" if status != "OK" else ""
            print(f"  {asset_name:<25} {len(b1):>10,} {len(b2):>10,} {size_diff:>+10,} {byte_diff:>12,}{marker}")
            
            if status != "OK":
                results['fixes_needed'].append(f"{asset_name}: {size_diff:+} bytes, {byte_diff} byte differences")
                
        except Exception as e:
            print(f"  {asset_name:<25} ERROR: {e}")


def compare_sides_deep(ctx1, ctx2, name1: str, name2: str, results: dict):
    """Deep comparison of SidesList - every property of every player."""
    s1 = ctx1.get_asset('SidesList')
    s2 = ctx2.get_asset('SidesList')
    
    if not s1 or not s2:
        print("  One or both maps missing SidesList")
        return
    
    print(f"  Player count: {name1}={len(s1.players)}, {name2}={len(s2.players)}")
    
    if len(s1.players) != len(s2.players):
        results['fixes_needed'].append(f"SidesList: player count {len(s1.players)} vs {len(s2.players)}")
    
    # Compare each player
    for i in range(max(len(s1.players), len(s2.players))):
        p1 = s1.players[i] if i < len(s1.players) else None
        p2 = s2.players[i] if i < len(s2.players) else None
        
        if not p1 or not p2:
            pn = p1.asset_property_collection.get_property('playerName') if p1 else None
            name = pn.data if pn else f'[{i}]'
            results['fixes_needed'].append(f"SidesList[{i}] {name}: missing in one map")
            continue
        
        pn = p1.asset_property_collection.get_property('playerName')
        player_name = pn.data if pn else f'[{i}]'
        
        # Compare all properties
        props1 = set(p1.asset_property_collection.property_map.keys())
        props2 = set(p2.asset_property_collection.property_map.keys())
        
        only1 = props1 - props2
        only2 = props2 - props1
        
        if only1:
            print(f"  [{i:>2}] {player_name}: missing in {name2}: {only1}")
            results['fixes_needed'].append(f"SidesList[{i}] {player_name}: {name2} missing props {only1}")
        
        if only2:
            print(f"  [{i:>2}] {player_name}: extra in {name2}: {only2}")
            results['fixes_needed'].append(f"SidesList[{i}] {player_name}: {name2} has extra props {only2}")
        
        # Compare common property values
        for prop_name in props1 & props2:
            v1 = p1.asset_property_collection.get_property(prop_name).data
            v2 = p2.asset_property_collection.get_property(prop_name).data
            if v1 != v2:
                print(f"  [{i:>2}] {player_name}.{prop_name}: {v1} vs {v2}")


def compare_teams_deep(ctx1, ctx2, name1: str, name2: str, results: dict):
    """Deep comparison of Teams - every property of every team."""
    t1 = ctx1.get_asset('Teams')
    t2 = ctx2.get_asset('Teams')
    
    if not t1 or not t2:
        print("  One or both maps missing Teams")
        return
    
    print(f"  Team count: {name1}={len(t1.teams)}, {name2}={len(t2.teams)}")
    
    if len(t1.teams) != len(t2.teams):
        results['fixes_needed'].append(f"Teams: count {len(t1.teams)} vs {len(t2.teams)}")
    
    # Compare each team
    for i in range(max(len(t1.teams), len(t2.teams))):
        tm1 = t1.teams[i] if i < len(t1.teams) else None
        tm2 = t2.teams[i] if i < len(t2.teams) else None
        
        if not tm1 or not tm2:
            tn = tm1.property_collection.get_property('teamName') if tm1 else None
            name = tn.data if tn else f'[{i}]'
            results['fixes_needed'].append(f"Teams[{i}] {name}: missing in one map")
            print(f"  [{i:>2}] {name}: MISSING")
            continue
        
        tn = tm1.property_collection.get_property('teamName')
        team_name = tn.data if tn else f'[{i}]'
        
        # Compare all properties
        props1 = set(tm1.property_collection.property_map.keys())
        props2 = set(tm2.property_collection.property_map.keys())
        
        only1 = props1 - props2
        only2 = props2 - props1
        
        if only1:
            print(f"  [{i:>2}] {team_name}: missing in {name2}: {only1}")
            results['fixes_needed'].append(f"Teams[{i}] {team_name}: {name2} missing props {only1}")
        
        if only2:
            print(f"  [{i:>2}] {team_name}: extra in {name2}: {only2}")


def compare_scripts_deep(ctx1, ctx2, name1: str, name2: str, results: dict):
    """Deep comparison of PlayerScriptsList."""
    psl1 = ctx1.get_asset('PlayerScriptsList')
    psl2 = ctx2.get_asset('PlayerScriptsList')
    
    if not psl1 or not psl2:
        print("  One or both maps missing PlayerScriptsList")
        return
    
    print(f"  Script list count: {name1}={len(psl1.script_lists)}, {name2}={len(psl2.script_lists)}")
    
    if len(psl1.script_lists) != len(psl2.script_lists):
        results['fixes_needed'].append(f"PlayerScriptsList: {len(psl1.script_lists)} vs {len(psl2.script_lists)} lists")
    
    def count_scripts(sl):
        def count_group(g):
            return len(g.scripts) + sum(count_group(sg) for sg in g.script_groups)
        return len(sl.scripts) + sum(count_group(g) for g in sl.script_groups)
    
    def get_script_names(sl):
        names = set()
        for s in sl.scripts:
            names.add(s.name)
        for g in sl.script_groups:
            for s in g.scripts:
                names.add(s.name)
            for sg in g.script_groups:
                for s in sg.scripts:
                    names.add(s.name)
        return names
    
    # Compare each list
    for i in range(max(len(psl1.script_lists), len(psl2.script_lists))):
        sl1 = psl1.script_lists[i] if i < len(psl1.script_lists) else None
        sl2 = psl2.script_lists[i] if i < len(psl2.script_lists) else None
        
        if not sl1 or not sl2:
            continue
        
        c1 = count_scripts(sl1)
        c2 = count_scripts(sl2)
        
        if c1 != c2:
            print(f"  [{i:>2}] Script count: {c1} vs {c2}")
            
            names1 = get_script_names(sl1)
            names2 = get_script_names(sl2)
            
            only1 = names1 - names2
            only2 = names2 - names1
            
            if only1:
                print(f"       Only in {name1}: {only1}")
            if only2:
                print(f"       Only in {name2}: {only2}")
                results['fixes_needed'].append(f"ScriptList[{i}]: {name2} has extra scripts {only2}")


def compare_objects_archon(ctx1, ctx2, name1: str, name2: str, results: dict):
    """Compare archon-specific objects."""
    o1 = ctx1.get_asset('ObjectsList')
    o2 = ctx2.get_asset('ObjectsList')
    
    if not o1 or not o2:
        print("  One or both maps missing ObjectsList")
        return
    
    print(f"  Total objects: {name1}={len(o1.map_objects)}, {name2}={len(o2.map_objects)}")
    
    # Focus on archon objects
    archon_types = ['EI_EasterIslandHeadDefense', 'MultiplayerBeacon', '*Waypoints/Waypoint']
    
    for obj_type in archon_types:
        objs1 = [o for o in o1.map_objects if o.type_name == obj_type]
        objs2 = [o for o in o2.map_objects if o.type_name == obj_type]
        
        if len(objs1) != len(objs2):
            print(f"  {obj_type}: {len(objs1)} vs {len(objs2)}")
            results['fixes_needed'].append(f"{obj_type}: count {len(objs1)} vs {len(objs2)}")


def compare_string_pools(ctx1, ctx2, name1, name2, results):
    """Compare string pools between two maps."""
    pool1 = ctx1.map_struct.string_pool
    pool2 = ctx2.map_struct.string_pool
    
    print(f"  {name1}: {len(pool1)} strings")
    print(f"  {name2}: {len(pool2)} strings")
    
    if len(pool1) != len(pool2):
        results['fixes_needed'].append(f"String pool: {len(pool1)} vs {len(pool2)} strings")
    
    set1 = set(pool1.keys())
    set2 = set(pool2.keys())
    
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    
    if only_in_1:
        print(f"\n  Strings only in {name1} ({len(only_in_1)}):")
        for s in sorted(only_in_1)[:10]:
            print(f"    - {s}")
        if len(only_in_1) > 10:
            print(f"    ... and {len(only_in_1) - 10} more")
        results['fixes_needed'].append(f"{name2} missing {len(only_in_1)} strings from {name1}")
    
    if only_in_2:
        print(f"\n  Strings only in {name2} ({len(only_in_2)}):")
        for s in sorted(only_in_2)[:10]:
            print(f"    + {s}")
        if len(only_in_2) > 10:
            print(f"    ... and {len(only_in_2) - 10} more")
        results['fixes_needed'].append(f"{name2} has {len(only_in_2)} extra strings")






def main():
    parser = argparse.ArgumentParser(
        description='Bit-perfect comparison of two RA3 map files'
    )
    parser.add_argument('map1', help='Path to first (TARGET/ground truth) map file')
    parser.add_argument('map2', help='Path to second (GENERATED) map file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show verbose output')
    
    args = parser.parse_args()
    
    results = compare_maps(args.map1, args.map2, args.verbose)
    
    # Return exit code based on fixes needed
    return 1 if results.get('fixes_needed') else 0


if __name__ == '__main__':
    sys.exit(main())
