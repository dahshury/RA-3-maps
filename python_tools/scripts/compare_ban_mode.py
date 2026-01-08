#!/usr/bin/env python3
"""
Compare a base map with its ban mode version to identify differences.
This helps understand what makes a ban mode map.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map


def compare_maps(base_path: str, ban_path: str):
    """Compare base map with ban mode version."""
    print(f"Loading base map: {base_path}")
    base_map = Ra3Map(base_path)
    base_map.parse()
    base_ctx = base_map.get_context()
    
    print(f"Loading ban mode map: {ban_path}")
    ban_map = Ra3Map(ban_path)
    ban_map.parse()
    ban_ctx = ban_map.get_context()
    
    print("\n" + "="*80)
    print("COMPARISON: Base Map vs Ban Mode Map")
    print("="*80)
    
    # Basic map info
    print(f"\nMap Dimensions:")
    print(f"  Base: {base_ctx.map_width}x{base_ctx.map_height}")
    print(f"  Ban:  {ban_ctx.map_width}x{ban_ctx.map_height}")
    
    # Player starts
    from map_processor.assets.objects.objects_list import ObjectsList
    from map_processor.assets.objects.map_object import MapObject
    
    base_objects = base_ctx.get_asset('ObjectsList')
    ban_objects = ban_ctx.get_asset('ObjectsList')
    
    base_starts = [o for o in base_objects.map_objects if hasattr(o, 'unique_id') and 'Player_' in o.unique_id and '_Start' in o.unique_id]
    ban_starts = [o for o in ban_objects.map_objects if hasattr(o, 'unique_id') and 'Player_' in o.unique_id and '_Start' in o.unique_id]
    
    print(f"\nPlayer Start Positions:")
    print(f"  Base: {len(base_starts)} starts")
    for s in sorted(base_starts, key=lambda x: x.unique_id):
        print(f"    {s.unique_id}: {s.position}")
    print(f"  Ban:  {len(ban_starts)} starts")
    for s in sorted(ban_starts, key=lambda x: x.unique_id):
        print(f"    {s.unique_id}: {s.position}")
    
    # SidesList comparison
    print(f"\n{'='*80}")
    print("SIDES LIST (Players)")
    print("="*80)
    base_sides = base_ctx.get_asset('SidesList')
    ban_sides = ban_ctx.get_asset('SidesList')
    
    if base_sides and ban_sides:
        print(f"  Base: {len(base_sides.players)} players")
        print(f"  Ban:  {len(ban_sides.players)} players")
        
        # Compare player properties
        for i in range(min(len(base_sides.players), len(ban_sides.players))):
            p1 = base_sides.players[i]
            p2 = ban_sides.players[i]
            
            pn1 = p1.asset_property_collection.get_property('playerName')
            pn2 = p2.asset_property_collection.get_property('playerName')
            name1 = pn1.data if pn1 else f'[{i}]'
            name2 = pn2.data if pn2 else f'[{i}]'
            
            if name1 != name2:
                print(f"    Player {i}: '{name1}' vs '{name2}'")
            
            # Compare key properties
            props_to_check = ['playerName', 'playerTeam', 'playerSide', 'playerStartMoney']
            for prop_name in props_to_check:
                prop1 = p1.asset_property_collection.get_property(prop_name)
                prop2 = p2.asset_property_collection.get_property(prop_name)
                if prop1 and prop2:
                    if prop1.data != prop2.data:
                        print(f"    {name1}.{prop_name}: {prop1.data} vs {prop2.data}")
    
    # Teams comparison
    print(f"\n{'='*80}")
    print("TEAMS")
    print("="*80)
    base_teams = base_ctx.get_asset('Teams')
    ban_teams = ban_ctx.get_asset('Teams')
    
    if base_teams and ban_teams:
        print(f"  Base: {len(base_teams.teams)} teams")
        print(f"  Ban:  {len(ban_teams.teams)} teams")
        
        # Get team names
        base_team_names = []
        for t in base_teams.teams:
            tn = t.property_collection.get_property('teamName')
            base_team_names.append(tn.data if tn else 'Unknown')
        
        ban_team_names = []
        for t in ban_teams.teams:
            tn = t.property_collection.get_property('teamName')
            ban_team_names.append(tn.data if tn else 'Unknown')
        
        print(f"\n  Base teams: {base_team_names}")
        print(f"  Ban teams:  {ban_team_names}")
        
        only_base = set(base_team_names) - set(ban_team_names)
        only_ban = set(ban_team_names) - set(base_team_names)
        
        if only_base:
            print(f"\n  Only in base: {only_base}")
        if only_ban:
            print(f"  Only in ban:  {only_ban}")
    
    # PlayerScriptsList comparison
    print(f"\n{'='*80}")
    print("PLAYER SCRIPTS")
    print("="*80)
    base_scripts = base_ctx.get_asset('PlayerScriptsList')
    ban_scripts = ban_ctx.get_asset('PlayerScriptsList')
    
    if base_scripts and ban_scripts:
        print(f"  Base: {len(base_scripts.script_lists)} script lists")
        print(f"  Ban:  {len(ban_scripts.script_lists)} script lists")
        
        if len(base_scripts.script_lists) != len(ban_scripts.script_lists):
            print(f"  WARNING: Different number of script lists!")
    
    # ObjectsList comparison - look for special objects
    print(f"\n{'='*80}")
    print("SPECIAL OBJECTS")
    print("="*80)
    
    def get_objects_by_type(objects_list, type_name):
        return [o for o in objects_list.map_objects if hasattr(o, 'type_name') and o.type_name == type_name]
    
    special_types = ['MultiplayerBeacon', 'EI_EasterIslandHeadDefense']
    for obj_type in special_types:
        base_objs = get_objects_by_type(base_objects, obj_type)
        ban_objs = get_objects_by_type(ban_objects, obj_type)
        print(f"  {obj_type}:")
        print(f"    Base: {len(base_objs)}")
        print(f"    Ban:  {len(ban_objs)}")
        if len(base_objs) != len(ban_objs):
            print(f"    *** DIFFERENCE ***")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print("\nKey differences identified:")
    differences = []
    
    if len(base_starts) != len(ban_starts):
        differences.append(f"Player starts: {len(base_starts)} vs {len(ban_starts)}")
    
    if base_sides and ban_sides and len(base_sides.players) != len(ban_sides.players):
        differences.append(f"Players in SidesList: {len(base_sides.players)} vs {len(ban_sides.players)}")
    
    if base_teams and ban_teams and len(base_teams.teams) != len(ban_teams.teams):
        differences.append(f"Teams: {len(base_teams.teams)} vs {len(ban_teams.teams)}")
        differences.append(f"  Base teams: {base_team_names}")
        differences.append(f"  Ban teams:  {ban_team_names}")
    
    if differences:
        for diff in differences:
            print(f"  - {diff}")
    else:
        print("  No major structural differences found in map data.")
        print("  Main difference is likely in XML metadata: IsMultiplayer='false' and NumPlayers='2'")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compare_ban_mode.py <base_map.map> <ban_mode_map.map>")
        sys.exit(1)
    
    base_path = Path(sys.argv[1]).resolve()
    ban_path = Path(sys.argv[2]).resolve()
    
    if not base_path.exists():
        print(f"Error: Base map not found: {base_path}")
        sys.exit(1)
    
    if not ban_path.exists():
        print(f"Error: Ban mode map not found: {ban_path}")
        sys.exit(1)
    
    compare_maps(str(base_path), str(ban_path))
