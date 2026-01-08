"""
Compare a normal RA3 map with an Archon-mode map to identify the differences.
This helps understand how to transform any map into Archon mode.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.multiplayer.mp_position_list import MPPositionList
from map_processor.assets.sides.sides_list import SidesList
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.assets.teams.teams import Teams
from map_processor.assets.scripts.player_scripts_list import PlayerScriptsList


def dump_player_info(context, label: str) -> Dict[str, Any]:
    """Dump player-related info from a map."""
    info = {"label": label}
    
    # MP Positions
    mp_positions = context.get_asset('MPPositionList')
    if mp_positions and hasattr(mp_positions, 'positions'):
        info["mp_positions"] = []
        for i, pos in enumerate(mp_positions.positions):
            pos_data = {
                "index": i,
                "is_human": pos.is_human,
                "is_computer": pos.is_computer,
                "load_ai_script": pos.load_ai_script,
                "team": pos.team,
                "side_restriction": pos.side_restriction,
            }
            info["mp_positions"].append(pos_data)
    
    # Sides List (players)
    sides_list = context.get_asset('SidesList')
    if sides_list and hasattr(sides_list, 'players'):
        info["players"] = []
        for i, player in enumerate(sides_list.players):
            player_data = {
                "index": i,
                "properties": {},
                "build_list_count": len(player.build_list_items) if player.build_list_items else 0,
            }
            # Get all properties
            for name, prop in player.asset_property_collection.property_map.items():
                player_data["properties"][name] = prop.data
            info["players"].append(player_data)
    
    # Teams
    teams = context.get_asset('Teams')
    if teams and hasattr(teams, 'teams'):
        info["teams"] = []
        for i, team in enumerate(teams.teams):
            team_data = {
                "index": i,
                "properties": {}
            }
            for name, prop in team.property_collection.property_map.items():
                team_data["properties"][name] = prop.data
            info["teams"].append(team_data)
    
    # Player Start Objects
    objects_list = context.get_asset('ObjectsList')
    player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start',
                        'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
    if objects_list and objects_list.map_objects:
        info["player_starts"] = []
        info["total_objects"] = len(objects_list.map_objects)
        for obj in objects_list.map_objects:
            unique_id = obj.unique_id
            if unique_id in player_start_ids or (unique_id and 'Player' in unique_id and 'Start' in unique_id):
                start_data = {
                    "unique_id": unique_id,
                    "type_name": obj.type_name,
                    "position": obj.position,
                    "angle": obj.angle,
                    "properties": {}
                }
                for name, prop in obj.asset_property_collection.property_map.items():
                    start_data["properties"][name] = prop.data
                info["player_starts"].append(start_data)
        
        # Also look for any objects with "archon" or "controller" in name/type
        info["archon_related_objects"] = []
        for obj in objects_list.map_objects:
            type_lower = obj.type_name.lower() if obj.type_name else ""
            unique_lower = (obj.unique_id or "").lower()
            if "archon" in type_lower or "controller" in type_lower or "builder" in type_lower:
                info["archon_related_objects"].append({
                    "unique_id": obj.unique_id,
                    "type_name": obj.type_name,
                    "position": obj.position,
                })
    
    # Scripts
    scripts = context.get_asset('PlayerScriptsList')
    if scripts and hasattr(scripts, 'script_lists'):
        info["script_lists"] = []
        for i, script_list in enumerate(scripts.script_lists):
            list_data = {
                "index": i,
                "name": getattr(script_list, 'name', ''),
                "scripts": [],
                "script_groups": [],
            }
            # Get scripts from the list
            if hasattr(script_list, 'scripts'):
                for script in script_list.scripts:
                    script_data = {
                        "name": script.name,
                        "comment": script.comment,
                        "is_active": script.is_active,
                        "is_subroutine": script.is_subroutine,
                        "conditions_count": len(script.script_or_conditions) if script.script_or_conditions else 0,
                        "actions_true_count": len(script.script_action_on_true) if script.script_action_on_true else 0,
                        "actions_false_count": len(script.script_action_on_false) if script.script_action_on_false else 0,
                    }
                    list_data["scripts"].append(script_data)
            
            # Get script groups from the list
            if hasattr(script_list, 'script_groups'):
                for group in script_list.script_groups:
                    group_data = dump_script_group(group)
                    list_data["script_groups"].append(group_data)
            
            info["script_lists"].append(list_data)
    
    return info


def dump_script_group(group) -> Dict[str, Any]:
    """Recursively dump script group info."""
    group_data = {
        "name": group.name,
        "is_active": group.is_active,
        "is_subroutine": group.is_subroutine,
        "scripts": [],
        "script_groups": [],
    }
    
    for script in group.scripts:
        script_data = {
            "name": script.name,
            "comment": script.comment,
            "is_active": script.is_active,
            "is_subroutine": script.is_subroutine,
            "conditions_count": len(script.script_or_conditions) if script.script_or_conditions else 0,
            "actions_true_count": len(script.script_action_on_true) if script.script_action_on_true else 0,
            "actions_false_count": len(script.script_action_on_false) if script.script_action_on_false else 0,
        }
        group_data["scripts"].append(script_data)
    
    for subgroup in group.script_groups:
        group_data["script_groups"].append(dump_script_group(subgroup))
    
    return group_data


def compare_maps(normal_path: str, archon_path: str):
    """Compare a normal map with an archon map."""
    print("=" * 80)
    print("COMPARING NORMAL MAP VS ARCHON MAP")
    print("=" * 80)
    
    parser = Ra3MapParser()
    
    print(f"\nParsing normal map: {normal_path}")
    normal_ctx = parser.parse(normal_path)
    normal_info = dump_player_info(normal_ctx, "normal")
    
    print(f"Parsing archon map: {archon_path}")
    archon_ctx = parser.parse(archon_path)
    archon_info = dump_player_info(archon_ctx, "archon")
    
    # Compare MP Positions
    print("\n" + "=" * 80)
    print("MP POSITIONS COMPARISON")
    print("=" * 80)
    
    print("\n--- Normal Map MP Positions ---")
    for pos in normal_info.get("mp_positions", []):
        print(f"  Position {pos['index']}: human={pos['is_human']}, computer={pos['is_computer']}, "
              f"team={pos['team']}, load_ai_script={pos['load_ai_script']}, "
              f"side_restriction={pos['side_restriction']}")
    
    print("\n--- Archon Map MP Positions ---")
    for pos in archon_info.get("mp_positions", []):
        print(f"  Position {pos['index']}: human={pos['is_human']}, computer={pos['is_computer']}, "
              f"team={pos['team']}, load_ai_script={pos['load_ai_script']}, "
              f"side_restriction={pos['side_restriction']}")
    
    # Compare Players (Sides)
    print("\n" + "=" * 80)
    print("PLAYERS (SIDES) COMPARISON")
    print("=" * 80)
    
    print(f"\n--- Normal Map Players ({len(normal_info.get('players', []))}) ---")
    for player in normal_info.get("players", []):
        props = player['properties']
        name = props.get('playerName', 'Unknown')
        display = props.get('playerDisplayName', '')
        is_human = props.get('playerIsHuman', False)
        faction = props.get('playerFaction', '')
        allies = props.get('playerAllies', '')
        enemies = props.get('playerEnemies', '')
        print(f"  [{player['index']}] {name}: display={display}, human={is_human}, "
              f"faction={faction}, allies={allies}, enemies={enemies}")
        print(f"       Build list items: {player['build_list_count']}")
    
    print(f"\n--- Archon Map Players ({len(archon_info.get('players', []))}) ---")
    for player in archon_info.get("players", []):
        props = player['properties']
        name = props.get('playerName', 'Unknown')
        display = props.get('playerDisplayName', '')
        is_human = props.get('playerIsHuman', False)
        faction = props.get('playerFaction', '')
        allies = props.get('playerAllies', '')
        enemies = props.get('playerEnemies', '')
        print(f"  [{player['index']}] {name}: display={display}, human={is_human}, "
              f"faction={faction}, allies={allies}, enemies={enemies}")
        print(f"       Build list items: {player['build_list_count']}")
        # Print all properties for debugging
        if len(props) > 5:
            print(f"       All properties: {list(props.keys())}")
    
    # Compare Teams
    print("\n" + "=" * 80)
    print("TEAMS COMPARISON")
    print("=" * 80)
    
    print(f"\n--- Normal Map Teams ({len(normal_info.get('teams', []))}) ---")
    for team in normal_info.get("teams", []):
        props = team['properties']
        name = props.get('teamName', 'Unknown')
        owner = props.get('teamOwner', '')
        print(f"  [{team['index']}] {name}: owner={owner}, props={list(props.keys())}")
    
    print(f"\n--- Archon Map Teams ({len(archon_info.get('teams', []))}) ---")
    for team in archon_info.get("teams", []):
        props = team['properties']
        name = props.get('teamName', 'Unknown')
        owner = props.get('teamOwner', '')
        print(f"  [{team['index']}] {name}: owner={owner}, props={list(props.keys())}")
    
    # Compare Player Starts
    print("\n" + "=" * 80)
    print("PLAYER START OBJECTS COMPARISON")
    print("=" * 80)
    
    print(f"\n--- Normal Map Player Starts ({len(normal_info.get('player_starts', []))}) ---")
    for start in normal_info.get("player_starts", []):
        pos = start['position']
        print(f"  {start['unique_id']}: type={start['type_name']}, "
              f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), angle={start['angle']:.1f}")
        if start['properties']:
            print(f"       Properties: {start['properties']}")
    
    print(f"\n--- Archon Map Player Starts ({len(archon_info.get('player_starts', []))}) ---")
    for start in archon_info.get("player_starts", []):
        pos = start['position']
        print(f"  {start['unique_id']}: type={start['type_name']}, "
              f"pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), angle={start['angle']:.1f}")
        if start['properties']:
            print(f"       Properties: {start['properties']}")
    
    # Archon-specific objects
    print("\n" + "=" * 80)
    print("ARCHON-RELATED OBJECTS")
    print("=" * 80)
    if archon_info.get("archon_related_objects"):
        for obj in archon_info["archon_related_objects"]:
            print(f"  {obj['unique_id']}: type={obj['type_name']}, pos={obj['position']}")
    else:
        print("  No archon-specific objects found by name pattern")
    
    # Total objects comparison
    print("\n" + "=" * 80)
    print("OBJECT COUNT COMPARISON")
    print("=" * 80)
    print(f"  Normal map total objects: {normal_info.get('total_objects', 0)}")
    print(f"  Archon map total objects: {archon_info.get('total_objects', 0)}")
    
    # Scripts comparison
    print("\n" + "=" * 80)
    print("SCRIPTS COMPARISON")
    print("=" * 80)
    
    def count_scripts_in_list(script_list_data):
        count = len(script_list_data.get("scripts", []))
        for group in script_list_data.get("script_groups", []):
            count += count_scripts_in_group(group)
        return count
    
    def count_scripts_in_group(group_data):
        count = len(group_data.get("scripts", []))
        for subgroup in group_data.get("script_groups", []):
            count += count_scripts_in_group(subgroup)
        return count
    
    def print_scripts_summary(info, label):
        print(f"\n--- {label} Scripts ---")
        total_scripts = 0
        for sl in info.get("script_lists", []):
            sl_count = count_scripts_in_list(sl)
            total_scripts += sl_count
            if sl_count > 0:
                print(f"  Script List [{sl['index']}] ({sl.get('name', 'unnamed')}): {sl_count} scripts")
                # Show script group names
                for group in sl.get("script_groups", []):
                    group_count = count_scripts_in_group(group)
                    if group_count > 0:
                        print(f"    Group '{group['name']}': {group_count} scripts")
                        # Show individual script names
                        for script in group.get("scripts", []):
                            print(f"      - {script['name']}: active={script['is_active']}, "
                                  f"conditions={script['conditions_count']}, actions={script['actions_true_count']}")
        print(f"  Total scripts: {total_scripts}")
    
    print_scripts_summary(normal_info, "Normal Map")
    print_scripts_summary(archon_info, "Archon Map")
    
    # Save detailed comparison to JSON
    output_path = Path(normal_path).parent / "archon_comparison.json"
    comparison = {
        "normal_map": normal_info,
        "archon_map": archon_info
    }
    
    # Convert tuples to lists for JSON serialization
    def convert_tuples(obj):
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_tuples(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tuples(i) for i in obj]
        return obj
    
    comparison = convert_tuples(comparison)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nDetailed comparison saved to: {output_path}")
    
    return normal_info, archon_info


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent / "RA3 Official maps" / "2 II"
    
    # Default: compare normal vs archon template
    normal_map = base_path / "map_mp_2_rao1.map"
    archon_map = base_path / "archon infinity isle [1.4]" / "archon infinity isle [1.4].map"
    
    # Check for command line args for custom comparison
    import sys
    if len(sys.argv) >= 3:
        normal_map = Path(sys.argv[1])
        archon_map = Path(sys.argv[2])
    
    compare_maps(str(normal_map), str(archon_map))

