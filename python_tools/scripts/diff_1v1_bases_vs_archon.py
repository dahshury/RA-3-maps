#!/usr/bin/env python3
"""
Comprehensive diff between 1v1 base maps and Archon GT maps.
Outputs all differences in markdown table format.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from map_processor.core.ra3map import Ra3Map
from io import BytesIO

def get_property_value(prop):
    """Get property value as string"""
    if prop is None:
        return "N/A"
    return str(prop.data)

def diff_map_pair(base_path, gt_path, map_name):
    """Compare a base map with its Archon GT map"""
    rows = []
    
    # Load both maps
    base = Ra3Map(base_path)
    base.parse()
    base_ctx = base.get_context()

    gt = Ra3Map(gt_path)
    gt.parse()
    gt_ctx = gt.get_context()
    
    # 1. String Pool
    base_strings = set(base_ctx.map_struct.string_pool.keys())
    gt_strings = set(gt_ctx.map_struct.string_pool.keys())
    only_base = sorted(list(base_strings - gt_strings))
    only_gt = sorted(list(gt_strings - base_strings))
    
    rows.append((map_name, "String Pool", "Count", f"Base: {len(base_strings)}, GT: {len(gt_strings)}", ""))
    if only_base:
        for s in only_base:
            rows.append((map_name, "String Pool", "Only in Base", s, ""))
    if only_gt:
        for s in only_gt:
            rows.append((map_name, "String Pool", "Only in GT", s, ""))
    
    # String ID differences
    diff_string_ids = []
    common_strings = base_strings & gt_strings
    for s in sorted(common_strings):
        base_id = base_ctx.map_struct.string_pool.get(s)
        gt_id = gt_ctx.map_struct.string_pool.get(s)
        if base_id != gt_id:
            diff_string_ids.append((s, base_id, gt_id))
    
    for s, base_id, gt_id in diff_string_ids[:20]:  # Limit to first 20
        rows.append((map_name, "String Pool", "ID Mismatch", s, f"Base ID: {base_id}, GT ID: {gt_id}"))
    
    # 2. Asset Counts
    base_asset_types = {}
    for a in base_ctx.map_struct.assets:
        name = a.get_asset_name()
        base_asset_types[name] = base_asset_types.get(name, 0) + 1

    gt_asset_types = {}
    for a in gt_ctx.map_struct.assets:
        name = a.get_asset_name()
        gt_asset_types[name] = gt_asset_types.get(name, 0) + 1

    all_types = sorted(set(base_asset_types.keys()) | set(gt_asset_types.keys()))
    for t in all_types:
        base_count = base_asset_types.get(t, 0)
        gt_count = gt_asset_types.get(t, 0)
        if base_count != gt_count:
            rows.append((map_name, "Asset Counts", t, f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 3. SidesList
    base_sides = base_ctx.get_asset("SidesList")
    gt_sides = gt_ctx.get_asset("SidesList")
    if base_sides and gt_sides:
        base_count = len(base_sides.players)
        gt_count = len(gt_sides.players)
        if base_count != gt_count:
            rows.append((map_name, "SidesList", "Player Count", f"Base: {base_count}", f"GT: {gt_count}"))
        
        # Compare player properties
        max_players = max(base_count, gt_count)
        for i in range(max_players):
            if i < base_count and i < gt_count:
                base_pl = base_sides.players[i]
                gt_pl = gt_sides.players[i]
                
                # aiPersonality
                base_ai = base_pl.asset_property_collection.get_property("aiPersonality")
                gt_ai = gt_pl.asset_property_collection.get_property("aiPersonality")
                if get_property_value(base_ai) != get_property_value(gt_ai):
                    rows.append((map_name, "SidesList", f"Player[{i}] aiPersonality", 
                               get_property_value(base_ai), get_property_value(gt_ai)))
                
                # playerColor
                base_color = base_pl.asset_property_collection.get_property("playerColor")
                gt_color = gt_pl.asset_property_collection.get_property("playerColor")
                if get_property_value(base_color) != get_property_value(gt_color):
                    rows.append((map_name, "SidesList", f"Player[{i}] playerColor", 
                               get_property_value(base_color), get_property_value(gt_color)))
                
                # playerRadarColor
                base_radar = base_pl.asset_property_collection.get_property("playerRadarColor")
                gt_radar = gt_pl.asset_property_collection.get_property("playerRadarColor")
                if get_property_value(base_radar) != get_property_value(gt_radar):
                    rows.append((map_name, "SidesList", f"Player[{i}] playerRadarColor", 
                               get_property_value(base_radar), get_property_value(gt_radar)))
    
    # 4. Teams
    base_teams = base_ctx.get_asset("Teams")
    gt_teams = gt_ctx.get_asset("Teams")
    if base_teams and gt_teams:
        base_count = len(base_teams.teams)
        gt_count = len(gt_teams.teams)
        if base_count != gt_count:
            rows.append((map_name, "Teams", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
        
        # Get team names
        def get_team_name(team):
            name_prop = team.property_collection.get_property("teamName")
            return name_prop.data if name_prop else "Unknown"
        
        base_team_names = {get_team_name(t): t for t in base_teams.teams}
        gt_team_names = {get_team_name(t): t for t in gt_teams.teams}
        
        # Teams only in GT
        only_gt_teams = sorted(set(gt_team_names.keys()) - set(base_team_names.keys()))
        for team_name in only_gt_teams:
            rows.append((map_name, "Teams", "Only in GT", team_name, ""))
        
        # exportWithScript differences
        for team_name in sorted(set(base_team_names.keys()) & set(gt_team_names.keys())):
            base_team = base_team_names[team_name]
            gt_team = gt_team_names[team_name]
            base_export = base_team.property_collection.get_property("exportWithScript")
            gt_export = gt_team.property_collection.get_property("exportWithScript")
            if get_property_value(base_export) != get_property_value(gt_export):
                rows.append((map_name, "Teams", f"{team_name} exportWithScript", 
                           get_property_value(base_export), get_property_value(gt_export)))
    
    # 5. ObjectsList
    base_objects = base_ctx.get_asset("ObjectsList")
    gt_objects = gt_ctx.get_asset("ObjectsList")
    if base_objects and gt_objects:
        base_count = len(base_objects.map_objects)
        gt_count = len(gt_objects.map_objects)
        if base_count != gt_count:
            rows.append((map_name, "ObjectsList", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
        
        # Find Archon-specific objects (keepalives, beacons, waypoints)
        def find_archon_objects(obj_list):
            keepalives = []
            beacons = []
            waypoints = []
            player_starts = []
            for obj in obj_list.map_objects:
                obj_name = obj.asset_property_collection.get_property("objectName")
                if obj_name and "EI_EasterIslandHeadDefense" in str(obj_name.data):
                    keepalives.append(obj)
                elif obj_name and "MultiplayerBeacon" in str(obj_name.data):
                    beacons.append(obj)
                elif obj_name and "Waypoint" in str(obj_name.data):
                    waypoints.append(obj)
                elif obj_name and "Player_" in str(obj_name.data) and "_Start" in str(obj_name.data):
                    player_starts.append(obj)
            return keepalives, beacons, waypoints, player_starts
        
        base_keep, base_beac, base_wp, base_ps = find_archon_objects(base_objects)
        gt_keep, gt_beac, gt_wp, gt_ps = find_archon_objects(gt_objects)
        
        if len(gt_keep) > len(base_keep):
            for i, obj in enumerate(gt_keep[len(base_keep):]):
                obj_name = obj.asset_property_collection.get_property("objectName")
                owner = obj.asset_property_collection.get_property("originalOwner")
                pos = obj.asset_property_collection.get_property("position")
                indestructible = obj.asset_property_collection.get_property("objectIndestructible")
                enabled = obj.asset_property_collection.get_property("objectEnabled")
                owner_str = get_property_value(owner)
                pos_str = get_property_value(pos)
                ind_str = get_property_value(indestructible)
                en_str = get_property_value(enabled)
                rows.append((map_name, "ObjectsList", f"Keepalive {obj_name.data if obj_name else 'Unknown'} {len(base_keep)+i}", 
                           f"owner={owner_str}, pos={pos_str}, indestructible={ind_str}, enabled={en_str}", ""))
        
        if len(gt_ps) > len(base_ps):
            for obj in gt_ps[len(base_ps):]:
                obj_name = obj.asset_property_collection.get_property("objectName")
                rows.append((map_name, "ObjectsList", "Archon Object (GT only)", 
                           obj_name.data if obj_name else "Unknown", ""))
    
    # 6. PlayerScriptsList
    base_psl = base_ctx.get_asset("PlayerScriptsList")
    gt_psl = gt_ctx.get_asset("PlayerScriptsList")
    if base_psl and gt_psl:
        base_count = len(base_psl.script_lists)
        gt_count = len(gt_psl.script_lists)
        if base_count != gt_count:
            rows.append((map_name, "PlayerScriptsList", "List Count", f"Base: {base_count}", f"GT: {gt_count}"))
        
        # Compare script lists
        max_lists = max(base_count, gt_count)
        for i in range(min(max_lists, 20)):  # Limit to first 20
            base_list = base_psl.script_lists[i] if i < base_count else None
            gt_list = gt_psl.script_lists[i] if i < gt_count else None
            
            if base_list and gt_list:
                base_scripts = len(base_list.scripts)
                base_groups = len(base_list.script_groups)
                gt_scripts = len(gt_list.scripts)
                gt_groups = len(gt_list.script_groups)
                if base_scripts != gt_scripts or base_groups != gt_groups:
                    rows.append((map_name, "PlayerScriptsList", f"List[{i}]", 
                               f"Base: {base_scripts} scripts, {base_groups} groups", 
                               f"GT: {gt_scripts} scripts, {gt_groups} groups"))
            elif gt_list:
                gt_scripts = len(gt_list.scripts)
                gt_groups = len(gt_list.script_groups)
                rows.append((map_name, "PlayerScriptsList", f"List[{i}]", 
                           "Base: N/A", f"GT: {gt_scripts} scripts, {gt_groups} groups"))
    
    # 7. BuildLists
    base_build = base_ctx.get_asset("BuildLists")
    gt_build = gt_ctx.get_asset("BuildLists")
    if base_build and gt_build:
        base_count = len(base_build.build_list)
        gt_count = len(gt_build.build_list)
        if base_count != gt_count:
            rows.append((map_name, "BuildLists", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 8. LibraryMapLists
    base_lib = base_ctx.get_asset("LibraryMapLists")
    gt_lib = gt_ctx.get_asset("LibraryMapLists")
    if base_lib and gt_lib:
        base_count = len(base_lib.library_maps)
        gt_count = len(gt_lib.library_maps)
        if base_count != gt_count:
            rows.append((map_name, "LibraryMapLists", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 9. MPPositionList
    base_mp = base_ctx.get_asset("MPPositionList")
    gt_mp = gt_ctx.get_asset("MPPositionList")
    if base_mp and gt_mp:
        base_count = len(base_mp.positions)
        gt_count = len(gt_mp.positions)
        if base_count != gt_count:
            rows.append((map_name, "MPPositionList", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 10. NamedCameras
    base_cam = base_ctx.get_asset("NamedCameras")
    gt_cam = gt_ctx.get_asset("NamedCameras")
    if base_cam and gt_cam:
        base_size = len(base_cam.cameras) if hasattr(base_cam, 'cameras') else 0
        gt_size = len(gt_cam.cameras) if hasattr(gt_cam, 'cameras') else 0
        if base_size != gt_size:
            rows.append((map_name, "NamedCameras", "Size", f"Base: {base_size} bytes", f"GT: {gt_size} bytes"))
    
    # 11. StandingWaveAreas
    base_wave = base_ctx.get_asset("StandingWaveAreas")
    gt_wave = gt_ctx.get_asset("StandingWaveAreas")
    if base_wave and gt_wave:
        base_count = len(base_wave.areas) if hasattr(base_wave, 'areas') else 0
        gt_count = len(gt_wave.areas) if hasattr(gt_wave, 'areas') else 0
        if base_count != gt_count:
            rows.append((map_name, "StandingWaveAreas", "Count", f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 12. WorldInfo
    base_wi = base_ctx.get_asset("WorldInfo")
    gt_wi = gt_ctx.get_asset("WorldInfo")
    if base_wi and gt_wi:
        base_tts = get_property_value(base_wi.properties.get_property("terrainTextureStrings"))
        gt_tts = get_property_value(gt_wi.properties.get_property("terrainTextureStrings"))
        if base_tts != gt_tts:
            rows.append((map_name, "WorldInfo", "terrainTextureStrings", 
                        base_tts[:100] if len(base_tts) > 100 else base_tts, 
                        gt_tts[:100] if len(gt_tts) > 100 else gt_tts))
    
    # 13. Asset Sizes
    for base_asset in base_ctx.map_struct.assets:
        name = base_asset.get_asset_name()
        gt_asset = gt_ctx.get_asset(name)
        if gt_asset:
            base_buf = BytesIO()
            base_asset.save(base_buf, base_ctx)
            base_size = len(base_buf.getvalue())
            
            gt_buf = BytesIO()
            gt_asset.save(gt_buf, gt_ctx)
            gt_size = len(gt_buf.getvalue())
            
            if base_size != gt_size:
                rows.append((map_name, "Asset Sizes", name, 
                           f"Base: {base_size} bytes", 
                           f"GT: {gt_size} bytes (diff: {gt_size-base_size:+d})"))
    
    return rows

def main():
    # Map base maps to their Archon GT counterparts
    base_dir = Path("../RA3 Official maps")
    archon_dir = base_dir / "Archon_Maps_1_player_examples/Archon Maps v1.4"
    
    map_pairs = [
        ("CE", base_dir / "2 CE/map_mp_2_feasel7.map", 
         archon_dir / "Archon BattleBase Beta [1.4]/Archon BattleBase Beta [1.4].map"),
        ("COC", base_dir / "2 COC/map_mp_2_feasel2.map",
         archon_dir / "Archon Industrial Strength [1.4]/Archon Industrial Strength [1.4].map"),
        ("CR", base_dir / "2 CR/map_mp_2_feasel1.map",
         archon_dir / "Archon Cabana Republic [1.4]/Archon Cabana Republic [1.4].map"),
        ("FI", base_dir / "2 FI/map_mp_2_feasel8.map",
         archon_dir / "Archon Fire Island [1.4]/Archon Fire Island [1.4].map"),
        ("II", base_dir / "2 II/map_mp_2_rao1.map",
         archon_dir / "Archon Infinity Isle [1.4]/Archon Infinity Isle [1.4].map"),
    ]
    
    all_rows = []
    for map_name, base_path, gt_path in map_pairs:
        if not base_path.exists():
            print(f"Warning: Base map not found: {base_path}", file=sys.stderr)
            continue
        if not gt_path.exists():
            print(f"Warning: GT map not found: {gt_path}", file=sys.stderr)
            continue
        
        print(f"Comparing {map_name}...", file=sys.stderr)
        rows = diff_map_pair(base_path, gt_path, map_name)
        all_rows.extend(rows)
    
    # Output markdown table
    print("| Map | Category | Property | Base Value | GT Value |")
    print("|-----|----------|----------|------------|----------|")
    for map_name, category, property_name, base_val, gt_val in all_rows:
        # Escape pipe characters
        map_name = str(map_name).replace("|", "\\|")
        category = str(category).replace("|", "\\|")
        property_name = str(property_name).replace("|", "\\|")
        base_val = str(base_val).replace("|", "\\|")
        gt_val = str(gt_val).replace("|", "\\|")
        print(f"| {map_name} | {category} | {property_name} | {base_val} | {gt_val} |")

if __name__ == "__main__":
    main()
