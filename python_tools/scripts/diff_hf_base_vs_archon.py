#!/usr/bin/env python3
"""
Comprehensive diff between base maps and Archon GT maps.
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
    # Load both maps
    base = Ra3Map('../RA3 Official maps/3 HF/map_mp_3_feasel3.map')
    base.parse()
    base_ctx = base.get_context()

    gt = Ra3Map('../RA3 Official maps/archon 3 player examples/[Archon]Hidden_Fortress_1.2/[Archon]Hidden_Fortress_1.2.map')
    gt.parse()
    gt_ctx = gt.get_context()

    rows = []
    
    # 1. String Pool
    base_strings = set(base_ctx.map_struct.string_pool.keys())
    gt_strings = set(gt_ctx.map_struct.string_pool.keys())
    only_base = sorted(list(base_strings - gt_strings))
    only_gt = sorted(list(gt_strings - base_strings))
    
    rows.append(("String Pool", "Count", f"Base: {len(base_strings)}, GT: {len(gt_strings)}", ""))
    if only_base:
        for s in only_base:
            rows.append(("String Pool", "Only in Base", s, ""))
    if only_gt:
        for s in only_gt:
            rows.append(("String Pool", "Only in GT", s, ""))
    
    # String ID differences
    diff_string_ids = []
    for s in base_strings & gt_strings:
        base_id = base_ctx.map_struct.string_pool[s]
        gt_id = gt_ctx.map_struct.string_pool[s]
        if base_id != gt_id:
            diff_string_ids.append((s, base_id, gt_id))
    
    for s, base_id, gt_id in sorted(diff_string_ids, key=lambda x: x[1])[:50]:  # Limit to first 50
        rows.append(("String Pool", "ID Mismatch", f"{s}", f"Base ID: {base_id}, GT ID: {gt_id}"))
    
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
            rows.append(("Asset Counts", t, f"Base: {base_count}", f"GT: {gt_count}"))
    
    # 3. SidesList
    base_sides = base_ctx.get_asset('SidesList')
    gt_sides = gt_ctx.get_asset('SidesList')
    rows.append(("SidesList", "Player Count", f"Base: {len(base_sides.players)}", f"GT: {len(gt_sides.players)}"))
    
    # Player details
    for i in range(min(len(base_sides.players), len(gt_sides.players))):
        bp = base_sides.players[i]
        gp = gt_sides.players[i]
        
        bp_name = get_property_value(bp.asset_property_collection.get_property('playerName'))
        gp_name = get_property_value(gp.asset_property_collection.get_property('playerName'))
        
        if bp_name != gp_name:
            rows.append(("SidesList", f"Player[{i}] Name", bp_name, gp_name))
        
        # Check properties
        bp_ai = get_property_value(bp.asset_property_collection.get_property('aiPersonality'))
        gp_ai = get_property_value(gp.asset_property_collection.get_property('aiPersonality'))
        if bp_ai != gp_ai:
            rows.append(("SidesList", f"Player[{i}] aiPersonality", bp_ai, gp_ai))
        
        bp_color = get_property_value(bp.asset_property_collection.get_property('playerColor'))
        gp_color = get_property_value(gp.asset_property_collection.get_property('playerColor'))
        if bp_color != gp_color:
            rows.append(("SidesList", f"Player[{i}] playerColor", bp_color, gp_color))
        
        bp_radar = get_property_value(bp.asset_property_collection.get_property('playerRadarColor'))
        gp_radar = get_property_value(gp.asset_property_collection.get_property('playerRadarColor'))
        if bp_radar != gp_radar:
            rows.append(("SidesList", f"Player[{i}] playerRadarColor", bp_radar, gp_radar))
    
    # 4. Teams
    base_teams = base_ctx.get_asset('Teams')
    gt_teams = gt_ctx.get_asset('Teams')
    rows.append(("Teams", "Count", f"Base: {len(base_teams.teams)}", f"GT: {len(gt_teams.teams)}"))
    
    # Team names
    base_team_names = set()
    for t in base_teams.teams:
        tn = t.property_collection.get_property('teamName')
        if tn:
            base_team_names.add(tn.data)

    gt_team_names = set()
    for t in gt_teams.teams:
        tn = t.property_collection.get_property('teamName')
        if tn:
            gt_team_names.add(tn.data)

    only_base_teams = sorted(list(base_team_names - gt_team_names))
    only_gt_teams = sorted(list(gt_team_names - base_team_names))
    
    for tn in only_base_teams:
        rows.append(("Teams", "Only in Base", tn, ""))
    for tn in only_gt_teams:
        rows.append(("Teams", "Only in GT", tn, ""))
    
    # Check exportWithScript property
    for t in gt_teams.teams:
        tn = get_property_value(t.property_collection.get_property('teamName'))
        export_prop = t.property_collection.get_property('exportWithScript')
        if export_prop and export_prop.data:
            # Check if base has this team
            base_has = False
            for bt in base_teams.teams:
                btn = get_property_value(bt.property_collection.get_property('teamName'))
                if btn == tn:
                    base_has = True
                    base_export = get_property_value(bt.property_collection.get_property('exportWithScript'))
                    if base_export != "True":
                        rows.append(("Teams", f"{tn} exportWithScript", base_export, "True"))
                    break
            if not base_has:
                rows.append(("Teams", f"{tn} exportWithScript", "N/A (team missing)", "True"))
    
    # 5. ObjectsList
    base_objects = base_ctx.get_asset('ObjectsList')
    gt_objects = gt_ctx.get_asset('ObjectsList')
    rows.append(("ObjectsList", "Count", f"Base: {len(base_objects.map_objects)}", f"GT: {len(gt_objects.map_objects)}"))
    
    # Archon-specific objects
    base_archon = []
    gt_archon = []
    for o in base_objects.map_objects:
        uid = o.asset_property_collection.get_property('uniqueID')
        if uid:
            uid_str = uid.data
            if any(x in uid_str for x in ['MultiplayerBeacon', 'EasterIslandHead', 'Linked Airfield', 'Apron', 'Player_']):
                base_archon.append(uid_str)

    for o in gt_objects.map_objects:
        uid = o.asset_property_collection.get_property('uniqueID')
        if uid:
            uid_str = uid.data
            if any(x in uid_str for x in ['MultiplayerBeacon', 'EasterIslandHead', 'Linked Airfield', 'Apron', 'Player_']):
                gt_archon.append(uid_str)
    
    base_archon_set = set(base_archon)
    gt_archon_set = set(gt_archon)
    
    for uid in sorted(gt_archon_set - base_archon_set):
        rows.append(("ObjectsList", "Archon Object (GT only)", uid, ""))
    
    # Keepalive details
    for o in gt_objects.map_objects:
        uid = o.asset_property_collection.get_property('uniqueID')
        if uid and 'EasterIslandHead' in uid.data:
            owner = get_property_value(o.asset_property_collection.get_property('originalOwner'))
            pos = o.position
            indestructible = get_property_value(o.asset_property_collection.get_property('objectIndestructible'))
            enabled = get_property_value(o.asset_property_collection.get_property('objectEnabled'))
            rows.append(("ObjectsList", f"Keepalive {uid.data}", f"owner={owner}, pos=({pos[0]:.1f},{pos[1]:.1f}), indestructible={indestructible}, enabled={enabled}", ""))
    
    # 6. PlayerScriptsList
    base_psl = base_ctx.get_asset('PlayerScriptsList')
    gt_psl = gt_ctx.get_asset('PlayerScriptsList')
    rows.append(("PlayerScriptsList", "List Count", f"Base: {len(base_psl.script_lists)}", f"GT: {len(gt_psl.script_lists)}"))
    
    for i in range(min(len(base_psl.script_lists), len(gt_psl.script_lists))):
        base_sl = base_psl.script_lists[i]
        gt_sl = gt_psl.script_lists[i]
        base_scripts = len(base_sl.scripts) + sum(len(g.scripts) for g in base_sl.script_groups)
        gt_scripts = len(gt_sl.scripts) + sum(len(g.scripts) for g in gt_sl.script_groups)
        base_groups = len(base_sl.script_groups)
        gt_groups = len(gt_sl.script_groups)
        
        if base_scripts != gt_scripts or base_groups != gt_groups:
            rows.append(("PlayerScriptsList", f"List[{i}]", f"Base: {base_scripts} scripts, {base_groups} groups", f"GT: {gt_scripts} scripts, {gt_groups} groups"))
    
    # 7. BuildLists
    base_builds = base_ctx.get_asset('BuildLists')
    gt_builds = gt_ctx.get_asset('BuildLists')
    rows.append(("BuildLists", "Count", f"Base: {len(base_builds.build_list)}", f"GT: {len(gt_builds.build_list)}"))
    
    # 8. LibraryMapLists
    base_libs = base_ctx.get_asset('LibraryMapLists')
    gt_libs = gt_ctx.get_asset('LibraryMapLists')
    rows.append(("LibraryMapLists", "Count", f"Base: {len(base_libs.library_maps)}", f"GT: {len(gt_libs.library_maps)}"))
    
    # 9. MPPositionList
    base_mp = base_ctx.get_asset('MPPositionList')
    gt_mp = gt_ctx.get_asset('MPPositionList')
    rows.append(("MPPositionList", "Count", f"Base: {len(base_mp.positions)}", f"GT: {len(gt_mp.positions)}"))
    
    for i, (bp, gp) in enumerate(zip(base_mp.positions, gt_mp.positions)):
        if bp.is_human != gp.is_human or bp.is_computer != gp.is_computer or bp.team != gp.team:
            rows.append(("MPPositionList", f"Position[{i}]", f"Base: human={bp.is_human}, comp={bp.is_computer}, team={bp.team}", f"GT: human={gp.is_human}, comp={gp.is_computer}, team={gp.team}"))
    
    # 10. NamedCameras
    base_cams = base_ctx.get_asset('NamedCameras')
    gt_cams = gt_ctx.get_asset('NamedCameras')
    if base_cams and hasattr(base_cams, 'data'):
        base_cam_size = len(base_cams.data)
    else:
        base_cam_size = 0
    if gt_cams and hasattr(gt_cams, 'data'):
        gt_cam_size = len(gt_cams.data)
    else:
        gt_cam_size = 0
    
    if base_cam_size != gt_cam_size:
        rows.append(("NamedCameras", "Size", f"Base: {base_cam_size} bytes", f"GT: {gt_cam_size} bytes"))
    
    # 11. StandingWaveAreas
    base_swa = base_ctx.get_asset('StandingWaveAreas')
    gt_swa = gt_ctx.get_asset('StandingWaveAreas')
    base_swa_count = len(base_swa.areas) if base_swa else 0
    gt_swa_count = len(gt_swa.areas) if gt_swa else 0
    if base_swa_count != gt_swa_count:
        rows.append(("StandingWaveAreas", "Count", f"Base: {base_swa_count}", f"GT: {gt_swa_count}"))
    
    # 12. WorldInfo
    base_wi = base_ctx.get_asset('WorldInfo')
    gt_wi = gt_ctx.get_asset('WorldInfo')
    if base_wi and gt_wi:
        base_tts = get_property_value(base_wi.properties.get_property('terrainTextureStrings'))
        gt_tts = get_property_value(gt_wi.properties.get_property('terrainTextureStrings'))
        if base_tts != gt_tts:
            rows.append(("WorldInfo", "terrainTextureStrings", base_tts[:100] if len(base_tts) > 100 else base_tts, gt_tts[:100] if len(gt_tts) > 100 else gt_tts))
    
    # 13. Asset sizes
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
                rows.append(("Asset Sizes", name, f"Base: {base_size} bytes", f"GT: {gt_size} bytes (diff: {gt_size-base_size:+d})"))
    
    # Output as markdown table
    print("| Category | Property | Base Value | GT Value |")
    print("|----------|----------|------------|----------|")
    for category, property_name, base_val, gt_val in rows:
        # Escape pipe characters in values
        base_val = str(base_val).replace("|", "\\|")
        gt_val = str(gt_val).replace("|", "\\|")
        property_name = str(property_name).replace("|", "\\|")
        category = str(category).replace("|", "\\|")
        print(f"| {category} | {property_name} | {base_val} | {gt_val} |")

if __name__ == "__main__":
    diff_maps()
