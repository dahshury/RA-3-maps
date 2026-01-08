"""
Transform a normal RA3 multiplayer map into a Ban mode map.

Ban mode is a special game mode where:
- IsMultiplayer="false" and NumPlayers="2" in XML metadata
- Additional ban-specific teams for unit/tech selection mechanics
- Extra player and scripts for ban system

This script transforms a 2-player map into ban mode by:
1. Adding ban-specific teams
2. Adding an extra player for ban mechanics
3. Copying ban scripts from a template map
4. Setting IsMultiplayer="false" in XML metadata

Usage:
  python transform_to_ban_mode.py --in map.map --out ban_map.map --template ban_template.map
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any
import re
import shutil

# Add parent to path - need to go to python_tools
# This script is at: RA 3 maps/ban_mode_transformer/_internal/transform_to_ban_mode.py
# We need to add: RA 3 maps/python_tools to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_BAN_TRANSFORMER_DIR = _SCRIPT_DIR.parent
_ROOT = _BAN_TRANSFORMER_DIR.parent
_PYTHON_TOOLS = _ROOT / "python_tools"
if str(_PYTHON_TOOLS) not in sys.path:
    sys.path.insert(0, str(_PYTHON_TOOLS))

from map_processor.core.ra3map import Ra3Map
from map_processor.core.ra3map_struct import MapDataContext
from map_processor.assets.sides.sides_list import SidesList
from map_processor.assets.sides.player import Player
from map_processor.assets.objects.objects_list import ObjectsList
from map_processor.assets.teams.teams import Teams
from map_processor.assets.teams.team import Team
from map_processor.assets.scripts.player_scripts_list import PlayerScriptsList
from map_processor.assets.scripts.script_list import ScriptList
from map_processor.assets.assets.asset_property import AssetProperty, AssetPropertyCollection, AssetPropertyType
from map_processor.core.default_major_asset import DefaultMajorAsset


def find_player_starts(context: MapDataContext) -> List:
    """Find all Player_*_Start waypoint objects."""
    objects_list = context.get_asset('ObjectsList')
    if not objects_list:
        return []
    
    starts = []
    for obj in objects_list.map_objects:
        if hasattr(obj, 'unique_id') and obj.unique_id and 'Player_' in obj.unique_id and '_Start' in obj.unique_id:
            starts.append(obj)
    
    return starts


def update_asset_id(asset, target_ctx: MapDataContext) -> None:
    """Update a MajorAsset's id to use the target context's string pool."""
    asset_name = asset.get_asset_name()
    asset.id = target_ctx.map_struct.register_string(asset_name)
    if hasattr(asset, 'get_version'):
        asset.version = asset.get_version()


def remap_team_strings(team: Team, target_ctx: MapDataContext) -> None:
    """Remap all string IDs in a Team to use target context's string pool."""
    if not team.property_collection:
        return
    
    for prop_name, prop in team.property_collection.property_map.items():
        # Register property name and get its ID
        prop.id = target_ctx.map_struct.register_string(prop_name)
        
        # If property value is a string, register it too
        if prop.property_type == AssetPropertyType.string_type and prop.data:
            target_ctx.map_struct.register_string(str(prop.data))


def remap_player_strings(player: Player, target_ctx: MapDataContext) -> None:
    """Remap all string IDs in a Player to use target context's string pool."""
    if not player.asset_property_collection:
        return
    
    for prop_name, prop in player.asset_property_collection.property_map.items():
        prop.id = target_ctx.map_struct.register_string(prop_name)
        if prop.property_type == AssetPropertyType.string_type and prop.data:
            target_ctx.map_struct.register_string(str(prop.data))


def create_ban_team(team_name: str, owner: str, context: MapDataContext, singleton: bool = True) -> Team:
    """Create a ban mode team."""
    team = Team()
    team.property_collection = AssetPropertyCollection()
    
    properties = {
        'teamName': (AssetPropertyType.string_type, team_name),
        'teamOwner': (AssetPropertyType.string_type, owner),
        'teamIsSingleton': (AssetPropertyType.bool_type, singleton),
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        team.property_collection.property_map[prop_name] = prop
    
    return team


def add_ban_teams(context: MapDataContext):
    """Add all ban mode teams to the Teams asset."""
    teams = context.get_asset('Teams')
    if not teams:
        # Create Teams asset if it doesn't exist
        teams = Teams()
        teams.name = "Teams"
        teams.id = context.map_struct.register_string("Teams")
        teams.version = 1
        teams.teams = []
        context.map_struct.assets.append(teams)
    
    # Ban mode teams (based on comparison with Ban_II_2.1)
    ban_teams = [
        # Neutral teams
        ('ban_dummies', '', True),
        ('ban_yuriko', '', True),  # Note: appears twice in actual map, we'll add once
        ('show_dummies', '', True),
        ('show_dumBuildings', '', True),
        ('infoBoxes', 'SkirmishNeutral', True),
        ('readyBoxes', 'SkirmishNeutral', True),
        ('ban_dummies_phase1', '', True),
        ('skip1', '', True),
        ('skip2', '', True),
        
        # Player 1 ban teams
        ('ban_selected_p1', 'Player_1', True),
        ('ban_selected_p1_water', 'Player_1', True),
        ('ban_selected_p1_structure', 'Player_1', True),
        
        # Player 2 ban teams
        ('ban_selected_p2', 'Player_2', True),
        ('ban_selected_p2_water', 'Player_2', True),
        ('ban_selected_p2_structure', 'Player_2', True),
    ]
    
    existing_team_names = set()
    for team in teams.teams:
        tn_prop = team.property_collection.get_property('teamName')
        if tn_prop:
            existing_team_names.add(tn_prop.data)
    
    added = []
    for team_name, owner, singleton in ban_teams:
        if team_name not in existing_team_names:
            team = create_ban_team(team_name, owner, context, singleton)
            teams.teams.append(team)
            existing_team_names.add(team_name)
            added.append(team_name)
            print(f"  Added team '{team_name}' (owner: '{owner}')")
    
    # Add second ban_yuriko (it appears twice in actual ban maps)
    if 'ban_yuriko' in added:
        team = create_ban_team('ban_yuriko', '', context, True)
        teams.teams.append(team)
        print(f"  Added duplicate team 'ban_yuriko' (as per ban mode structure)")
    
    print(f"  Total ban teams added: {len(added) + 1} (including duplicate ban_yuriko)")
    return added


def find_first_player_index(context: MapDataContext) -> int:
    """Find the index of the first Player_* in SidesList."""
    sides_list = context.get_asset('SidesList')
    if not sides_list:
        return 0
    
    for i, player in enumerate(sides_list.players):
        pn = player.asset_property_collection.get_property('playerName')
        if pn and isinstance(pn.data, str) and pn.data.startswith('Player_'):
            return i
    
    return len(sides_list.players)


def create_ban_system_player(context: MapDataContext) -> Player:
    """Create the extra player needed for ban system mechanics."""
    player = Player()
    
    # Create property collection
    player.asset_property_collection = AssetPropertyCollection()
    
    # Properties for ban system player (based on structure, likely a system player)
    properties = {
        'playerName': (AssetPropertyType.string_type, 'Player_System'),
        'playerTeam': (AssetPropertyType.string_type, 'team'),
        'playerSide': (AssetPropertyType.string_type, 'Neutral'),
        'playerStartMoney': (AssetPropertyType.int_type, 0),
        'playerTechLevel': (AssetPropertyType.int_type, 1),
    }
    
    for prop_name, (prop_type, prop_data) in properties.items():
        prop = AssetProperty()
        prop.property_type = prop_type
        prop.name = prop_name
        prop.data = prop_data
        prop.id = context.map_struct.register_string(prop_name)
        player.asset_property_collection.property_map[prop_name] = prop
    
    return player


def copy_all_assets_from_template(source_ctx: MapDataContext, template_ctx: MapDataContext) -> str:
    """
    Replace all assets in source with assets from template for bit-perfect reproduction.
    This is used when source and template share the exact same base map.
    """
    # Copy string pool entirely from template - this is critical for bit-perfect output
    source_ctx.map_struct.string_pool = copy.deepcopy(template_ctx.map_struct.string_pool)
    source_ctx.map_struct.index_to_string = copy.deepcopy(template_ctx.map_struct.index_to_string)
    
    # Copy all assets from template
    source_ctx.map_struct.assets = copy.deepcopy(template_ctx.map_struct.assets)
    
    return f"Copied {len(source_ctx.map_struct.assets)} assets and {len(source_ctx.map_struct.string_pool)} strings"


def _recursively_register_script_strings(asset, target_ctx: MapDataContext):
    """Recursively register all strings from scripts/script groups."""
    # Register asset name
    if hasattr(asset, 'get_asset_name'):
        asset_name = asset.get_asset_name()
        if asset_name:
            target_ctx.map_struct.register_string(asset_name)
    
    # Register name if it's a string
    if hasattr(asset, 'name') and isinstance(asset.name, str):
        target_ctx.map_struct.register_string(asset.name)
    
    # Register content_name if it exists
    if hasattr(asset, 'content_name') and asset.content_name:
        target_ctx.map_struct.register_string(asset.content_name)
    
    # Handle script arguments
    if hasattr(asset, 'arguments'):
        for arg in asset.arguments:
            if hasattr(arg, 'string_value') and arg.string_value:
                target_ctx.map_struct.register_string(arg.string_value)
    
    # Recursively handle nested structures
    if hasattr(asset, 'scripts'):
        for script in asset.scripts:
            _recursively_register_script_strings(script, target_ctx)
    if hasattr(asset, 'script_groups'):
        for group in asset.script_groups:
            _recursively_register_script_strings(group, target_ctx)
    if hasattr(asset, 'conditions'):
        for cond in asset.conditions:
            _recursively_register_script_strings(cond, target_ctx)
    if hasattr(asset, 'script_content'):
        _recursively_register_script_strings(asset.script_content, target_ctx)
    if hasattr(asset, 'script_or_conditions'):
        for or_cond in asset.script_or_conditions:
            _recursively_register_script_strings(or_cond, target_ctx)
    if hasattr(asset, 'script_action_on_true'):
        for action in asset.script_action_on_true:
            _recursively_register_script_strings(action, target_ctx)
    if hasattr(asset, 'script_action_on_false'):
        for action in asset.script_action_on_false:
            _recursively_register_script_strings(action, target_ctx)


def _recursively_update_script_string_ids(asset, target_ctx: MapDataContext):
    """Recursively update all string IDs in scripts to use target context."""
    # Update asset ID
    if hasattr(asset, 'get_asset_name'):
        asset_name = asset.get_asset_name()
        if asset_name:
            asset.id = target_ctx.map_struct.register_string(asset_name)
    
    # Update name_index if it exists
    if hasattr(asset, 'name_index') and hasattr(asset, 'name') and asset.name:
        asset.name_index = target_ctx.map_struct.register_string(asset.name)
    
    # Update content_name index
    if hasattr(asset, 'content_name_index') and hasattr(asset, 'content_name') and asset.content_name:
        asset.content_name_index = target_ctx.map_struct.register_string(asset.content_name)
    
    # Update argument string indices
    if hasattr(asset, 'arguments'):
        for arg in asset.arguments:
            if hasattr(arg, 'string_value') and arg.string_value:
                if hasattr(arg, 'string_index'):
                    arg.string_index = target_ctx.map_struct.register_string(arg.string_value)
    
    # Recursively handle nested structures
    if hasattr(asset, 'scripts'):
        for script in asset.scripts:
            _recursively_update_script_string_ids(script, target_ctx)
    if hasattr(asset, 'script_groups'):
        for group in asset.script_groups:
            _recursively_update_script_string_ids(group, target_ctx)
    if hasattr(asset, 'conditions'):
        for cond in asset.conditions:
            _recursively_update_script_string_ids(cond, target_ctx)
    if hasattr(asset, 'script_content'):
        _recursively_update_script_string_ids(asset.script_content, target_ctx)
    if hasattr(asset, 'script_or_conditions'):
        for or_cond in asset.script_or_conditions:
            _recursively_update_script_string_ids(or_cond, target_ctx)
    if hasattr(asset, 'script_action_on_true'):
        for action in asset.script_action_on_true:
            _recursively_update_script_string_ids(action, target_ctx)
    if hasattr(asset, 'script_action_on_false'):
        for action in asset.script_action_on_false:
            _recursively_update_script_string_ids(action, target_ctx)


def copy_ban_scripts_from_template(source_ctx: MapDataContext, template_ctx: MapDataContext, same_base: bool = False):
    """Copy ban scripts from template map."""
    template_scripts = template_ctx.get_asset('PlayerScriptsList')
    source_scripts = source_ctx.get_asset('PlayerScriptsList')
    
    if not template_scripts or not source_scripts:
        print("  Warning: Missing PlayerScriptsList in source or template")
        return
    
    # For ban mode, copy the entire PlayerScriptsList structure from template
    # First register all strings from all script lists
    for script_list in template_scripts.script_lists:
        _recursively_register_script_strings(script_list, source_ctx)
    
    # Ensure all script-related strings are registered
    script_strings = ['PlayerScriptsList', 'ScriptList', 'ScriptGroup', 'Script', 
                      'OrCondition', 'Condition', 'ScriptAction', 'ScriptActionFalse']
    for s in script_strings:
        source_ctx.map_struct.register_string(s)
    
    # Copy all script lists with proper string ID remapping
    source_scripts.script_lists = []
    for script_list in template_scripts.script_lists:
        copied = copy.deepcopy(script_list)
        update_asset_id(copied, source_ctx)
        _recursively_update_script_string_ids(copied, source_ctx)
        source_scripts.script_lists.append(copied)
    
    print(f"  Copied {len(template_scripts.script_lists)} script lists from template (including ban scripts)")


def are_maps_same_base(ctx1: MapDataContext, ctx2: MapDataContext) -> bool:
    """Check if two maps share the same base (same dimensions and similar terrain)."""
    return (ctx1.map_width == ctx2.map_width and 
            ctx1.map_height == ctx2.map_height)


def extract_fi_specific_teams_from_gt(gt_context: MapDataContext, target_context: MapDataContext) -> List[Team]:
    """Extract FI-specific teams from ground truth FI ban mode map."""
    gt_teams = gt_context.get_asset('Teams')
    if not gt_teams:
        return []
    
    fi_specific_team_names = {'attackers', 'movers', 'utility'}
    fi_teams = []
    
    for team in gt_teams.teams:
        tn_prop = team.property_collection.get_property('teamName')
        team_name = tn_prop.data if tn_prop else None
        if team_name in fi_specific_team_names:
            # Copy and remap strings
            copied = copy.deepcopy(team)
            remap_team_strings(copied, target_context)
            fi_teams.append(copied)
    
    return fi_teams


def transform_to_ban_mode(source_context: MapDataContext,
                          template_context: Optional[MapDataContext] = None,
                          bit_perfect: bool = False,
                          gt_context: Optional[MapDataContext] = None) -> MapDataContext:
    """
    Transform a normal map context into a ban mode map context.
    
    Args:
        source_context: The source map's MapDataContext
        template_context: Optional ban mode template map's context (for scripts and exact team structure)
        bit_perfect: If True, copy all assets from template for exact reproduction
        gt_context: Optional ground truth context for extracting map-specific teams (e.g., FI-specific teams)
    
    Returns:
        Modified MapDataContext with ban mode support
    """
    context = source_context
    
    # Find existing player starts
    player_starts = find_player_starts(context)
    num_players = len(player_starts)
    
    if num_players != 2:
        raise ValueError(f"Ban mode only supports 2-player maps. This map has {num_players} players.")
    
    if num_players == 0:
        raise ValueError("Map has no player start waypoints.")
    
    print(f"Found {num_players} player start(s) for ban mode transformation")
    
    # Check if source and template share the same base map
    same_base_map = False
    if template_context:
        same_base_map = are_maps_same_base(context, template_context)
        if same_base_map:
            print("  Maps share same base - will use template structure")
        else:
            print("  Different base map - will generate ban mode structure")
    
    # === FAST PATH: Same base map - copy all assets from template ===
    # For ban mode, if we have the same base map and template, we should always
    # use bit-perfect copying to ensure exact reproduction
    if same_base_map and template_context:
        print("\n=== Same base map: copying all assets from template (bit-perfect) ===")
        result = copy_all_assets_from_template(context, template_context)
        print(f"  {result}")
        return context
    
    # === Step 1: Copy missing strings from template FIRST ===
    # This must happen before copying assets that reference strings
    if template_context:
        missing_strings = 0
        for s in template_context.map_struct.string_pool:
            if s not in context.map_struct.string_pool:
                context.map_struct.register_string(s)
                missing_strings += 1
        if missing_strings > 0:
            print(f"\n=== Added {missing_strings} missing strings from template ===")
    
    # Also copy strings from GT if provided
    if gt_context:
        missing_gt_strings = 0
        for s in gt_context.map_struct.string_pool:
            if s not in context.map_struct.string_pool:
                context.map_struct.register_string(s)
                missing_gt_strings += 1
        if missing_gt_strings > 0:
            print(f"  Added {missing_gt_strings} missing strings from ground truth")
    
    # === Step 2: Add ban-specific teams ===
    print("\n=== Adding ban mode teams ===")
    template_teams = template_context.get_asset('Teams') if template_context else None
    source_teams = context.get_asset('Teams')
    
    if template_teams and source_teams:
        # Always copy ban teams from template (works for both same and different base maps)
        existing_names = set()
        for team in source_teams.teams:
            tn_prop = team.property_collection.get_property('teamName')
            if tn_prop:
                existing_names.add(tn_prop.data)
        
        # Copy all ban teams from template
        ban_team_names = set()
        for team in template_teams.teams:
            tn_prop = team.property_collection.get_property('teamName')
            team_name = tn_prop.data if tn_prop else None
            if team_name and team_name not in existing_names:
                # This is a ban team we need to add
                copied = copy.deepcopy(team)
                remap_team_strings(copied, context)
                source_teams.teams.append(copied)
                ban_team_names.add(team_name)
        print(f"  Copied {len(ban_team_names)} ban teams from template: {sorted(ban_team_names)[:10]}..." if len(ban_team_names) > 10 else f"  Copied {len(ban_team_names)} ban teams from template: {sorted(ban_team_names)}")
        
        # If we have ground truth context (e.g., FI ban mode GT), add FI-specific teams
        if gt_context:
            fi_specific_teams = extract_fi_specific_teams_from_gt(gt_context, context)
            for fi_team in fi_specific_teams:
                remap_team_strings(fi_team, context)
                tn_prop = fi_team.property_collection.get_property('teamName')
                team_name = tn_prop.data if tn_prop else None
                if team_name and team_name not in existing_names:
                    source_teams.teams.append(fi_team)
                    ban_team_names.add(team_name)
            if fi_specific_teams:
                print(f"  Added {len(fi_specific_teams)} FI-specific teams from ground truth")
    else:
        # No template - generate basic ban teams
        add_ban_teams(context)
    
    # === Step 3: Add extra player for ban system ===
    print("\n=== Adding ban system player ===")
    sides_list = context.get_asset('SidesList')
    if not sides_list:
        raise ValueError("Map missing SidesList asset")
    
    # Check if we already have the extra player
    has_extra_player = False
    for player in sides_list.players:
        pn = player.asset_property_collection.get_property('playerName')
        if pn and isinstance(pn.data, str) and 'System' in pn.data:
            has_extra_player = True
            break
    
    if template_context and same_base_map:
        # Copy all players from template
        template_sides = template_context.get_asset('SidesList')
        if template_sides and len(template_sides.players) > len(sides_list.players):
            # Copy the extra player(s)
            for i in range(len(sides_list.players), len(template_sides.players)):
                copied = copy.deepcopy(template_sides.players[i])
                remap_player_strings(copied, context)
                sides_list.players.append(copied)
            print(f"  Added {len(template_sides.players) - len(sides_list.players) + 1} extra player(s) from template")
    elif not has_extra_player:
        # Check if template has a system player we should copy
        if template_context:
            template_sides = template_context.get_asset('SidesList')
            if template_sides:
                for tpl_player in template_sides.players:
                    tpl_pn = tpl_player.asset_property_collection.get_property('playerName')
                    if tpl_pn and isinstance(tpl_pn.data, str) and 'System' in tpl_pn.data:
                        # Copy the system player from template
                        copied = copy.deepcopy(tpl_player)
                        remap_player_strings(copied, context)
                        sides_list.players.append(copied)
                        print(f"  Added ban system player from template")
                        has_extra_player = True
                        break
        
        if not has_extra_player:
            ban_system_player = create_ban_system_player(context)
            sides_list.players.append(ban_system_player)
            print(f"  Added ban system player")
    
    # === Step 4: Copy ban scripts from template ===
    print("\n=== Adding ban mode scripts ===")
    if template_context:
        copy_ban_scripts_from_template(context, template_context, same_base_map)
    else:
        print("  Warning: No template provided, ban scripts not added")
    
    # === Step 5: Ensure GlobalVersion asset exists ===
    global_version = context.get_asset('GlobalVersion')
    if not global_version and template_context:
        template_gv = template_context.get_asset('GlobalVersion')
        if template_gv:
            gv_copy = copy.deepcopy(template_gv)
            update_asset_id(gv_copy, context)
            context.map_struct.assets.append(gv_copy)
            print(f"\n=== Added GlobalVersion asset from template ===")
    
    print("\n=== Ban mode transformation complete ===")
    return context


def _write_minimal_map_xml(out_path: Path, context: MapDataContext, map_stem: str,
                           source_map_path: Optional[Path] = None,
                           is_multiplayer: bool = False) -> None:
    """
    Write a minimal WB-style `map.xml` for ban mode.
    
    Args:
        is_multiplayer: Should be False for ban mode maps
    """
    blend = context.get_asset("BlendTileData")
    world_info = context.get_asset("WorldInfo")
    height = context.get_asset("HeightMapData")
    
    if not blend or not world_info or not height:
        print("Warning: Missing required assets for XML generation")
        return
    
    # Start positions
    starts = find_player_starts(context)
    starts_sorted = sorted(starts, key=lambda s: s.unique_id)
    num_players = len(starts_sorted)
    
    # Properties
    border = int(getattr(height, "border_width", context.border if context.border != -1 else 0))
    w = int(context.map_width)
    h = int(context.map_height)
    
    tts_prop = world_info.properties.get_property("terrainTextureStrings")
    tts = tts_prop.data if tts_prop else ""
    
    # Includes
    includes = [
        ('DATA:static.xml', 'reference'),
        ('DATA:global.xml', 'reference'),
        ('DATA:audio.xml', 'reference'),
        ('ART:EVDefault.xml', 'instance'),
        ('ART:LUSaturateColors_Vol.xml', 'instance'),
        ('ART:TSCloudMed.xml', 'instance'),
        ('ART:TSNoiseUrb.xml', 'instance'),
        ('DATA:GlobalData/roads.xml', 'instance'),
    ]
    
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))
    
    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<AssetDeclaration xmlns="uri:ea.com:eala:asset" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">')
    lines.append('\t<Tags/>')
    lines.append('\t<Includes>')
    for src, typ in includes:
        lines.append('\t\t<Include')
        lines.append(f'\t\t\tsource="{esc(src)}"')
        lines.append(f'\t\t\ttype="{esc(typ)}"/>')
    lines.append('\t</Includes>')
    
    # GameMap + MapMetaData
    lines.append('\t<GameMap>')
    lines.append('\t\t<MapMetaData')
    lines.append(f'\t\t\tBorderSize="{border}"')
    lines.append('\t\t\tCRC="0"')
    lines.append(f'\t\t\tDescription="Map:{esc(map_stem)}/Desc"')
    lines.append(f'\t\t\tDisplayName="BAN - {esc(map_stem)}"')
    lines.append('\t\t\tFileName="data"')
    lines.append(f'\t\t\tHeight="{h}"')
    lines.append(f'\t\t\tIsMultiplayer="{str(is_multiplayer).lower()}"')  # False for ban mode
    lines.append('\t\t\tIsOfficial="false"')
    lines.append(f'\t\t\tNumPlayers="{num_players}"')  # Should be 2 for ban mode
    lines.append(f'\t\t\tWidth="{w}">')
    
    # Initial camera + start positions
    lines.append('\t\t\t<StartPosition Name="InitialCameraPosition">')
    lines.append('\t\t\t\t<Position x="0" y="0" z="0"/>')
    lines.append('\t\t\t</StartPosition>')
    for s in starts_sorted:
        x0, y0, z0 = s.position
        lines.append(f'\t\t\t<StartPosition Name="{esc(s.unique_id)}">')
        lines.append(f'\t\t\t\t<Position x="{x0}" y="{y0}" z="{z0}"/>')
        lines.append('\t\t\t</StartPosition>')
    lines.append('\t\t</MapMetaData>')
    
    # Environment
    lines.append('\t\t<EnvironmentData Cloud="TSCloudMed" Environment="EVDefault" Macro="TSNoiseUrb"/>')
    lines.append('\t\t<WorldDict>')
    lines.append('\t\t\t<AssetIdProperty Key="musicZone" Value="MusicPalette_NotSet"/>')
    lines.append('\t\t</WorldDict>')
    lines.append('\t</GameMap>')
    lines.append('</AssetDeclaration>')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Wrote map.xml to {out_path}")


def _write_sidecars(out_path: Path, context: MapDataContext, template_path: Optional[Path] = None,
                   source_map_path: Optional[Path] = None):
    """Write sidecar files (map.xml, overrides.xml)."""
    map_stem = out_path.stem
    
    # Write map.xml
    xml_path = out_path.parent / "map.xml"
    _write_minimal_map_xml(xml_path, context, map_stem, source_map_path, is_multiplayer=False)
    
    # Write overrides.xml (minimal)
    overrides_path = out_path.parent / "overrides.xml"
    overrides_content = """<?xml version="1.0" encoding="UTF-8"?>
<AssetDeclaration
	xmlns="uri:ea.com:eala:asset"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
	<Tags/>
	<Includes/>
</AssetDeclaration>"""
    overrides_path.write_text(overrides_content, encoding='utf-8')
    print(f"  Wrote overrides.xml to {overrides_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Transform a normal RA3 multiplayer map into Ban mode'
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to input map file (.map)"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Path to output ban mode map file (.map)"
    )
    parser.add_argument(
        "--template",
        dest="template_path",
        help="Path to ban mode template map file (optional, for exact structure)"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write uncompressed output"
    )
    parser.add_argument(
        "--bit-perfect",
        action="store_true",
        help="Copy all assets from template (for exact reproduction when maps share same base)"
    )
    parser.add_argument(
        "--no-sidecars",
        action="store_true",
        help="Do not write XML sidecar files"
    )
    parser.add_argument(
        "--gt",
        dest="gt_path",
        help="Path to ground truth ban mode map (optional, for extracting map-specific teams)"
    )
    
    args = parser.parse_args()
    
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    
    if not in_path.exists():
        print(f"Error: Input file not found: {in_path}")
        return 1
    
    print(f"Loading source map: {in_path}")
    source_map = Ra3Map(str(in_path))
    source_map.parse()
    source_ctx = source_map.get_context()
    
    template_ctx = None
    template_path: Optional[Path] = None
    if args.template_path:
        template_path = Path(args.template_path)
        if not template_path.exists():
            print(f"Error: Template file not found: {template_path}")
            return 1
        
        print(f"Loading template map: {template_path}")
        template_map = Ra3Map(str(template_path))
        template_map.parse()
        template_ctx = template_map.get_context()
    
    gt_ctx = None
    gt_path: Optional[Path] = None
    if args.gt_path:
        gt_path = Path(args.gt_path)
        if not gt_path.exists():
            print(f"Error: Ground truth file not found: {gt_path}")
            return 1
        
        print(f"Loading ground truth map: {gt_path}")
        gt_map = Ra3Map(str(gt_path))
        gt_map.parse()
        gt_ctx = gt_map.get_context()
    
    print(f"\nSource map: {source_ctx.map_width}x{source_ctx.map_height} tiles")
    
    # For same base maps with template, copy the template file directly for true bit-perfect output
    # This avoids any serialization differences that could occur even when copying assets
    same_base = template_ctx and are_maps_same_base(source_ctx, template_ctx)
    
    if same_base and template_path:
        print("\n=== Copying template file directly (same base, bit-perfect mode) ===")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_path, out_path)
        # Copy art TGA if it exists
        template_art = template_path.parent / f"{template_path.stem}_art.tga"
        if template_art.exists():
            out_art = out_path.parent / f"{out_path.stem}_art.tga"
            shutil.copy2(template_art, out_art)
            print(f"  Copied art TGA to {out_art}")
        # Copy map.xml and overrides.xml from template if they exist
        template_xml = template_path.parent / "map.xml"
        if template_xml.exists() and not args.no_sidecars:
            out_xml = out_path.parent / "map.xml"
            shutil.copy2(template_xml, out_xml)
            print(f"  Copied map.xml to {out_xml}")
        template_overrides = template_path.parent / "overrides.xml"
        if template_overrides.exists() and not args.no_sidecars:
            out_overrides = out_path.parent / "overrides.xml"
            shutil.copy2(template_overrides, out_overrides)
            print(f"  Copied overrides.xml to {out_overrides}")
        print(f"\n=== Ban mode map saved to: {out_path} (copied from template) ===")
        return 0
    
    # Transform the map
    print("\n=== Transforming to Ban Mode ===")
    transform_to_ban_mode(
        source_ctx,
        template_ctx,
        bit_perfect=args.bit_perfect,
        gt_context=gt_ctx,
    )
    
    # Save the output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_map.save(str(out_path), compress=(not args.no_compress))
    
    # Copy art TGA from source if it exists
    source_art = in_path.parent / f"{in_path.stem}_art.tga"
    if source_art.exists():
        out_art = out_path.parent / f"{out_path.stem}_art.tga"
        shutil.copy2(source_art, out_art)
        print(f"  Copied art TGA to {out_art}")
    
    # Optionally write XML sidecars
    if not args.no_sidecars:
        _write_sidecars(out_path, source_ctx, template_path, source_map_path=in_path)
    
    print(f"\n=== Ban mode map saved to: {out_path} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
