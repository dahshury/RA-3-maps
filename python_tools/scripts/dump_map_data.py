"""
Parse a map and dump ALL relevant data to see what we're working with.
This helps evaluate if LLM-based generation is feasible.
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData
from map_processor.assets.terrain.blend_tile_data import BlendTileData
from map_processor.utils.object_categories import ObjectCategoryConfig


def dump_map_data(map_path: str):
    """Parse map and print all relevant data."""
    print(f"=" * 80)
    print(f"PARSING MAP: {map_path}")
    print(f"=" * 80)
    
    parser = Ra3MapParser()
    context = parser.parse(map_path)
    
    # Basic info
    print(f"\n{'='*40}")
    print("MAP METADATA")
    print(f"{'='*40}")
    print(f"  Map Width: {context.map_width}")
    print(f"  Map Height: {context.map_height}")
    print(f"  Border: {context.border}")
    print(f"  Map Name: {context.map_name}")
    print(f"  Total cells: {context.map_width * context.map_height:,}")
    
    # Height Map Data
    print(f"\n{'='*40}")
    print("HEIGHT MAP DATA")
    print(f"{'='*40}")
    height_data = context.get_asset_by_type(HeightMapData)
    if height_data and height_data.elevations is not None:
        elevations = np.array(height_data.elevations)
        print(f"  Shape: {elevations.shape}")
        print(f"  Min height: {np.min(elevations):.2f}")
        print(f"  Max height: {np.max(elevations):.2f}")
        print(f"  Mean height: {np.mean(elevations):.2f}")
        print(f"  Height range: {np.max(elevations) - np.min(elevations):.2f}")
        
        # Count unique heights (rounded)
        rounded = np.round(elevations, 0)
        unique_heights = np.unique(rounded)
        print(f"  Unique heights (rounded): {len(unique_heights)}")
        
        # Show histogram of heights
        print(f"\n  Height distribution:")
        hist, bins = np.histogram(elevations, bins=10)
        for i in range(len(hist)):
            pct = hist[i] / elevations.size * 100
            print(f"    {bins[i]:6.1f} - {bins[i+1]:6.1f}: {hist[i]:6,} cells ({pct:5.1f}%)")
    
    # Blend Tile Data
    print(f"\n{'='*40}")
    print("BLEND TILE DATA")
    print(f"{'='*40}")
    blend_data = context.get_asset('BlendTileData')
    if blend_data:
        # Tiles
        if hasattr(blend_data, 'tiles') and blend_data.tiles is not None:
            tiles = np.array(blend_data.tiles)
            print(f"\n  TILES:")
            print(f"    Shape: {tiles.shape}")
            unique_tiles = np.unique(tiles)
            print(f"    Unique tile IDs: {len(unique_tiles)}")
            print(f"    Tile ID range: {np.min(tiles)} - {np.max(tiles)}")
        
        # Textures
        if hasattr(blend_data, 'textures') and blend_data.textures:
            print(f"\n  TEXTURES ({len(blend_data.textures)} total):")
            for i, tex in enumerate(blend_data.textures):
                name = tex.name if hasattr(tex, 'name') else str(tex)
                print(f"    [{i}] {name}")
        
        # Passability
        if hasattr(blend_data, 'passability') and blend_data.passability is not None:
            pass_arr = np.array(blend_data.passability)
            print(f"\n  PASSABILITY:")
            print(f"    Shape: {pass_arr.shape}")
            unique_pass = np.unique(pass_arr)
            print(f"    Unique values: {unique_pass}")
            for val in unique_pass:
                count = np.sum(pass_arr == val)
                pct = count / pass_arr.size * 100
                print(f"    Value {val}: {count:,} cells ({pct:.1f}%)")
        
        # Impassable
        if hasattr(blend_data, 'impassable') and blend_data.impassable is not None:
            imp = np.array(blend_data.impassable)
            print(f"\n  IMPASSABLE:")
            print(f"    Shape: {imp.shape}")
            impassable_count = np.sum(imp)
            pct = impassable_count / imp.size * 100
            print(f"    Impassable cells: {impassable_count:,} ({pct:.2f}%)")
            print(f"    Passable cells: {imp.size - impassable_count:,} ({100-pct:.2f}%)")
        
        # Buildability
        if hasattr(blend_data, 'buildability') and blend_data.buildability is not None:
            build = np.array(blend_data.buildability)
            print(f"\n  BUILDABILITY:")
            print(f"    Shape: {build.shape}")
            buildable_count = np.sum(build)
            pct = buildable_count / build.size * 100
            print(f"    Buildable cells: {buildable_count:,} ({pct:.2f}%)")
            print(f"    Non-buildable cells: {build.size - buildable_count:,} ({100-pct:.2f}%)")
        
        # Visibility
        if hasattr(blend_data, 'visibility') and blend_data.visibility is not None:
            vis = np.array(blend_data.visibility)
            print(f"\n  VISIBILITY:")
            print(f"    Shape: {vis.shape}")
            visible_count = np.sum(vis)
            pct = visible_count / vis.size * 100
            print(f"    Visible cells: {visible_count:,} ({pct:.2f}%)")
    
    # Water Areas
    print(f"\n{'='*40}")
    print("WATER AREAS")
    print(f"{'='*40}")
    water_areas = context.get_asset('StandingWaterAreas')
    if water_areas and hasattr(water_areas, 'water_areas') and water_areas.water_areas:
        print(f"  Standing water areas: {len(water_areas.water_areas)}")
        for i, area in enumerate(water_areas.water_areas):
            if hasattr(area, 'points') and area.points:
                print(f"    Area {i}: {len(area.points)} points")
                # Print first few points
                for j, pt in enumerate(area.points[:5]):
                    print(f"      Point {j}: ({pt[0]:.1f}, {pt[1]:.1f})")
                if len(area.points) > 5:
                    print(f"      ... and {len(area.points) - 5} more points")
    else:
        print("  No standing water areas")
    
    river_areas = context.get_asset('RiverAreas')
    if river_areas and hasattr(river_areas, 'areas') and river_areas.areas:
        print(f"  River areas: {len(river_areas.areas)}")
    else:
        print("  No river areas")
    
    # Objects - FILTERED BY TRAINING CONFIG
    print(f"\n{'='*40}")
    print("OBJECTS (Training-relevant only)")
    print(f"{'='*40}")
    objects_list = context.get_asset('ObjectsList')
    
    # Categories we want
    category_config = ObjectCategoryConfig()
    training_categories = ['ore_node', 'oil_derrick', 'garrison_tikihut', 'garrison_house', 
                          'garrison_warehouse', 'garrison_other', 'building_observation_post',
                          'building_hospital', 'building_garage', 'building_snowy',
                          'building_convention_center', 'building_port_structure',
                          'building_airport', 'building_military', 'building_cargo_container',
                          'building_supply', 'building_veterancy', 'building_shipyard',
                          'building_tech_structure', 'building_soviet', 'building_other',
                          'player_start']
    
    if objects_list and objects_list.map_objects:
        print(f"  Total objects in map: {len(objects_list.map_objects)}")
        
        # Categorize and filter
        categorized = {}
        uncategorized = []
        player_starts = []
        
        player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start', 
                           'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
        
        for obj in objects_list.map_objects:
            # Check for player starts first
            is_player_start = False
            if hasattr(obj, 'unique_id') and obj.unique_id:
                if obj.unique_id in player_start_ids:
                    is_player_start = True
                    pos = obj.position
                    player_starts.append({
                        'unique_id': obj.unique_id,
                        'type': obj.type_name,
                        'x': pos[0] if hasattr(pos, '__getitem__') else pos.x,
                        'y': pos[1] if hasattr(pos, '__getitem__') else pos.y,
                        'z': pos[2] if hasattr(pos, '__getitem__') else pos.z,
                    })
            
            if not is_player_start:
                category, should_draw = category_config.get_category_for_object(obj.type_name)
                if should_draw and category:
                    cat_name = category.name
                    if cat_name not in categorized:
                        categorized[cat_name] = []
                    pos = obj.position
                    categorized[cat_name].append({
                        'type': obj.type_name,
                        'x': pos[0] if hasattr(pos, '__getitem__') else pos.x,
                        'y': pos[1] if hasattr(pos, '__getitem__') else pos.y,
                        'z': pos[2] if hasattr(pos, '__getitem__') else pos.z,
                        'angle': obj.angle if hasattr(obj, 'angle') else 0,
                    })
        
        # Print player starts
        if player_starts:
            print(f"\n  PLAYER STARTS ({len(player_starts)}):")
            for ps in player_starts:
                print(f"    {ps['unique_id']}: type={ps['type']}, pos=({ps['x']:.1f}, {ps['y']:.1f}, {ps['z']:.1f})")
        
        # Print categorized objects
        total_training_objects = len(player_starts)
        for cat_name, objs in sorted(categorized.items()):
            total_training_objects += len(objs)
            print(f"\n  {cat_name.upper()} ({len(objs)}):")
            for obj in objs[:10]:  # Show first 10
                print(f"    {obj['type']}: pos=({obj['x']:.1f}, {obj['y']:.1f}, {obj['z']:.1f}), angle={obj['angle']:.1f}")
            if len(objs) > 10:
                print(f"    ... and {len(objs) - 10} more")
        
        print(f"\n  SUMMARY:")
        print(f"    Total objects in map: {len(objects_list.map_objects)}")
        print(f"    Training-relevant objects: {total_training_objects}")
        print(f"    Filtered out (decorative/excluded): {len(objects_list.map_objects) - total_training_objects}")
    
    # MP Positions
    print(f"\n{'='*40}")
    print("MULTIPLAYER POSITIONS")
    print(f"{'='*40}")
    mp_positions = context.get_asset('MPPositionList')
    if mp_positions and hasattr(mp_positions, 'positions') and mp_positions.positions:
        print(f"  MP Positions: {len(mp_positions.positions)}")
        for i, pos in enumerate(mp_positions.positions):
            if hasattr(pos, 'position'):
                p = pos.position
                print(f"    Player {i+1}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
    
    # Summary
    print(f"\n{'='*80}")
    print("DATA SIZE SUMMARY FOR LLM GENERATION")
    print(f"{'='*80}")
    
    total_grid_cells = context.map_width * context.map_height
    print(f"\n  Grid dimensions: {context.map_width} x {context.map_height} = {total_grid_cells:,} cells")
    
    print(f"\n  If represented as raw arrays:")
    print(f"    Height map: {total_grid_cells:,} floats")
    print(f"    Tile map: {total_grid_cells:,} ushorts")
    print(f"    Passability: {total_grid_cells:,} bools")
    print(f"    Buildability: {total_grid_cells:,} bools")
    print(f"    = {total_grid_cells * 4:,} total values")
    
    print(f"\n  If represented as REGIONS:")
    print(f"    Height regions: ~5-20 polygons (estimate based on terrain complexity)")
    print(f"    Texture regions: ~{len(blend_data.textures) if blend_data and hasattr(blend_data, 'textures') else 'N/A'} texture types")
    print(f"    Water areas: {len(water_areas.water_areas) if water_areas and hasattr(water_areas, 'water_areas') and water_areas.water_areas else 0}")
    print(f"    Objects: {total_training_objects} (exact positions needed)")
    print(f"    = Likely ~2,000-5,000 tokens total")
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    print(f"""
  RAW ARRAY approach: {total_grid_cells * 4:,} values = ~{total_grid_cells * 4 // 500:,}K tokens
    → TOO LARGE for LLM output (output limit ~8-32K tokens)
  
  REGION-BASED approach: ~2,000-5,000 tokens
    → FEASIBLE for LLM generation!
    
  The region approach works because:
    - Height has only {len(unique_heights) if 'unique_heights' in dir() else 'few'} distinct levels (create height regions)
    - Passability is {pct if 'pct' in dir() else 'mostly'} sparse (only list impassable regions)
    - Objects are sparse ({total_training_objects} objects vs {total_grid_cells:,} cells)
    - Water is defined by polygons already
""")


if __name__ == "__main__":
    map_path = Path(__file__).parent.parent.parent / "RA3 Official maps" / "2 II" / "map_mp_2_rao1.map"
    dump_map_data(str(map_path))









