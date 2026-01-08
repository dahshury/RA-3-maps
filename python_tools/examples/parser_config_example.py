"""
Example: Using ParserConfig to filter assets and objects for AI training
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map
from map_processor.parsing.parser_config import ParserConfig

# Example 1: Use training config (includes only training assets)
print("Example 1: Using training config")
print("=" * 60)

training_config = ParserConfig.training_config()
map_file = Path(__file__).parent.parent.parent / 'RA3 Official maps' / '2 II' / 'map_mp_2_rao1.map'

if map_file.exists():
    ra3map = Ra3Map(str(map_file), config=training_config)
    ra3map.parse()
    context = ra3map.get_context()
    
    # Check what assets were included
    print(f"Parsed map: {context.map_name}")
    print(f"Assets in map: {len(context.map_struct.assets)}")
    print(f"Excluded assets: {len(context.excluded_assets)}")
    
    # Check objects
    objects_list = context.get_asset('ObjectsList')
    if objects_list:
        print(f"Total objects: {len(objects_list.map_objects)}")
        
        # Count by category
        from map_processor.utils.object_categories import ObjectCategoryConfig
        category_config = ObjectCategoryConfig()
        counts = {}
        for obj in objects_list.map_objects:
            category, _ = category_config.get_category_for_object(obj.type_name)
            if category:
                counts[category.name] = counts.get(category.name, 0) + 1
        
        print("Objects by category:")
        for cat, count in sorted(counts.items()):
            print(f"  {cat}: {count}")
    
    # Save filtered map
    output_path = Path(__file__).parent.parent / 'test_output' / 'filtered_map.map'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ra3map.save(str(output_path), compress=False)
    print(f"\nSaved filtered map to: {output_path}")
else:
    print(f"Map file not found: {map_file}")

# Example 2: Custom config
print("\n\nExample 2: Custom config")
print("=" * 60)

from map_processor.utils.constants import (
    ASSET_HeightMapData, ASSET_BlendTileData, ASSET_ObjectsList
)

# Only include height map and blend tile data
custom_config = ParserConfig(
    included_assets={ASSET_HeightMapData, ASSET_BlendTileData, ASSET_ObjectsList},
    included_object_categories={'ore_node', 'oil_derrick'}  # Only ores and oil
)

print(f"Custom config created:")
print(f"  Included assets: {custom_config.included_assets}")
print(f"  Included object categories: {custom_config.included_object_categories}")

# Example 3: Exclude specific assets
print("\n\nExample 3: Exclude specific assets")
print("=" * 60)

from map_processor.utils.constants import (
    ASSET_PlayerScriptsList, ASSET_TriggerAreas, ASSET_MissionHotSpots
)

exclude_config = ParserConfig(
    excluded_assets={
        ASSET_PlayerScriptsList, 
        ASSET_TriggerAreas, 
        ASSET_MissionHotSpots
    }
)

print(f"Exclude config created:")
print(f"  Excluded assets: {exclude_config.excluded_assets}")











