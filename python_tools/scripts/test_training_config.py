"""
Test script to verify training config works correctly
Parses map with training config and visualizes to ensure nothing is missed
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.core.ra3map import Ra3Map
from map_processor.parsing.parser_config import ParserConfig
from map_processor.utils.map_visualizer import MapVisualizer

def main():
    # Use training config
    config = ParserConfig.training_config()
    
    print("=" * 70)
    print("Testing Training Config")
    print("=" * 70)
    print(f"Included assets: {len(config.included_assets)}")
    print(f"Excluded assets: {len(config.excluded_assets) if config.excluded_assets else 0}")
    print(f"Included object categories: {config.included_object_categories}")
    print()
    
    # Parse map
    map_file = Path(__file__).parent.parent.parent / 'RA3 Official maps' / '2 FI' / 'map_mp_2_feasel8.map'
    
    if not map_file.exists():
        print(f"ERROR: Map file not found: {map_file}")
        return
    
    print(f"Parsing map: {map_file.name}")
    print("-" * 70)
    
    ra3map = Ra3Map(str(map_file), config=config)
    ra3map.parse()
    context = ra3map.get_context()
    
    print(f"Map parsed: {context.map_name}")
    print(f"Dimensions: {context.map_width} x {context.map_height} (border: {context.border})")
    print()
    
    # Check assets
    print("Assets in map:")
    print("-" * 70)
    included_count = 0
    excluded_count = 0
    
    for asset in context.map_struct.assets:
        asset_name = asset.get_asset_name()
        if config.is_asset_included(asset_name):
            included_count += 1
            print(f"  [INCLUDED] {asset_name}")
        else:
            excluded_count += 1
            print(f"  [EXCLUDED] {asset_name} (excluded)")
    
    if context.excluded_assets:
        print(f"\nExcluded assets (read but not stored): {len(context.excluded_assets)}")
        for asset_name in sorted(context.excluded_assets.keys()):
            print(f"  [EMPTY] {asset_name} (stored as empty)")
    
    print()
    print(f"Total included assets: {included_count}")
    print(f"Total excluded assets: {excluded_count + len(context.excluded_assets)}")
    print()
    
    # Check objects
    objects_list = context.get_asset('ObjectsList')
    if objects_list:
        print("Objects:")
        print("-" * 70)
        print(f"Total objects in file: {len(objects_list.map_objects)}")
        
        # Count by category
        from map_processor.utils.object_categories import ObjectCategoryConfig
        category_config = ObjectCategoryConfig()
        counts = {}
        uncategorized = []
        
        for obj in objects_list.map_objects:
            category, should_draw = category_config.get_category_for_object(obj.type_name)
            if category:
                counts[category.name] = counts.get(category.name, 0) + 1
            elif should_draw:
                uncategorized.append(obj.type_name)
        
        print("\nObjects by category (included):")
        for cat, count in sorted(counts.items()):
            cat_key = cat.lower().replace(' ', '_')
            status = "[INCLUDED]" if cat_key in config.included_object_categories else "[EXCLUDED]"
            print(f"  {status} {cat}: {count}")
        
        if uncategorized:
            print(f"\nUncategorized objects (should be excluded): {len(set(uncategorized))}")
            for obj_type in sorted(set(uncategorized))[:10]:  # Show first 10
                print(f"    - {obj_type}")
            if len(set(uncategorized)) > 10:
                print(f"    ... and {len(set(uncategorized)) - 10} more")
    
    print()
    print("=" * 70)
    print("Generating visualization...")
    print("=" * 70)
    
    # Visualize
    output_dir = Path(__file__).parent.parent / 'test_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = MapVisualizer.visualize_map(
        context, 
        str(output_dir), 
        'map_mp_2_feasel8_training'
    )
    
    print()
    print("=" * 70)
    print("Visualization complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 70)
    
    # Verify critical assets are present
    print("\nVerification:")
    print("-" * 70)
    critical_assets = [
        'HeightMapData', 'BlendTileData', 'ObjectsList', 
        'WorldInfo', 'StandingWaterAreas', 'RiverAreas'
    ]
    
    all_present = True
    for asset_name in critical_assets:
        asset = context.get_asset(asset_name)
        if asset:
            print(f"  [OK] {asset_name}")
        else:
            print(f"  [MISSING] {asset_name} - MISSING!")
            all_present = False
    
    if all_present:
        print("\n[SUCCESS] All critical assets are present!")
    else:
        print("\n[ERROR] Some critical assets are missing!")

if __name__ == '__main__':
    main()

