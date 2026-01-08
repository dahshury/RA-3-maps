"""
Complete analysis of ALL objects in maps - not just training-relevant ones.
This includes trees, ambient sounds, bridges, decorative objects, etc.
"""
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData


def analyze_all_objects(map_path: Path) -> dict:
    """Analyze ALL objects in a map, categorized by type."""
    parser = Ra3MapParser()
    context = parser.parse(str(map_path))
    
    result = {
        'map_name': map_path.stem,
        'map_folder': map_path.parent.name,
        'size': [context.map_width, context.map_height],
        'border': context.border,
        'total_cells': context.map_width * context.map_height,
    }
    
    # Height data
    height_data = context.get_asset_by_type(HeightMapData)
    if height_data and height_data.elevations is not None:
        arr = np.array(height_data.elevations)
        result['height'] = {
            'min': float(arr.min()),
            'max': float(arr.max()),
            'range': float(arr.max() - arr.min()),
            'unique_rounded': int(len(np.unique(np.round(arr, 0)))),
        }
        # Height bands
        rounded = np.round(arr / 10) * 10
        bands = {}
        for h in np.unique(rounded):
            count = int(np.sum(rounded == h))
            if count > arr.size * 0.01:  # Only bands > 1%
                bands[f'{int(h)}'] = round(count / arr.size * 100, 1)
        result['height']['major_bands'] = bands
    
    # Textures
    blend = context.get_asset('BlendTileData')
    if blend and hasattr(blend, 'textures'):
        result['textures'] = [t.name for t in blend.textures]
        result['texture_count'] = len(blend.textures)
    
    # Passability
    if blend and hasattr(blend, 'impassable') and blend.impassable is not None:
        imp = np.array(blend.impassable)
        result['impassable_pct'] = round(np.sum(imp) / imp.size * 100, 2)
    
    # Water
    water = context.get_asset('StandingWaterAreas')
    if water and hasattr(water, 'water_areas') and water.water_areas:
        result['water_areas'] = []
        for area in water.water_areas:
            if hasattr(area, 'points') and area.points:
                result['water_areas'].append({
                    'points': len(area.points),
                    'coords': [(round(p[0], 0), round(p[1], 0)) for p in area.points]
                })
    
    # ALL objects - categorize by prefix/type
    objects_list = context.get_asset('ObjectsList')
    if objects_list and objects_list.map_objects:
        result['total_objects'] = len(objects_list.map_objects)
        
        # Group by prefix (e.g., YU_, CC_, etc.) and type
        by_prefix = defaultdict(list)
        by_category = defaultdict(list)
        
        for obj in objects_list.map_objects:
            type_name = obj.type_name
            pos = obj.position
            angle = obj.angle if hasattr(obj, 'angle') else 0
            unique_id = obj.unique_id if hasattr(obj, 'unique_id') else None
            
            obj_data = {
                'type': type_name,
                'pos': [round(pos[0], 0), round(pos[1], 0), round(pos[2], 0)],
                'angle': round(angle, 0),
            }
            if unique_id:
                obj_data['id'] = unique_id
            
            # Categorize
            type_lower = type_name.lower()
            
            # Get prefix (first part before underscore or first word)
            if '_' in type_name:
                prefix = type_name.split('_')[0].upper()
            else:
                prefix = 'GAME'
            
            # Determine category
            if 'tree' in type_lower or 'palm' in type_lower or 'bamboo' in type_lower:
                category = 'TREES'
            elif 'bush' in type_lower or 'grass' in type_lower or 'plant' in type_lower or 'fern' in type_lower:
                category = 'VEGETATION'
            elif 'rock' in type_lower or 'boulder' in type_lower or 'stone' in type_lower:
                category = 'ROCKS'
            elif 'coral' in type_lower or 'reef' in type_lower:
                category = 'CORAL'
            elif 'cliff' in type_lower:
                category = 'CLIFFS'
            elif 'bridge' in type_lower:
                category = 'BRIDGES'
            elif 'ambient' in type_lower or 'sound' in type_lower or 'amb_' in type_lower.replace(' ', '_'):
                category = 'AMBIENT_SOUNDS'
            elif 'waypoint' in type_lower:
                category = 'WAYPOINTS'
            elif 'derrick' in type_lower:
                category = 'OIL_DERRICKS'
            elif 'ore' in type_lower or 'node' in type_lower:
                category = 'ORE_NODES'
            elif 'house' in type_lower or 'hut' in type_lower or 'church' in type_lower or 'hotel' in type_lower or 'restaurant' in type_lower or 'apartment' in type_lower:
                category = 'BUILDINGS_GARRISON'
            elif 'building' in type_lower or 'structure' in type_lower or 'tech' in type_lower:
                category = 'BUILDINGS_TECH'
            elif 'lamp' in type_lower or 'light' in type_lower or 'post' in type_lower:
                category = 'LIGHTING'
            elif 'fence' in type_lower or 'wall' in type_lower:
                category = 'FENCES_WALLS'
            elif 'road' in type_lower or 'path' in type_lower or 'dock' in type_lower:
                category = 'ROADS_PATHS'
            elif 'sign' in type_lower or 'board' in type_lower:
                category = 'SIGNS'
            elif 'vehicle' in type_lower or 'car' in type_lower or 'boat' in type_lower or 'ship' in type_lower:
                category = 'VEHICLES'
            elif 'crate' in type_lower or 'barrel' in type_lower or 'container' in type_lower or 'cargo' in type_lower:
                category = 'PROPS'
            elif 'statue' in type_lower or 'idol' in type_lower or 'monument' in type_lower:
                category = 'DECORATIVE'
            elif prefix in ['CC', 'YU', 'CS', 'IL', 'HV', 'AM', 'GC', 'MY', 'SA', 'MJ', 'TH', 'HW']:
                category = f'SCENERY_{prefix}'
            else:
                category = 'OTHER'
            
            by_prefix[prefix].append(obj_data)
            by_category[category].append(obj_data)
        
        # Summarize by category
        result['objects_by_category'] = {}
        for cat, objs in sorted(by_category.items()):
            result['objects_by_category'][cat] = {
                'count': len(objs),
                'types': list(set(o['type'] for o in objs))[:10],  # First 10 unique types
                'sample_positions': [o['pos'][:2] for o in objs[:5]]  # First 5 positions
            }
        
        # Full object list (for LLM context)
        result['all_objects'] = []
        for obj in objects_list.map_objects:
            pos = obj.position
            result['all_objects'].append({
                't': obj.type_name,
                'p': [int(pos[0]), int(pos[1])],
                'a': int(obj.angle) if hasattr(obj, 'angle') else 0
            })
    
    return result


def main():
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    
    # Select 10 diverse maps
    map_folders = [
        "2 CE", "2 II", "2 SS",  # 2-player
        "3 Caledra of Chaos", "3 HF",  # 3-player
        "4 Rock Ridge", "4 Pool Party", "4 Ring of Fire",  # 4-player
        "5 Circus Maximus",  # 5-player
        "6 Carville",  # 6-player
    ]
    
    all_results = []
    total_objects = 0
    category_totals = defaultdict(int)
    
    for folder in map_folders:
        folder_path = maps_dir / folder
        if not folder_path.exists():
            print(f"Skipping {folder} - not found")
            continue
        
        # Find .map file
        map_files = list(folder_path.glob("*.map"))
        if not map_files:
            print(f"Skipping {folder} - no .map file")
            continue
        
        map_file = map_files[0]
        print(f"Analyzing: {folder} / {map_file.name}")
        
        try:
            result = analyze_all_objects(map_file)
            all_results.append(result)
            total_objects += result.get('total_objects', 0)
            
            for cat, data in result.get('objects_by_category', {}).items():
                category_totals[cat] += data['count']
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPLETE OBJECT ANALYSIS - 10 MAPS")
    print("="*80)
    
    print(f"\nTotal objects across all maps: {total_objects:,}")
    print(f"Average per map: {total_objects // len(all_results):,}")
    
    print("\nOBJECTS BY CATEGORY (all maps combined):")
    for cat, count in sorted(category_totals.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {count:5,}")
    
    # Calculate token estimates
    print("\n" + "="*80)
    print("TOKEN ESTIMATION FOR LLM CONTEXT")
    print("="*80)
    
    # Sample one map in detail
    sample = all_results[5]  # 4 Rock Ridge - complex map
    print(f"\nSample map: {sample['map_folder']} ({sample['total_objects']} objects)")
    
    # Compact format for all objects
    compact_objects = json.dumps(sample['all_objects'], separators=(',', ':'))
    print(f"All objects (compact JSON): {len(compact_objects):,} chars = ~{len(compact_objects)//4:,} tokens")
    
    # With full metadata
    full_spec = {
        'metadata': {
            'name': sample['map_folder'],
            'size': sample['size'],
            'border': sample['border']
        },
        'terrain': {
            'height_range': sample.get('height', {}).get('range', 0),
            'major_bands': sample.get('height', {}).get('major_bands', {}),
        },
        'textures': sample.get('textures', []),
        'water_areas': sample.get('water_areas', []),
        'impassable_pct': sample.get('impassable_pct', 0),
        'objects': sample['all_objects']
    }
    full_json = json.dumps(full_spec, separators=(',', ':'))
    print(f"Full map spec (compact): {len(full_json):,} chars = ~{len(full_json)//4:,} tokens")
    
    # Pretty printed for readability
    pretty_json = json.dumps(full_spec, indent=2)
    print(f"Full map spec (pretty): {len(pretty_json):,} chars = ~{len(pretty_json)//4:,} tokens")
    
    print("\n" + "="*80)
    print("VIABILITY ASSESSMENT")
    print("="*80)
    
    avg_objects = total_objects // len(all_results)
    avg_chars = len(compact_objects)
    
    print(f"""
WITH 1 MILLION TOKEN CONTEXT WINDOW:

Single map in context:
  - Objects: ~{avg_objects:,} average
  - Tokens needed: ~{avg_chars//4:,} tokens (compact)
  - = {avg_chars//4/1000000*100:.3f}% of context

10 maps as examples:
  - Tokens needed: ~{avg_chars//4*10:,} tokens
  - = {avg_chars//4*10/1000000*100:.2f}% of context

50 maps as examples (full dataset):
  - Tokens needed: ~{avg_chars//4*50:,} tokens  
  - = {avg_chars//4*50/1000000*100:.1f}% of context

VERDICT: ✅ COMPLETELY VIABLE!
Even with 50 complete maps, you'd use only ~{avg_chars//4*50/1000000*100:.0f}% of a 1M context window.
""")
    
    # Save full analysis
    output_path = Path(__file__).parent.parent / "full_map_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Full analysis saved to: {output_path}")


if __name__ == "__main__":
    main()









