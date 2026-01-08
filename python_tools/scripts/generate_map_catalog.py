#!/usr/bin/env python
"""
RA3 Map Catalog Generator for LLM Training

Generates compressed JSON specifications for maps that can be fed to an LLM.
Each JSON includes:
- Automatic natural language description (prompt)
- Terrain data (heights, passability, buildability, visibility)
- Textures
- Water areas
- All gameplay objects (exact positions)
- Decorative patterns (zone-based compression)

Usage:
    python generate_map_catalog.py                           # All maps in RA3 Official maps
    python generate_map_catalog.py --folder "2 CE"           # Specific folder
    python generate_map_catalog.py --limit 5                 # First 5 maps only
    python generate_map_catalog.py --folder "4 Rock Ridge" --limit 1  # Single map
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData


class MapCatalogGenerator:
    """Generates compressed map specifications for LLM training."""
    
    # Gameplay object keywords - these get exact positions
    GAMEPLAY_KEYWORDS = [
        'OreNode', 'OilDerrick', 'Player', 'Waypoint', 'Observation',
        'Airport', 'Hospital', 'House', 'Hotel', 'Restaurant', 'Church',
        'Apartment', 'Villa', 'Hut', 'TikiHut', 'Mansion', 'Shack',
        'Dwelling', 'Garage', 'Bridge', 'Tech', 'Structure', 'Veteran',
        'Supply', 'Cargo', 'Command', 'Military', 'Port', 'Shipyard'
    ]
    
    # Style detection keywords
    STYLE_KEYWORDS = {
        'snowy': ['snow', 'ice', 'frozen', 'winter', 'cold'],
        'tropical': ['hawaii', 'tropical', 'palm', 'beach'],
        'jungle': ['yucatan', 'jungle', 'rainforest'],
        'desert': ['desert', 'sand', 'dune', 'arid'],
        'urban': ['city', 'urban', 'pavement', 'concrete'],
        'volcanic': ['volcanic', 'lava', 'magma', 'fire'],
        'coastal': ['cannes', 'coast', 'shore', 'ocean'],
        'mediterranean': ['mykonos', 'greece', 'mediterranean'],
        'temperate': ['capecod', 'heidelberg', 'temperate', 'grass']
    }
    
    def __init__(self):
        self.parser = Ra3MapParser()
    
    def detect_style(self, textures: List[str]) -> str:
        """Detect map style from texture names."""
        text = ' '.join(t.lower() for t in textures)
        
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return style
        return 'temperate'
    
    def is_gameplay_object(self, type_name: str) -> bool:
        """Check if object is gameplay-relevant (needs exact position)."""
        type_lower = type_name.lower()
        return any(kw.lower() in type_lower for kw in self.GAMEPLAY_KEYWORDS)
    
    def analyze_height_terrain(self, elevations: np.ndarray) -> Dict[str, Any]:
        """Analyze height map for terrain features."""
        result = {
            'min': int(elevations.min()),
            'max': int(elevations.max()),
            'range': int(elevations.max() - elevations.min()),
            'mean': int(elevations.mean()),
        }
        
        # Major height bands (rounded to 20 units)
        rounded = np.round(elevations / 20) * 20
        bands = {}
        for h in np.unique(rounded):
            pct = np.sum(rounded == h) / elevations.size * 100
            if pct > 1:  # Only bands > 1%
                bands[int(h)] = round(pct, 1)
        result['major_bands'] = bands
        
        # Terrain features
        result['has_cliffs'] = result['range'] > 80
        result['is_mostly_flat'] = result['range'] < 30
        result['has_multiple_levels'] = len(bands) >= 3
        
        return result
    
    def analyze_passability(self, blend_data) -> Dict[str, Any]:
        """Analyze passability, buildability, visibility."""
        result = {}
        
        if hasattr(blend_data, 'impassable') and blend_data.impassable is not None:
            imp = np.array(blend_data.impassable)
            result['impassable_pct'] = round(np.sum(imp) / imp.size * 100, 2)
        
        if hasattr(blend_data, 'buildability') and blend_data.buildability is not None:
            build = np.array(blend_data.buildability)
            result['buildable_pct'] = round(np.sum(build) / build.size * 100, 2)
        
        if hasattr(blend_data, 'visibility') and blend_data.visibility is not None:
            vis = np.array(blend_data.visibility)
            result['visible_pct'] = round(np.sum(vis) / vis.size * 100, 2)
        
        if hasattr(blend_data, 'passability') and blend_data.passability is not None:
            pass_arr = np.array(blend_data.passability)
            unique_vals = np.unique(pass_arr)
            result['passability_values'] = [int(v) for v in unique_vals]
        
        return result
    
    def extract_water_areas(self, context) -> List[Dict[str, Any]]:
        """Extract water area polygons."""
        water_areas = []
        
        standing_water = context.get_asset('StandingWaterAreas')
        if standing_water and hasattr(standing_water, 'water_areas') and standing_water.water_areas:
            for area in standing_water.water_areas:
                if hasattr(area, 'points') and area.points:
                    water_areas.append({
                        'type': 'standing_water',
                        'points': [[int(p[0]), int(p[1])] for p in area.points]
                    })
        
        river_areas = context.get_asset('RiverAreas')
        if river_areas and hasattr(river_areas, 'areas') and river_areas.areas:
            for area in river_areas.areas:
                if hasattr(area, 'points') and area.points:
                    water_areas.append({
                        'type': 'river',
                        'points': [[int(p[0]), int(p[1])] for p in area.points]
                    })
        
        return water_areas
    
    def categorize_objects(self, objects_list) -> Tuple[List[Dict], List[Dict]]:
        """Separate objects into gameplay (exact) and decorative (patterns)."""
        gameplay_objects = []
        decorative_by_type = defaultdict(list)
        
        player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start',
                           'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
        
        for obj in objects_list.map_objects:
            pos = obj.position
            obj_data = {
                'type': obj.type_name,
                'pos': [int(pos[0]), int(pos[1]), int(pos[2])],
                'angle': int(obj.angle) if hasattr(obj, 'angle') else 0
            }
            
            # Check for player starts
            if hasattr(obj, 'unique_id') and obj.unique_id:
                if obj.unique_id in player_start_ids:
                    obj_data['id'] = obj.unique_id
                    obj_data['is_player_start'] = True
                    gameplay_objects.append(obj_data)
                    continue
            
            # Categorize by gameplay relevance
            if self.is_gameplay_object(obj.type_name):
                if hasattr(obj, 'unique_id') and obj.unique_id:
                    obj_data['id'] = obj.unique_id
                gameplay_objects.append(obj_data)
            else:
                decorative_by_type[obj.type_name].append(obj_data)
        
        # Compress decorative objects into patterns
        decorative_patterns = []
        for type_name, objs in decorative_by_type.items():
            if len(objs) < 3:
                # Few objects - keep individual
                gameplay_objects.extend(objs)
            else:
                # Create zone pattern
                positions = np.array([o['pos'][:2] for o in objs])
                pattern = {
                    'type': type_name,
                    'count': len(objs),
                    'zone': {
                        'center': [int(positions[:, 0].mean()), int(positions[:, 1].mean())],
                        'spread': [int(positions[:, 0].std()), int(positions[:, 1].std())],
                        'min': [int(positions[:, 0].min()), int(positions[:, 1].min())],
                        'max': [int(positions[:, 0].max()), int(positions[:, 1].max())]
                    },
                    'sample_positions': [o['pos'][:2] for o in objs[:3]]
                }
                decorative_patterns.append(pattern)
        
        return gameplay_objects, decorative_patterns
    
    def generate_description(self, spec: Dict[str, Any]) -> str:
        """Generate a natural language description like a user prompt."""
        size = spec['metadata']['size']
        players = spec['metadata']['player_count']
        style = spec['metadata']['style']
        terrain = spec.get('terrain', {})
        gameplay = spec.get('gameplay_objects', [])
        decorative = spec.get('decorative_patterns', [])
        passability = spec.get('passability', {})
        
        # Map size category
        area = size[0] * size[1]
        if area < 200 * 200:
            size_desc = "small"
        elif area < 400 * 400:
            size_desc = "medium-sized"
        elif area < 600 * 600:
            size_desc = "large"
        else:
            size_desc = "extra large"
        
        # Style name mapping
        style_names = {
            'snowy': 'snowy winter',
            'tropical': 'tropical island',
            'jungle': 'lush jungle',
            'desert': 'arid desert',
            'urban': 'urban city',
            'volcanic': 'volcanic',
            'coastal': 'coastal beach',
            'mediterranean': 'Mediterranean coastal',
            'temperate': 'temperate grassland'
        }
        style_name = style_names.get(style, style)
        
        # Opening
        prompt = f"Create a {size_desc} {players}-player {style_name} map"
        
        # Terrain description
        terrain_parts = []
        height_range = terrain.get('range', 0)
        if height_range > 150:
            terrain_parts.append("dramatic cliffs and elevation changes")
        elif height_range > 80:
            terrain_parts.append("varied elevation with ridges")
        elif height_range > 30:
            terrain_parts.append("gentle rolling hills")
        else:
            terrain_parts.append("mostly flat terrain")
        
        num_bands = len(terrain.get('major_bands', {}))
        if num_bands >= 4:
            terrain_parts.append(f"{num_bands} distinct height levels")
        
        if terrain_parts:
            prompt += f" with {' and '.join(terrain_parts)}"
        
        # Water features
        water = spec.get('water', [])
        if water:
            water_types = [w.get('type', 'water') for w in water]
            if 'river' in water_types:
                prompt += ", featuring a river"
            elif len(water) > 1:
                prompt += f", with {len(water)} water areas"
            else:
                prompt += ", with a central lake or water feature"
        
        prompt += ". "
        
        # Resources
        ore_count = sum(1 for o in gameplay if 'OreNode' in o.get('type', ''))
        oil_count = sum(1 for o in gameplay if 'OilDerrick' in o.get('type', '') or 'Derrick' in o.get('type', ''))
        
        resource_desc = []
        if ore_count > 0:
            per_player = ore_count // players if players > 0 else ore_count
            if per_player > 3:
                resource_desc.append(f"abundant ore deposits ({ore_count} total, ~{per_player} per player)")
            else:
                resource_desc.append(f"{ore_count} ore nodes")
        
        if oil_count > 0:
            per_player = oil_count // players if players > 0 else oil_count
            resource_desc.append(f"{oil_count} oil derricks ({per_player} per player)")
        
        if resource_desc:
            prompt += "Include " + " and ".join(resource_desc) + ". "
        
        # Buildings & tech structures
        garrison_types = ['House', 'Hotel', 'Restaurant', 'Church', 'Apartment', 'Hut', 
                         'Villa', 'TikiHut', 'Shack', 'Dwelling']
        garrison_objs = [o for o in gameplay 
                        if any(gt.lower() in o.get('type', '').lower() for gt in garrison_types)]
        
        tech_objs = [o for o in gameplay 
                    if 'Tech' in o.get('type', '') or 'Observation' in o.get('type', '') 
                    or 'Airport' in o.get('type', '') or 'Hospital' in o.get('type', '')]
        
        building_parts = []
        if garrison_objs:
            # Group by general type
            building_parts.append(f"{len(garrison_objs)} garrisonable civilian buildings")
        if tech_objs:
            tech_types = set()
            for o in tech_objs:
                if 'Observation' in o.get('type', ''):
                    tech_types.add('observation post')
                elif 'Airport' in o.get('type', ''):
                    tech_types.add('airport')
                elif 'Hospital' in o.get('type', ''):
                    tech_types.add('hospital')
                else:
                    tech_types.add('tech structure')
            building_parts.append(f"tech structures ({', '.join(tech_types)})")
        
        if building_parts:
            prompt += "Add " + " and ".join(building_parts) + ". "
        
        # Decorative elements
        tree_patterns = [p for p in decorative 
                        if any(t in p.get('type', '').lower() 
                              for t in ['tree', 'palm', 'bamboo', 'pine'])]
        tree_count = sum(p.get('count', 0) for p in tree_patterns)
        
        rock_patterns = [p for p in decorative 
                        if any(t in p.get('type', '').lower() 
                              for t in ['rock', 'boulder', 'cliff'])]
        rock_count = sum(p.get('count', 0) for p in rock_patterns)
        
        bush_patterns = [p for p in decorative 
                        if any(t in p.get('type', '').lower() 
                              for t in ['bush', 'plant', 'grass', 'fern'])]
        bush_count = sum(p.get('count', 0) for p in bush_patterns)
        
        decor_parts = []
        if tree_count > 200:
            decor_parts.append("dense tree coverage")
        elif tree_count > 50:
            decor_parts.append("scattered trees")
        
        if rock_count > 100:
            decor_parts.append("rocky terrain decorations")
        
        if bush_count > 200:
            decor_parts.append("lush ground vegetation")
        
        # Roads and infrastructure
        road_patterns = [p for p in decorative 
                        if any(t in p.get('type', '').lower() 
                              for t in ['road', 'path', 'sidewalk'])]
        road_count = sum(p.get('count', 0) for p in road_patterns)
        if road_count > 100:
            decor_parts.append("road network")
        
        if decor_parts:
            prompt += "Decorate with " + ", ".join(decor_parts) + ". "
        
        # Player placement
        player_starts = [o for o in gameplay if o.get('is_player_start')]
        if player_starts and len(player_starts) >= 2:
            positions = [ps.get('pos', [0, 0]) for ps in player_starts]
            # Check if symmetric
            center = [size[0] * 5, size[1] * 5]  # World coords are 10x tile coords
            corners = ['northwest', 'northeast', 'southwest', 'southeast']
            prompt += f"Player starting positions should be symmetrically balanced. "
        
        # Passability note
        imp_pct = passability.get('impassable_pct', 0)
        if imp_pct > 15:
            prompt += f"Include {imp_pct:.0f}% impassable terrain for tactical chokepoints."
        elif imp_pct > 5:
            prompt += f"Natural terrain obstacles should cover about {imp_pct:.0f}% of the map."
        
        return prompt.strip()
    
    def generate_spec(self, map_path: Path) -> Dict[str, Any]:
        """Generate complete map specification."""
        context = self.parser.parse(str(map_path))
        
        spec = {}
        
        # Metadata
        spec['metadata'] = {
            'name': map_path.stem,
            'folder': map_path.parent.name,
            'size': [context.map_width, context.map_height],
            'border': context.border,
            'player_count': 0,  # Will be updated
            'style': 'temperate'  # Will be updated
        }
        
        # Terrain/Height
        height_data = context.get_asset_by_type(HeightMapData)
        if height_data and height_data.elevations is not None:
            elevations = np.array(height_data.elevations)
            spec['terrain'] = self.analyze_height_terrain(elevations)
        
        # Textures
        blend_data = context.get_asset('BlendTileData')
        if blend_data and hasattr(blend_data, 'textures') and blend_data.textures:
            textures = [t.name for t in blend_data.textures]
            spec['textures'] = textures
            spec['metadata']['style'] = self.detect_style(textures)
        
        # Passability, buildability, visibility
        if blend_data:
            spec['passability'] = self.analyze_passability(blend_data)
        
        # Water
        spec['water'] = self.extract_water_areas(context)
        
        # Objects
        objects_list = context.get_asset('ObjectsList')
        if objects_list and objects_list.map_objects:
            gameplay, decorative = self.categorize_objects(objects_list)
            spec['gameplay_objects'] = gameplay
            spec['decorative_patterns'] = decorative
            
            # Count players
            player_starts = [o for o in gameplay if o.get('is_player_start')]
            spec['metadata']['player_count'] = len(player_starts)
            
            # Total object counts
            spec['object_counts'] = {
                'total': len(objects_list.map_objects),
                'gameplay': len(gameplay),
                'decorative_patterns': len(decorative),
                'decorative_total': sum(p.get('count', 0) for p in decorative)
            }
        
        # Generate natural language description
        spec['description'] = self.generate_description(spec)
        
        return spec
    
    def save_spec(self, spec: Dict[str, Any], output_path: Path) -> None:
        """Save specification to JSON file."""
        # Create a formatted output with description as "prompt"
        output = {
            'prompt': spec.pop('description'),
            **spec
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Calculate size
        compact = json.dumps(output, separators=(',', ':'))
        tokens = len(compact) // 4
        
        print(f"  Saved: {output_path.name}")
        print(f"  Size: {len(compact):,} chars = ~{tokens:,} tokens")
        return tokens


def main():
    parser = argparse.ArgumentParser(description='Generate RA3 map catalog for LLM training')
    parser.add_argument('--folder', type=str, help='Specific map folder to process')
    parser.add_argument('--limit', type=int, help='Limit number of maps to process')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as map folder)')
    args = parser.parse_args()
    
    # Find maps directory
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    
    if not maps_dir.exists():
        print(f"Error: Maps directory not found: {maps_dir}")
        sys.exit(1)
    
    # Find map folders
    if args.folder:
        folders = [maps_dir / args.folder]
        if not folders[0].exists():
            print(f"Error: Folder not found: {folders[0]}")
            sys.exit(1)
    else:
        # Get all folders with .map files
        folders = sorted([f for f in maps_dir.iterdir() if f.is_dir()])
    
    # Collect all map files
    map_files = []
    for folder in folders:
        for map_file in folder.glob("*.map"):
            map_files.append(map_file)
        # Check subdirectories too
        for subdir in folder.iterdir():
            if subdir.is_dir():
                for map_file in subdir.glob("*.map"):
                    map_files.append(map_file)
    
    if args.limit:
        map_files = map_files[:args.limit]
    
    print(f"Found {len(map_files)} map files to process")
    print("=" * 60)
    
    generator = MapCatalogGenerator()
    total_tokens = 0
    successful = 0
    
    for map_file in map_files:
        print(f"\nProcessing: {map_file.parent.name}/{map_file.name}")
        
        try:
            spec = generator.generate_spec(map_file)
            
            # Output path
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{map_file.stem}_llm_spec.json"
            else:
                output_path = map_file.parent / f"{map_file.stem}_llm_spec.json"
            
            tokens = generator.save_spec(spec, output_path)
            total_tokens += tokens
            successful += 1
            
            # Print summary
            print(f"  Players: {spec['metadata']['player_count']}")
            print(f"  Style: {spec['metadata']['style']}")
            print(f"  Objects: {spec['object_counts']['gameplay']} gameplay, "
                  f"{spec['object_counts']['decorative_total']} decorative")
            print(f"  Prompt: {spec.get('prompt', spec.get('description', ''))[:100]}...")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {successful}/{len(map_files)} maps")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average per map: {total_tokens // successful if successful > 0 else 0:,} tokens")
    print(f"\nWith 1M context window:")
    print(f"  Can fit: {1_000_000 // (total_tokens // successful) if successful > 0 else 0} complete maps")


if __name__ == "__main__":
    main()

