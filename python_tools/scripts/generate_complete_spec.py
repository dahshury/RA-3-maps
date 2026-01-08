#!/usr/bin/env python
"""
Generate COMPLETE map specifications for LLM training.

This spec is designed to:
1. Capture all information needed to rebuild a map
2. Be small enough for LLM context (~3-5K tokens per map)
3. Include spatial information (grids) not just statistics

The renderer would:
- Interpolate height grids
- Derive passability from height gradients
- Apply texture blending based on height + style
- Expand decorative zones into individual objects
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData


class CompleteSpecGenerator:
    """Generates complete, rebuildable map specifications."""
    
    STYLE_DETECTION = {
        'snowy': ['snow', 'ice', 'iceland', 'frozen'],
        'tropical': ['hawaii', 'cannes', 'santamonica'],
        'jungle': ['yucatan', 'havana'],
        'mediterranean': ['mykonos', 'greece'],
        'golf': ['golf', 'gypsy'],
        'desert': ['desert'],
        'temperate': ['capecod', 'heidelberg']
    }
    
    GAMEPLAY_KEYWORDS = [
        'OreNode', 'OilDerrick', 'Observation', 'Tech', 'Garage', 'Airport', 'Hospital',
        'House', 'Hut', 'Warehouse', 'Hotel', 'Restaurant', 'Church', 'Villa',
        'Apartment', 'TikiHut', 'Player', 'Waypoint'
    ]
    
    DECORATIVE_GROUPS = {
        'trees': ['tree', 'palm', 'bamboo', 'pine'],
        'bushes': ['bush', 'plant', 'fern', 'shrub'],
        'grass': ['grass'],
        'rocks': ['rock', 'cliff', 'boulder', 'stone'],
        'roads': ['road', 'sidewalk', 'marking', 'line', 'path'],
        'coral': ['coral'],
        'sounds': ['amb_'],
        'vehicles': ['truck', 'auto', 'car', 'boat'],
        'lights': ['lamp', 'light', 'post']
    }
    
    def __init__(self, grid_resolution: int = 16):
        self.parser = Ra3MapParser()
        self.grid_res = grid_resolution
    
    def detect_style(self, texture_names: List[str]) -> str:
        """Detect map style from textures."""
        text = ' '.join(t.lower() for t in texture_names)
        for style, keywords in self.STYLE_DETECTION.items():
            if any(kw in text for kw in keywords):
                return style
        return 'temperate'
    
    def generate_height_grid(self, elevations: np.ndarray) -> Dict[str, Any]:
        """Generate height grid with actual values."""
        h_step = elevations.shape[0] // self.grid_res
        w_step = elevations.shape[1] // self.grid_res
        
        grid = []
        for j in range(self.grid_res):
            row = []
            for i in range(self.grid_res):
                region = elevations[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                row.append(int(region.mean()))
            grid.append(row)
        
        return {
            'values': grid,
            'min': int(elevations.min()),
            'max': int(elevations.max())
        }
    
    def generate_passability_grid(self, impassable: np.ndarray) -> List[List[int]]:
        """Generate passability grid (% impassable per cell)."""
        h_step = impassable.shape[0] // self.grid_res
        w_step = impassable.shape[1] // self.grid_res
        
        grid = []
        for j in range(self.grid_res):
            row = []
            for i in range(self.grid_res):
                region = impassable[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                pct = int(region.sum() / region.size * 100)
                row.append(pct)
            grid.append(row)
        return grid
    
    def extract_water(self, context) -> List[Dict[str, Any]]:
        """Extract water polygons."""
        water_areas = []
        world_size = context.map_width * 10
        
        standing_water = context.get_asset('StandingWaterAreas')
        if standing_water and hasattr(standing_water, 'water_areas') and standing_water.water_areas:
            for area in standing_water.water_areas:
                if hasattr(area, 'points') and area.points:
                    points = area.points
                    # Convert to percentage coordinates
                    points_pct = [[int(p[0]/world_size*100), int(p[1]/world_size*100)] for p in points]
                    
                    # Detect if ocean (extends beyond map) or inland lake
                    min_x = min(p[0] for p in points)
                    max_x = max(p[0] for p in points)
                    is_ocean = min_x < 0 or max_x > world_size
                    
                    water_areas.append({
                        'type': 'ocean' if is_ocean else 'lake',
                        'polygon_pct': points_pct
                    })
        
        return water_areas
    
    def categorize_objects(self, objects_list, world_size: int) -> Dict[str, List[Dict]]:
        """Categorize objects into gameplay and decorative."""
        categories = {
            'player_starts': [],
            'ore_nodes': [],
            'oil_derricks': [],
            'tech_structures': [],
            'garrisons': [],
            'decorative': defaultdict(list)
        }
        
        for obj in objects_list.map_objects:
            pos = obj.position
            pos_pct = [int(pos[0] / world_size * 100), int(pos[1] / world_size * 100)]
            angle = int(obj.angle) if hasattr(obj, 'angle') else 0
            type_lower = obj.type_name.lower()
            
            # Player starts
            if hasattr(obj, 'unique_id') and obj.unique_id:
                if 'Player' in obj.unique_id and 'Start' in obj.unique_id:
                    categories['player_starts'].append({
                        'id': obj.unique_id,
                        'pos_pct': pos_pct
                    })
                    continue
            
            # Resources
            if 'OreNode' in obj.type_name:
                categories['ore_nodes'].append({'pos_pct': pos_pct, 'angle': angle})
                continue
            if 'OilDerrick' in obj.type_name or 'Derrick' in obj.type_name:
                categories['oil_derricks'].append({'pos_pct': pos_pct, 'angle': angle})
                continue
            
            # Tech structures
            tech_keywords = ['observation', 'tech', 'garage', 'airport', 'hospital']
            if any(kw in type_lower for kw in tech_keywords):
                categories['tech_structures'].append({
                    'type': obj.type_name,
                    'pos_pct': pos_pct,
                    'angle': angle
                })
                continue
            
            # Garrisons
            garrison_keywords = ['house', 'hut', 'warehouse', 'hotel', 'restaurant', 
                               'church', 'villa', 'apartment', 'tikihut', 'shack']
            if any(kw in type_lower for kw in garrison_keywords):
                categories['garrisons'].append({
                    'type': obj.type_name,
                    'pos_pct': pos_pct,
                    'angle': angle
                })
                continue
            
            # Decorative - group by type
            dec_group = 'misc'
            for group, keywords in self.DECORATIVE_GROUPS.items():
                if any(kw in type_lower for kw in keywords):
                    dec_group = group
                    break
            
            categories['decorative'][dec_group].append({
                'type': obj.type_name,
                'pos_pct': pos_pct
            })
        
        return categories
    
    def compress_decorative(self, decorative: Dict[str, List]) -> Dict[str, List[Dict]]:
        """Compress decorative objects into zone patterns."""
        patterns = {}
        
        for group_name, objects in decorative.items():
            if not objects:
                continue
            
            # Group by type
            by_type = defaultdict(list)
            for obj in objects:
                by_type[obj['type']].append(obj['pos_pct'])
            
            # Create patterns for types with 3+ instances
            group_patterns = []
            for type_name, positions in sorted(by_type.items(), key=lambda x: -len(x[1])):
                if len(positions) < 3:
                    continue
                
                arr = np.array(positions)
                group_patterns.append({
                    'type': type_name,
                    'count': len(positions),
                    'zone': {
                        'center': [int(arr[:, 0].mean()), int(arr[:, 1].mean())],
                        'spread': [int(arr[:, 0].std()), int(arr[:, 1].std())]
                    }
                })
            
            if group_patterns:
                patterns[group_name] = group_patterns
        
        return patterns
    
    def generate_prompt(self, spec: Dict[str, Any]) -> str:
        """Generate natural language prompt for the map."""
        meta = spec['metadata']
        size_category = 'small' if meta['size'][0] < 300 else 'medium' if meta['size'][0] < 500 else 'large'
        
        parts = [
            f"Create a {size_category} {meta['player_count']}-player {meta['style']} map"
        ]
        
        # Terrain
        height = spec.get('height', {})
        if height.get('max', 0) - height.get('min', 0) > 150:
            parts.append("with dramatic cliffs and height variation")
        elif height.get('max', 0) - height.get('min', 0) > 50:
            parts.append("with varied elevation")
        else:
            parts.append("with mostly flat terrain")
        
        # Water
        water = spec.get('water', [])
        if water:
            water_types = [w['type'] for w in water]
            if 'lake' in water_types:
                parts.append(", featuring an inland lake")
            elif 'ocean' in water_types:
                parts.append(", surrounded by ocean")
        
        parts.append(". ")
        
        # Resources
        ore_count = len(spec.get('ore_nodes', []))
        oil_count = len(spec.get('oil_derricks', []))
        if ore_count > 0:
            per_player = ore_count // meta['player_count']
            parts.append(f"Include {ore_count} ore nodes (~{per_player} per player)")
        if oil_count > 0:
            parts.append(f" and {oil_count} oil derricks")
        parts.append(". ")
        
        # Buildings
        tech_count = len(spec.get('tech_structures', []))
        garrison_count = len(spec.get('garrisons', []))
        if garrison_count > 0:
            parts.append(f"Add {garrison_count} garrisonable buildings")
        if tech_count > 0:
            parts.append(f" and {tech_count} tech structures")
        parts.append(". ")
        
        # Decorative summary
        decorative = spec.get('decorative_patterns', {})
        tree_count = sum(p['count'] for p in decorative.get('trees', []))
        if tree_count > 100:
            parts.append("Dense vegetation. ")
        elif tree_count > 30:
            parts.append("Scattered trees. ")
        
        # Player placement
        starts = spec.get('player_starts', [])
        if len(starts) >= 2:
            parts.append("Player starts should be balanced and symmetrical.")
        
        return ''.join(parts).strip()
    
    def generate_spec(self, map_path: Path) -> Dict[str, Any]:
        """Generate complete map specification."""
        context = self.parser.parse(str(map_path))
        world_size = context.map_width * 10
        
        spec = {}
        
        # Get blend data
        blend_data = context.get_asset('BlendTileData')
        texture_names = [t.name for t in blend_data.textures] if blend_data else []
        
        # Metadata
        spec['metadata'] = {
            'name': map_path.stem,
            'size': [context.map_width, context.map_height],
            'border': context.border,
            'player_count': 0,  # Updated later
            'style': self.detect_style(texture_names)
        }
        
        # Height grid
        height_data = context.get_asset_by_type(HeightMapData)
        if height_data and height_data.elevations is not None:
            spec['height'] = self.generate_height_grid(np.array(height_data.elevations))
        
        # Passability grid
        if blend_data and hasattr(blend_data, 'impassable') and blend_data.impassable is not None:
            spec['passability'] = self.generate_passability_grid(np.array(blend_data.impassable))
        
        # Textures
        spec['textures'] = texture_names
        
        # Water
        spec['water'] = self.extract_water(context)
        
        # Objects
        objects_list = context.get_asset('ObjectsList')
        if objects_list:
            categories = self.categorize_objects(objects_list, world_size)
            
            spec['player_starts'] = categories['player_starts']
            spec['metadata']['player_count'] = len(categories['player_starts'])
            
            spec['ore_nodes'] = categories['ore_nodes']
            spec['oil_derricks'] = categories['oil_derricks']
            spec['tech_structures'] = categories['tech_structures']
            spec['garrisons'] = categories['garrisons']
            
            # Compress decorative
            spec['decorative_patterns'] = self.compress_decorative(categories['decorative'])
        
        # Generate prompt
        spec['prompt'] = self.generate_prompt(spec)
        
        return spec


def main():
    parser = argparse.ArgumentParser(description='Generate complete map specs')
    parser.add_argument('--folder', type=str, help='Specific folder')
    parser.add_argument('--limit', type=int, help='Limit maps')
    parser.add_argument('--grid-res', type=int, default=16, help='Grid resolution (default: 16)')
    args = parser.parse_args()
    
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    
    # Find map files
    map_files = []
    if args.folder:
        folder = maps_dir / args.folder
        map_files = list(folder.glob("*.map"))
    else:
        for folder in maps_dir.iterdir():
            if folder.is_dir():
                map_files.extend(folder.glob("*.map"))
    
    if args.limit:
        map_files = map_files[:args.limit]
    
    print(f"Processing {len(map_files)} maps...")
    
    generator = CompleteSpecGenerator(grid_resolution=args.grid_res)
    total_tokens = 0
    
    for map_file in map_files:
        print(f"\n{map_file.parent.name}/{map_file.name}")
        
        try:
            spec = generator.generate_spec(map_file)
            
            # Save
            output_path = map_file.parent / f"{map_file.stem}_complete_spec.json"
            with open(output_path, 'w') as f:
                json.dump(spec, f, indent=2)
            
            # Stats
            spec_json = json.dumps(spec)
            tokens = len(spec_json) // 4
            total_tokens += tokens
            
            print(f"  {tokens:,} tokens | {spec['metadata']['player_count']} players | {spec['metadata']['style']}")
            print(f"  Prompt: {spec['prompt'][:80]}...")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_tokens:,} tokens across {len(map_files)} maps")
    print(f"Average: {total_tokens // len(map_files):,} tokens per map")


if __name__ == "__main__":
    main()









