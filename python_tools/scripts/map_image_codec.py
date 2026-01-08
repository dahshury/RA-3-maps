#!/usr/bin/env python
"""
Map Image Codec: Reversible map <-> image conversion.

This allows you to:
1. Convert a .map file to a stylized image
2. Edit the image with AI image tools (Midjourney, DALL-E, Stable Diffusion)
3. Convert the modified image back to map data

The image uses a topographic color scheme that encodes:
- Height bands as distinct colors
- Objects as colored markers
- Water as blue regions

Usage:
    # Encode: Map -> Image
    python map_image_codec.py encode path/to/map.map
    
    # Decode: Image -> Map Spec
    python map_image_codec.py decode path/to/image.png --style tropical
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData


# =============================================================================
# COLOR PALETTE (designed for AI-compatibility)
# =============================================================================

# Height bands - distinct, visually pleasing topographic colors
HEIGHT_BANDS = [
    (0, 50, (0, 51, 102)),         # Deep Water - Navy Blue
    (50, 80, (51, 102, 153)),      # Shallow Water - Steel Blue
    (80, 100, (194, 178, 128)),    # Wet Sand - Khaki
    (100, 130, (210, 180, 140)),   # Beach/Coast - Tan
    (130, 160, (107, 142, 35)),    # Lowland - Olive Drab
    (160, 190, (34, 139, 34)),     # Plains - Forest Green
    (190, 220, (144, 238, 144)),   # Midland - Light Green
    (220, 260, (189, 183, 107)),   # Highland - Dark Khaki
    (260, 300, (205, 133, 63)),    # Mountain - Peru
    (300, 350, (139, 90, 43)),     # High Mountain - Sienna
    (350, 400, (128, 128, 128)),   # Peak - Gray
    (400, 600, (255, 255, 255)),   # Summit - White
]

# Object markers - HIGHLY DISTINCT colors that AI won't confuse with terrain
# Using pure RGB primaries and secondaries for maximum distinctiveness
OBJECT_COLORS = {
    'player_start': {
        'color': (255, 0, 0),       # Pure Red - VERY distinct
        'outline': (255, 255, 255), # White outline
        'shape': 'circle',
        'size': 25,                 # Large for visibility
    },
    'ore_node': {
        'color': (255, 255, 0),     # Pure Yellow - distinct from gold terrain
        'outline': (0, 0, 0),       # Black outline
        'shape': 'square',
        'size': 12,
    },
    'oil_derrick': {
        'color': (128, 0, 255),     # Purple - not in terrain palette!
        'outline': (255, 255, 255), # White outline
        'shape': 'diamond',
        'size': 14,
    },
    'tech_structure': {
        'color': (255, 0, 255),     # Magenta - distinct
        'outline': (255, 255, 255), # White outline
        'shape': 'triangle',
        'size': 14,
    },
    'garrison': {
        'color': (0, 255, 255),     # Cyan - not in terrain palette!
        'outline': (0, 0, 0),       # Black outline
        'shape': 'circle',
        'size': 10,
    },
    'bridge': {
        'color': (255, 128, 0),     # Orange - distinct
        'outline': (0, 0, 0),
        'shape': 'rectangle',
        'size': 18,
    },
}


# =============================================================================
# ENCODER: Map -> Image
# =============================================================================

class MapEncoder:
    """Encode a .map file to a reversible image."""
    
    def __init__(self):
        self.parser = Ra3MapParser()
    
    def height_to_color(self, height):
        """Convert height value to band color."""
        for h_min, h_max, rgb in HEIGHT_BANDS:
            if h_min <= height < h_max:
                return rgb
        return HEIGHT_BANDS[-1][2]
    
    def draw_marker(self, draw, x, y, obj_type):
        """Draw an object marker at position."""
        if obj_type not in OBJECT_COLORS:
            return
        
        cfg = OBJECT_COLORS[obj_type]
        color = cfg['color']
        outline = cfg['outline']
        size = cfg['size']
        shape = cfg['shape']
        
        if shape == 'circle':
            draw.ellipse([x-size, y-size, x+size, y+size], 
                        fill=color, outline=outline, width=2)
        elif shape == 'square':
            draw.rectangle([x-size, y-size, x+size, y+size],
                          fill=color, outline=outline, width=2)
        elif shape == 'diamond':
            points = [(x, y-size), (x+size, y), (x, y+size), (x-size, y)]
            draw.polygon(points, fill=color, outline=outline)
        elif shape == 'triangle':
            points = [(x, y-size), (x+size, y+size), (x-size, y+size)]
            draw.polygon(points, fill=color, outline=outline)
        elif shape == 'rectangle':
            draw.rectangle([x-size, y-size//2, x+size, y+size//2],
                          fill=color, outline=outline, width=2)
    
    def encode(self, map_path: Path, output_path: Path = None) -> Path:
        """Encode map to image."""
        context = self.parser.parse(str(map_path))
        
        # Get height data
        height_data = context.get_asset_by_type(HeightMapData)
        heights = np.array(height_data.elevations)
        h, w = heights.shape
        
        # Create terrain image
        img_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                img_arr[y, x] = self.height_to_color(heights[y, x])
        
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        
        # Add object markers
        world_scale = 10
        objects_list = context.get_asset('ObjectsList')
        
        object_stats = {'player_start': 0, 'ore_node': 0, 'oil_derrick': 0, 
                       'tech_structure': 0, 'garrison': 0}
        
        for obj in objects_list.map_objects:
            pos = obj.position
            px = int(pos[0] / world_scale)
            py = int(pos[1] / world_scale)
            
            if not (0 <= px < w and 0 <= py < h):
                continue
            
            type_lower = obj.type_name.lower()
            
            # Determine object type
            obj_type = None
            
            if hasattr(obj, 'unique_id') and obj.unique_id:
                if 'Player' in obj.unique_id and 'Start' in obj.unique_id:
                    obj_type = 'player_start'
            
            if not obj_type:
                if 'orenode' in type_lower:
                    obj_type = 'ore_node'
                elif 'derrick' in type_lower:
                    obj_type = 'oil_derrick'
                elif any(kw in type_lower for kw in ['observation', 'techstructure', 'garage', 'airport', 'hospital']):
                    obj_type = 'tech_structure'
                elif any(kw in type_lower for kw in ['house', 'warehouse', 'hotel', 'church', 'hut', 'villa']):
                    obj_type = 'garrison'
                elif 'bridge' in type_lower:
                    obj_type = 'bridge'
            
            if obj_type:
                self.draw_marker(draw, px, py, obj_type)
                object_stats[obj_type] = object_stats.get(obj_type, 0) + 1
        
        # Save
        if output_path is None:
            output_path = map_path.parent / f"{map_path.stem}_editable.png"
        
        img.save(output_path)
        
        # Also save metadata for later
        metadata = {
            'original_map': str(map_path),
            'size': [w, h],
            'height_range': [float(heights.min()), float(heights.max())],
            'objects': object_stats,
            'height_bands': [[h_min, h_max, list(rgb)] for h_min, h_max, rgb in HEIGHT_BANDS],
            'object_colors': {k: list(v['color']) for k, v in OBJECT_COLORS.items()},
        }
        
        meta_path = output_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path, metadata


# =============================================================================
# DECODER: Image -> Map Spec
# =============================================================================

class MapDecoder:
    """Decode an edited image back to map data."""
    
    def __init__(self, style='temperate'):
        self.style = style
    
    def color_distance(self, c1, c2):
        """Euclidean distance between two colors."""
        return np.sqrt(np.sum((np.array(c1, dtype=float) - np.array(c2, dtype=float))**2))
    
    def find_closest_height(self, pixel, tolerance=80):
        """Find the height band for a pixel color."""
        min_dist = float('inf')
        best_height = None
        
        for h_min, h_max, band_color in HEIGHT_BANDS:
            dist = self.color_distance(pixel, band_color)
            if dist < min_dist:
                min_dist = dist
                best_height = (h_min + h_max) // 2
        
        if min_dist > tolerance:
            return None  # Unknown color
        return best_height
    
    def find_objects(self, img_arr, color, tolerance=50, min_size=30, max_size=3000):
        """Find connected components of a marker color."""
        h, w = img_arr.shape[:2]
        
        # Create mask of matching pixels
        mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if self.color_distance(img_arr[y, x, :3], color) < tolerance:
                    mask[y, x] = True
        
        # Find connected components
        visited = np.zeros_like(mask)
        objects = []
        
        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    # Flood fill
                    blob = []
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if 0 <= cy < h and 0 <= cx < w and mask[cy, cx] and not visited[cy, cx]:
                            visited[cy, cx] = True
                            blob.append((cx, cy))
                            stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
                    
                    if min_size <= len(blob) <= max_size:
                        xs = [p[0] for p in blob]
                        ys = [p[1] for p in blob]
                        objects.append({
                            'x': int(np.mean(xs)),
                            'y': int(np.mean(ys)),
                            'size': len(blob)
                        })
        
        return objects
    
    def decode(self, image_path: Path) -> dict:
        """Decode image to map specification."""
        img = Image.open(image_path).convert('RGB')
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        
        # Parse height map
        height_map = np.zeros((h, w), dtype=np.float32)
        marker_mask = np.zeros((h, w), dtype=bool)
        
        # Check for marker colors first
        marker_colors = [cfg['color'] for cfg in OBJECT_COLORS.values()]
        
        for y in range(h):
            for x in range(w):
                pixel = tuple(img_arr[y, x, :3])
                
                # Check if it's a marker
                is_marker = any(self.color_distance(pixel, mc) < 50 for mc in marker_colors)
                
                if is_marker:
                    marker_mask[y, x] = True
                else:
                    height = self.find_closest_height(pixel)
                    if height is not None:
                        height_map[y, x] = height
        
        # Fill marker areas with neighbor heights
        for y in range(h):
            for x in range(w):
                if marker_mask[y, x]:
                    neighbors = []
                    for dy in range(-15, 16):
                        for dx in range(-15, 16):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not marker_mask[ny, nx]:
                                if height_map[ny, nx] > 0:
                                    neighbors.append(height_map[ny, nx])
                    if neighbors:
                        height_map[y, x] = np.median(neighbors)
        
        # Detect objects
        objects = {}
        for obj_type, cfg in OBJECT_COLORS.items():
            min_size = 50 if obj_type == 'player_start' else 20
            found = self.find_objects(img_arr, cfg['color'], 
                                      min_size=min_size, max_size=2000)
            if found:
                objects[obj_type] = found
        
        # Detect water (blue regions at low height)
        water_mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                pixel = img_arr[y, x, :3]
                # Check if it's a water color
                for h_min, h_max, color in HEIGHT_BANDS[:2]:  # Deep and shallow water
                    if self.color_distance(pixel, color) < 60:
                        water_mask[y, x] = True
                        break
        
        # Build spec
        spec = {
            'metadata': {
                'size': [w, h],
                'style': self.style,
                'source': 'image_decoded',
            },
            'height': {
                'grid': self._downsample_height(height_map, 32),
                'min': float(height_map[height_map > 0].min()) if (height_map > 0).any() else 0,
                'max': float(height_map.max()),
            },
            'water': {
                'has_water': bool(water_mask.any()),
                'coverage_pct': float(water_mask.sum() / water_mask.size * 100),
            },
            'objects': {},
        }
        
        # Add objects with percentage positions
        world_scale = 10
        for obj_type, obj_list in objects.items():
            spec['objects'][obj_type] = [
                {
                    'pos_pct': [int(o['x'] / w * 100), int(o['y'] / h * 100)],
                    'pos_world': [o['x'] * world_scale, o['y'] * world_scale],
                }
                for o in obj_list
            ]
        
        return spec, height_map
    
    def _downsample_height(self, heights, grid_size):
        """Downsample height map to grid."""
        h, w = heights.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        grid = []
        for j in range(grid_size):
            row = []
            for i in range(grid_size):
                region = heights[j*cell_h:(j+1)*cell_h, i*cell_w:(i+1)*cell_w]
                # Use median to ignore marker areas
                valid = region[region > 0]
                row.append(int(np.median(valid)) if len(valid) > 0 else 0)
            grid.append(row)
        return grid


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Map Image Codec')
    parser.add_argument('action', choices=['encode', 'decode'], 
                       help='Action to perform')
    parser.add_argument('input', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--style', type=str, default='temperate',
                       help='Map style for decoding (tropical, arctic, etc.)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.action == 'encode':
        encoder = MapEncoder()
        output_path = Path(args.output) if args.output else None
        result_path, metadata = encoder.encode(input_path, output_path)
        
        print(f"Encoded: {input_path}")
        print(f"Output: {result_path}")
        print(f"Size: {metadata['size']}")
        print(f"Height range: {metadata['height_range']}")
        print(f"Objects: {metadata['objects']}")
        print()
        print("You can now edit this image with an AI image tool!")
        print("When done, run: python map_image_codec.py decode <edited_image.png>")
    
    elif args.action == 'decode':
        decoder = MapDecoder(style=args.style)
        spec, height_map = decoder.decode(input_path)
        
        # Save spec
        output_path = Path(args.output) if args.output else input_path.with_suffix('.spec.json')
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Decoded: {input_path}")
        print(f"Spec saved: {output_path}")
        print(f"Size: {spec['metadata']['size']}")
        print(f"Height range: {spec['height']['min']:.0f} - {spec['height']['max']:.0f}")
        print(f"Water coverage: {spec['water']['coverage_pct']:.1f}%")
        print(f"Objects found:")
        for obj_type, obj_list in spec['objects'].items():
            print(f"  {obj_type}: {len(obj_list)}")


if __name__ == "__main__":
    main()

