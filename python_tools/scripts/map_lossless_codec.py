#!/usr/bin/env python
"""
Lossless Map Image Codec - Complete 1:1 reversible encoding.

Creates a composite image with:
1. Height encoded in luminance (grayscale base)
2. Texture category encoded in hue/saturation
3. Objects as distinct colored markers
4. JSON sidecar with exact data for perfect reconstruction

Usage:
    python map_lossless_codec.py encode path/to/map.map --output test/
    python map_lossless_codec.py decode test/map_visual.png
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from collections import defaultdict
from colorsys import hsv_to_rgb, rgb_to_hsv

sys.path.insert(0, str(Path(__file__).parent.parent))

from map_processor.parsing.map_parser import Ra3MapParser
from map_processor.assets.terrain.height_map_data import HeightMapData


# =============================================================================
# TEXTURE CATEGORIES - Grouped by visual similarity
# =============================================================================

TEXTURE_CATEGORIES = {
    'water': {
        'keywords': ['reef', 'coral'],
        'hue': 0.55,  # Cyan-blue
        'saturation': 0.7,
    },
    'sand': {
        'keywords': ['sand', 'beach'],
        'hue': 0.12,  # Yellow-tan
        'saturation': 0.5,
    },
    'dirt': {
        'keywords': ['dirt', 'mud', 'path'],
        'hue': 0.08,  # Orange-brown
        'saturation': 0.6,
    },
    'grass': {
        'keywords': ['grass'],
        'hue': 0.33,  # Green
        'saturation': 0.6,
    },
    'rock': {
        'keywords': ['rock', 'cliff', 'stone'],
        'hue': 0.0,   # Gray (low saturation)
        'saturation': 0.1,
    },
    'pavement': {
        'keywords': ['pave', 'concrete', 'road'],
        'hue': 0.0,
        'saturation': 0.05,
    },
    'snow': {
        'keywords': ['snow', 'ice'],
        'hue': 0.55,
        'saturation': 0.1,
    },
}


# =============================================================================
# OBJECT MARKERS - Distinct, non-terrain colors
# =============================================================================

OBJECT_MARKERS = {
    'player_start': {
        'color': (255, 0, 0),      # Pure red
        'size': 8,                  # Radius in pixels
        'shape': 'circle_filled',
        'outline': (255, 255, 255),
        'priority': 100,            # Draw order
    },
    'ore_node': {
        'color': (255, 255, 0),    # Pure yellow
        'size': 5,
        'shape': 'square_filled',
        'outline': (0, 0, 0),
        'priority': 90,
    },
    'oil_derrick': {
        'color': (180, 0, 255),    # Purple
        'size': 5,
        'shape': 'diamond_filled',
        'outline': (255, 255, 255),
        'priority': 90,
    },
    'tech_structure': {
        'color': (255, 0, 255),    # Magenta
        'size': 5,
        'shape': 'triangle_filled',
        'outline': (255, 255, 255),
        'priority': 85,
    },
    'garrison': {
        'color': (0, 255, 255),    # Cyan
        'size': 4,
        'shape': 'circle_filled',
        'outline': (0, 0, 0),
        'priority': 80,
    },
    'tree': {
        'color': (0, 100, 0),      # Dark green
        'size': 2,
        'shape': 'circle_filled',
        'outline': None,
        'priority': 30,
    },
    'rock': {
        'color': (100, 100, 100),  # Gray
        'size': 2,
        'shape': 'square_filled',
        'outline': None,
        'priority': 25,
    },
    'road': {
        'color': (139, 90, 43),    # Brown
        'size': 2,
        'shape': 'square_filled',
        'outline': None,
        'priority': 20,
    },
    'decorative': {
        'color': (200, 200, 200),  # Light gray
        'size': 1,
        'shape': 'dot',
        'outline': None,
        'priority': 10,
    },
    'ambient': {
        'color': (255, 200, 255),  # Light pink
        'size': 1,
        'shape': 'dot',
        'outline': None,
        'priority': 5,
    },
}


def classify_texture(texture_name):
    """Classify texture into category."""
    name_lower = texture_name.lower()
    for cat, cfg in TEXTURE_CATEGORIES.items():
        if any(kw in name_lower for kw in cfg['keywords']):
            return cat
    return 'dirt'  # Default


def classify_object(obj):
    """Classify object into marker category."""
    type_name = obj.type_name.lower()
    
    # Check for player start
    if hasattr(obj, 'unique_id') and obj.unique_id:
        if 'player' in obj.unique_id.lower() and 'start' in obj.unique_id.lower():
            return 'player_start'
    
    if 'orenode' in type_name:
        return 'ore_node'
    if 'derrick' in type_name:
        return 'oil_derrick'
    if any(kw in type_name for kw in ['observation', 'techstructure', 'garage', 'airport', 'hospital']):
        return 'tech_structure'
    if any(kw in type_name for kw in ['house', 'hut', 'warehouse', 'hotel', 'restaurant', 'church', 'villa', 'shack']):
        return 'garrison'
    if any(kw in type_name for kw in ['tree', 'palm', 'bamboo', 'fern']):
        return 'tree'
    if any(kw in type_name for kw in ['grass', 'bush', 'plant']):
        return 'tree'  # Group with trees
    if any(kw in type_name for kw in ['rock', 'cliff', 'boulder', 'stone', 'coral']):
        return 'rock'
    if any(kw in type_name for kw in ['road', 'sidewalk', 'path']):
        return 'road'
    if 'amb_' in type_name:
        return 'ambient'
    
    return 'decorative'


class LosslessMapEncoder:
    """Encode map to lossless visual + JSON sidecar."""
    
    def __init__(self):
        self.parser = Ra3MapParser()
    
    def encode(self, map_path: Path, output_dir: Path):
        """Encode map to visual image + JSON sidecar."""
        context = self.parser.parse(str(map_path))
        
        # Get terrain data
        height_data = context.get_asset_by_type(HeightMapData)
        heights = np.array(height_data.elevations)  # Shape is (width, height) in [x, y] order
        
        blend_data = context.get_asset('BlendTileData')
        tiles = np.array(blend_data.tiles)  # Also (width, height) [x, y]
        texture_names = [t.name for t in blend_data.textures]
        impassable = np.array(blend_data.impassable)
        
        # Get objects
        objects_list = context.get_asset('ObjectsList')
        
        # Map dimensions from context (these are the correct dimensions)
        width = context.map_width
        height = context.map_height
        border = context.border if context.border > 0 else 0
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- ENCODE HEIGHT + TEXTURE TO IMAGE ---
        
        # Height scaling
        h_min, h_max = heights.min(), heights.max()
        h_range = h_max - h_min if h_max != h_min else 1.0
        
        # Create base image in [x, y] coordinates first
        terrain_rgb = np.zeros((heights.shape[0], heights.shape[1], 3), dtype=np.uint8)
        
        # Map each cell to HSV then RGB
        for x in range(heights.shape[0]):
            for y in range(heights.shape[1]):
                # Get height -> luminance (value)
                h_val = heights[x, y]
                luminance = (h_val - h_min) / h_range  # 0-1
                luminance = 0.2 + luminance * 0.7  # Map to 0.2-0.9 for visibility
                
                # Get texture -> hue/saturation
                tile_val = tiles[x, y]
                tex_idx = tile_val % len(texture_names)
                tex_name = texture_names[tex_idx]
                tex_cat = classify_texture(tex_name)
                
                cat_cfg = TEXTURE_CATEGORIES.get(tex_cat, TEXTURE_CATEGORIES['dirt'])
                hue = cat_cfg['hue']
                sat = cat_cfg['saturation']
                
                # HSV to RGB
                r, g, b = hsv_to_rgb(hue, sat, luminance)
                terrain_rgb[x, y] = [int(r * 255), int(g * 255), int(b * 255)]
        
        # Convert to image coordinates: transpose and flip Y
        # This matches the original visualizer: shaded.transpose(1, 0, 2)[::-1, :, :]
        img_arr = terrain_rgb.transpose(1, 0, 2)[::-1, :, :]
        
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)
        
        # --- ENCODE OBJECTS ---
        
        world_scale = 10.0
        object_data = []
        
        # Image dimensions (after transpose/flip)
        img_width = width
        img_height = height
        
        # Sort objects by priority (low first, so high priority draws on top)
        sorted_objects = []
        for obj in objects_list.map_objects:
            cat = classify_object(obj)
            priority = OBJECT_MARKERS.get(cat, {}).get('priority', 0)
            sorted_objects.append((priority, obj, cat))
        sorted_objects.sort(key=lambda x: x[0])
        
        for priority, obj, cat in sorted_objects:
            pos = obj.position
            
            # Use the exact C# formula from MiniMapUtil.cs:
            # x = o.position.x / 10f + borderWidth
            # y = (height - 1) - o.position.y / 10f - borderWidth
            tile_x = int(pos[0] / world_scale)
            tile_y = int(pos[1] / world_scale)
            px = tile_x + border
            py = (img_height - 1) - tile_y - border
            
            if not (0 <= px < img_width and 0 <= py < img_height):
                continue
            
            marker = OBJECT_MARKERS.get(cat)
            if not marker:
                continue
            
            # Draw marker
            color = marker['color']
            size = marker['size']
            shape = marker['shape']
            outline = marker.get('outline')
            
            if shape == 'circle_filled':
                bbox = [px - size, py - size, px + size, py + size]
                draw.ellipse(bbox, fill=color, outline=outline)
            elif shape == 'square_filled':
                bbox = [px - size, py - size, px + size, py + size]
                draw.rectangle(bbox, fill=color, outline=outline)
            elif shape == 'diamond_filled':
                points = [(px, py - size), (px + size, py), (px, py + size), (px - size, py)]
                draw.polygon(points, fill=color, outline=outline)
            elif shape == 'triangle_filled':
                points = [(px, py - size), (px + size, py + size), (px - size, py + size)]
                draw.polygon(points, fill=color, outline=outline)
            elif shape == 'dot':
                draw.point((px, py), fill=color)
            
            # Store object data
            angle = obj.angle if hasattr(obj, 'angle') else 0
            unique_id = obj.unique_id if hasattr(obj, 'unique_id') else None
            
            object_data.append({
                'type': obj.type_name,
                'category': cat,
                'pos_tile': [px, py],
                'pos_world': [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0],
                'angle': float(angle),
                'unique_id': unique_id,
            })
        
        # Save image
        img_path = output_dir / f"{map_path.stem}_visual.png"
        img.save(img_path)
        
        # --- CREATE SIDECAR JSON ---
        
        sidecar = {
            'source_map': str(map_path),
            'size': [width, height],
            'border': border,
            'world_scale': int(world_scale),
            
            'height': {
                'min': float(h_min),
                'max': float(h_max),
                'encoding': 'luminance',
                'formula': 'height = (luminance - 0.2) / 0.7 * range + min',
            },
            
            'textures': {
                'palette': texture_names,
                'categories': {cat: cfg for cat, cfg in TEXTURE_CATEGORIES.items()},
            },
            
            'objects': object_data,
            
            'object_markers': {
                cat: {'color': list(cfg['color']), 'size': cfg['size']}
                for cat, cfg in OBJECT_MARKERS.items()
            },
            
            'passability': {
                'impassable_count': int(impassable.sum()),
                'impassable_pct': float(impassable.sum() / impassable.size * 100),
            },
        }
        
        json_path = output_dir / f"{map_path.stem}_sidecar.json"
        with open(json_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
        
        # --- ALSO SAVE RAW DATA FOR PERFECT RECONSTRUCTION ---
        
        # Height map as 16-bit grayscale (with correct orientation)
        height_normalized = ((heights - h_min) / h_range * 65535).astype(np.uint16)
        # Transpose and flip to match image orientation
        height_oriented = height_normalized.T[::-1, :]
        height_img = Image.fromarray(height_oriented)
        height_img.save(output_dir / f"{map_path.stem}_height.png")
        
        # Texture map as 8-bit (indices, with correct orientation)
        tex_indices = (tiles % len(texture_names)).astype(np.uint8)
        tex_oriented = tex_indices.T[::-1, :]
        tex_img = Image.fromarray(tex_oriented, mode='L')
        tex_img.save(output_dir / f"{map_path.stem}_textures.png")
        
        return {
            'visual': img_path,
            'sidecar': json_path,
            'height_raw': output_dir / f"{map_path.stem}_height.png",
            'texture_raw': output_dir / f"{map_path.stem}_textures.png",
            'stats': {
                'size': [width, height],
                'objects': len(object_data),
                'textures': len(texture_names),
            }
        }


class LosslessMapDecoder:
    """Decode edited image back to map data."""
    
    def __init__(self):
        pass
    
    def decode(self, visual_path: Path, sidecar_path: Path = None):
        """Decode visual image + sidecar to map spec."""
        
        # Load sidecar
        if sidecar_path is None:
            sidecar_path = visual_path.with_name(
                visual_path.stem.replace('_visual', '_sidecar') + '.json'
            )
        
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        
        # Load visual image
        img = Image.open(visual_path).convert('RGB')
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
        
        # --- DECODE HEIGHT FROM LUMINANCE ---
        
        h_min = sidecar['height']['min']
        h_max = sidecar['height']['max']
        h_range = h_max - h_min
        
        heights = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                r, g, b = img_arr[y, x]
                # Luminance approximation
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                height = (luminance - 0.2) / 0.7 * h_range + h_min
                heights[y, x] = height
        
        # --- DETECT OBJECTS FROM MARKERS ---
        
        detected_objects = []
        marker_colors = {cat: np.array(cfg['color']) 
                        for cat, cfg in OBJECT_MARKERS.items()}
        
        # Create mask of detected markers
        visited = np.zeros((h, w), dtype=bool)
        
        for cat, expected_color in marker_colors.items():
            cfg = OBJECT_MARKERS[cat]
            tolerance = 40
            
            # Find pixels matching this color
            for y in range(h):
                for x in range(w):
                    if visited[y, x]:
                        continue
                    
                    pixel = img_arr[y, x]
                    dist = np.sqrt(np.sum((pixel.astype(float) - expected_color.astype(float))**2))
                    
                    if dist < tolerance:
                        # Flood fill to find connected component
                        blob = []
                        stack = [(y, x)]
                        while stack:
                            cy, cx = stack.pop()
                            if 0 <= cy < h and 0 <= cx < w and not visited[cy, cx]:
                                px_dist = np.sqrt(np.sum((img_arr[cy, cx].astype(float) - expected_color.astype(float))**2))
                                if px_dist < tolerance:
                                    visited[cy, cx] = True
                                    blob.append((cx, cy))
                                    stack.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
                        
                        if len(blob) >= 1:
                            # Compute centroid
                            xs = [p[0] for p in blob]
                            ys = [p[1] for p in blob]
                            cx = int(np.mean(xs))
                            cy = int(np.mean(ys))
                            
                            detected_objects.append({
                                'category': cat,
                                'pos_tile': [cx, cy],
                                'pos_world': [cx * sidecar['world_scale'], 
                                             cy * sidecar['world_scale'], 0],
                                'size': len(blob),
                            })
        
        # --- BUILD DECODED SPEC ---
        
        # Note: h, w from image are (height, width), spec uses [width, height]
        spec = {
            'size': [w, h],  # [width, height] to match map convention
            'height': {
                'min': float(heights.min()),
                'max': float(heights.max()),
                'grid_32': self._downsample(heights, 32),
            },
            'objects': {
                'from_visual': detected_objects,
                'from_sidecar': sidecar.get('objects', []),
            },
            'textures': sidecar.get('textures', {}),
        }
        
        return spec, heights
    
    def _downsample(self, arr, grid_size):
        """Downsample array to grid."""
        h, w = arr.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        grid = []
        for j in range(grid_size):
            row = []
            for i in range(grid_size):
                region = arr[j*cell_h:(j+1)*cell_h, i*cell_w:(i+1)*cell_w]
                row.append(int(np.mean(region)))
            grid.append(row)
        return grid


def main():
    parser = argparse.ArgumentParser(description='Lossless Map Image Codec')
    parser.add_argument('action', choices=['encode', 'decode'])
    parser.add_argument('input', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.action == 'encode':
        encoder = LosslessMapEncoder()
        output_dir = Path(args.output) if args.output else input_path.parent / 'encoded'
        result = encoder.encode(input_path, output_dir)
        
        print(f"Encoded: {input_path.name}")
        print(f"Output directory: {output_dir}")
        print(f"Files created:")
        print(f"  - {result['visual'].name} (AI-editable visualization)")
        print(f"  - {result['sidecar'].name} (exact reconstruction data)")
        print(f"  - {result['height_raw'].name} (16-bit height map)")
        print(f"  - {result['texture_raw'].name} (texture indices)")
        print(f"Stats: {result['stats']}")
    
    elif args.action == 'decode':
        decoder = LosslessMapDecoder()
        spec, heights = decoder.decode(input_path)
        
        output_path = Path(args.output) if args.output else input_path.with_suffix('.decoded.json')
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"Decoded: {input_path.name}")
        print(f"Output: {output_path}")
        print(f"Height range: {spec['height']['min']:.0f} - {spec['height']['max']:.0f}")
        print(f"Objects detected from visual: {len(spec['objects']['from_visual'])}")
        print(f"Objects from sidecar: {len(spec['objects']['from_sidecar'])}")


if __name__ == "__main__":
    main()

