"""
Generate minimap TGA from map data.
Based on MiniMap.cs from MapCoreLib.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from map_processor.core.ra3map_struct import MapDataContext

# Terrain style color palettes (from MiniMap.cs)
STYLE_COLORS = {
    # 0: Grassland (草原)
    0: [
        (94, 95, 53),
        (151, 130, 85),
        (176, 158, 113),
        (181, 186, 145),
    ],
    # 1: Snow (雪地)
    1: [
        (145, 146, 151),
        (162, 165, 174),
        (183, 188, 191),
    ],
    # 2: Floating Fortress (浮岛要塞)
    2: [
        (88, 98, 123),
        (116, 131, 152),
    ],
    # 3: Castle (古堡)
    3: [
        (150, 140, 85),
        (101, 74, 38),
        (89, 78, 63),
    ],
    # 4: Imperial Garden (帝国花园)
    4: [
        (52, 65, 0),
        (120, 98, 4),
        (162, 150, 116),
    ],
    # 5: Desert (沙漠)
    5: [
        (119, 84, 54),
        (151, 130, 86),
        (164, 149, 102),
    ],
    # 6: Hot Springs (温热泉)
    6: [
        (89, 113, 14),
        (129, 154, 11),
        (174, 168, 110),
    ],
    # 7: Tropical Rainforest (热带雨林)
    7: [
        (35, 71, 0),
        (129, 129, 0),
        (154, 123, 45),
    ],
    # 8: Desolate Ghost Village (荒凉鬼村)
    8: [
        (156, 121, 77),
        (154, 123, 45),
        (195, 164, 84),
    ],
    # 9: Volcano Island (火山岛)
    9: [
        (94, 94, 52),
        (51, 53, 28),
        (140, 140, 100),
    ],
}

# Water color
WATER_COLOR = (68, 94, 106)

# Impassable color
IMPASSABLE_COLOR = (0, 0, 0)


def detect_terrain_style(context: 'MapDataContext') -> int:
    """
    Try to detect terrain style from map textures.
    Returns style index (0-9), defaults to 0 (grassland).
    """
    blend_tile = context.get_asset("BlendTileData")
    if not blend_tile or not blend_tile.textures:
        return 0
    
    # Check texture names for hints
    texture_names = [t.name.lower() for t in blend_tile.textures if hasattr(t, 'name') and t.name]
    
    # Simple heuristics based on common texture names
    all_names = " ".join(texture_names)
    
    if "snow" in all_names or "ice" in all_names or "arctic" in all_names:
        return 1  # Snow
    elif "desert" in all_names or "sand" in all_names:
        return 5  # Desert
    elif "volcano" in all_names or "lava" in all_names:
        return 9  # Volcano
    elif "jungle" in all_names or "tropical" in all_names:
        return 7  # Tropical
    elif "castle" in all_names or "stone" in all_names:
        return 3  # Castle
    elif "float" in all_names:
        return 2  # Floating Fortress
    
    return 0  # Default to grassland


def find_height_levels(height_data: np.ndarray, water_height: float = 200.0) -> np.ndarray:
    """
    Find distinct height levels above water.
    Returns sorted array of common height values.
    """
    # Get land heights only
    land_heights = height_data[height_data > water_height]
    
    if len(land_heights) == 0:
        return np.array([210.0])  # Default single level
    
    # Round to nearest integer to group similar heights
    rounded = np.round(land_heights).astype(np.int32)
    
    # Count occurrences of each height
    unique, counts = np.unique(rounded, return_counts=True)
    
    # Filter to heights that appear at least 100 times
    common_mask = counts > 100
    common_heights = unique[common_mask].astype(np.float32)
    
    if len(common_heights) == 0:
        # Fallback: just use the most common heights
        sorted_indices = np.argsort(counts)[::-1]
        common_heights = unique[sorted_indices[:4]].astype(np.float32)
    
    return np.sort(common_heights)


def interpolate_color(height: float, h1: float, h2: float, c1: tuple, c2: tuple) -> tuple:
    """Interpolate between two colors based on height."""
    if h2 == h1:
        return c1
    
    ratio = (height - h1) / (h2 - h1)
    ratio = max(0.0, min(1.0, ratio))
    
    r = int(c1[0] + ratio * (c2[0] - c1[0]))
    g = int(c1[1] + ratio * (c2[1] - c1[1]))
    b = int(c1[2] + ratio * (c2[2] - c1[2]))
    
    return (r, g, b)


def generate_minimap(context: 'MapDataContext', style: Optional[int] = None, 
                     show_impassable: bool = True) -> Optional[Image.Image]:
    """
    Generate a minimap image from map data.
    
    Args:
        context: Map data context with HeightMapData and BlendTileData
        style: Terrain style (0-9). If None, auto-detect.
        show_impassable: Whether to show impassable areas in black
        
    Returns:
        PIL Image of the minimap, or None if generation fails
    """
    # Get height data
    height_asset = context.get_asset("HeightMapData")
    if height_asset is None or height_asset.elevations is None:
        return None
    
    height_data = height_asset.elevations
    map_width = height_asset.map_width
    map_height = height_asset.map_height
    border = height_asset.border_width
    
    # Get playable area dimensions
    playable_width = map_width - 2 * border
    playable_height = map_height - 2 * border
    
    if playable_width <= 0 or playable_height <= 0:
        return None
    
    # Extract playable area heights
    playable_heights = height_data[border:border+playable_width, border:border+playable_height]
    
    # Get impassable data if available
    impassable = None
    if show_impassable:
        blend_tile = context.get_asset("BlendTileData")
        if blend_tile is not None and blend_tile.impassable is not None:
            impassable = blend_tile.impassable[border:border+playable_width, border:border+playable_height]
    
    # Determine terrain style
    if style is None:
        style = detect_terrain_style(context)
    
    colors = STYLE_COLORS.get(style, STYLE_COLORS[0])
    
    # Find height levels
    water_height = 200.0
    height_levels = find_height_levels(playable_heights, water_height)
    
    # Create image (RGB)
    img = Image.new('RGB', (playable_width, playable_height), WATER_COLOR)
    pixels = img.load()
    
    for x in range(playable_width):
        for y in range(playable_height):
            h = playable_heights[x, y]
            
            # Flip Y coordinate (game uses bottom-left origin, image uses top-left)
            img_y = playable_height - 1 - y
            
            # Water
            if h <= water_height:
                pixels[x, img_y] = WATER_COLOR
                continue
            
            # Impassable
            if impassable is not None and impassable[x, y]:
                pixels[x, img_y] = IMPASSABLE_COLOR
                continue
            
            # Find color based on height
            if len(height_levels) == 0:
                pixels[x, img_y] = colors[0]
                continue
            
            # Check if height matches a level exactly
            matched = False
            for i, level in enumerate(height_levels):
                if abs(h - level) < 0.5:
                    color_idx = min(i, len(colors) - 1)
                    pixels[x, img_y] = colors[color_idx]
                    matched = True
                    break
            
            if matched:
                continue
            
            # Interpolate between levels
            if h > height_levels[-1]:
                # Above highest level
                pixels[x, img_y] = colors[-1]
            elif h < height_levels[0]:
                # Below lowest level (but above water)
                pixels[x, img_y] = colors[0]
            else:
                # Find levels to interpolate between
                for i in range(len(height_levels) - 1):
                    if height_levels[i] < h < height_levels[i + 1]:
                        c1_idx = min(i, len(colors) - 1)
                        c2_idx = min(i + 1, len(colors) - 1)
                        color = interpolate_color(h, height_levels[i], height_levels[i + 1],
                                                  colors[c1_idx], colors[c2_idx])
                        pixels[x, img_y] = color
                        break
                else:
                    pixels[x, img_y] = colors[0]
    
    return img


def save_minimap_tga(context: 'MapDataContext', output_path: Path, 
                     style: Optional[int] = None) -> bool:
    """
    Generate and save minimap as TGA file.
    
    Args:
        context: Map data context
        output_path: Path to save the TGA file
        style: Terrain style (0-9), auto-detected if None
        
    Returns:
        True if successful, False otherwise
    """
    img = generate_minimap(context, style=style)
    if img is None:
        return False
    
    try:
        # TGA format, uncompressed
        img.save(output_path, format='TGA')
        return True
    except Exception as e:
        print(f"  Warning: Could not save minimap: {e}")
        return False

