"""
Map visualization utilities
Create images from map data (heights, tiles, textures, water, etc.)
"""
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Tuple, Dict, List
from pathlib import Path

from ..assets.terrain.height_map_data import HeightMapData
from ..assets.terrain.passability import Passability


class MapVisualizer:
    """
    Create visualizations from RA3 map data
    """
    
    # Texture name to color mapping based on texture name patterns
    TEXTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
        # Grass textures - various greens
        'Grass': (50, 150, 50),  # Bright green
        'TGrass': (50, 150, 50),
        
        # Dirt textures - browns
        'Dirt': (139, 90, 43),  # Saddle brown
        'TDirt': (139, 90, 43),
        'Mud': (101, 67, 33),  # Darker brown
        
        # Rock textures - grays
        'Rock': (128, 128, 128),  # Gray
        'TRock': (128, 128, 128),
        'Gravel': (169, 169, 169),  # Dark gray
        'TGravel': (169, 169, 169),
        'BB_Gravel': (169, 169, 169),  # Dark gray
        
        # Sand textures - beige/yellow
        'Sand': (238, 203, 173),  # Peach puff
        'TSand': (238, 203, 173),
        
        # Pavement - light gray
        'Pavement': (192, 192, 192),  # Silver
        'Dock': (105, 105, 105),  # Dim gray
        
        # Water/Reef - blue
        'Reef': (0, 100, 200),  # Deep blue
        'TReef': (0, 100, 200),

        # Snow / Ice - whites and pale blues
        'Snow': (235, 240, 248),  # Near-white with cool tint
        'Ice': (200, 220, 235),  # Pale blue
        'Cliff_Iceland': (180, 195, 210),  # Bluish gray for icy cliffs

        # Industrial / urban surfaces
        'SteelDeck': (110, 115, 120),  # Steel gray
        'Asphalt': (60, 60, 64),  # Dark asphalt
        'Cliff': (115, 95, 75),  # Brownish cliff
        'BB': (140, 130, 110),  # Construction tan

        # Transition - mix of colors
        'Transition': (107, 142, 35),  # Olive drab
        
        # Elevation - light brown/gray
        'Elevation': (160, 82, 45),  # Sienna
        
        # Grid - light gray
        'Grid': (220, 220, 220),  # Gainsboro
        'RA3Grid': (220, 220, 220),
    }
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        """
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    @staticmethod
    def _get_xp(use_gpu: bool):
        """
        Return array module for computation.
        - CPU: numpy
        - GPU: cupy (if installed)
        """
        if not use_gpu:
            return np
        try:
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "GPU acceleration requested but CuPy is not installed. "
                "Install an appropriate package (e.g. cupy-cuda12x) or run with use_gpu=False."
            ) from e

        # Validate that CuPy can actually JIT kernels (NVRTC present) and a CUDA runtime is usable.
        # Cache the check so we don't repeatedly trigger compilation overhead.
        cached = getattr(MapVisualizer, "_GPU_RUNTIME_OK", None)
        if cached is None:
            try:
                _ = cp.cuda.runtime.getDeviceCount()
                # Trigger a tiny kernel compilation/execution to verify NVRTC is available.
                _ = (cp.arange(1, dtype=cp.float32) + 1).sum()
                MapVisualizer._GPU_RUNTIME_OK = True
            except Exception as e:  # pragma: no cover
                MapVisualizer._GPU_RUNTIME_OK = False
                raise RuntimeError(
                    "GPU acceleration requested and CuPy is installed, but the CUDA runtime compilation "
                    "component (NVRTC) could not be loaded.\n\n"
                    "Common fix on Windows:\n"
                    "- Install the NVIDIA CUDA Toolkit 12.x (so `nvrtc64_120_0.dll` is available)\n"
                    "- Ensure the CUDA bin directory is on PATH (or set CUDA_PATH), then retry.\n\n"
                    "Original error: "
                    f"{e}"
                ) from e
        elif cached is False:  # pragma: no cover
            raise RuntimeError(
                "GPU acceleration requested but the CUDA runtime/NVRTC check previously failed in this process."
            )
        return cp

    @staticmethod
    def _to_numpy(arr):
        """
        Convert numpy/cupy array to numpy array (no-op for numpy).

        Important: avoid importing CuPy on CPU-only runs (it can emit environment warnings).
        """
        # CuPy arrays provide `.get()` -> numpy
        get = getattr(arr, "get", None)
        if callable(get):  # pragma: no cover
            try:
                return get()
            except Exception:
                return arr
        return arr

    @staticmethod
    def _world_points_to_pixels(points: List[Tuple[float, float]], width: int, height: int, border: int) -> List[Tuple[int, int]]:
        """
        Convert world coordinate points (RA3) to pixel coordinates used in the generated image.
        Matches the object drawing formula (world units -> tiles by /10, and Y is flipped).
        """
        world_to_tile = 10.0
        out: List[Tuple[int, int]] = []
        for px, py in points:
            tile_x = int(px / world_to_tile)
            tile_y = int(py / world_to_tile)
            x = tile_x + border
            y = height - 1 - tile_y - border
            out.append((x, y))
        return out
    
    @staticmethod
    def _get_texture_color(texture_name: str) -> Tuple[int, int, int]:
        """
        Get color for a texture based on its name.
        Uses pattern matching to determine appropriate color.
        """
        texture_name_lower = texture_name.lower()
        
        # Check for exact matches first
        for pattern, color in MapVisualizer.TEXTURE_COLORS.items():
            if pattern.lower() in texture_name_lower:
                # Adjust brightness based on texture name variations
                base_color = list(color)
                
                # Hawaii textures - brighter greens
                if 'hawaii' in texture_name_lower:
                    if 'grass' in texture_name_lower:
                        base_color = [60, 180, 60]  # Brighter green
                
                # Yucatan textures - warmer colors
                if 'yucatan' in texture_name_lower:
                    if 'grass' in texture_name_lower:
                        base_color = [60, 160, 60]  # Medium green
                    elif 'dirt' in texture_name_lower:
                        base_color = [139, 90, 43]  # Standard brown
                    elif 'rock' in texture_name_lower:
                        base_color = [140, 140, 140]  # Light gray
                    elif 'reef' in texture_name_lower:
                        base_color = [0, 120, 220]  # Bright blue
                
                # HotSprings - reddish browns
                if 'hotsprings' in texture_name_lower:
                    if 'dirt' in texture_name_lower:
                        base_color = [150, 100, 50]  # Reddish brown
                    elif 'grass' in texture_name_lower:
                        base_color = [70, 140, 50]  # Duller green
                
                # Romania - darker, more muted
                if 'romania' in texture_name_lower:
                    base_color = [int(c * 0.8) for c in base_color]
                
                # Vlad - grayish
                if 'vlad' in texture_name_lower:
                    base_color = [int((c + 128) / 2) for c in base_color]
                
                # Cannes - sandy/beachy
                if 'cannes' in texture_name_lower:
                    if 'sand' in texture_name_lower:
                        base_color = [240, 210, 180]  # Light beige
                
                # Havana - tropical
                if 'havana' in texture_name_lower:
                    if 'mud' in texture_name_lower:
                        base_color = [120, 80, 50]  # Dark brown
                    elif 'reef' in texture_name_lower:
                        base_color = [0, 150, 250]  # Bright blue
                
                return tuple(base_color)
        
        # Default: light brown/gray
        return (160, 140, 120)
    
    @staticmethod
    def create_height_map_image(
        height_data: HeightMapData,
        output_path: Optional[str] = None,
        colormap: str = 'grayscale',
        use_gpu: bool = False,
    ) -> Image.Image:
        """
        Create an image from height map data.
        
        Args:
            height_data: HeightMapData asset
            output_path: Optional path to save the image
            colormap: Color scheme ('grayscale', 'terrain', 'elevation')
            
        Returns:
            PIL Image
        """
        if height_data.elevations is None:
            raise ValueError("Height data has no elevations")
        
        xp = MapVisualizer._get_xp(use_gpu)
        elevations = xp.asarray(height_data.elevations)
        width, height = height_data.map_width, height_data.map_height
        
        # Normalize elevations to 0-255 range
        min_elev = xp.min(elevations)
        max_elev = xp.max(elevations)
        elev_range = xp.where(max_elev != min_elev, (max_elev - min_elev), 1.0).astype(xp.float32)
        normalized = xp.clip(((elevations - min_elev) / elev_range) * 255.0, 0, 255).astype(xp.uint8)
        
        # Apply colormap
        if colormap == 'grayscale':
            # Simple grayscale
            image_array = normalized.T  # (height, width)
            
        elif colormap == 'terrain' or colormap == 'elevation':
            # Terrain-like colormap (green for low, brown/gray for high)
            val_yx = (normalized.T.astype(xp.float32) / 255.0)  # (height, width)
            image_array = xp.zeros((height, width, 3), dtype=xp.uint8)

            # Water/low
            m0 = val_yx < 0.3
            t0 = xp.where(m0, val_yx / 0.3, 0.0)
            image_array[..., 0] = xp.where(m0, 0, image_array[..., 0])
            image_array[..., 1] = xp.where(m0, xp.clip(100.0 * t0, 0, 255).astype(xp.uint8), image_array[..., 1])
            image_array[..., 2] = xp.where(m0, xp.clip(200.0 * t0, 0, 255).astype(xp.uint8), image_array[..., 2])

            # Land mid
            m1 = (val_yx >= 0.3) & (val_yx < 0.6)
            t1 = xp.where(m1, (val_yx - 0.3) / 0.3, 0.0)
            r1 = xp.clip(50.0 * t1, 0, 255).astype(xp.uint8)
            g1 = xp.clip(150.0 + 105.0 * t1, 0, 255).astype(xp.uint8)
            b1 = xp.clip(50.0 * (1.0 - t1), 0, 255).astype(xp.uint8)
            image_array[..., 0] = xp.where(m1, r1, image_array[..., 0])
            image_array[..., 1] = xp.where(m1, g1, image_array[..., 1])
            image_array[..., 2] = xp.where(m1, b1, image_array[..., 2])

            # Mountain high
            m2 = val_yx >= 0.6
            t2 = xp.where(m2, (val_yx - 0.6) / 0.4, 0.0)
            r2 = xp.clip(100.0 + 155.0 * t2, 0, 255).astype(xp.uint8)
            g2 = xp.clip(80.0 + 175.0 * t2, 0, 255).astype(xp.uint8)
            b2 = xp.clip(50.0 + 205.0 * t2, 0, 255).astype(xp.uint8)
            image_array[..., 0] = xp.where(m2, r2, image_array[..., 0])
            image_array[..., 1] = xp.where(m2, g2, image_array[..., 1])
            image_array[..., 2] = xp.where(m2, b2, image_array[..., 2])
            
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
        
        image_array = MapVisualizer._to_numpy(image_array)

        # Create PIL Image
        if colormap == 'grayscale':
            image = Image.fromarray(image_array, mode='L')
        else:
            image = Image.fromarray(image_array, mode='RGB')
        
        # Save if path provided
        if output_path:
            image.save(output_path)
            print(f"Height map image saved to: {output_path}")
        
        return image
    
    @staticmethod
    def create_comprehensive_terrain_image(context, output_path: Optional[str] = None, use_gpu: bool = False) -> Image.Image:
        """
        Create a comprehensive terrain visualization combining:
        - Real texture colors from BlendTileData
        - Height data for shading
        - Water areas
        - Passability information
        
        Args:
            context: MapDataContext
            output_path: Optional path to save the image
            
        Returns:
            PIL Image
        """
        # Get assets
        height_data = context.get_asset_by_type(HeightMapData)
        blend_tile_data = context.get_asset('BlendTileData')
        standing_water = context.get_asset('StandingWaterAreas')
        river_areas = context.get_asset('RiverAreas')
        
        if not height_data or not blend_tile_data:
            raise ValueError("HeightMapData and BlendTileData are required")
        
        width = context.map_width
        height = context.map_height
        border = context.border if getattr(context, 'border', 0) and context.border > 0 else 0
        
        xp = MapVisualizer._get_xp(use_gpu)

        # GPU/CPU arrays
        elevations = xp.asarray(height_data.elevations)  # (width, height) [x,y]
        tiles = xp.asarray(blend_tile_data.tiles)        # (width, height) [x,y]

        # Elevation shading
        min_elev = xp.min(elevations)
        max_elev = xp.max(elevations)
        elev_range = xp.where(max_elev != min_elev, (max_elev - min_elev), 1.0).astype(xp.float32)
        elev_norm = ((elevations - min_elev) / elev_range).astype(xp.float32)  # (w,h)
        shade = (0.7 + 0.3 * elev_norm).astype(xp.float32)  # (w,h)

        # Vectorized texture index calculation (matches BlendTileData.get_texture)
        x = xp.arange(width, dtype=xp.int32)[:, None]
        y = xp.arange(height, dtype=xp.int32)[None, :]
        row_first = ((y % 8) // 2) * 16 + ((y % 2) * 2)
        current = ((x % 8) // 2) * 4 + (x % 2) + row_first
        texture_idx = ((tiles.astype(xp.int32) - current) // 64).astype(xp.int32)  # (w,h)

        # Precompute per-texture colors once (host), then index in a vectorized way.
        texture_colors = np.array(
            [MapVisualizer._get_texture_color(t.name) for t in blend_tile_data.textures],
            dtype=np.uint8,
        )
        if texture_colors.size == 0:
            texture_colors = np.array([(160, 140, 120)], dtype=np.uint8)

        colors_xp = xp.asarray(texture_colors)
        max_idx = colors_xp.shape[0] - 1
        idx_clipped = xp.clip(texture_idx, 0, max_idx)
        base_rgb = colors_xp[idx_clipped]  # (w,h,3) uint8

        # Apply shading
        shaded = xp.clip(
            xp.rint(base_rgb.astype(xp.float32) * shade[..., None]),
            0,
            255,
        ).astype(xp.uint8)  # (w,h,3)

        # Convert to image coordinates: (height, width, 3) with flipped Y
        image_array = shaded.transpose(1, 0, 2)[::-1, :, :]

        # Overlay impassable areas (darken)
        if blend_tile_data.passability is not None:
            passability = xp.asarray(blend_tile_data.passability)  # (w,h)
            imp_mask = (passability == int(Passability.Impassable)).T[::-1, :]  # (h,w)
            factor = xp.where(imp_mask, xp.float32(0.5), xp.float32(1.0))
            image_array = xp.clip(
                xp.rint(image_array.astype(xp.float32) * factor[..., None]),
                0,
                255,
            ).astype(xp.uint8)

        image_array = MapVisualizer._to_numpy(image_array)

        # Create PIL Image as RGBA for fast alpha overlays
        image = Image.fromarray(image_array, mode='RGB').convert('RGBA')
        draw = ImageDraw.Draw(image)
        
        # Fast alpha overlays (water + rivers)
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        if standing_water and getattr(standing_water, 'water_areas', None):
            for water_area in standing_water.water_areas:
                pts = getattr(water_area, 'points', None)
                if not pts or len(pts) < 3:
                    continue
                pix = MapVisualizer._world_points_to_pixels(pts, width, height, border)
                # Skip obviously off-image polygons quickly
                if all((x < 0 or x >= width or y < 0 or y >= height) for x, y in pix):
                    continue
                odraw.polygon(pix, fill=(0, 80, 160, 140))
            
        # Draw river areas
        if river_areas and hasattr(river_areas, 'areas'):
            for river_area in river_areas.areas:
                if hasattr(river_area, 'points') and river_area.points:
                    # Points are already tuples (x, y) in world coordinates
                    # Convert to pixel coordinates (divide by 10, world units to tiles)
                    points = MapVisualizer._world_points_to_pixels(river_area.points, width, height, border)
                    if len(points) >= 2:
                        # Draw river as lines
                        for i in range(len(points) - 1):
                            odraw.line([points[i], points[i+1]], fill=(0, 120, 220, 180), width=3)

        image = Image.alpha_composite(image, overlay)
        draw = ImageDraw.Draw(image)
        
        # Draw objects (buildings, refineries, oil derricks, etc.)
        objects_list = context.get_asset('ObjectsList')
        if objects_list and objects_list.map_objects:
            from .object_categories import ObjectCategoryConfig
            
            # Initialize object category configuration
            category_config = ObjectCategoryConfig()
            
            # World coordinates to pixel coordinates: divide by 10 (tile size)
            # Each tile is 10x10 world units
            world_to_tile = 10.0
            
            # Count objects by category for statistics
            object_counts = {}
            
            # Player start unique IDs
            player_start_ids = {'Player_1_Start', 'Player_2_Start', 'Player_3_Start', 
                               'Player_4_Start', 'Player_5_Start', 'Player_6_Start'}
            
            for obj in objects_list.map_objects:
                pos = obj.position
                
                # Convert world coordinates to pixel coordinates
                # Use the exact C# formula from MiniMapUtil.cs:
                # x = o.position.x / 10f + borderWidth
                # y = (height - 1) - o.position.y / 10f - borderWidth
                border = context.border if context.border > 0 else 0
                tile_x = int(pos[0] / world_to_tile)
                tile_y = int(pos[1] / world_to_tile)
                pixel_x = tile_x + border
                pixel_y = height - 1 - tile_y - border
                
                # Skip if outside image bounds
                if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                    continue
                
                # Check if this is a player start (check uniqueID first)
                is_player_start = False
                if hasattr(obj, 'unique_id') and obj.unique_id:
                    if obj.unique_id in player_start_ids:
                        is_player_start = True
                
                if is_player_start:
                    # Draw player start
                    player_start_category = category_config.categories.get('player_start')
                    if player_start_category and player_start_category.enabled:
                        bbox = [
                            max(0, pixel_x - player_start_category.size), max(0, pixel_y - player_start_category.size),
                            min(width - 1, pixel_x + player_start_category.size), min(height - 1, pixel_y + player_start_category.size)
                        ]
                        # Draw filled circle
                        draw.ellipse(bbox, fill=player_start_category.color, outline=(0, 0, 0), width=2)
                        # Draw a small cross in the center for better visibility
                        draw.line([pixel_x - 2, pixel_y, pixel_x + 2, pixel_y], fill=(0, 0, 0), width=1)
                        draw.line([pixel_x, pixel_y - 2, pixel_x, pixel_y + 2], fill=(0, 0, 0), width=1)
                        
                        # Count objects by category
                        object_counts[player_start_category.name] = object_counts.get(player_start_category.name, 0) + 1
                else:
                    # Get category for this object
                    category, should_draw = category_config.get_category_for_object(obj.type_name)
                    
                    if should_draw and category:
                        # Draw object as a filled circle with outline
                        bbox = [
                            max(0, pixel_x - category.size), max(0, pixel_y - category.size),
                            min(width - 1, pixel_x + category.size), min(height - 1, pixel_y + category.size)
                        ]
                        # Roads are tiny markers; the standard 2px black outline + center
                        # cross dwarf the fill and produce black-dot artifacts. Render
                        # them as solid filled circles in their own color instead.
                        if category.name == 'Road':
                            draw.ellipse(bbox, fill=category.color, outline=category.color, width=1)
                        else:
                            draw.ellipse(bbox, fill=category.color, outline=(0, 0, 0), width=2)
                            # Small cross in the center for better visibility
                            draw.line([pixel_x - 2, pixel_y, pixel_x + 2, pixel_y], fill=(0, 0, 0), width=1)
                            draw.line([pixel_x, pixel_y - 2, pixel_x, pixel_y + 2], fill=(0, 0, 0), width=1)

                        # Count objects by category
                        object_counts[category.name] = object_counts.get(category.name, 0) + 1
            
            # Print statistics
            if object_counts:
                print("\nObjects drawn on map:")
                for obj_type, count in sorted(object_counts.items()):
                    print(f"  {obj_type}: {count}")
            
            # Draw legend in top right corner (only show categories that actually exist on the map)
            if objects_list and objects_list.map_objects and object_counts:
                MapVisualizer._draw_legend(image, width, height, category_config, object_counts)
        
        # Save if path provided
        if output_path:
            image.convert('RGB').save(output_path)
            print(f"Comprehensive terrain image saved to: {output_path}")
        
        return image.convert('RGB')
    
    @staticmethod
    def _draw_legend(image: Image.Image, image_width: int, image_height: int, category_config, object_counts: dict):
        """
        Draw a legend in the top right corner showing object categories.
        Only shows categories that actually exist on the map.
        
        Args:
            draw: PIL ImageDraw object
            image_width: Width of the image
            image_height: Height of the image
            category_config: ObjectCategoryConfig instance
            object_counts: Dictionary of category names to counts (only show categories that have counts > 0)
        """
        from .object_categories import ObjectCategoryConfig
        
        # Only show categories that actually exist on the map (have objects)
        categories_to_show = []
        for cat_key, cat in category_config.categories.items():
            if cat.enabled and cat.name in object_counts and object_counts[cat.name] > 0:
                categories_to_show.append(cat)
        
        # Sort by category name for consistent ordering
        categories_to_show.sort(key=lambda c: c.name)
        
        if not categories_to_show:
            return
        
        # Legend box dimensions
        legend_padding = 10
        legend_item_height = 20
        legend_item_spacing = 5
        circle_size = 12
        text_offset_x = 25
        
        legend_width = 180
        legend_height = len(categories_to_show) * (legend_item_height + legend_item_spacing) + legend_padding * 2
        
        # Position in top right with margin
        margin = 15
        legend_x = image_width - legend_width - margin
        legend_y = margin
        
        draw = ImageDraw.Draw(image)

        # Semi-transparent background (requires RGBA image; safe even if RGB)
        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
            fill=(40, 40, 40, 160),
            outline=(255, 255, 255, 255),
            width=2,
        )
        
        # Draw legend items
        try:
            from PIL import ImageFont
            # Try to use a default font, fallback to basic if not available
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        y_pos = legend_y + legend_padding
        for category in categories_to_show:
            # Draw colored circle
            circle_x = legend_x + legend_padding
            circle_y = y_pos + (legend_item_height - circle_size) // 2
            
            # Draw filled circle with outline
            bbox = [circle_x, circle_y, circle_x + circle_size, circle_y + circle_size]
            draw.ellipse(bbox, fill=category.color, outline=(255, 255, 255), width=1)
            
            # Draw small cross in center
            center_x = circle_x + circle_size // 2
            center_y = circle_y + circle_size // 2
            draw.line([center_x - 2, center_y, center_x + 2, center_y], fill=(0, 0, 0), width=1)
            draw.line([center_x, center_y - 2, center_x, center_y + 2], fill=(0, 0, 0), width=1)
            
            # Draw text label
            text_x = circle_x + text_offset_x
            text_y = y_pos + (legend_item_height - 12) // 2
            draw.text((text_x, text_y), category.name, fill=(255, 255, 255), font=font)
            
            y_pos += legend_item_height + legend_item_spacing
    
    @staticmethod
    def create_tile_id_image(tile_data, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Create an image from tile ID data.
        
        Note: BlendTileData is currently stored as DefaultMajorAsset (raw bytes),
        so this would require full parsing. For now, returns None.
        
        Args:
            tile_data: BlendTileData asset (would need to be parsed)
            output_path: Optional path to save the image
            
        Returns:
            PIL Image or None if tile data not parsed
        """
        # TODO: Implement when BlendTileData is fully parsed
        # For now, tile data is in DefaultMajorAsset (raw bytes)
        return None
    
    @staticmethod
    def visualize_map(context, output_dir: str, map_name: str = "map", 
                     generate_heightmap: bool = False, 
                     generate_comprehensive: bool = True,
                     use_gpu: bool = False) -> dict:
        """
        Create visualizations for a map (heights, textures, comprehensive terrain)
        
        Args:
            context: MapDataContext
            output_dir: Directory to save images
            map_name: Base name for output files
            generate_heightmap: Whether to generate heightmap images (default: False)
            generate_comprehensive: Whether to generate comprehensive terrain image (default: True)
            
        Returns:
            dict with paths to generated images
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Create height map visualization (optional)
        if generate_heightmap:
            height_data = context.get_asset_by_type(HeightMapData)
            if height_data:
                # Grayscale height map
                grayscale_path = output_path / f"{map_name}_heightmap_grayscale.png"
                MapVisualizer.create_height_map_image(height_data, str(grayscale_path), colormap='grayscale', use_gpu=use_gpu)
                results['heightmap_grayscale'] = str(grayscale_path)
                
                # Terrain colormap height map
                terrain_path = output_path / f"{map_name}_heightmap_terrain.png"
                MapVisualizer.create_height_map_image(height_data, str(terrain_path), colormap='terrain', use_gpu=use_gpu)
                results['heightmap_terrain'] = str(terrain_path)
        
        # Create comprehensive terrain visualization with real textures (default: enabled)
        if generate_comprehensive:
            try:
                comprehensive_path = output_path / f"{map_name}_terrain_comprehensive.png"
                MapVisualizer.create_comprehensive_terrain_image(context, str(comprehensive_path), use_gpu=use_gpu)
                results['terrain_comprehensive'] = str(comprehensive_path)
            except Exception as e:
                # If GPU was explicitly requested, fail loudly so the caller can see the real issue.
                if use_gpu:
                    raise
                print(f"Warning: Could not create comprehensive terrain image: {e}")
        
        return results
