"""
Map visualization utilities
Create images from map data (heights, tiles, etc.)
"""
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from pathlib import Path

from ..assets.terrain.height_map_data import HeightMapData


class MapVisualizer:
    """
    Create visualizations from RA3 map data
    """
    
    @staticmethod
    def create_height_map_image(height_data: HeightMapData, output_path: Optional[str] = None, 
                                colormap: str = 'grayscale') -> Image.Image:
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
        
        elevations = height_data.elevations
        width, height = height_data.map_width, height_data.map_height
        
        # Normalize elevations to 0-255 range
        min_elev = np.min(elevations)
        max_elev = np.max(elevations)
        elev_range = max_elev - min_elev
        
        if elev_range == 0:
            # Flat map
            normalized = np.zeros((height, width), dtype=np.uint8)
        else:
            normalized = ((elevations - min_elev) / elev_range * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == 'grayscale':
            # Simple grayscale
            image_array = normalized.T  # Transpose for correct orientation (x,y -> row,col)
            
        elif colormap == 'terrain' or colormap == 'elevation':
            # Terrain-like colormap (green for low, brown/gray for high)
            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create a simple terrain colormap
            # Low (water): Blue
            # Mid (land): Green to Yellow
            # High (mountain): Brown to Gray
            # Note: normalized is [width, height], need to transpose for image
            normalized_float = normalized.astype(np.float32) / 255.0
            image_array = np.zeros((height, width, 3), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    val = normalized_float[x, y]  # Note: x,y indexing for normalized array
                    if val < 0.3:
                        # Water/Low: Blue
                        image_array[y, x] = [0, int(100 * val / 0.3), int(200 * val / 0.3)]
                    elif val < 0.6:
                        # Land: Green to Yellow
                        t = (val - 0.3) / 0.3
                        image_array[y, x] = [int(50 * t), int(150 + 105 * t), int(50 * (1 - t))]
                    else:
                        # Mountain: Brown to Gray
                        t = (val - 0.6) / 0.4
                        image_array[y, x] = [
                            int(100 + 155 * t),
                            int(80 + 175 * t),
                            int(50 + 205 * t)
                        ]
            
        else:
            raise ValueError(f"Unknown colormap: {colormap}")
        
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
    def visualize_map(context, output_dir: str, map_name: str = "map") -> dict:
        """
        Create visualizations for a map (heights, etc.)
        
        Args:
            context: MapDataContext
            output_dir: Directory to save images
            map_name: Base name for output files
            
        Returns:
            dict with paths to generated images
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Create height map visualization
        height_data = context.get_asset_by_type(HeightMapData)
        if height_data:
            # Grayscale height map
            grayscale_path = output_path / f"{map_name}_heightmap_grayscale.png"
            MapVisualizer.create_height_map_image(height_data, str(grayscale_path), colormap='grayscale')
            results['heightmap_grayscale'] = str(grayscale_path)
            
            # Terrain colormap height map
            terrain_path = output_path / f"{map_name}_heightmap_terrain.png"
            MapVisualizer.create_height_map_image(height_data, str(terrain_path), colormap='terrain')
            results['heightmap_terrain'] = str(terrain_path)
        
        return results

