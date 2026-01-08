"""
Test map visualization
"""
import pytest
import tempfile
from pathlib import Path
from PIL import Image

from map_processor.ra3map import Ra3Map
from map_processor.map_visualizer import MapVisualizer
from map_processor.height_map_data import HeightMapData


def test_create_height_map_image():
    """Test creating height map image from map data"""
    # Find test map
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    map_file = maps_dir / "2 II" / "map_mp_2_rao1.map"
    
    if not map_file.exists():
        pytest.skip(f"Map file not found: {map_file}")
    
    # Parse map
    ra3map = Ra3Map(str(map_file))
    ra3map.parse()
    context = ra3map.get_context()
    
    # Get height data
    height_data = context.get_asset_by_type(HeightMapData)
    assert height_data is not None, "HeightMapData should exist"
    
    # Create visualization
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test grayscale
        grayscale_path = Path(tmpdir) / "heightmap_grayscale.png"
        image = MapVisualizer.create_height_map_image(height_data, str(grayscale_path), colormap='grayscale')
        
        assert image is not None
        assert grayscale_path.exists()
        assert image.size == (height_data.map_width, height_data.map_height)
        assert image.mode == 'L'  # Grayscale
        
        # Test terrain colormap
        terrain_path = Path(tmpdir) / "heightmap_terrain.png"
        image_terrain = MapVisualizer.create_height_map_image(height_data, str(terrain_path), colormap='terrain')
        
        assert image_terrain is not None
        assert terrain_path.exists()
        assert image_terrain.size == (height_data.map_width, height_data.map_height)
        assert image_terrain.mode == 'RGB'  # Color
        
        print(f"\nCreated visualizations:")
        print(f"  Grayscale: {grayscale_path}")
        print(f"  Terrain: {terrain_path}")
        print(f"  Image size: {image.size}")
        print(f"  Height data size: {height_data.map_width}x{height_data.map_height}")


def test_visualize_map():
    """Test full map visualization"""
    # Find test map
    maps_dir = Path(__file__).parent.parent.parent / "RA3 Official maps"
    map_file = maps_dir / "2 II" / "map_mp_2_rao1.map"
    
    if not map_file.exists():
        pytest.skip(f"Map file not found: {map_file}")
    
    # Parse map
    ra3map = Ra3Map(str(map_file))
    ra3map.parse()
    context = ra3map.get_context()
    
    # Create visualizations
    with tempfile.TemporaryDirectory() as tmpdir:
        results = MapVisualizer.visualize_map(context, tmpdir, "test_map")
        
        assert 'heightmap_grayscale' in results
        assert 'heightmap_terrain' in results
        
        # Verify files exist
        assert Path(results['heightmap_grayscale']).exists()
        assert Path(results['heightmap_terrain']).exists()
        
        print(f"\nVisualization results: {results}")











