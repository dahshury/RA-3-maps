"""
Tests for map parsing functionality
"""
import pytest
import json
import os
from pathlib import Path


def test_find_map_files(maps_directory):
    """Test finding map files in directory"""
    from map_processor.utils import find_map_files
    
    map_files = find_map_files(maps_directory)
    
    assert len(map_files) > 0, "Should find at least one map file"
    assert all(f.endswith('.map') for f in map_files), "All files should be .map files"
    assert all(os.path.exists(f) for f in map_files), "All files should exist"


def test_infer_player_count_from_path():
    """Test inferring player count from path"""
    from map_processor.utils import infer_player_count_from_path
    
    # Test cases
    test_cases = [
        ("RA 3 maps/RA3 Official maps/2 CE/map.map", 2),
        ("RA 3 maps/RA3 Official maps/3 Caledra of Chaos/map.map", 3),
        ("RA 3 maps/RA3 Official maps/4 Blitzen's Back/map.map", 4),
        ("RA 3 maps/RA3 Official maps/5 Circus Maximus/map.map", 5),
        ("RA 3 maps/RA3 Official maps/6 Burnt out Paradise/map.map", 6),
        ("RA 3 maps/RA3 Official maps/BattleBase/2 BBB/map.map", 2),
    ]
    
    for path, expected_count in test_cases:
        result = infer_player_count_from_path(path)
        assert result == expected_count, f"Failed for path: {path}"


def test_get_map_info(sample_map_file):
    """Test getting map information"""
    from map_processor.utils import get_map_info
    
    info = get_map_info(sample_map_file)
    
    assert "path" in info
    assert "name" in info
    assert "size" in info
    assert info["path"] == sample_map_file
    assert os.path.exists(info["path"])


def test_parse_map_python(python_processor, sample_map_file):
    """Test parsing a map using pure Python implementation"""
    ra3map = python_processor(sample_map_file)
    ra3map.parse()
    
    context = ra3map.get_context()
    
    # Verify context was created
    assert context is not None, "Context should be created"
    assert context.map_struct is not None, "MapStruct should exist"
    
    # Verify basic map properties
    assert context.map_width > 0, "Map width should be positive"
    assert context.map_height > 0, "Map height should be positive"
    
    print(f"Parsed map: {context.map_name}")
    print(f"Size: {context.map_width}x{context.map_height}")
    print(f"Assets: {len(context.map_struct.assets)}")


def test_parse_map_height_data(python_processor, sample_map_file):
    """Test parsing HeightMapData from a map"""
    from map_processor.height_map_data import HeightMapData
    
    ra3map = python_processor(sample_map_file)
    ra3map.parse()
    
    context = ra3map.get_context()
    
    # Get HeightMapData
    height_map = context.get_asset_by_type(HeightMapData)
    
    if height_map is None:
        pytest.skip("HeightMapData not found in this map (may be using DefaultMajorAsset)")
    
    # Verify height map data
    assert height_map.elevations is not None, "Elevations should exist"
    assert height_map.map_width > 0, "Map width should be positive"
    assert height_map.map_height > 0, "Map height should be positive"
    assert height_map.elevations.shape == (height_map.map_width, height_map.map_height), \
        "Elevations shape should match map dimensions"
    
    print(f"Height map: {height_map.map_width}x{height_map.map_height}")
    print(f"Elevation range: {height_map.elevations.min():.2f} - {height_map.elevations.max():.2f}")


def test_parse_multiple_maps_python(python_processor, all_map_files):
    """Test parsing multiple maps"""
    if len(all_map_files) == 0:
        pytest.skip("No map files available")
    
    # Test first 5 maps to keep test fast
    test_files = all_map_files[:5]
    
    success_count = 0
    for map_file in test_files:
        try:
            ra3map = python_processor(map_file)
            ra3map.parse()
            context = ra3map.get_context()
            assert context.map_width > 0
            assert context.map_height > 0
            success_count += 1
            print(f"✓ Parsed: {Path(map_file).name}")
        except Exception as e:
            print(f"✗ Failed to parse {Path(map_file).name}: {e}")
    
    # At least some maps should parse successfully
    assert success_count > 0, "Should parse at least some maps"


@pytest.mark.parametrize("map_index", [0, 1, 2])
def test_parse_map_parametrized_python(python_processor, all_map_files, map_index):
    """Parametrized test for parsing maps"""
    if len(all_map_files) <= map_index:
        pytest.skip(f"Not enough map files (need at least {map_index + 1})")
    
    map_file = all_map_files[map_index]
    
    ra3map = python_processor(map_file)
    ra3map.parse()
    context = ra3map.get_context()
    
    assert context.map_width > 0
    assert context.map_height > 0

