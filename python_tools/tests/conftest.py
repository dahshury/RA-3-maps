"""
Pytest configuration and fixtures
"""
import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def maps_directory():
    """Fixture providing the path to the RA 3 maps directory"""
    current_file = Path(__file__)
    # Look for maps directory - could be "RA 3 maps" or "RA3 Official maps"
    possible_dirs = [
        current_file.parent.parent.parent / "RA 3 maps",
        current_file.parent.parent.parent / "RA3 Official maps",
        current_file.parent.parent.parent.parent / "RA 3 maps",
    ]
    
    for maps_dir in possible_dirs:
        if maps_dir.exists():
            # Find .map files in subdirectories
            map_files = list(maps_dir.rglob("*.map"))
            if map_files:
                return str(maps_dir)
    
    pytest.skip(f"Maps directory not found. Tried: {[str(d) for d in possible_dirs]}")


@pytest.fixture
def sample_map_file(maps_directory):
    """Fixture providing a sample map file for testing"""
    from map_processor.utils import find_map_files
    
    map_files = find_map_files(maps_directory)
    
    if not map_files:
        pytest.skip("No map files found in maps directory")
    
    return map_files[0]


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def python_processor():
    """
    Fixture providing a Python map processor instance.
    """
    from map_processor.ra3map import Ra3Map
    return Ra3Map


@pytest.fixture
def all_map_files(maps_directory):
    """Fixture providing all map files in the directory"""
    from map_processor.utils import find_map_files
    
    map_files = find_map_files(maps_directory)
    
    if not map_files:
        pytest.skip("No map files found")
    
    return map_files

