"""
Utility functions for map processing
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional


def find_map_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Find all .map files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of paths to .map files
    """
    directory_path = Path(directory)
    pattern = "**/*.map" if recursive else "*.map"
    
    map_files = list(directory_path.glob(pattern))
    return [str(f) for f in map_files]


def infer_player_count_from_path(map_path: str) -> Optional[int]:
    """
    Infer player count from map file path.
    
    Maps are often in directories named like "2 CE", "3 Caledra", "4 Blitzen", etc.
    
    Args:
        map_path: Path to the map file
        
    Returns:
        Player count (2-6) or None if cannot be determined
    """
    path = Path(map_path)
    
    # Check parent directory name
    parent_name = path.parent.name
    
    # Look for patterns like "2 CE", "3 Caledra", etc.
    parts = parent_name.split()
    if parts and parts[0].isdigit():
        count = int(parts[0])
        if 2 <= count <= 6:
            return count
    
    # Check if parent's parent has number (e.g., "BattleBase/2 BBB")
    grandparent_name = path.parent.parent.name if path.parent.parent != path.parent else None
    if grandparent_name:
        parts = grandparent_name.split()
        if parts and parts[0].isdigit():
            count = int(parts[0])
            if 2 <= count <= 6:
                return count
    
    return None


def get_map_info(map_path: str) -> dict:
    """
    Get basic information about a map file.
    
    Args:
        map_path: Path to the map file
        
    Returns:
        Dictionary with map information
    """
    path = Path(map_path)
    
    return {
        "path": str(path),
        "name": path.stem,
        "directory": str(path.parent),
        "size": path.stat().st_size if path.exists() else 0,
        "player_count": infer_player_count_from_path(map_path)
    }


def compare_maps(original_data: dict, reconstructed_data: dict, tolerance: float = 1e-5) -> dict:
    """
    Compare two parsed map data structures.
    
    Args:
        original_data: Original map data
        reconstructed_data: Reconstructed map data
        tolerance: Tolerance for float comparisons
        
    Returns:
        Dictionary with comparison results
    """
    differences = []
    
    # Compare basic metadata
    for key in ["mapWidth", "mapHeight", "borderWidth"]:
        if key in original_data and key in reconstructed_data:
            if original_data[key] != reconstructed_data[key]:
                differences.append(f"{key}: {original_data[key]} != {reconstructed_data[key]}")
    
    # Compare height maps (if available)
    if "heightMap" in original_data and "heightMap" in reconstructed_data:
        import numpy as np
        
        orig_heights = np.array(original_data["heightMap"])
        recon_heights = np.array(reconstructed_data["heightMap"])
        
        if orig_heights.shape != recon_heights.shape:
            differences.append(f"Height map shape mismatch: {orig_heights.shape} != {recon_heights.shape}")
        else:
            max_diff = np.abs(orig_heights - recon_heights).max()
            if max_diff > tolerance:
                differences.append(f"Height map max difference: {max_diff}")
    
    return {
        "identical": len(differences) == 0,
        "differences": differences,
        "difference_count": len(differences)
    }

