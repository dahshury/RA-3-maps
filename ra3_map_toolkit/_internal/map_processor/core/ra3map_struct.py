"""
RA3 Map Structure - Core data structures
Based on Ra3MapStruct.cs and MapDataContext.cs
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .major_asset import MajorAsset


@dataclass
class MapDataContext:
    """
    Context holding all map data.
    Based on MapDataContext.cs
    """
    map_struct: 'Ra3MapStruct'
    map_width: int = -1
    map_height: int = -1
    border: int = -1
    map_name: str = ""
    extra: Dict[Any, Any] = field(default_factory=dict)
    # Store excluded assets (for reconstruction)
    excluded_assets: Dict[str, Any] = field(default_factory=dict)
    
    def get_asset(self, asset_name: str) -> Optional['MajorAsset']:
        """Get an asset by name"""
        return self.map_struct.get_asset_by_name(asset_name)
    
    def get_asset_by_type(self, asset_type: type) -> Optional['MajorAsset']:
        """Get first asset of given type"""
        for asset in self.map_struct.assets:
            if isinstance(asset, asset_type):
                return asset
        return None


class Ra3MapStruct:
    """
    Map structure container with string pool and assets.
    Based on Ra3MapStruct.cs
    """
    
    def __init__(self):
        self.string_pool: Dict[str, int] = {}  # string -> index
        self.index_to_string: Dict[int, str] = {}  # index -> string
        self.assets: List['MajorAsset'] = []
    
    def register_string(self, s: str, index: Optional[int] = None) -> int:
        """
        Register a string in the string pool.
        
        Args:
            s: String to register
            index: Optional index (if None, uses next available index)
            
        Returns:
            Index of the string
        """
        if s in self.string_pool:
            return self.string_pool[s]
        
        if index is None:
            # Use max existing index + 1 to avoid collisions
            # (don't use len() because indices may have gaps)
            if self.index_to_string:
                index = max(self.index_to_string.keys()) + 1
            else:
                index = 0
        
        self.string_pool[s] = index
        self.index_to_string[index] = s
        return index
    
    def get_string_index(self, s: str) -> int:
        """Get index for a string, or -1 if not found"""
        return self.string_pool.get(s, -1)
    
    def find_string_by_index(self, index: int) -> Optional[str]:
        """Find string by index, or None if not found"""
        return self.index_to_string.get(index)
    
    def add_asset(self, asset: 'MajorAsset') -> None:
        """Add an asset to the map structure"""
        self.assets.append(asset)
    
    def get_asset_by_name(self, name: str) -> Optional['MajorAsset']:
        """Get asset by name"""
        for asset in self.assets:
            if asset.get_asset_name() == name:
                return asset
        return None

