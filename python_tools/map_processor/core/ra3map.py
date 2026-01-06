"""
RA3Map - Main API for working with RA3 maps
Based on Ra3Map.cs
"""
from pathlib import Path
from typing import Optional

from .ra3map_struct import Ra3MapStruct, MapDataContext
from ..parsing.map_parser import Ra3MapParser
from ..parsing.map_reconstructor import Ra3MapReconstructor


class Ra3Map:
    """
    Main class for working with RA3 maps.
    Based on Ra3Map.cs
    """
    
    def __init__(self, map_path: str):
        """
        Initialize Ra3Map.
        
        Args:
            map_path: Path to the map file
        """
        self.map_path = map_path
        self.context: Optional[MapDataContext] = None
        self._parser = Ra3MapParser()
        self._reconstructor = Ra3MapReconstructor()
    
    def parse(self) -> None:
        """
        Parse the map file.
        Based on parse method in Ra3Map.cs
        """
        self.context = self._parser.parse(self.map_path)
        self.context.map_name = Path(self.map_path).stem
    
    def get_context(self) -> MapDataContext:
        """Get the map data context"""
        if self.context is None:
            raise ValueError("Map must be parsed first. Call parse() before get_context()")
        return self.context
    
    def save(self, output_path: str, compress: bool = True) -> None:
        """
        Save the map to a file.
        Based on save method in Ra3Map.cs
        
        Args:
            output_path: Path to save the map file
            compress: Whether to compress the output
        """
        if self.context is None:
            raise ValueError("Map must be parsed first. Call parse() before save()")
        
        self._reconstructor.save_map(self.context, output_path, compress)

