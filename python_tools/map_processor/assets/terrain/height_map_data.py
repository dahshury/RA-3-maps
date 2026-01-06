"""
HeightMapData asset
Based on HeightMapData.cs
"""
import struct
import numpy as np
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_HeightMapData
from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class HeightMapBorder:
    """Height map border (simplified - full implementation would parse all fields)"""
    
    def __init__(self):
        self.x: int = 0
        self.y: int = 0
        self.width: int = 0
        self.height: int = 0
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'HeightMapBorder':
        # Simplified - would need full HeightMapBorder.cs implementation
        # For now, just read basic structure
        self.x = struct.unpack('<i', br.read(4))[0]
        self.y = struct.unpack('<i', br.read(4))[0]
        self.width = struct.unpack('<i', br.read(4))[0]
        self.height = struct.unpack('<i', br.read(4))[0]
        # Skip remaining fields for now
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        # Simplified - would need full implementation
        bw.write(struct.pack('<i', self.x))
        bw.write(struct.pack('<i', self.y))
        bw.write(struct.pack('<i', self.width))
        bw.write(struct.pack('<i', self.height))


class HeightMapData(MajorAsset):
    """
    Height map data asset.
    Based on HeightMapData.cs
    """
    
    def __init__(self):
        super().__init__()
        self._elevations_raw: np.ndarray = None  # uint16 array [mapWidth, mapHeight] - original SageFloat16 values for bit-perfect preservation
        self.elevations: np.ndarray = None  # float array [mapWidth, mapHeight] - computed from raw values
        self.borders: List[HeightMapBorder] = []
        self.map_width: int = 0
        self.map_height: int = 0
        self.border_width: int = 0
        self.playable_width: int = 0
        self.playable_height: int = 0
        self.area: int = 0
    
    def get_asset_name(self) -> str:
        return ASSET_HeightMapData
    
    def get_version(self) -> int:
        return 6
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse height map data.
        Based on parseData in HeightMapData.cs
        """
        # Read dimensions
        self.map_width = struct.unpack('<i', br.read(4))[0]
        self.map_height = struct.unpack('<i', br.read(4))[0]
        self.border_width = struct.unpack('<i', br.read(4))[0]
        
        # Update context
        context.map_width = self.map_width
        context.map_height = self.map_height
        context.border = self.border_width
        
        # Calculate playable area
        self.playable_width = self.map_width - 2 * self.border_width
        self.playable_height = self.map_height - 2 * self.border_width
        
        # Read borders
        border_count = struct.unpack('<i', br.read(4))[0]
        self.borders = []
        for i in range(border_count):
            border = HeightMapBorder()
            border.from_stream(br, context)
            self.borders.append(border)
        
        # Read area
        self.area = struct.unpack('<i', br.read(4))[0]
        
        # Read elevations (SageFloat16 format)
        # Store original uint16 values for bit-perfect preservation
        self._elevations_raw = np.zeros((self.map_width, self.map_height), dtype=np.uint16)
        self.elevations = np.zeros((self.map_width, self.map_height), dtype=np.float32)
        for y in range(self.map_height):
            for x in range(self.map_width):
                value_uint16 = struct.unpack('<H', br.read(2))[0]
                self._elevations_raw[x, y] = value_uint16  # Store original for bit-perfect save
                self.elevations[x, y] = BinaryUtils.from_sage_float16(value_uint16)  # Convert to float for use
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save height map data.
        Based on saveData in HeightMapData.cs
        """
        # Write dimensions
        bw.write(struct.pack('<i', self.map_width))
        bw.write(struct.pack('<i', self.map_height))
        bw.write(struct.pack('<i', self.border_width))
        
        # Write borders
        bw.write(struct.pack('<i', len(self.borders)))
        for border in self.borders:
            border.save_data(bw, context)
        
        # Write area
        bw.write(struct.pack('<i', self.area))
        
        # Write elevations (SageFloat16 format)
        # Use original uint16 values for bit-perfect preservation
        for y in range(self.map_height):
            for x in range(self.map_width):
                if self._elevations_raw is not None:
                    # Use original raw value for bit-perfect reconstruction
                    value_uint16 = self._elevations_raw[x, y]
                else:
                    # Fallback: convert from float (for newly created maps)
                    value_uint16 = BinaryUtils.to_sage_float16(float(self.elevations[x, y]))
                bw.write(struct.pack('<H', value_uint16))

