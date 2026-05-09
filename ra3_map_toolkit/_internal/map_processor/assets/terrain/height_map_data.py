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
        nbytes = self.map_width * self.map_height * 2
        data = br.read(nbytes)
        if len(data) != nbytes:
            raise EOFError("Unexpected end of stream while reading HeightMapData elevations")

        # File order is row-major by Y then X. Internally we store arrays as [x, y].
        raw_yx = np.frombuffer(data, dtype='<u2').reshape((self.map_height, self.map_width))
        self._elevations_raw = raw_yx.T.copy()

        # SageFloat16 is a CUSTOM format (NOT IEEE float16):
        # height = upper_byte * 10.0 + lower_byte * 9.96 / 256.0
        upper = (raw_yx >> 8).astype(np.float32)
        lower = (raw_yx & 0xFF).astype(np.float32)
        elev_yx = upper * 10.0 + lower * 9.96 / 256.0
        self.elevations = elev_yx.T.copy()
    
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
        if self._elevations_raw is not None:
            raw_yx = np.asarray(self._elevations_raw, dtype=np.uint16).T  # (height, width)
            bw.write(raw_yx.astype('<u2', copy=False).tobytes(order='C'))
            return

        # Fallback: convert from float (for newly created maps)
        # Convert to IEEE float16 then write raw bits, row-major by Y then X.
        elev_xy = np.asarray(self.elevations, dtype=np.float32)
        elev_yx_f16 = elev_xy.T.astype('<f2', copy=False)  # (height, width)
        bw.write(elev_yx_f16.view('<u2').tobytes(order='C'))

