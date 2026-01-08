"""
BlendInfo class
Based on BlendInfo.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ..terrain.blend_direction import BlendDirection

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class BlendInfo:
    """
    Blend information (not a MajorAsset, used within BlendTileData).
    Based on BlendInfo.cs
    """
    
    def __init__(self):
        self.secondary_texture_tile: int = 0
        self.i3: int = 0
        self.i4: int = 0
        self.blend_direction: BlendDirection = BlendDirection(0)
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'BlendInfo':
        """
        Parse blend info from stream.
        Based on fromStream in BlendInfo.cs
        """
        self.secondary_texture_tile = struct.unpack('<i', br.read(4))[0]
        
        # Read 6 bytes for blend direction
        # Preserve original bytes for bit-perfect reconstruction (bytes can be > 1)
        bytes_data = br.read(6)
        self._blend_direction_raw = bytes_data  # Store original bytes
        self.blend_direction = self._to_blend_direction(bytes_data)
        
        self.i3 = struct.unpack('<I', br.read(4))[0]
        self.i4 = struct.unpack('<I', br.read(4))[0]
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save blend info data.
        Based on saveData in BlendInfo.cs
        """
        bw.write(struct.pack('<i', self.secondary_texture_tile))
        # Use original raw bytes for bit-perfect reconstruction (preserves bytes > 1)
        if hasattr(self, '_blend_direction_raw'):
            bw.write(self._blend_direction_raw)
        else:
            # Fallback to conversion if raw bytes not available
            bytes_data = self._from_blend_direction(self.blend_direction)
            bw.write(bytes_data)
        bw.write(struct.pack('<I', self.i3))
        bw.write(struct.pack('<I', self.i4))
    
    def _to_blend_direction(self, bytes_data: bytes) -> BlendDirection:
        """
        Convert 6 bytes to BlendDirection.
        Based on ToBlendDirection in BlendInfo.cs
        """
        value = 0
        for i in range(len(bytes_data)):
            if bytes_data[i] == 1:
                value |= (1 << i)
        return BlendDirection(value)
    
    def _from_blend_direction(self, bd: BlendDirection) -> bytes:
        """
        Convert BlendDirection to 6 bytes.
        Based on FromBlendDirection in BlendInfo.cs
        """
        bytes_data = bytearray(6)
        value = int(bd)
        for i in range(6):
            if (value & (1 << i)) != 0:
                bytes_data[i] = 1
        return bytes(bytes_data)

