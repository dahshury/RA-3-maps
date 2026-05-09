"""
Texture asset
Based on Texture.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Texture:
    """
    Texture information (not a MajorAsset, used within BlendTileData).
    Based on Texture.cs
    """
    
    def __init__(self):
        self.cell_start: int = 0
        self.cell_count: int = 0
        self.cell_size: int = 0
        self.magic_value: int = 0  # Must be 0
        self.name: str = ""
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'Texture':
        """
        Parse texture from stream.
        Based on fromStream in Texture.cs
        """
        self.cell_start = struct.unpack('<i', br.read(4))[0]
        self.cell_count = struct.unpack('<i', br.read(4))[0]
        self.cell_size = struct.unpack('<i', br.read(4))[0]
        self.magic_value = struct.unpack('<i', br.read(4))[0]
        self.name = BinaryUtils.read_string_default(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save texture data.
        Based on saveData in Texture.cs
        """
        bw.write(struct.pack('<i', self.cell_start))
        bw.write(struct.pack('<i', self.cell_count))
        bw.write(struct.pack('<i', self.cell_size))
        bw.write(struct.pack('<i', self.magic_value))
        BinaryUtils.write_string_default(bw, self.name)
    
    @staticmethod
    def new_instance(start: int, name: str) -> 'Texture':
        """
        Create new texture instance.
        Based on newInstance in Texture.cs
        """
        texture = Texture()
        texture.cell_start = start
        texture.cell_count = 16
        texture.cell_size = 4
        texture.magic_value = 0
        texture.name = name
        return texture

