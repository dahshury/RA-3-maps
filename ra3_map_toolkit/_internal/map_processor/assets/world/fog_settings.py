"""
FogSettings asset
Based on FogSettings.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_FogSettings

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class FogSettings(MajorAsset):
    """
    Fog settings asset.
    Based on FogSettings.cs from MapCreatorCore
    """
    
    def __init__(self):
        super().__init__()
        self.enabled: bool = True
        self.start: float = 0.0
        self.end: float = 20.0
        self.r: float = 0.5
        self.g: float = 0.5
        self.b: float = 0.5
    
    def get_asset_name(self) -> str:
        return ASSET_FogSettings
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse fog settings data.
        Based on FogSettings(BinaryReader br) in FogSettings.cs
        """
        enabled_int = struct.unpack('<i', br.read(4))[0]
        self.enabled = enabled_int == 1
        self.start = struct.unpack('<f', br.read(4))[0]
        self.end = struct.unpack('<f', br.read(4))[0]
        self.r = struct.unpack('<f', br.read(4))[0]
        self.g = struct.unpack('<f', br.read(4))[0]
        self.b = struct.unpack('<f', br.read(4))[0]
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save fog settings data.
        Based on SaveData in FogSettings.cs
        """
        bw.write(struct.pack('<i', 1 if self.enabled else 0))
        bw.write(struct.pack('<f', self.start))
        bw.write(struct.pack('<f', self.end))
        bw.write(struct.pack('<f', self.r))
        bw.write(struct.pack('<f', self.g))
        bw.write(struct.pack('<f', self.b))

