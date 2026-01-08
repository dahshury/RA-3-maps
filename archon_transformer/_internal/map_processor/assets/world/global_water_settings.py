"""
GlobalWaterSettings asset
Based on GlobalWaterSettings.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_GlobalWaterSettings

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class GlobalWaterSettings(MajorAsset):
    """
    Global water settings asset.
    Based on GlobalWaterSettings.cs
    """
    
    def __init__(self):
        super().__init__()
        self.reflection: bool = True
        self.reflection_plane_z: float = 200.0
    
    def get_asset_name(self) -> str:
        return ASSET_GlobalWaterSettings
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse global water settings data.
        Based on saveData pattern (reverse engineering - C# only has saveData)
        """
        reflection_int = struct.unpack('<i', br.read(4))[0]
        self.reflection = reflection_int != 0
        self.reflection_plane_z = struct.unpack('<f', br.read(4))[0]
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save global water settings data.
        Based on saveData in GlobalWaterSettings.cs
        """
        bw.write(struct.pack('<i', 1 if self.reflection else 0))
        bw.write(struct.pack('<f', self.reflection_plane_z))

