"""
EnvironmentData asset
Based on EnvironmentData.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_EnvironmentData
from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class EnvironmentData(MajorAsset):
    """
    Environment data asset.
    Based on EnvironmentData.cs from MapCreatorCore
    """
    
    def __init__(self):
        super().__init__()
        self.water_max_alpha_depth: float = 20.0
        self.water_max_alpha: float = 255.0
        self.macro_texture: str = "TSNoiseUrb"
        self.cloud_texture: str = "TSCloudMed"
        self.environment_map: str = "EVDefault"
    
    def get_asset_name(self) -> str:
        return ASSET_EnvironmentData
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse environment data.
        Based on EnvironmentData(BinaryReader br) in EnvironmentData.cs
        Note: C# code uses IOUtility.ReadString which is readDefaultString
        """
        self.water_max_alpha_depth = struct.unpack('<f', br.read(4))[0]
        self.water_max_alpha = struct.unpack('<f', br.read(4))[0] * 255.0
        self.macro_texture = BinaryUtils.read_string_default(br)
        self.cloud_texture = BinaryUtils.read_string_default(br)
        self.environment_map = BinaryUtils.read_string_default(br)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save environment data.
        Based on SaveData in EnvironmentData.cs
        """
        bw.write(struct.pack('<f', self.water_max_alpha_depth))
        bw.write(struct.pack('<f', self.water_max_alpha / 255.0))
        BinaryUtils.write_string_default(bw, self.macro_texture)
        BinaryUtils.write_string_default(bw, self.cloud_texture)
        BinaryUtils.write_string_default(bw, self.environment_map)

