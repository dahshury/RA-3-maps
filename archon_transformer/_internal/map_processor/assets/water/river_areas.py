"""
RiverAreas asset
Based on RiverAreas.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_RiverAreas
from .river_area import RiverArea

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class RiverAreas(MajorAsset):
    """
    River areas asset.
    Based on RiverAreas.cs from MapCreatorCore
    """
    
    def __init__(self):
        super().__init__()
        self.areas: List[RiverArea] = []
    
    def get_asset_name(self) -> str:
        return ASSET_RiverAreas
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse river areas data.
        Based on RiverAreas(BinaryReader br) in RiverAreas.cs
        """
        areas_count = struct.unpack('<i', br.read(4))[0]
        self.areas = []
        for i in range(areas_count):
            area = RiverArea()
            area.from_stream(br, context)
            self.areas.append(area)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save river areas data.
        Based on SaveData in RiverAreas.cs
        """
        bw.write(struct.pack('<i', len(self.areas)))
        for area in self.areas:
            area.save_data(bw, context)

