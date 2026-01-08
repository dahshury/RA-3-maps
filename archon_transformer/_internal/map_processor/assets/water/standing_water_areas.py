"""
StandingWaterAreas asset
Based on StandingWaterAreas.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_StandingWaterAreas
from .standing_water_area import StandingWaterArea

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class StandingWaterAreas(MajorAsset):
    """
    Standing water areas list asset.
    Based on StandingWaterAreas.cs
    """
    
    def __init__(self):
        super().__init__()
        self.water_areas: List[StandingWaterArea] = []
    
    def get_asset_name(self) -> str:
        return ASSET_StandingWaterAreas
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse standing water areas data.
        Based on parseData in StandingWaterAreas.cs
        """
        import struct
        length = struct.unpack('<i', br.read(4))[0]
        self.water_areas = []
        for i in range(length):
            water_area = StandingWaterArea()
            water_area.from_stream(br, context)
            self.water_areas.append(water_area)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save standing water areas data.
        Based on saveData in StandingWaterAreas.cs
        """
        import struct
        bw.write(struct.pack('<i', len(self.water_areas)))
        for water_area in self.water_areas:
            water_area.save_data(bw, context)

