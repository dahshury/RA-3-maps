"""
StandingWaveAreas asset
Based on StandingWaveAreas.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_StandingWaveAreas
from .standing_wave_area import StandingWaveArea

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class StandingWaveAreas(MajorAsset):
    """
    Standing wave areas asset.
    Based on StandingWaveAreas.cs from MapCreatorCore
    """
    
    def __init__(self):
        super().__init__()
        self.areas: List[StandingWaveArea] = []
    
    def get_asset_name(self) -> str:
        return ASSET_StandingWaveAreas
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse standing wave areas data.
        Based on StandingWaveAreas(BinaryReader br) in StandingWaveAreas.cs
        """
        areas_count = struct.unpack('<i', br.read(4))[0]
        self.areas = []
        for i in range(areas_count):
            area = StandingWaveArea()
            area.from_stream(br, context)
            self.areas.append(area)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save standing wave areas data.
        Based on SaveData in StandingWaveAreas.cs
        """
        bw.write(struct.pack('<i', len(self.areas)))
        for area in self.areas:
            area.save_data(bw, context)

