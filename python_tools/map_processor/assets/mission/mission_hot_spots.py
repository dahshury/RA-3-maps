"""
MissionHotSpots asset
Based on MissionHotSpots.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_MissionHotSpots
from .mission_hot_spot import MissionHotSpot

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MissionHotSpots(MajorAsset):
    """
    Mission hot spots asset.
    Based on MissionHotSpots.cs from MapCreatorCore
    """
    
    def __init__(self):
        super().__init__()
        self.spots: List[MissionHotSpot] = []
    
    def get_asset_name(self) -> str:
        return ASSET_MissionHotSpots
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse mission hot spots data.
        Based on MissionHotSpots(BinaryReader br) in MissionHotSpots.cs
        """
        spots_count = struct.unpack('<i', br.read(4))[0]
        self.spots = []
        for i in range(spots_count):
            spot = MissionHotSpot()
            spot.from_stream(br, context)
            self.spots.append(spot)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save mission hot spots data.
        Based on SaveData in MissionHotSpots.cs
        """
        bw.write(struct.pack('<i', len(self.spots)))
        for spot in self.spots:
            spot.save_data(bw, context)

