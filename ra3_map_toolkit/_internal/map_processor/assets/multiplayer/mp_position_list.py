"""
MPPositionList asset
Based on MPPositionList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_MPPositionList
from .mp_position_info import MPPositionInfo

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MPPositionList(MajorAsset):
    """
    Multiplayer position list asset.
    Based on MPPositionList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.positions: List[MPPositionInfo] = []
    
    def get_asset_name(self) -> str:
        return ASSET_MPPositionList
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse MP position list data.
        Based on parseData in MPPositionList.cs
        """
        # Fixed array of 6 MPPositionInfo (each is a MajorAsset)
        self.positions = []
        for i in range(6):
            position_info = MPPositionInfo()
            position_info.from_stream(br, context)
            self.positions.append(position_info)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save MP position list data.
        Based on saveData in MPPositionList.cs
        """
        # Note: C# code calls base.saveData which does nothing, but we need to save positions
        for position_info in self.positions:
            position_info.save(bw, context)

