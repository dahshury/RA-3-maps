"""
WorldInfo asset
Based on WorldInfo.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_WorldInfo
from ..assets.asset_property import AssetPropertyCollection

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class WorldInfo(MajorAsset):
    """
    World information asset (map metadata, properties, etc.).
    Based on WorldInfo.cs
    """
    
    def __init__(self):
        super().__init__()
        self.properties: AssetPropertyCollection = AssetPropertyCollection()
    
    def get_asset_name(self) -> str:
        return ASSET_WorldInfo
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse world info data.
        Based on parseData in WorldInfo.cs
        """
        self.properties = AssetPropertyCollection()
        self.properties.from_stream(br, context)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save world info data.
        Based on saveData in WorldInfo.cs
        """
        self.properties.save_data(bw, context)

