"""
Team class
Based on Team.cs
"""
from typing import BinaryIO, TYPE_CHECKING

from ..assets.asset_property import AssetPropertyCollection

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Team:
    """
    Team definition (NOT a MajorAsset, used within Teams list).
    Based on Team.cs
    """
    
    def __init__(self):
        self.property_collection = AssetPropertyCollection()
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'Team':
        """
        Parse team from stream.
        Based on fromStream in Team.cs
        """
        self.property_collection = AssetPropertyCollection()
        self.property_collection.from_stream(br, context)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save team data.
        Based on saveData in Team.cs
        """
        self.property_collection.save_data(bw, context)

