"""
ObjectsList asset
Based on ObjectsList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_ObjectsList
from ..objects.map_object import MapObject

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ObjectsList(MajorAsset):
    """
    List of map objects.
    Based on ObjectsList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.map_objects: List[MapObject] = []
    
    def get_asset_name(self) -> str:
        return ASSET_ObjectsList
    
    def get_version(self) -> int:
        return 3
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ObjectsList':
        """
        Parse objects list from stream.
        Based on fromStream in ObjectsList.cs
        """
        # Call base to read header (sets self.data_start_pos)
        super().from_stream(br, context)
        
        # Read objects until we've consumed all data
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        while br.tell() - self.data_start_pos < self.data_size:
            obj = MapObject()
            obj.from_stream(br, context)
            self.map_objects.append(obj)
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save objects list data.
        Based on saveData in ObjectsList.cs
        """
        for obj in self.map_objects:
            obj.save(bw, context)

