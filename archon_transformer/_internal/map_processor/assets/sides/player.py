"""
Player class
Based on Player.cs
"""
from typing import BinaryIO, List, TYPE_CHECKING

from ..assets.asset_property import AssetPropertyCollection
from ..build.build_list_item import BuildListItem

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Player:
    """
    Player definition.
    Based on Player.cs
    """
    
    def __init__(self):
        self.asset_property_collection = AssetPropertyCollection()
        self.build_list_items: List[BuildListItem] = []
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'Player':
        """
        Parse player from stream.
        Based on fromStream in Player.cs
        """
        self.asset_property_collection.from_stream(br, context)
        
        import struct
        build_list_item_count = struct.unpack('<i', br.read(4))[0]
        self.build_list_items = []
        for i in range(build_list_item_count):
            item = BuildListItem()
            item.from_stream(br, context)
            self.build_list_items.append(item)
        
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save player data.
        Based on saveData in Player.cs
        """
        self.asset_property_collection.save_data(bw, context)
        
        import struct
        bw.write(struct.pack('<i', len(self.build_list_items)))
        for item in self.build_list_items:
            item.save_data(bw, context)

