"""
MapObject asset
Based on MapObject.cs
"""
import math
import struct
from typing import BinaryIO, TYPE_CHECKING, Tuple, Optional

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_MapObject
from ...utils.binary_utils import BinaryUtils
from ..assets.asset_property import AssetPropertyCollection

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MapObject(MajorAsset):
    """
    Map object (placed on the map).
    Based on MapObject.cs
    """
    
    def __init__(self):
        super().__init__()
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.angle: float = 0.0  # In degrees
        self.road_option: int = 0
        self.type_name: str = ""
        self.asset_property_collection = AssetPropertyCollection()
    
    def get_asset_name(self) -> str:
        return ASSET_MapObject
    
    def get_version(self) -> int:
        return 3
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse map object data.
        Based on parseData in MapObject.cs
        """
        # Read Vec3D position
        self.position = BinaryUtils.read_vec3d(br)
        
        # Read angle (in radians, convert to degrees)
        angle_rad = struct.unpack('<f', br.read(4))[0]
        self.angle = angle_rad * 180.0 / math.pi
        
        # Read road option
        self.road_option = struct.unpack('<i', br.read(4))[0]
        
        # Read type name (default string)
        self.type_name = BinaryUtils.read_string_default(br)
        
        # Read property collection
        self.asset_property_collection.from_stream(br, context)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save map object data.
        Based on saveData in MapObject.cs
        """
        # Write Vec3D position
        BinaryUtils.write_vec3d(bw, self.position[0], self.position[1], self.position[2])
        
        # Write angle (convert degrees to radians)
        angle_rad = self.angle * math.pi / 180.0
        bw.write(struct.pack('<f', angle_rad))
        
        # Write road option
        bw.write(struct.pack('<i', self.road_option))
        
        # Write type name
        BinaryUtils.write_string_default(bw, self.type_name)
        
        # Write property collection
        self.asset_property_collection.save_data(bw, context)
    
    @property
    def original_owner(self) -> Optional[str]:
        """Get original owner property"""
        prop = self.asset_property_collection.get_property("originalOwner")
        return str(prop.data) if prop and prop.data else None
    
    @property
    def unique_id(self) -> Optional[str]:
        """Get unique ID property"""
        prop = self.asset_property_collection.get_property("uniqueID")
        return str(prop.data) if prop and prop.data else None

