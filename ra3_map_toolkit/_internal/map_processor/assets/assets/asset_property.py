"""
Asset Property classes
Based on AssetProperty.cs and AssetPropertyCollection.cs
"""
import struct
from typing import BinaryIO, Dict, Optional, TYPE_CHECKING
from enum import IntEnum

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class AssetPropertyType(IntEnum):
    """Property types for asset properties"""
    bool_type = 0
    int_type = 1
    float_type = 2
    string_type = 3
    string_unicode_type = 4
    string_name_value_type = 5
    unknown_type = 255


class AssetProperty:
    """
    Asset property with type and value.
    Based on AssetProperty.cs
    """
    
    def __init__(self):
        self.property_type: AssetPropertyType = AssetPropertyType.unknown_type
        self.id: int = 0
        self.name: str = ""
        self.data: Optional[object] = None
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'AssetProperty':
        """
        Parse property from binary stream.
        Based on fromStream in AssetProperty.cs
        """
        self.property_type = AssetPropertyType(struct.unpack('B', br.read(1))[0])
        self.id = BinaryUtils.read_uint24(br)
        self.name = context.map_struct.find_string_by_index(self.id) or ""
        
        if self.property_type == AssetPropertyType.bool_type:
            self.data = struct.unpack('?', br.read(1))[0]
        elif self.property_type == AssetPropertyType.int_type:
            self.data = struct.unpack('<i', br.read(4))[0]
        elif self.property_type == AssetPropertyType.float_type:
            self.data = struct.unpack('<f', br.read(4))[0]
        elif self.property_type == AssetPropertyType.string_type:
            self.data = BinaryUtils.read_string_default(br)
        elif self.property_type == AssetPropertyType.string_unicode_type:
            self.data = BinaryUtils.read_unicode_string(br)
        elif self.property_type == AssetPropertyType.string_name_value_type:
            self.data = BinaryUtils.read_string_default(br)
        else:
            # Unknown type - skip for now
            pass
        
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save property to binary stream.
        Based on saveData in AssetProperty.cs
        """
        bw.write(struct.pack('B', self.property_type))
        BinaryUtils.write_uint24(bw, self.id)
        
        if self.property_type == AssetPropertyType.bool_type:
            bw.write(struct.pack('?', bool(self.data)))
        elif self.property_type == AssetPropertyType.int_type:
            bw.write(struct.pack('<i', int(self.data)))
        elif self.property_type == AssetPropertyType.float_type:
            bw.write(struct.pack('<f', float(self.data)))
        elif self.property_type == AssetPropertyType.string_type:
            BinaryUtils.write_string_default(bw, str(self.data))
        elif self.property_type == AssetPropertyType.string_unicode_type:
            BinaryUtils.write_unicode_string(bw, str(self.data))
        elif self.property_type == AssetPropertyType.string_name_value_type:
            BinaryUtils.write_string_default(bw, str(self.data))
    
    @staticmethod
    def of(name: str, data: object, context: 'MapDataContext') -> 'AssetProperty':
        """
        Create a new AssetProperty.
        Based on of method in AssetProperty.cs
        """
        prop = AssetProperty()
        prop.data = data
        prop.name = name
        
        if isinstance(data, bool):
            prop.property_type = AssetPropertyType.bool_type
        elif isinstance(data, int):
            prop.property_type = AssetPropertyType.int_type
        elif isinstance(data, float):
            prop.property_type = AssetPropertyType.float_type
        elif isinstance(data, str):
            if name == "playerDisplayName":
                prop.property_type = AssetPropertyType.string_unicode_type
            else:
                prop.property_type = AssetPropertyType.string_type
        elif isinstance(data, list):
            prop.property_type = AssetPropertyType.string_name_value_type
        else:
            prop.property_type = AssetPropertyType.int_type
        
        prop.id = context.map_struct.register_string(name)
        return prop


class AssetPropertyCollection:
    """
    Collection of asset properties.
    Based on AssetPropertyCollection.cs
    """
    
    def __init__(self):
        self.property_map: Dict[str, AssetProperty] = {}
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'AssetPropertyCollection':
        """
        Parse property collection from binary stream.
        Based on fromStream in AssetPropertyCollection.cs
        """
        property_count = struct.unpack('<h', br.read(2))[0]
        for i in range(property_count):
            prop = AssetProperty()
            prop.from_stream(br, context)
            self.property_map[prop.name] = prop
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save property collection to binary stream.
        Based on saveData in AssetPropertyCollection.cs
        """
        bw.write(struct.pack('<H', len(self.property_map)))
        for prop in self.property_map.values():
            prop.save_data(bw, context)
    
    def add_property(self, name: str, data: object, context: 'MapDataContext') -> None:
        """
        Add a property to the collection.
        Based on addProperty in AssetPropertyCollection.cs
        """
        if name in self.property_map:
            return
        prop = AssetProperty.of(name, data, context)
        self.property_map[name] = prop
    
    def get_property(self, name: str) -> Optional[AssetProperty]:
        """
        Get a property by name.
        Based on getProperty in AssetPropertyCollection.cs
        """
        return self.property_map.get(name)
    
    def set_property(self, name: str, data: object) -> None:
        """
        Set a property's data value.
        Based on setProperty in AssetPropertyCollection.cs
        """
        if name not in self.property_map:
            return
        self.property_map[name].data = data

