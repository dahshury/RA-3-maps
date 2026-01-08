"""
Condition asset
Based on Condition.cs - this is a MajorAsset that wraps condition content
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_Condition
from ...utils.binary_utils import BinaryUtils
from ..assets.asset_property import AssetPropertyType
from ..scripts.script_argument import ScriptArgument

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Condition(MajorAsset):
    """
    Condition asset - a MajorAsset wrapper around condition content.
    The inner data is ScriptContent-style data (but NOT wrapped in another header)
    plus an is_inverted flag.
    Based on Condition.cs
    """
    
    def __init__(self):
        super().__init__()
        # ScriptContent-like data (without header)
        self.content_type: int = 0
        self.asset_property_type: AssetPropertyType = AssetPropertyType.int_type
        self.content_name: str = ""
        self.name_index: int = 0
        self.enable: bool = True
        self.arguments: list = []
        # Condition-specific
        self.is_inverted: bool = False
    
    def get_asset_name(self) -> str:
        return ASSET_Condition
    
    def get_version(self) -> int:
        return 6  # As seen in the GT data
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'Condition':
        """
        Parse condition from stream.
        """
        super().from_stream(br, context)
        
        # Read ScriptContent-style data (no inner header!)
        self.content_type = struct.unpack('<i', br.read(4))[0]
        self.asset_property_type = AssetPropertyType(struct.unpack('B', br.read(1))[0])
        self.name_index = BinaryUtils.read_uint24(br)
        self.content_name = context.map_struct.find_string_by_index(self.name_index) or ""
        
        arg_nums = struct.unpack('<i', br.read(4))[0]
        self.arguments = []
        for i in range(arg_nums):
            arg = ScriptArgument()
            arg.from_stream(br, context)
            self.arguments.append(arg)
        
        enable_int = struct.unpack('<i', br.read(4))[0]
        self.enable = enable_int == 1
        
        # Read is_inverted
        is_inverted_int = struct.unpack('<i', br.read(4))[0]
        self.is_inverted = is_inverted_int == 1
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save condition data.
        """
        # Write ScriptContent-style data (no inner header!)
        bw.write(struct.pack('<i', self.content_type))
        bw.write(struct.pack('B', self.asset_property_type))
        BinaryUtils.write_uint24(bw, self.name_index)
        bw.write(struct.pack('<i', len(self.arguments)))
        for arg in self.arguments:
            arg.save_data(bw, context)
        bw.write(struct.pack('<i', 1 if self.enable else 0))
        
        # Write is_inverted
        bw.write(struct.pack('<i', 1 if self.is_inverted else 0))


