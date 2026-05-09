"""
ScriptArgument class
Based on ScriptArgument.cs
"""
import struct
from typing import BinaryIO, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ScriptArgument:
    """
    Script argument (parameter).
    Based on ScriptArgument.cs
    """
    
    def __init__(self):
        self.argument_type: int = 0
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.int_value: int = 0
        self.float_value: float = 0.0
        self.string_value: str = ""
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ScriptArgument':
        """
        Parse script argument from stream.
        Based on fromStream in ScriptArgument.cs
        """
        self.argument_type = struct.unpack('<I', br.read(4))[0]
        if self.argument_type == 16:
            # Position coordinate
            self.position = BinaryUtils.read_vec3d(br)
        else:
            self.int_value = struct.unpack('<i', br.read(4))[0]
            self.float_value = struct.unpack('<f', br.read(4))[0]
            self.string_value = BinaryUtils.read_string_default(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save script argument data.
        Based on saveData in ScriptArgument.cs
        """
        bw.write(struct.pack('<I', self.argument_type))
        if self.argument_type == 16:
            BinaryUtils.write_vec3d(bw, self.position[0], self.position[1], self.position[2])
        else:
            bw.write(struct.pack('<i', self.int_value))
            bw.write(struct.pack('<f', self.float_value))
            BinaryUtils.write_string_default(bw, self.string_value)

