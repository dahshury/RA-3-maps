"""
ScriptCondition class
Based on ScriptCondition.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from ..scripts.script_content import ScriptContent

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ScriptCondition:
    """
    Script condition (if statement).
    Based on ScriptCondition.cs
    Note: ScriptCondition is NOT a MajorAsset, it contains a ScriptContent
    """
    
    def __init__(self):
        self.script_content = ScriptContent()
        self.is_inverted: bool = False
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ScriptCondition':
        """
        Parse script condition from stream.
        Based on fromStream in ScriptCondition.cs
        """
        self.script_content.from_stream(br, context)
        is_inverted_int = struct.unpack('<i', br.read(4))[0]
        self.is_inverted = is_inverted_int == 1
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save script condition data.
        Based on saveData in ScriptCondition.cs
        """
        self.script_content.save(bw, context)
        bw.write(struct.pack('<i', 1 if self.is_inverted else 0))

