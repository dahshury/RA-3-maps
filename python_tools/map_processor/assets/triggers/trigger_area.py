"""
TriggerArea class
Based on TriggerArea.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class TriggerArea:
    """
    TriggerArea (NOT a MajorAsset, used within TriggerAreas).
    Based on TriggerArea.cs from MapCreatorCore
    """
    
    count = 0  # Static counter
    
    def __init__(self):
        self.name: str = ""
        self.id: int = 0
        self.points: List[Tuple[float, float]] = []  # Vec2D array
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'TriggerArea':
        """
        Parse trigger area from stream.
        Based on TriggerArea(BinaryReader br) in TriggerArea.cs
        """
        self.name = BinaryUtils.read_string_ascii(br)  # IOUtility.ReadString
        # C# reads a short here, which is expected to be 0. We'll read and ignore.
        short_val = struct.unpack('<h', br.read(2))[0]
        if short_val != 0:
            pass  # C# logs a warning, but we'll just continue
        self.id = struct.unpack('<i', br.read(4))[0]
        points_count = struct.unpack('<i', br.read(4))[0]
        self.points = []
        for i in range(points_count):
            self.points.append(BinaryUtils.read_vec2d(br))
        # C# reads an int here, expected to be 0. We'll read and ignore.
        int_val = struct.unpack('<i', br.read(4))[0]
        if int_val != 0:
            pass  # C# logs a warning, but we'll just continue
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save trigger area data.
        Based on Save(BinaryWriter bw) in TriggerArea.cs
        """
        BinaryUtils.write_string_ascii(bw, self.name)  # IOUtility.WriteString
        bw.write(struct.pack('<h', 0))  # Expected short 0
        bw.write(struct.pack('<i', self.id))
        bw.write(struct.pack('<i', len(self.points)))
        for x, y in self.points:
            BinaryUtils.write_vec2d(bw, x, y)
        bw.write(struct.pack('<i', 0))  # Expected int 0

