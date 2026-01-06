"""
StandingWaterArea class
Based on StandingWaterArea.cs
"""
import struct
from typing import BinaryIO, List, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class StandingWaterArea:
    """
    Standing water area information (NOT a MajorAsset, used within StandingWaterAreas).
    Based on StandingWaterArea.cs
    """
    
    def __init__(self):
        self.id: int = 0
        self.name: str = ""
        self.uv_scroll_speed: float = 0.0
        self.additive_blending: bool = False
        self.bumpmap_texture: str = ""
        self.sky_texture: str = ""
        self.points: List[Tuple[float, float]] = []  # Vec2D array
        self.water_height: int = 0
        self.fx_shader: str = ""
        self.depth_colors: str = ""
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'StandingWaterArea':
        """
        Parse standing water area from stream.
        Based on fromStream in StandingWaterArea.cs
        """
        self.id = struct.unpack('<i', br.read(4))[0]
        self.name = BinaryUtils.read_string_default(br)
        br.read(2)  # ReadInt16() - skip 2 bytes (unused)
        self.uv_scroll_speed = struct.unpack('<f', br.read(4))[0]
        self.additive_blending = struct.unpack('?', br.read(1))[0]
        self.bumpmap_texture = BinaryUtils.read_string_default(br)
        self.sky_texture = BinaryUtils.read_string_default(br)
        points_count = struct.unpack('<i', br.read(4))[0]
        self.points = []
        for i in range(points_count):
            # Vec2D: two floats (x, y)
            x = struct.unpack('<f', br.read(4))[0]
            y = struct.unpack('<f', br.read(4))[0]
            self.points.append((x, y))
        self.water_height = struct.unpack('<i', br.read(4))[0]
        self.fx_shader = BinaryUtils.read_string_default(br)
        self.depth_colors = BinaryUtils.read_string_default(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save standing water area data.
        Based on saveData in StandingWaterArea.cs
        """
        bw.write(struct.pack('<i', self.id))
        BinaryUtils.write_string_default(bw, self.name)
        bw.write(struct.pack('<h', 0))  # WriteInt16(0)
        bw.write(struct.pack('<f', self.uv_scroll_speed))
        bw.write(struct.pack('?', self.additive_blending))
        BinaryUtils.write_string_default(bw, self.bumpmap_texture)
        BinaryUtils.write_string_default(bw, self.sky_texture)
        bw.write(struct.pack('<i', len(self.points)))
        for x, y in self.points:
            bw.write(struct.pack('<f', x))
            bw.write(struct.pack('<f', y))
        bw.write(struct.pack('<i', self.water_height))
        BinaryUtils.write_string_default(bw, self.fx_shader)
        BinaryUtils.write_string_default(bw, self.depth_colors)
