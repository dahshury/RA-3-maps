"""
RiverArea class
Based on RiverArea.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class RiverArea:
    """
    River area (NOT a MajorAsset, used within RiverAreas).
    Based on RiverArea.cs from MapCreatorCore
    """
    
    def __init__(self):
        self.id: int = 0
        self.name: str = ""
        self.river_texture: str = ""
        self.normal_map: str = ""
        self.low_lod_noise_texture: str = ""
        self.low_lod_sparkle_texture: str = ""
        self.color: int = 0  # uint32
        self.alpha: float = 0.0
        self.additive_blending: bool = False
        self.water_height: int = 0
        self.uv_scroll_speed: float = 0.0
        self.river_type: str = ""
        self.minimum_water_lod: str = ""
        self.points: List[Tuple[float, float]] = []  # Vec2D array
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'RiverArea':
        """
        Parse river area from stream.
        Based on RiverArea(BinaryReader br) in RiverArea.cs
        Uses IOUtility.ReadString which is read_string_ascii (ushort length + ASCII)
        """
        self.id = struct.unpack('<i', br.read(4))[0]
        self.name = BinaryUtils.read_string_ascii(br)
        # Read and check short 0
        short_val = struct.unpack('<h', br.read(2))[0]
        if short_val != 0:
            pass  # Expected 0, but continue
        self.uv_scroll_speed = struct.unpack('<f', br.read(4))[0]
        self.additive_blending = struct.unpack('<?', br.read(1))[0]
        self.river_texture = BinaryUtils.read_string_ascii(br)
        self.normal_map = BinaryUtils.read_string_ascii(br)
        self.low_lod_noise_texture = BinaryUtils.read_string_ascii(br)
        self.low_lod_sparkle_texture = BinaryUtils.read_string_ascii(br)
        self.color = struct.unpack('<I', br.read(4))[0]
        self.alpha = struct.unpack('<f', br.read(4))[0] * 255.0
        self.water_height = struct.unpack('<i', br.read(4))[0]
        self.river_type = BinaryUtils.read_string_ascii(br)
        self.minimum_water_lod = BinaryUtils.read_string_ascii(br)
        points_count = struct.unpack('<i', br.read(4))[0]
        # Note: C# code creates array of size points_count * 2, then reads Vec2D into it
        # This means we read points_count Vec2D pairs
        self.points = []
        for i in range(points_count * 2):
            x, y = BinaryUtils.read_vec2d(br)
            self.points.append((x, y))
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save river area data.
        Based on Save(BinaryWriter bw) in RiverArea.cs
        """
        bw.write(struct.pack('<i', self.id))
        BinaryUtils.write_string_ascii(bw, self.name)
        bw.write(struct.pack('<h', 0))  # short 0
        bw.write(struct.pack('<f', self.uv_scroll_speed))
        bw.write(struct.pack('<?', self.additive_blending))
        BinaryUtils.write_string_ascii(bw, self.river_texture)
        BinaryUtils.write_string_ascii(bw, self.normal_map)
        BinaryUtils.write_string_ascii(bw, self.low_lod_noise_texture)
        BinaryUtils.write_string_ascii(bw, self.low_lod_sparkle_texture)
        bw.write(struct.pack('<I', self.color))
        bw.write(struct.pack('<f', self.alpha / 255.0))
        bw.write(struct.pack('<i', self.water_height))
        BinaryUtils.write_string_ascii(bw, self.river_type)
        BinaryUtils.write_string_ascii(bw, self.minimum_water_lod)
        bw.write(struct.pack('<i', len(self.points) // 2))  # points_count
        for x, y in self.points:
            BinaryUtils.write_vec2d(bw, x, y)

