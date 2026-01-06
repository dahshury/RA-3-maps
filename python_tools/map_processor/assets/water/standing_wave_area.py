"""
StandingWaveArea class
Based on StandingWaveArea.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, Tuple, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class StandingWaveArea:
    """
    Standing wave area (NOT a MajorAsset, used within StandingWaveAreas).
    Based on StandingWaveArea.cs from MapCreatorCore
    """
    
    def __init__(self):
        self.id: int = 0
        self.name: str = ""
        self.particle_effect: str = ""
        self.uv_scroll_speed: float = 0.0
        self.additive_blending: bool = False
        self.points: List[Tuple[float, float]] = []  # Vec2D array
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'StandingWaveArea':
        """
        Parse standing wave area from stream.
        Based on StandingWaveArea(BinaryReader br) in StandingWaveArea.cs
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
        points_count = struct.unpack('<i', br.read(4))[0]
        self.points = []
        for i in range(points_count):
            x, y = BinaryUtils.read_vec2d(br)
            self.points.append((x, y))
        # Read and check int 0
        int_val = struct.unpack('<i', br.read(4))[0]
        if int_val != 0:
            pass  # Expected 0, but continue
        self.particle_effect = BinaryUtils.read_string_ascii(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save standing wave area data.
        Based on Save(BinaryWriter bw) in StandingWaveArea.cs
        """
        bw.write(struct.pack('<i', self.id))
        BinaryUtils.write_string_ascii(bw, self.name)
        bw.write(struct.pack('<h', 0))  # short 0
        bw.write(struct.pack('<f', self.uv_scroll_speed))
        bw.write(struct.pack('<?', self.additive_blending))
        bw.write(struct.pack('<i', len(self.points)))
        for x, y in self.points:
            BinaryUtils.write_vec2d(bw, x, y)
        bw.write(struct.pack('<i', 0))  # int 0
        BinaryUtils.write_string_ascii(bw, self.particle_effect)

