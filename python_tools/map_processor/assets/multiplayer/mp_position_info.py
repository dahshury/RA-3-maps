"""
MPPositionInfo asset
Based on MPPositionInfo.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_MPPositionInfo
from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MPPositionInfo(MajorAsset):
    """
    Multiplayer position information (MajorAsset).
    Based on MPPositionInfo.cs
    """
    
    def __init__(self):
        super().__init__()
        self.is_human: bool = False
        self.is_computer: bool = False
        self.load_ai_script: bool = False
        self.team: int = 0
        self.side_restriction: List[str] = []
    
    def get_asset_name(self) -> str:
        return ASSET_MPPositionInfo
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse MP position info data.
        Based on parseData in MPPositionInfo.cs
        """
        self.is_human = struct.unpack('?', br.read(1))[0]
        self.is_computer = struct.unpack('?', br.read(1))[0]
        self.load_ai_script = struct.unpack('?', br.read(1))[0]
        self.team = struct.unpack('<I', br.read(4))[0]
        side_restriction_count = struct.unpack('<i', br.read(4))[0]
        self.side_restriction = []
        for i in range(side_restriction_count):
            self.side_restriction.append(BinaryUtils.read_string_default(br))
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save MP position info data.
        Based on saveData in MPPositionInfo.cs
        """
        bw.write(struct.pack('?', self.is_human))
        bw.write(struct.pack('?', self.is_computer))
        bw.write(struct.pack('?', self.load_ai_script))
        bw.write(struct.pack('<I', self.team))
        bw.write(struct.pack('<i', len(self.side_restriction)))
        for side in self.side_restriction:
            BinaryUtils.write_string_default(bw, side)
