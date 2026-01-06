"""
MissionHotSpot class
Based on MissionHotSpot.cs from MapCreatorCore
"""
from typing import BinaryIO, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MissionHotSpot:
    """
    Mission hot spot (NOT a MajorAsset, used within MissionHotSpots).
    Based on MissionHotSpot.cs from MapCreatorCore
    Note: Uses IOUtility.ReadString which reads ushort length + ASCII string
    """
    
    def __init__(self):
        self.id: str = ""
        self.title: str = ""
        self.description: str = ""
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'MissionHotSpot':
        """
        Parse mission hot spot from stream.
        Based on MissionHotSpot(BinaryReader br) in MissionHotSpot.cs
        Uses IOUtility.ReadString which is read_string_ascii (ushort length + ASCII)
        """
        self.id = BinaryUtils.read_string_ascii(br)
        self.title = BinaryUtils.read_string_ascii(br)
        self.description = BinaryUtils.read_string_ascii(br)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save mission hot spot data.
        Based on Save(BinaryWriter bw) in MissionHotSpot.cs
        """
        BinaryUtils.write_string_ascii(bw, self.id)
        BinaryUtils.write_string_ascii(bw, self.title)
        BinaryUtils.write_string_ascii(bw, self.description)

