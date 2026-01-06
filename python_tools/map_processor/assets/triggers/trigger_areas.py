"""
TriggerAreas asset
Based on TriggerAreas.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_TriggerAreas
from .trigger_area import TriggerArea

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class TriggerAreas(MajorAsset):
    """
    TriggerAreas asset.
    Based on TriggerAreas.cs from MapCreatorCore
    Note: The C# implementation has parsing commented out and just skips data,
    but TriggerArea has parsing implemented, so we implement the parsing here.
    """
    
    def __init__(self):
        super().__init__()
        self.areas: List[TriggerArea] = []
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse asset-specific data.
        Based on TriggerAreas(BinaryReader br) in TriggerAreas.cs
        Note: The C# implementation has this commented out and just skips to the end,
        but since TriggerArea has parsing, we implement it here.
        """
        count = struct.unpack('<i', br.read(4))[0]
        self.areas = []
        for i in range(count):
            area = TriggerArea()
            area.from_stream(br, context)
            self.areas.append(area)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset-specific data.
        Based on SaveData(BinaryWriter bw) in TriggerAreas.cs
        """
        bw.write(struct.pack('<i', len(self.areas)))
        for area in self.areas:
            area.save_data(bw, context)
    
    def get_asset_name(self) -> str:
        return ASSET_TriggerAreas
    
    def get_version(self) -> int:
        return 1

