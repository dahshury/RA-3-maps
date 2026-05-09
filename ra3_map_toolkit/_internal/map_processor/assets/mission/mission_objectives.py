"""
MissionObjectives asset
Based on MissionObjectives.cs from MapCreatorCore
"""
import struct
from typing import BinaryIO, Dict, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_MissionObjectives
from .mission_objective import MissionObjective

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class MissionObjectives(MajorAsset):
    """
    MissionObjectives asset.
    Based on MissionObjectives.cs from MapCreatorCore
    Note: The C# implementation has parsing commented out and just skips data,
    but MissionObjective has parsing implemented, so we implement the parsing here.
    """
    
    def __init__(self):
        super().__init__()
        self.objectives: Dict[str, MissionObjective] = {}
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse asset-specific data.
        Based on MissionObjectives(BinaryReader br) in MissionObjectives.cs
        Note: The C# implementation has this commented out and just skips to the end,
        but since MissionObjective has parsing, we implement it here.
        """
        count = struct.unpack('<i', br.read(4))[0]
        self.objectives = {}
        for i in range(count):
            objective = MissionObjective()
            objective.from_stream(br, context)
            self.objectives[objective.id] = objective  # Use id as key (dictionary in C#)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset-specific data.
        Based on SaveData(BinaryWriter bw) in MissionObjectives.cs
        """
        bw.write(struct.pack('<i', len(self.objectives)))
        for objective in self.objectives.values():
            objective.save_data(bw, context)
    
    def get_asset_name(self) -> str:
        return ASSET_MissionObjectives
    
    def get_version(self) -> int:
        return 1

