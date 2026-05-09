"""
MissionObjective class
Based on MissionObjective.cs from MapCreatorCore
"""
import struct
from enum import IntEnum
from typing import BinaryIO, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ObjectiveType(IntEnum):
    """
    Mission objective type.
    Based on MissionObjective.ObjectiveType enum in MissionObjective.cs
    """
    Attack = 0
    Build = 3
    Capture = 4
    Move = 5
    Protect = 6


class MissionObjective:
    """
    MissionObjective (NOT a MajorAsset, used within MissionObjectives).
    Based on MissionObjective.cs from MapCreatorCore
    """
    
    count = 0  # Static counter
    
    def __init__(self):
        self.id: str = ""
        self.text: str = ""
        self.description: str = ""
        self.bonus_objective: bool = False
        self.objective_type: ObjectiveType = ObjectiveType.Attack
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'MissionObjective':
        """
        Parse mission objective from stream.
        Based on MissionObjective(BinaryReader br) in MissionObjective.cs
        """
        self.id = BinaryUtils.read_string_ascii(br)  # IOUtility.ReadString
        self.text = BinaryUtils.read_string_ascii(br)  # IOUtility.ReadString
        self.description = BinaryUtils.read_string_ascii(br)  # IOUtility.ReadString
        self.bonus_objective = struct.unpack('?', br.read(1))[0]  # ReadBoolean
        self.objective_type = ObjectiveType(struct.unpack('<i', br.read(4))[0])
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save mission objective data.
        Based on Save(BinaryWriter bw) in MissionObjective.cs
        """
        BinaryUtils.write_string_ascii(bw, self.id)  # IOUtility.WriteString
        BinaryUtils.write_string_ascii(bw, self.text)  # IOUtility.WriteString
        BinaryUtils.write_string_ascii(bw, self.description)  # IOUtility.WriteString
        bw.write(struct.pack('?', self.bonus_objective))  # Write(bool)
        bw.write(struct.pack('<i', self.objective_type.value))  # Write((int)objectiveType)

