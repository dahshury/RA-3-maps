"""
OrCondition asset
Based on OrCondition.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_OrCondition
from ..scripts.condition import Condition

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class OrCondition(MajorAsset):
    """
    OR condition (group of conditions).
    Based on OrCondition.cs
    """
    
    def __init__(self):
        super().__init__()
        self.conditions: List[Condition] = []
    
    def get_asset_name(self) -> str:
        return ASSET_OrCondition
    
    def get_version(self) -> int:
        return 1
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'OrCondition':
        """
        Parse OR condition from stream.
        Based on fromStream in OrCondition.cs
        """
        super().from_stream(br, context)
        
        # Read Condition assets until we've consumed all data
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        while br.tell() - self.data_start_pos < self.data_size:
            condition = Condition()
            condition.from_stream(br, context)
            self.conditions.append(condition)
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save OR condition data.
        Based on saveData in OrCondition.cs
        """
        for condition in self.conditions:
            condition.save(bw, context)  # Use save(), not save_data(), because Condition is a MajorAsset

