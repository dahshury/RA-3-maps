"""
PlayerScriptsList asset
Based on PlayerScriptsList.cs
"""
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_PlayerScriptsList
from ..scripts.script_list import ScriptList

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class PlayerScriptsList(MajorAsset):
    """
    List of script lists (one per player).
    Based on PlayerScriptsList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.script_lists: List[ScriptList] = []
    
    def get_asset_name(self) -> str:
        return ASSET_PlayerScriptsList
    
    def get_version(self) -> int:
        return 1
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'PlayerScriptsList':
        """
        Parse player scripts list from stream.
        Based on fromStream in PlayerScriptsList.cs
        """
        super().from_stream(br, context)
        
        # Read script lists until we've consumed all data
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        while br.tell() - self.data_start_pos < self.data_size:
            script_list = ScriptList()
            script_list.from_stream(br, context)
            self.script_lists.append(script_list)
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save player scripts list data.
        Based on saveData in PlayerScriptsList.cs
        """
        for script_list in self.script_lists:
            script_list.save(bw, context)

