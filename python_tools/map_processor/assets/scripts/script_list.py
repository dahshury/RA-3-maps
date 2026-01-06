"""
ScriptList asset
Based on ScriptList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_ScriptList, ASSET_Script, ASSET_ScriptGroup
from ..scripts.script import Script
from ..scripts.script_group import ScriptGroup

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ScriptList(MajorAsset):
    """
    List of scripts for a player.
    Based on ScriptList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.scripts: List[Script] = []
        self.script_groups: List[ScriptGroup] = []
    
    def get_asset_name(self) -> str:
        return ASSET_ScriptList
    
    def get_version(self) -> int:
        return 1
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ScriptList':
        """
        Parse script list from stream.
        Based on fromStream in ScriptList.cs
        """
        super().from_stream(br, context)
        
        # Read child assets (Script, ScriptGroup)
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        while br.tell() - self.data_start_pos < self.data_size:
            asset_id_pos = br.tell()
            asset_id = struct.unpack('<i', br.read(4))[0]
            br.seek(asset_id_pos)
            asset_name = context.map_struct.find_string_by_index(asset_id)
            
            if asset_name == ASSET_Script:
                script = Script()
                script.from_stream(br, context)
                self.scripts.append(script)
            elif asset_name == ASSET_ScriptGroup:
                script_group = ScriptGroup()
                script_group.from_stream(br, context)
                self.script_groups.append(script_group)
            else:
                # Unknown asset type - skip
                break
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save script list data.
        Based on saveData in ScriptList.cs
        """
        # Write scripts
        for script in self.scripts:
            script.save(bw, context)
        
        # Write script groups
        for script_group in self.script_groups:
            script_group.save(bw, context)

