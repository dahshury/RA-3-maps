"""
ScriptGroup asset
Based on ScriptGroup.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_ScriptGroup, ASSET_Script
from ...utils.binary_utils import BinaryUtils
from ..scripts.script import Script

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ScriptGroup(MajorAsset):
    """
    Group of scripts.
    Based on ScriptGroup.cs
    """
    
    def __init__(self):
        super().__init__()
        self.name: str = ""
        self.is_active: bool = True
        self.is_subroutine: bool = False
        self.scripts: List[Script] = []
        self.script_groups: List['ScriptGroup'] = []
    
    def get_asset_name(self) -> str:
        return ASSET_ScriptGroup
    
    def get_version(self) -> int:
        return 3
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'ScriptGroup':
        """
        Parse script group from stream.
        Based on fromStream in ScriptGroup.cs
        """
        super().from_stream(br, context)
        
        self.name = BinaryUtils.read_string_default(br)
        self.is_active = struct.unpack('?', br.read(1))[0]
        self.is_subroutine = struct.unpack('?', br.read(1))[0]
        
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
        Save script group data.
        Based on saveData in ScriptGroup.cs
        """
        BinaryUtils.write_string_default(bw, self.name)
        bw.write(struct.pack('?', self.is_active))
        bw.write(struct.pack('?', self.is_subroutine))
        
        # Write scripts
        for script in self.scripts:
            script.save(bw, context)
        
        # Write script groups
        for script_group in self.script_groups:
            script_group.save(bw, context)

