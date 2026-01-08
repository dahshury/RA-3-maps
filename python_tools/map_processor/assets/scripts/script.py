"""
Script asset
Based on Script.cs
"""
import struct
from typing import BinaryIO, List, Tuple, Union, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_Script, ASSET_OrCondition, ASSET_ScriptAction, ASSET_ScriptActionFalse
from ...utils.binary_utils import BinaryUtils
from ..scripts.or_condition import OrCondition
from ..scripts.script_action import ScriptAction, ScriptActionFalse

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class Script(MajorAsset):
    """
    Script definition.
    Based on Script.cs
    """
    
    def __init__(self):
        super().__init__()
        self.name: str = ""
        self.comment: str = ""
        self.condition_comment: str = ""
        self.action_comment: str = ""
        self.is_subroutine: bool = False
        self.is_active: bool = True
        self.deactivate_upon_success: bool = True
        self.active_in_easy: bool = True
        self.active_in_medium: bool = True
        self.active_in_hard: bool = True
        self.evaluation_interval: int = 0
        self.actions_fire_sequentially: bool = False
        self.loop_actions: bool = False
        self.loop_count: int = 0
        self.sequential_target_type: int = 1
        self.sequential_target_name: str = ""
        self.unknown: str = ""
        self.script_or_conditions: List[OrCondition] = []
        self.script_action_on_true: List[ScriptAction] = []
        self.script_action_on_false: List[ScriptActionFalse] = []
        # Preserve original order of child assets for bit-perfect serialization
        # Each entry is a tuple: ('condition'|'action_true'|'action_false', index_in_list)
        self._child_order: List[Tuple[str, int]] = []
    
    def get_asset_name(self) -> str:
        return ASSET_Script
    
    def get_version(self) -> int:
        return 4
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'Script':
        """
        Parse script from stream.
        Based on fromStream in Script.cs
        """
        super().from_stream(br, context)
        
        self.name = BinaryUtils.read_string_default(br)
        self.comment = BinaryUtils.read_string_default(br)
        self.condition_comment = BinaryUtils.read_string_default(br)
        self.action_comment = BinaryUtils.read_string_default(br)
        
        self.is_active = struct.unpack('?', br.read(1))[0]
        self.deactivate_upon_success = struct.unpack('?', br.read(1))[0]
        self.active_in_easy = struct.unpack('?', br.read(1))[0]
        self.active_in_medium = struct.unpack('?', br.read(1))[0]
        self.active_in_hard = struct.unpack('?', br.read(1))[0]
        self.is_subroutine = struct.unpack('?', br.read(1))[0]
        
        self.evaluation_interval = struct.unpack('<i', br.read(4))[0]
        self.actions_fire_sequentially = struct.unpack('?', br.read(1))[0]
        self.loop_actions = struct.unpack('?', br.read(1))[0]
        self.loop_count = struct.unpack('<i', br.read(4))[0]
        self.sequential_target_type = struct.unpack('B', br.read(1))[0]
        self.sequential_target_name = BinaryUtils.read_string_default(br)
        self.unknown = BinaryUtils.read_string_default(br)
        
        # Read child assets (OrCondition, ScriptAction, ScriptActionFalse)
        # data_start_pos is set by base.from_stream() - use self.data_start_pos
        # Track original order for bit-perfect serialization
        self._child_order = []
        while br.tell() - self.data_start_pos < self.data_size:
            asset_id_pos = br.tell()
            asset_id = struct.unpack('<i', br.read(4))[0]
            br.seek(asset_id_pos)
            asset_name = context.map_struct.find_string_by_index(asset_id)
            
            if asset_name == ASSET_OrCondition:
                condition = OrCondition()
                condition.from_stream(br, context)
                self._child_order.append(('condition', len(self.script_or_conditions)))
                self.script_or_conditions.append(condition)
            elif asset_name == ASSET_ScriptAction:
                action = ScriptAction()
                action.from_stream(br, context)
                self._child_order.append(('action_true', len(self.script_action_on_true)))
                self.script_action_on_true.append(action)
            elif asset_name == ASSET_ScriptActionFalse:
                action_false = ScriptActionFalse()
                action_false.from_stream(br, context)
                self._child_order.append(('action_false', len(self.script_action_on_false)))
                self.script_action_on_false.append(action_false)
            else:
                # Unknown asset type - skip
                break
        
        return self
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Not used - parsing handled in from_stream override"""
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save script data.
        Based on saveData in Script.cs
        """
        BinaryUtils.write_string_default(bw, self.name)
        BinaryUtils.write_string_default(bw, self.comment)
        BinaryUtils.write_string_default(bw, self.condition_comment)
        BinaryUtils.write_string_default(bw, self.action_comment)
        
        bw.write(struct.pack('?', self.is_active))
        bw.write(struct.pack('?', self.deactivate_upon_success))
        bw.write(struct.pack('?', self.active_in_easy))
        bw.write(struct.pack('?', self.active_in_medium))
        bw.write(struct.pack('?', self.active_in_hard))
        bw.write(struct.pack('?', self.is_subroutine))
        
        bw.write(struct.pack('<i', self.evaluation_interval))
        bw.write(struct.pack('?', self.actions_fire_sequentially))
        bw.write(struct.pack('?', self.loop_actions))
        bw.write(struct.pack('<i', self.loop_count))
        bw.write(struct.pack('B', self.sequential_target_type))
        BinaryUtils.write_string_default(bw, self.sequential_target_name)
        BinaryUtils.write_string_default(bw, self.unknown)
        
        # Write child assets in original order (for bit-perfect preservation)
        if self._child_order:
            for child_type, idx in self._child_order:
                if child_type == 'condition':
                    self.script_or_conditions[idx].save(bw, context)
                elif child_type == 'action_true':
                    self.script_action_on_true[idx].save(bw, context)
                elif child_type == 'action_false':
                    self.script_action_on_false[idx].save(bw, context)
        else:
            # Fallback: default order (for newly created scripts)
            for action in self.script_action_on_true:
                action.save(bw, context)
            for condition in self.script_or_conditions:
                condition.save(bw, context)
            for action in self.script_action_on_false:
                action.save(bw, context)

