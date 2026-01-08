"""
ScriptAction classes
Based on ScriptAction.cs and ScriptActionFalse.cs
"""
from typing import BinaryIO, TYPE_CHECKING

from ..scripts.script_content import ScriptContent
from ...utils.constants import ASSET_ScriptAction, ASSET_ScriptActionFalse

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class ScriptAction(ScriptContent):
    """
    Script action (what to do).
    Based on ScriptAction.cs
    """
    
    def get_asset_name(self) -> str:
        return ASSET_ScriptAction
    
    def get_version(self) -> int:
        return 3


class ScriptActionFalse(ScriptAction):
    """
    Script action for false branch.
    Based on ScriptActionFalse.cs
    """
    
    def get_asset_name(self) -> str:
        return ASSET_ScriptActionFalse

