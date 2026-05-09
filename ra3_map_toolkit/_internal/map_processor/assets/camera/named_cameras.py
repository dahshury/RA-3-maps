"""
NamedCameras asset
Based on NamedCameras.cs
"""
from typing import BinaryIO, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_NamedCameras

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class NamedCameras(MajorAsset):
    """
    Named cameras asset.
    Based on NamedCameras.cs
    Note: C# implementation has empty parseData and saveData methods
    """
    
    def get_asset_name(self) -> str:
        return ASSET_NamedCameras
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse named cameras data.
        Based on parseData in NamedCameras.cs (empty in C#)
        """
        # C# code has empty implementation - just pass through
        pass
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save named cameras data.
        Based on saveData in NamedCameras.cs (empty in C#)
        """
        # C# code has empty implementation - just pass through
        pass

