"""
Default Major Asset - for assets that aren't fully implemented
Based on DefaultMajorAsset.cs
"""
import struct
from typing import BinaryIO, TYPE_CHECKING

from .major_asset import MajorAsset

if TYPE_CHECKING:
    from .ra3map_struct import MapDataContext


class DefaultMajorAsset(MajorAsset):
    """
    Default asset that stores raw binary data for unknown asset types.
    Based on DefaultMajorAsset.cs
    """
    
    def __init__(self, asset_name: str = ""):
        super().__init__()
        self._asset_name = asset_name
        self.data: bytes = b""
    
    def get_asset_name(self) -> str:
        return self._asset_name or "UnknownAsset"
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """Store raw data"""
        # data_size was already read in from_stream
        # Read the data
        self.data = br.read(self.data_size)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """Write raw data"""
        bw.write(self.data)

