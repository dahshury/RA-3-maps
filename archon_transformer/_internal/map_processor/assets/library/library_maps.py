"""
LibraryMaps asset
Based on LibraryMaps.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_LibraryMaps
from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class LibraryMaps(MajorAsset):
    """
    Library maps asset.
    Based on LibraryMaps.cs
    """
    
    def __init__(self):
        super().__init__()
        self.library_maps: List[str] = []
    
    def get_asset_name(self) -> str:
        return ASSET_LibraryMaps
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse library maps data.
        Based on parseData in LibraryMaps.cs
        Note: C# code doesn't show parseData, but saveData writes count then strings
        """
        library_maps_count = struct.unpack('<i', br.read(4))[0]
        self.library_maps = []
        for i in range(library_maps_count):
            self.library_maps.append(BinaryUtils.read_string_default(br))
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save library maps data.
        Based on saveData in LibraryMaps.cs
        """
        bw.write(struct.pack('<i', len(self.library_maps)))
        for library_map in self.library_maps:
            BinaryUtils.write_string_default(bw, library_map)

