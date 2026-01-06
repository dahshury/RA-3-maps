"""
LibraryMapLists asset
Based on LibraryMapLists.cs
"""
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_LibraryMapLists
from .library_maps import LibraryMaps

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class LibraryMapLists(MajorAsset):
    """
    Library map lists asset.
    Based on LibraryMapLists.cs
    """
    
    def __init__(self):
        super().__init__()
        self.library_maps: List[LibraryMaps] = []
    
    def get_asset_name(self) -> str:
        return ASSET_LibraryMapLists
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse library map lists data.
        Based on parseData in LibraryMapLists.cs
        Note: C# code doesn't show parseData, but saveData calls libraryMap.Save for each
        So we need to read until data_size is consumed, parsing LibraryMaps (which are MajorAssets)
        """
        self.library_maps = []
        while br.tell() - self.data_start_pos < self.data_size:
            library_map = LibraryMaps()
            library_map.from_stream(br, context)
            self.library_maps.append(library_map)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save library map lists data.
        Based on saveData in LibraryMapLists.cs
        """
        for library_map in self.library_maps:
            library_map.save(bw, context)

