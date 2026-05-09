"""
AssetList asset
Based on AssetList.cs
"""
import struct
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_AssetList

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class AssetBlock:
    """
    Asset block (type and instance ID).
    Based on AssetBlock class in AssetList.cs
    """
    
    def __init__(self):
        self.type_id: int = 0
        self.instance_id: int = 0
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'AssetBlock':
        """
        Parse asset block from stream.
        Based on fromStream in AssetBlock.cs
        """
        self.type_id = struct.unpack('<I', br.read(4))[0]
        self.instance_id = struct.unpack('<I', br.read(4))[0]
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset block data.
        Based on saveData in AssetBlock.cs
        """
        bw.write(struct.pack('<I', self.type_id))
        bw.write(struct.pack('<I', self.instance_id))


class AssetList(MajorAsset):
    """
    List of asset blocks.
    Based on AssetList.cs
    """
    
    def __init__(self):
        super().__init__()
        self.asset_blocks: List[AssetBlock] = []
    
    def get_asset_name(self) -> str:
        return ASSET_AssetList
    
    def get_version(self) -> int:
        return 1
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse asset list data.
        Based on parseData in AssetList.cs
        """
        asset_block_count = struct.unpack('<i', br.read(4))[0]
        self.asset_blocks = []
        for i in range(asset_block_count):
            block = AssetBlock()
            block.from_stream(br, context)
            self.asset_blocks.append(block)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save asset list data.
        Based on saveData in AssetList.cs
        """
        bw.write(struct.pack('<i', len(self.asset_blocks)))
        for block in self.asset_blocks:
            block.save_data(bw, context)

