"""
PostEffectsChunk asset
Based on PostEffectsChunk.cs
"""
from typing import BinaryIO, List, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_PostEffectsChunk
from .post_effect import PostEffect

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class PostEffectsChunk(MajorAsset):
    """
    Post effects chunk asset.
    Based on PostEffectsChunk.cs
    """
    
    def __init__(self):
        super().__init__()
        self.effects: List[PostEffect] = []
    
    def get_asset_name(self) -> str:
        return ASSET_PostEffectsChunk
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse post effects chunk data.
        Based on parseData in PostEffectsChunk.cs
        Note: C# code doesn't show parseData, but saveData writes effects.Count then effects
        """
        import struct
        effects_count = struct.unpack('<i', br.read(4))[0]
        self.effects = []
        for i in range(effects_count):
            effect = PostEffect()
            effect.from_stream(br, context)
            self.effects.append(effect)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save post effects chunk data.
        Based on saveData in PostEffectsChunk.cs
        """
        import struct
        bw.write(struct.pack('<i', len(self.effects)))
        for effect in self.effects:
            effect.save_data(bw, context)

