"""
PostEffect class
Based on PostEffect.cs
"""
import struct
from typing import Any, BinaryIO, List, TYPE_CHECKING

from ...utils.binary_utils import BinaryUtils

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class PostEffectParameter:
    """
    Post effect parameter (internal class).
    Based on Parameter struct in PostEffect.cs
    """
    
    def __init__(self, name: str = "", param_type: str = "", data: Any = None):
        self.name: str = name
        self.param_type: str = param_type
        self.data: Any = data
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'PostEffectParameter':
        """
        Parse parameter from stream.
        Based on Parameter(BinaryReader br) in PostEffect.cs
        """
        self.name = BinaryUtils.read_string_default(br)
        self.param_type = BinaryUtils.read_string_default(br)
        
        if self.param_type == "Float":
            self.data = struct.unpack('<f', br.read(4))[0]
        elif self.param_type == "Float4":
            self.data = [
                struct.unpack('<f', br.read(4))[0],
                struct.unpack('<f', br.read(4))[0],
                struct.unpack('<f', br.read(4))[0],
                struct.unpack('<f', br.read(4))[0]
            ]
        elif self.param_type == "Texture":
            self.data = BinaryUtils.read_string_default(br)
        elif self.param_type == "Int":
            self.data = struct.unpack('<i', br.read(4))[0]
        else:
            self.data = None
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save parameter data.
        Based on Save(BinaryWriter bw) in PostEffect.cs
        """
        BinaryUtils.write_string_default(bw, self.name)
        BinaryUtils.write_string_default(bw, self.param_type)
        
        if self.param_type == "Float":
            bw.write(struct.pack('<f', self.data))
        elif self.param_type == "Float4":
            for val in self.data:
                bw.write(struct.pack('<f', val))
        elif self.param_type == "Texture":
            BinaryUtils.write_string_default(bw, self.data)
        elif self.param_type == "Int":
            bw.write(struct.pack('<i', self.data))


class PostEffect:
    """
    Post-processing effect (NOT a MajorAsset, used within PostEffectsChunk).
    Based on PostEffect.cs
    """
    
    def __init__(self, name: str = "", parameters: List[PostEffectParameter] = None):
        self.name: str = name
        self.parameters: List[PostEffectParameter] = parameters or []
    
    def from_stream(self, br: BinaryIO, context: 'MapDataContext') -> 'PostEffect':
        """
        Parse post effect from stream.
        Based on PostEffect(BinaryReader br) in PostEffect.cs
        """
        self.name = BinaryUtils.read_string_default(br)
        param_count = struct.unpack('<i', br.read(4))[0]
        self.parameters = []
        for i in range(param_count):
            param = PostEffectParameter()
            param.from_stream(br, context)
            self.parameters.append(param)
        return self
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save post effect data.
        Based on Save(BinaryWriter bw) in PostEffect.cs
        """
        BinaryUtils.write_string_default(bw, self.name)
        bw.write(struct.pack('<i', len(self.parameters)))
        for param in self.parameters:
            param.save_data(bw, context)
