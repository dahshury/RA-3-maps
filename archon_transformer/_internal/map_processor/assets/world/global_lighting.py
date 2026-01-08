"""
GlobalLighting asset
Based on GlobalLighting.cs
"""
import struct
from typing import BinaryIO, List, Tuple, TYPE_CHECKING

from ...core.major_asset import MajorAsset
from ...utils.constants import ASSET_GlobalLighting
from .global_lighting_configuration import GlobalLightingConfiguration

if TYPE_CHECKING:
    from ...core.ra3map_struct import MapDataContext


class GlobalLighting(MajorAsset):
    """
    Global lighting asset.
    Based on GlobalLighting.cs
    """
    
    def __init__(self):
        super().__init__()
        self.time: int = 0
        self.lighting_configurations: List[GlobalLightingConfiguration] = []  # Array of 4
        self.shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 0)  # MapColorArgb: (A, R, G, B)
        self.no_cloud_factor: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # ColorRgbF: (R, G, B)
    
    def get_asset_name(self) -> str:
        return ASSET_GlobalLighting
    
    def parse_data(self, br: BinaryIO, context: 'MapDataContext') -> None:
        """
        Parse global lighting data.
        Based on parseData in GlobalLighting.cs
        """
        self.time = struct.unpack('<i', br.read(4))[0]
        self.lighting_configurations = []
        for i in range(4):
            config = GlobalLightingConfiguration()
            config.from_stream(br, context)
            self.lighting_configurations.append(config)
        
        # Read MapColorArgb (packed as uint32: ARGB)
        argb_value = struct.unpack('<I', br.read(4))[0]
        self.shadow_color = (
            (argb_value >> 24) & 0xFF,  # A
            (argb_value >> 16) & 0xFF,  # R
            (argb_value >> 8) & 0xFF,   # G
            argb_value & 0xFF           # B
        )
        
        # Read ColorRgbF (3 floats: R, G, B)
        r = struct.unpack('<f', br.read(4))[0]
        g = struct.unpack('<f', br.read(4))[0]
        b = struct.unpack('<f', br.read(4))[0]
        self.no_cloud_factor = (r, g, b)
    
    def save_data(self, bw: BinaryIO, context: 'MapDataContext') -> None:
        """
        Save global lighting data.
        Based on saveData in GlobalLighting.cs
        """
        bw.write(struct.pack('<i', self.time))
        for config in self.lighting_configurations:
            config.save_data(bw, context)
        
        # Write MapColorArgb (packed as uint32: ARGB)
        a, r, g, b = self.shadow_color
        argb_value = (a << 24) | (r << 16) | (g << 8) | b
        bw.write(struct.pack('<I', argb_value))
        
        # Write ColorRgbF (3 floats: R, G, B)
        r, g, b = self.no_cloud_factor
        bw.write(struct.pack('<f', r))
        bw.write(struct.pack('<f', g))
        bw.write(struct.pack('<f', b))

